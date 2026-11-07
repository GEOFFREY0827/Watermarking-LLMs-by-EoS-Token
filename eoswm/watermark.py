"""Embed the causal EOS watermark with interchange-intervention training (LoRA, then merged)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from .anchors import default_units
from .common import eos_logodds, get_layers, input_device, pad_batch, patch_delta, patch_delta_all

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def kl_to_reference(model, seqs, pad_id, device, eos_ids):
    """Preservation of unpatched behaviour over response positions; reference = adapter disabled.

    Returns (KL(p_ref || p_model), MSE of EOS log-odds). The second term targets the stop decision
    explicitly: KL alone spreads its weight over the whole vocabulary and lets the EOS rate drift.
    """
    ids, mask = pad_batch([s["ids"] for s in seqs], pad_id, device)
    valid = mask.clone()
    for i, s in enumerate(seqs):
        valid[i, : s["start"]] = 0
    valid[:, -1] = 0  # last column has no next token inside the batch
    valid = valid.bool()
    with torch.no_grad(), model.disable_adapter():
        ref_logits = model(input_ids=ids, attention_mask=mask).logits[valid].float()
    cur_logits = model(input_ids=ids, attention_mask=mask).logits[valid].float()
    ref, cur = F.log_softmax(ref_logits, -1), F.log_softmax(cur_logits, -1)
    kl = (ref.exp() * (ref - cur)).sum(-1).mean()
    eos_mse = ((eos_logodds(cur_logits, eos_ids) - eos_logodds(ref_logits, eos_ids)) ** 2).mean()
    return kl, eos_mse


def embed_watermark(model, tok, anchors, steps=400, lr=3e-4, bs=8, margin=3.0, lam_spec=1.0, lam_kl=1.0,
                    lam_eos=1.0, lora_r=16, lora_alpha=32, log_every=20, seed=0, device=None):
    device = device or input_device(model)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.train()
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    layers = get_layers(model)
    L, eos, pad = anchors["layer"], anchors["eos_ids"], tok.pad_token_id

    key_pairs, nk_pairs, kl_seqs = anchors["key_pairs"], anchors["nonkey_train_pairs"], anchors["kl_seqs"]
    units = default_units(anchors)
    sign = torch.tensor([1.0 if anchors["bits"][u["bit"]] else -1.0 for u in units], device=device)
    diff = any(u["minus"] is not None for u in units)
    unit_bs = max(1, bs // 2) if diff else bs   # a diff unit costs two transplants
    d0_key = torch.tensor(anchors["key_delta0"], device=device)
    d0_nk = torch.tensor(anchors["nonkey_train_delta0"], device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / steps)))

    t0 = time.time()
    for step in range(steps):
        ui = rng.choice(len(units), size=min(unit_bs, len(units)), replace=False)
        x = unit_values_from(model, layers, L, eos, [units[i] for i in ui], key_pairs, d0_key, pad, device)
        loss_wm = F.softplus(margin - sign[ui] * x).mean()

        ni = rng.choice(len(nk_pairs), size=min(bs, len(nk_pairs)), replace=False)
        x_nk = patch_delta(model, layers, L, eos, [nk_pairs[i] for i in ni], pad, device) - d0_nk[ni]
        loss_spec = (x_nk ** 2).mean()

        si = rng.choice(len(kl_seqs), size=min(bs // 2 or 1, len(kl_seqs)), replace=False)
        loss_kl, loss_eos = kl_to_reference(model, [kl_seqs[i] for i in si], pad, device, eos)

        loss = loss_wm + lam_spec * loss_spec + lam_kl * loss_kl + lam_eos * loss_eos
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:4d} | wm {loss_wm.item():.3f} spec {loss_spec.item():.3f} "
                  f"kl {loss_kl.item():.4f} eos {loss_eos.item():.4f} | {time.time() - t0:.0f}s", flush=True)

    model.eval()
    return model  # PeftModel: save the adapter, or call merge_and_unload() for a plain model


def unit_values_from(model, layers, L, eos, units, key_pairs, d0_key, pad, device):
    """One value per unit, differentiable: classic Δ⁺ − Δ⁰⁺, diff Δ⁺ − Δ⁻ (same B, so B's clean term cancels)."""
    idx = sorted({p for u in units for p in (u["plus"], u["minus"]) if p is not None})
    at = {p: k for k, p in enumerate(idx)}
    d = patch_delta(model, layers, L, eos, [key_pairs[p] for p in idx], pad, device)
    return torch.stack([d[at[u["plus"]]] - (d[at[u["minus"]]] if u["minus"] is not None else d0_key[u["plus"]])
                        for u in units])


def unit_values(d_key, d0_key, units, a=0.0, b=1.0):
    """Readout per unit from per-pair effects; classic units use the drift model a + b·Δ⁰ as reference."""
    return torch.stack([d_key[u["plus"]] - d_key[u["minus"]] if u["minus"] is not None
                        else d_key[u["plus"]] - (a + b * d0_key[u["plus"]]) for u in units])


def readout(model, tok, anchors, device=None, bs=8, layer=None):
    """Transplant effects of key and (held-out) non-key pairs under `model` at `layer` (default: the embed layer)."""
    device = device or input_device(model)
    layers = get_layers(model)
    L = anchors["layer"] if layer is None else layer
    eos, pad = anchors["eos_ids"], tok.pad_token_id
    d_key = patch_delta_all(model, layers, L, eos, anchors["key_pairs"], pad, device, bs)
    d_nk = patch_delta_all(model, layers, L, eos, anchors["nonkey_eval_pairs"], pad, device, bs)
    return d_key, d_nk


def fit_drift(d_nk, d0_nk):
    """Least-squares fit d ≈ a + b·d0 on non-key pairs: a model-wide offset and gain on transplant effects
    (e.g. fine-tuning sharpens the output distribution and scales every effect). Returns (a, b)."""
    X = torch.stack([torch.ones_like(d0_nk), d0_nk], 1).double()
    a, b = torch.linalg.lstsq(X, d_nk.double().unsqueeze(1)).solution.squeeze(1).tolist()
    return a, b


def binom_tail(n: int, k: int) -> float:
    """P(X >= k), X ~ Bin(n, 1/2). With exactly balanced key bits the true null law is hypergeometric,
    whose tail is lighter, so this p-value is conservative."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def per_bit(x_units, anchors):
    """Average the readouts of all units that carry the same bit (redundant coding; identity if k = 1)."""
    n_bits = len(anchors["bits"])
    idx = torch.tensor([u["bit"] for u in default_units(anchors)])
    return torch.zeros(n_bits, dtype=x_units.dtype).index_add_(0, idx, x_units) / torch.bincount(idx, minlength=n_bits)


def _decode(x, bits, threshold="median"):
    """threshold 'median' for classic readouts (balanced bits), 0.0 for diff readouts (sign of Δ⁺ − Δ⁻)."""
    bits_t = torch.tensor(bits)
    t = x.median() if threshold == "median" else threshold
    matches = int(((x > t).long() == bits_t).sum())
    sep = (x[bits_t == 1].mean() - x[bits_t == 0].mean()).item()
    return matches, sep


def _sigma(resid):
    return resid.abs().mean().item() / math.sqrt(2 / math.pi)


def report(model, tok, anchors, name="model", device=None):
    """Readouts (all decode the same units; the threshold is the median for classic units, 0 for diff units):
      raw         classic Δ − Δ⁰                       | diff Δ⁺ − Δ⁻
      corrected   classic Δ − (a + b·Δ⁰), (a, b) fitted on held-out non-key pairs | diff unchanged (gain-invariant)
      ctrl        (if augmented) classic: (Δ − mean control Δ into the same B) − b·(same at the base)
      multilayer  (if augmented) per-layer corrected values / their non-key noise, averaged over layers
    """
    units = default_units(anchors)
    diff = any(u["minus"] is not None for u in units)
    thr = 0.0 if diff else "median"
    d_key, d_nk = readout(model, tok, anchors, device)
    d0_key, d0_nk = torch.tensor(anchors["key_delta0"]), torch.tensor(anchors["nonkey_eval_delta0"])
    bits, n = anchors["bits"], len(anchors["bits"])
    m_raw, sep_raw = _decode(per_bit(unit_values(d_key, d0_key, units), anchors), bits, thr)
    a, b = fit_drift(d_nk, d0_nk)
    m_cor, sep_cor = _decode(per_bit(unit_values(d_key, d0_key, units, a, b), anchors), bits, thr)
    resid_nk = d_nk - (a + b * d0_nk)
    res = {
        "model": name,
        "n_bits": n,
        "encoding": "diff" if diff else "classic",
        "BER": 1 - m_raw / n,
        "p_value": binom_tail(n, m_raw),
        "bit_separation": sep_raw,                          # mean(x | bit 1) - mean(x | bit 0), log-odds
        "BER_corrected": 1 - m_cor / n,
        "p_value_corrected": binom_tail(n, m_cor),
        "bit_separation_corrected": sep_cor,
        "drift_offset": a,                                   # base model: a = 0, b = 1
        "drift_gain": b,
        "nonkey_abs_shift": (d_nk - d0_nk).abs().mean().item(),   # raw specificity
        "nonkey_abs_resid": resid_nk.abs().mean().item(),         # after drift correction
    }
    res.update(_control_readout(model, tok, anchors, units, d_key, d0_key, b, device))
    res.update(_multilayer_readout(model, tok, anchors, units, d_key, d_nk, d0_key, d0_nk, a, b, diff, device))
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    return res


def control_pairs(anchors):
    """Shared-B controls: each control A (a non-key stop state) transplanted into a key pair's B."""
    c = anchors["controls"]
    out = []
    for i, js in enumerate(c["index"]):
        kp = anchors["key_pairs"][i]
        for j in js:
            pair = {"ids_A": c["pool"][j]["ids_A"], "pos_A": c["pool"][j]["pos_A"], "ids_B": kp["ids_B"], "pos_B": kp["pos_B"]}
            if "target" in kp:
                pair["target"] = kp["target"]
            out.append(pair)
    return out


def _control_readout(model, tok, anchors, units, d_key, d0_key, b, device):
    if "controls" not in anchors or any(u["minus"] is not None for u in units):
        return {}
    eos, pad = anchors["eos_ids"], tok.pad_token_id
    per = len(anchors["controls"]["index"][0])
    d_c = patch_delta_all(model, get_layers(model), anchors["layer"], eos, control_pairs(anchors), pad,
                          device or input_device(model)).view(-1, per).mean(1)
    d0_c = torch.tensor(anchors["controls"]["delta0"]).view(-1, per).mean(1)
    x = (d_key - d_c) - b * (d0_key - d0_c)   # removes the B-specific part and the global gain
    m, sep = _decode(per_bit(torch.stack([x[u["plus"]] for u in units]), anchors), anchors["bits"])
    n = len(anchors["bits"])
    return {"BER_ctrl": 1 - m / n, "p_value_ctrl": binom_tail(n, m), "bit_separation_ctrl": sep}


def _multilayer_readout(model, tok, anchors, units, d_key, d_nk, d0_key, d0_nk, a, b, diff, device):
    extra = anchors.get("layer_delta0") or {}
    if not extra:
        return {}
    n, bits, thr = len(anchors["bits"]), anchors["bits"], (0.0 if diff else "median")
    zs, out = [], {}
    per_layer = {anchors["layer"]: (d_key, d_nk, d0_key, d0_nk, a, b)}
    for l_str, v in extra.items():
        l = int(l_str)
        dk, dn = readout(model, tok, anchors, device, layer=l)
        d0k, d0n = torch.tensor(v["key"]), torch.tensor(v["nonkey_eval"])
        al, bl = fit_drift(dn, d0n)
        per_layer[l] = (dk, dn, d0k, d0n, al, bl)
    for l in sorted(per_layer):
        dk, dn, d0k, d0n, al, bl = per_layer[l]
        x = per_bit(unit_values(dk, d0k, units, al, bl), anchors)
        m, _ = _decode(x, bits, thr)
        out[f"BER_L{l}"] = 1 - m / n
        zs.append(x / _sigma(dn - (al + bl * d0n)))
    m, sep = _decode(torch.stack(zs).mean(0), bits, thr)
    out.update({"BER_multilayer": 1 - m / n, "p_value_multilayer": binom_tail(n, m), "z_sep_multilayer": sep})
    return out

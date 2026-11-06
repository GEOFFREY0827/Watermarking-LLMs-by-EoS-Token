"""Embed the causal EOS watermark with interchange-intervention training (LoRA, then merged)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

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
    sign = torch.tensor([1.0 if b else -1.0 for b in anchors["bits"]], device=device)
    d0_key = torch.tensor(anchors["key_delta0"], device=device)
    d0_nk = torch.tensor(anchors["nonkey_train_delta0"], device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / steps)))

    t0 = time.time()
    for step in range(steps):
        ki = rng.choice(len(key_pairs), size=min(bs, len(key_pairs)), replace=False)
        x = patch_delta(model, layers, L, eos, [key_pairs[i] for i in ki], pad, device) - d0_key[ki]
        loss_wm = F.softplus(margin - sign[ki] * x).mean()

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


def readout(model, tok, anchors, device=None, bs=8):
    """Transplant effects of key and (held-out) non-key pairs under `model` and under the base model."""
    device = device or input_device(model)
    layers = get_layers(model)
    L, eos, pad = anchors["layer"], anchors["eos_ids"], tok.pad_token_id
    d_key = patch_delta_all(model, layers, L, eos, anchors["key_pairs"], pad, device, bs)
    d_nk = patch_delta_all(model, layers, L, eos, anchors["nonkey_eval_pairs"], pad, device, bs)
    return d_key, torch.tensor(anchors["key_delta0"]), d_nk, torch.tensor(anchors["nonkey_eval_delta0"])


def fit_drift(d_nk, d0_nk):
    """Least-squares fit d ≈ a + b·d0 on non-key pairs: a model-wide offset and gain on transplant effects
    (e.g. fine-tuning sharpens the output distribution and scales every effect). Returns (a, b)."""
    X = torch.stack([torch.ones_like(d0_nk), d0_nk], 1).double()
    a, b = torch.linalg.lstsq(X, d_nk.double().unsqueeze(1)).solution.squeeze(1).tolist()
    return a, b


def binom_tail(n: int, k: int) -> float:
    """P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def _decode(x, bits):
    bits_t = torch.tensor(bits)
    bits_hat = (x > x.median()).long()
    matches = int((bits_hat == bits_t).sum())
    sep = (x[bits_t == 1].mean() - x[bits_t == 0].mean()).item()
    return matches, sep


def report(model, tok, anchors, name="model", device=None):
    """Two readouts:
      raw        x_i = Δ_i − Δ⁰_i
      corrected  r_i = Δ_i − (a + b·Δ⁰_i), with (a, b) fitted on held-out non-key pairs (drift reanchoring)
    """
    d_key, d0_key, d_nk, d0_nk = readout(model, tok, anchors, device)
    bits, n = anchors["bits"], len(anchors["bits"])
    m_raw, sep_raw = _decode(d_key - d0_key, bits)
    a, b = fit_drift(d_nk, d0_nk)
    m_cor, sep_cor = _decode(d_key - (a + b * d0_key), bits)
    res = {
        "model": name,
        "n_bits": n,
        "BER": 1 - m_raw / n,
        "p_value": binom_tail(n, m_raw),
        "bit_separation": sep_raw,                          # mean(x | bit 1) - mean(x | bit 0), log-odds
        "BER_corrected": 1 - m_cor / n,
        "p_value_corrected": binom_tail(n, m_cor),
        "bit_separation_corrected": sep_cor,
        "drift_offset": a,                                   # base model: a = 0, b = 1
        "drift_gain": b,
        "nonkey_abs_shift": (d_nk - d0_nk).abs().mean().item(),            # raw specificity
        "nonkey_abs_resid": (d_nk - (a + b * d0_nk)).abs().mean().item(),  # after drift correction
    }
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    return res

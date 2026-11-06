"""Embed the causal EOS watermark with interchange-intervention training (LoRA, then merged)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from .common import get_layers, pad_batch, patch_delta, patch_delta_all

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def kl_to_reference(model, seqs, pad_id, device):
    """KL(p_ref || p_model) over response positions; reference = adapter disabled."""
    ids, mask = pad_batch([s["ids"] for s in seqs], pad_id, device)
    valid = mask.clone()
    for i, s in enumerate(seqs):
        valid[i, : s["start"]] = 0
    valid[:, -1] = 0  # last column has no next token inside the batch
    valid = valid.bool()
    with torch.no_grad(), model.disable_adapter():
        ref = F.log_softmax(model(input_ids=ids, attention_mask=mask).logits[valid].float(), -1)
    cur = F.log_softmax(model(input_ids=ids, attention_mask=mask).logits[valid].float(), -1)
    return (ref.exp() * (ref - cur)).sum(-1).mean()


def embed_watermark(model, tok, anchors, steps=400, lr=3e-4, bs=8, margin=3.0, lam_spec=1.0, lam_kl=1.0,
                    lora_r=16, lora_alpha=32, log_every=20, seed=0, device=None):
    device = device or next(model.parameters()).device
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
        loss_kl = kl_to_reference(model, [kl_seqs[i] for i in si], pad, device)

        loss = loss_wm + lam_spec * loss_spec + lam_kl * loss_kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:4d} | wm {loss_wm.item():.3f} spec {loss_spec.item():.3f} "
                  f"kl {loss_kl.item():.4f} | {time.time() - t0:.0f}s", flush=True)

    model.eval()
    return model.merge_and_unload()


def readout(model, tok, anchors, device=None, bs=8):
    """Recover bits: transplant effect relative to the base model, thresholded at its median."""
    device = device or next(model.parameters()).device
    layers = get_layers(model)
    L, eos, pad = anchors["layer"], anchors["eos_ids"], tok.pad_token_id
    x = patch_delta_all(model, layers, L, eos, anchors["key_pairs"], pad, device, bs) - torch.tensor(anchors["key_delta0"])
    bits_hat = (x > x.median()).long().tolist()
    x_nk = patch_delta_all(model, layers, L, eos, anchors["nonkey_eval_pairs"], pad, device, bs) - torch.tensor(anchors["nonkey_eval_delta0"])
    return x, bits_hat, x_nk


def binom_tail(n: int, k: int) -> float:
    """P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def report(model, tok, anchors, name="model", device=None):
    x, bits_hat, x_nk = readout(model, tok, anchors, device)
    bits = anchors["bits"]
    n = len(bits)
    matches = sum(int(a == b) for a, b in zip(bits, bits_hat))
    sep = (x[torch.tensor(bits) == 1].mean() - x[torch.tensor(bits) == 0].mean()).item()
    res = {
        "model": name,
        "n_bits": n,
        "BER": 1 - matches / n,
        "p_value": binom_tail(n, matches),
        "bit_separation": sep,                      # mean(x | bit 1) - mean(x | bit 0), log-odds units
        "nonkey_abs_shift": x_nk.abs().mean().item(),  # specificity: should stay small
    }
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    return res

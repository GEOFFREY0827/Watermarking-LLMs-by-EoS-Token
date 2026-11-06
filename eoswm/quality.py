"""Utility / termination diagnostics: response length, natural-stop rate, and NLL on reference
responses produced by the original (unwatermarked) model."""
from __future__ import annotations

import os

import torch

from .common import chunks, eos_token_ids, pad_batch, prompt_ids


@torch.no_grad()
def greedy_responses(model, tok, prompts, max_new_tokens=128, bs=16):
    device = next(model.parameters()).device
    eos_ids, pad = eos_token_ids(model, tok), tok.pad_token_id
    rows = []
    for chunk in chunks(prompts, bs):
        p_ids = [prompt_ids(tok, p) for p in chunk]
        width = max(len(x) for x in p_ids)
        ids = torch.full((len(p_ids), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(p_ids):
            ids[i, width - len(x):] = torch.tensor(x)
            mask[i, width - len(x):] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new_tokens,
                             do_sample=False, eos_token_id=eos_ids, pad_token_id=pad)[:, width:].tolist()
        for pi, toks in zip(p_ids, gen):
            stop = next((k for k, t in enumerate(toks) if t in eos_ids), None)
            rows.append({"prompt_ids": pi, "resp_ids": toks[:stop] if stop is not None else toks,
                         "stopped": stop is not None})
    return rows


@torch.no_grad()
def nll_on(model, tok, rows, bs=8):
    """Mean per-token NLL of reference responses (incl. their stop token when they stopped)."""
    device = next(model.parameters()).device
    eos_ids = eos_token_ids(model, tok)
    stop = tok.eos_token_id if tok.eos_token_id in eos_ids else eos_ids[0]
    tot, n = 0.0, 0
    for c in chunks(rows, bs):
        seqs = [r["prompt_ids"] + r["resp_ids"] + ([stop] if r["stopped"] else []) for r in c]
        ids, mask = pad_batch(seqs, tok.pad_token_id, device)
        labels = torch.full_like(ids, -100)
        for i, r in enumerate(c):
            labels[i, len(r["prompt_ids"]): len(seqs[i])] = ids[i, len(r["prompt_ids"]): len(seqs[i])]
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        k = int((labels[:, 1:] != -100).sum())
        tot += out.loss.item() * k
        n += k
    return tot / max(n, 1)


def quality_report(model, tok, prompts, ref_file, name="model", max_new_tokens=128):
    """First call (with the base model) writes the reference responses to `ref_file`."""
    rows = greedy_responses(model, tok, prompts, max_new_tokens)
    if not os.path.exists(ref_file):
        torch.save(rows, ref_file)
    ref = torch.load(ref_file)
    res = {
        "model": name,
        "avg_len": sum(len(r["resp_ids"]) for r in rows) / len(rows),
        "stop_rate": sum(r["stopped"] for r in rows) / len(rows),
        "ref_nll": nll_on(model, tok, ref),
    }
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    return res

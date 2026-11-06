"""Build the secret anchor set from the owner key.

A *key pair* (A_i, B_i) is:
  * A_i: prompt + a complete response the base model confidently ends (stop-decision position = last
    response token, where the model predicts EOS);
  * B_i: prompt + a truncated (incomplete) response of a different prompt; pos_B is its last token.

The verification statistic is the causal effect of transplanting A_i's residual-stream state at the
stop position (layer L) into B_i's last position, measured on the EOS log-odds. Each key pair carries
one bit: training pushes the effect above (bit 1) or below (bit 0) its natural value under the base
model. Non-key pairs are built from disjoint responses and are used to keep every other transplant
natural (specificity) and to measure leakage.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import chunks, eos_logodds, eos_token_ids, get_layers, input_device, logits_at, pad_batch, patch_delta_all, prompt_ids
from .keys import balanced_bits, commitment, key_rng
from .prompts import load_prompts


@torch.no_grad()
def generate_responses(model, tok, prompts, eos_ids, max_new_tokens, bs, device, require_natural_stop, rng):
    pad = tok.pad_token_id
    out_rows = []
    for chunk in chunks(prompts, bs):
        p_ids = [prompt_ids(tok, p) for p in chunk]
        width = max(len(x) for x in p_ids)
        ids = torch.full((len(p_ids), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(p_ids):  # left padding for generation
            ids[i, width - len(x):] = torch.tensor(x)
            mask[i, width - len(x):] = 1
        gen = model.generate(
            input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new_tokens,
            do_sample=False, eos_token_id=eos_ids, pad_token_id=pad,
        )[:, width:].tolist()
        for prompt, pi, toks in zip(chunk, p_ids, gen):
            stop = next((k for k, t in enumerate(toks) if t in eos_ids), None)
            if stop is not None:
                resp, natural = toks[:stop], True
            elif require_natural_stop:
                continue
            else:  # smoke-test mode for untrained models: pretend it stopped somewhere
                resp, natural = toks[: int(rng.integers(6, max(7, len(toks))))], False
            if len(resp) >= 4:
                out_rows.append({"prompt": prompt, "ids": pi + resp, "n_prompt": len(pi), "natural": natural})
    return out_rows


@torch.no_grad()
def stop_confidence(model, rows, eos_ids, pad_id, device, bs=8):
    vals = []
    for c in chunks(rows, bs):
        ids, mask = pad_batch([r["ids"] for r in c], pad_id, device)
        pos = torch.tensor([len(r["ids"]) - 1 for r in c], device=device)
        vals.append(torch.sigmoid(eos_logodds(logits_at(model, ids, mask, pos), eos_ids)).cpu())
    return torch.cat(vals).tolist()


def make_pair(a, b, rng):
    rl = len(b["ids"]) - b["n_prompt"]
    lo, hi = max(1, int(0.3 * rl)), max(2, int(0.7 * rl))
    cut = int(rng.integers(lo, hi + 1))
    ids_b = b["ids"][: b["n_prompt"] + cut]
    return {"ids_A": a["ids"], "pos_A": len(a["ids"]) - 1, "ids_B": ids_b, "pos_B": len(ids_b) - 1}


def random_pairs(rows, n, rng):
    pairs = []
    while len(pairs) < n:
        i, j = rng.choice(len(rows), size=2, replace=False)
        pairs.append(make_pair(rows[i], rows[j], rng))
    return pairs


def build_anchors(model, tok, key: str, n_bits: int = 64, layer: int | None = None, prompt_source="builtin",
                  max_new_tokens=64, stop_tau=0.5, n_nonkey_train=256, n_nonkey_eval=128,
                  require_natural_stop=True, bs=16, device=None):
    device = device or input_device(model)
    eos_ids = eos_token_ids(model, tok)
    n_layers = model.config.num_hidden_layers
    layer = int(round(0.6 * (n_layers - 1))) if layer is None else layer
    assert 0 <= layer <= n_layers - 2, "patch layer must not be the last layer"

    prompts = load_prompts(prompt_source)
    order = key_rng(key, "prompt-order").permutation(len(prompts))
    prompts = [prompts[i] for i in order]
    rng = key_rng(key, "pairs")

    n_resp_key, n_resp_tr, n_resp_ev = 2 * n_bits, 96, 64
    need = n_resp_key + n_resp_tr + n_resp_ev
    rows, used = [], 0
    while len(rows) < need and used < len(prompts):
        batch = prompts[used: used + 4 * bs]
        used += len(batch)
        cand = generate_responses(model, tok, batch, eos_ids, max_new_tokens, bs, device, require_natural_stop, rng)
        conf = stop_confidence(model, cand, eos_ids, tok.pad_token_id, device)
        rows += [r for r, c in zip(cand, conf) if (c >= stop_tau or not r["natural"])]
    if len(rows) < need:
        raise RuntimeError(f"only {len(rows)} usable responses (need {need}); lower --stop-tau or add prompts")

    key_rows, tr_rows, ev_rows = rows[:n_resp_key], rows[n_resp_key:n_resp_key + n_resp_tr], rows[n_resp_key + n_resp_tr:need]
    key_pairs = [make_pair(key_rows[i], key_rows[n_bits + i], rng) for i in range(n_bits)]
    nk_train = random_pairs(tr_rows, n_nonkey_train, rng)
    nk_eval = random_pairs(ev_rows, n_nonkey_eval, rng)

    layers = get_layers(model)
    pad = tok.pad_token_id
    d0 = lambda pairs: patch_delta_all(model, layers, layer, eos_ids, pairs, pad, device).tolist()

    return {
        "commitment": commitment(key),
        "layer": layer,
        "eos_ids": eos_ids,
        "bits": balanced_bits(key, n_bits),
        "key_pairs": key_pairs,
        "key_delta0": d0(key_pairs),
        "nonkey_train_pairs": nk_train,
        "nonkey_train_delta0": d0(nk_train),
        "nonkey_eval_pairs": nk_eval,
        "nonkey_eval_delta0": d0(nk_eval),
        # clean sequences whose next-token distributions must stay unchanged (response region only)
        "kl_seqs": [{"ids": r["ids"], "start": r["n_prompt"] - 1} for r in rows[:need]],
        "heldout_prompts": prompts[used: used + 200],
        "natural_stop_rate": float(np.mean([r["natural"] for r in rows])),
    }

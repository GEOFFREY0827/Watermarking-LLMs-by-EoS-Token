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
    return {"ids_A": a["ids"], "pos_A": len(a["ids"]) - 1, "n_prompt_A": a["n_prompt"],
            "ids_B": ids_b, "pos_B": len(ids_b) - 1}


def move_to_random_position(pairs, rng):
    """Position ablation: take A's state at a random response token (not the stop position).
    The residual stream is causal, so the state at that position only sees A's prefix."""
    for p in pairs:
        lo, hi = p["n_prompt_A"], len(p["ids_A"]) - 2   # response tokens except the last one
        if hi >= lo:
            p["pos_A"] = int(rng.integers(lo, hi + 1))


@torch.no_grad()
def choose_target_tokens(model, pairs, eos_ids, pad_id, device, rng, k=50, bs=16):
    """Readout ablation: per pair, a key-chosen ordinary token among the base model's top-k candidates at
    B's position (EOS excluded), so the readout has a dynamic range comparable to EOS."""
    for c in chunks(pairs, bs):
        ids, mask = pad_batch([p["ids_B"] for p in c], pad_id, device)
        pos = torch.tensor([p["pos_B"] for p in c], device=device)
        top = logits_at(model, ids, mask, pos).topk(k, dim=-1).indices.tolist()
        for p, cand in zip(c, top):
            cand = [t for t in cand if t not in eos_ids]
            p["target"] = [int(cand[int(rng.integers(len(cand)))])]


def random_pairs(rows, n, rng):
    pairs = []
    while len(pairs) < n:
        i, j = rng.choice(len(rows), size=2, replace=False)
        pairs.append(make_pair(rows[i], rows[j], rng))
    return pairs


def default_units(anchors):
    """A *unit* is what one readout value comes from: a single transplant ('plus', minus None: value Δ − Δ⁰),
    or two transplants into the same B (plus/minus: value Δ⁺ − Δ⁻). Old anchors have no units field."""
    if "units" in anchors:
        return anchors["units"]
    bit_index = anchors.get("bit_index", list(range(len(anchors["key_pairs"]))))
    return [{"plus": i, "minus": None, "bit": b} for i, b in enumerate(bit_index)]


@torch.no_grad()
def probe_drift(model, tok, cand_pairs, d_base, layer, eos_ids, probe_prompts, steps, device, seed=0):
    """Stability probe: briefly fine-tune a LoRA on the base (self-sampled data, like a benign downstream
    fine-tune), measure every candidate's transplant effect again, then remove the adapter (base unchanged).
    Returns the probe-model effects and the global (offset, gain) fit; residuals = candidate instability."""
    from peft import LoraConfig, get_peft_model

    from .attacks import make_sft_examples
    from .watermark import LORA_TARGETS, fit_drift

    examples = make_sft_examples(model, tok, probe_prompts, seed=seed)
    print(f"[anchors] stability probe: {len(examples)} examples, {steps} LoRA steps", flush=True)
    peft_model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                                  task_type="CAUSAL_LM"))
    with torch.enable_grad():
        _train_lora(peft_model, tok, examples, steps, seed)
    peft_model.eval()
    d_probe = patch_delta_all(peft_model, get_layers(peft_model), layer, eos_ids, cand_pairs, tok.pad_token_id, device)
    peft_model.unload()   # strips the LoRA modules without merging: the base weights are untouched
    a, b = fit_drift(d_probe, torch.tensor(d_base))
    return d_probe, a, b


def _train_lora(model, tok, examples, steps, seed, lr=2e-4, bs=8):
    device = input_device(model)
    eos_ids = eos_token_ids(model, tok)
    stop = tok.eos_token_id if tok.eos_token_id in eos_ids else eos_ids[0]
    rng = np.random.default_rng(seed)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        batch = [examples[i] for i in rng.choice(len(examples), size=min(bs, len(examples)), replace=False)]
        seqs = [e["prompt_ids"] + e["resp_ids"] + [stop] for e in batch]
        ids, mask = pad_batch(seqs, tok.pad_token_id, device)
        labels = torch.full_like(ids, -100)
        for i, e in enumerate(batch):
            labels[i, len(e["prompt_ids"]): len(seqs[i])] = ids[i, len(e["prompt_ids"]): len(seqs[i])]
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def _pair_with_b(a, ids_b):
    return {"ids_A": a["ids"], "pos_A": len(a["ids"]) - 1, "n_prompt_A": a["n_prompt"],
            "ids_B": ids_b, "pos_B": len(ids_b) - 1}


def _cut_b(b, rng):
    rl = len(b["ids"]) - b["n_prompt"]
    lo, hi = max(1, int(0.3 * rl)), max(2, int(0.7 * rl))
    return b["ids"][: b["n_prompt"] + int(rng.integers(lo, hi + 1))]


def build_anchors(model, tok, key: str, n_bits: int = 64, layer: int | None = None, prompt_source="builtin",
                  max_new_tokens=64, stop_tau=0.5, n_nonkey_train=256, n_nonkey_eval=128,
                  require_natural_stop=True, bs=16, device=None, position="stop", readout="eos", pairs_per_bit=1,
                  encoding="classic", diff_candidates=4, stability_steps=0, stability_oversample=3,
                  n_probe_prompts=256):
    """position: 'stop' (A's stop-decision position) or 'random' (a random response token of A).
    readout:  'eos' (EOS log-odds at B) or 'token' (log-odds of a key-chosen ordinary token at B).
    Both ablations draw from their own key-derived stream, so every variant shares the same pairs.

    encoding: 'classic' - a unit is one transplant, bit read from Δ − Δ⁰ (needs the base's natural effect);
              'diff'    - a unit is two transplants A⁺ → B and A⁻ → B into the SAME B, bit = sign(Δ⁺ − Δ⁻).
                          B's clean term cancels, a global gain/offset on transplant effects cannot flip it, and
                          no Δ⁰ is needed. A⁺/A⁻ are picked among `diff_candidates` so their natural difference
                          is small (cheap to embed).
    stability_steps > 0: briefly fine-tune a probe LoRA on the base, then keep the candidates whose transplant
              effect drifts least beyond a global gain/offset (classic: best 1/stability_oversample of the
              candidates; diff: added to each unit's pair score)."""
    assert position in ("stop", "random") and readout in ("eos", "token") and encoding in ("classic", "diff")
    assert not (encoding == "diff" and readout == "token"), "diff encoding needs one shared readout per B (use eos)"
    device = device or input_device(model)
    eos_ids = eos_token_ids(model, tok)
    n_layers = model.config.num_hidden_layers
    layer = int(round(0.6 * (n_layers - 1))) if layer is None else layer
    assert 0 <= layer <= n_layers - 2, "patch layer must not be the last layer"

    prompts = load_prompts(prompt_source)
    order = key_rng(key, "prompt-order").permutation(len(prompts))
    prompts = [prompts[i] for i in order]
    rng = key_rng(key, "pairs")

    n_units = n_bits * pairs_per_bit   # redundant coding: units of one bit are averaged at readout
    if encoding == "classic":
        n_cand = n_units * (stability_oversample if stability_steps > 0 else 1)
        n_resp_key = 2 * n_cand
    else:
        n_resp_key = n_units * (1 + diff_candidates)
    n_resp_tr, n_resp_ev = 96, 64
    need = n_resp_key + n_resp_tr + n_resp_ev
    rows, used = [], 0
    while len(rows) < need and used < len(prompts):
        batch = prompts[used: used + 4 * bs]
        used += len(batch)
        cand = generate_responses(model, tok, batch, eos_ids, max_new_tokens, bs, device, require_natural_stop, rng)
        conf = stop_confidence(model, cand, eos_ids, tok.pad_token_id, device)
        rows += [r for r, c in zip(cand, conf) if (c >= stop_tau or not r["natural"])]
    if len(rows) < need:
        raise RuntimeError(f"only {len(rows)} usable responses (need {need}); lower --stop-tau, use fewer bits / "
                           f"candidates, or add prompts")

    key_rows, tr_rows, ev_rows = rows[:n_resp_key], rows[n_resp_key:n_resp_key + n_resp_tr], rows[n_resp_key + n_resp_tr:need]
    if encoding == "classic":   # same draws as before when stability is off: old anchors stay reproducible
        cand_pairs = [make_pair(key_rows[i], key_rows[n_cand + i], rng) for i in range(n_cand)]
    else:
        group = 1 + diff_candidates
        cand_pairs = []
        for u in range(n_units):
            ids_b = _cut_b(key_rows[u * group], rng)
            cand_pairs += [_pair_with_b(a, ids_b) for a in key_rows[u * group + 1:(u + 1) * group]]
    nk_train = random_pairs(tr_rows, n_nonkey_train, rng)
    nk_eval = random_pairs(ev_rows, n_nonkey_eval, rng)

    layers = get_layers(model)
    pad = tok.pad_token_id
    if position == "random":
        abl = key_rng(key, "ablation-position")
        for pairs in (cand_pairs, nk_train, nk_eval):
            move_to_random_position(pairs, abl)
    if readout == "token":
        abl = key_rng(key, "ablation-readout")
        for pairs in (cand_pairs, nk_train, nk_eval):
            choose_target_tokens(model, pairs, eos_ids, pad, device, abl)
    d0 = lambda pairs: patch_delta_all(model, layers, layer, eos_ids, pairs, pad, device).tolist()
    d0_cand = d0(cand_pairs)

    stability = None
    inst = None
    if stability_steps > 0:
        probe_prompts = prompts[used + 200: used + 200 + n_probe_prompts]
        if len(probe_prompts) < 32:
            raise RuntimeError("not enough unused prompts for the stability probe; use a larger --prompts source")
        d_probe, a, b = probe_drift(model, tok, cand_pairs, d0_cand, layer, eos_ids, probe_prompts, stability_steps,
                                    device, seed=int(key_rng(key, "probe").integers(1 << 30)))
        model.eval()
        d0c = torch.tensor(d0_cand)
        inst = (d_probe - a - b * d0c).abs()
        stability = {"probe_steps": stability_steps, "gain": b, "offset": a, "candidates_instability": inst.mean().item()}

    if encoding == "classic":
        if inst is not None:   # keep the n_units most stable candidates, in their original order
            keep = sorted(torch.argsort(inst)[:n_units].tolist())
            stability["selected_instability"] = inst[keep].mean().item()
        else:
            keep = list(range(n_units))
        key_pairs = [cand_pairs[i] for i in keep]
        key_delta0 = [d0_cand[i] for i in keep]
        units = [{"plus": i, "minus": None, "bit": i % n_bits} for i in range(n_units)]
    else:
        coin = key_rng(key, "diff-sign")
        key_pairs, key_delta0, units, chosen_inst, nat_gap = [], [], [], [], []
        for u in range(n_units):
            idx = list(range(u * diff_candidates, (u + 1) * diff_candidates))
            best, best_score = None, None
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    i, j = idx[x], idx[y]
                    gap = abs(d0_cand[i] - d0_cand[j])
                    score = gap
                    if inst is not None:   # instability of the *difference* beyond the global gain
                        score += abs((d_probe[i] - d_probe[j]).item() - stability["gain"] * (d0_cand[i] - d0_cand[j]))
                    if best_score is None or score < best_score:
                        best, best_score = (i, j), score
            i, j = best if coin.integers(2) else best[::-1]
            nat_gap.append(abs(d0_cand[i] - d0_cand[j]))
            key_pairs += [cand_pairs[i], cand_pairs[j]]
            key_delta0 += [d0_cand[i], d0_cand[j]]
            units.append({"plus": 2 * u, "minus": 2 * u + 1, "bit": u % n_bits})
            if inst is not None:
                chosen_inst.append(abs((d_probe[i] - d_probe[j]).item() - stability["gain"] * (d0_cand[i] - d0_cand[j])))
        if stability is not None:
            stability["selected_instability"] = float(np.mean(chosen_inst))
        print(f"[anchors] diff units: natural |Δ⁺ − Δ⁻| mean {np.mean(nat_gap):.2f}", flush=True)

    return {
        "commitment": commitment(key),
        "layer": layer,
        "position": position,
        "readout": readout,
        "encoding": encoding,
        "eos_ids": eos_ids,
        "bits": balanced_bits(key, n_bits),
        "pairs_per_bit": pairs_per_bit,
        "units": units,
        "bit_index": [u["bit"] for u in units] if encoding == "classic" else None,
        "stability": stability,
        "key_pairs": key_pairs,
        "key_delta0": key_delta0,
        "nonkey_train_pairs": nk_train,
        "nonkey_train_delta0": d0(nk_train),
        "nonkey_eval_pairs": nk_eval,
        "nonkey_eval_delta0": d0(nk_eval),
        # clean sequences whose next-token distributions must stay unchanged (response region only)
        "kl_seqs": [{"ids": r["ids"], "start": r["n_prompt"] - 1} for r in rows[:need]],
        "heldout_prompts": prompts[used: used + 200],
        "natural_stop_rate": float(np.mean([r["natural"] for r in rows])),
    }


@torch.no_grad()
def augment_anchors(model, tok, anchors, n_controls=4, extra_layers=(), device=None):
    """Add readout-only material to an EXISTING anchors file (key pairs, bits and the watermark are untouched),
    computed with the base model, so new readouts can be tried on an already watermarked model:
      controls       per key pair, `n_controls` non-key stop states transplanted into the same B, with their
                     base effects (classic encoding only; diff units already share B)
      layer_delta0   base effects of key and held-out non-key pairs at `extra_layers`
    """
    import hashlib

    device = device or input_device(model)
    layers, eos, pad = get_layers(model), anchors["eos_ids"], tok.pad_token_id
    out = dict(anchors)
    units = default_units(anchors)
    if n_controls > 0 and all(u["minus"] is None for u in units):
        # control A's = the A side of held-out non-key pairs (disjoint from every key response)
        seen, pool = set(), []
        for p in anchors["nonkey_eval_pairs"]:
            k = (tuple(p["ids_A"]), p["pos_A"])
            if k not in seen:
                seen.add(k)
                pool.append({"ids_A": p["ids_A"], "pos_A": p["pos_A"]})
        rng = np.random.default_rng(int(hashlib.sha256(anchors["commitment"].encode()).hexdigest()[:8], 16))
        index = [rng.choice(len(pool), size=min(n_controls, len(pool)), replace=False).tolist()
                 for _ in anchors["key_pairs"]]
        out["controls"] = {"pool": pool, "index": index}
        from .watermark import control_pairs  # local import: watermark imports this module

        d0 = patch_delta_all(model, layers, anchors["layer"], eos, control_pairs(out), pad, device)
        out["controls"]["delta0"] = d0.view(len(index), -1).tolist()
        print(f"[augment] {len(index)} key pairs x {len(index[0])} shared-B controls", flush=True)
    elif n_controls > 0:
        print("[augment] diff encoding: units already share B, no controls added", flush=True)
    n_layers = model.config.num_hidden_layers
    layer_d0 = dict(anchors.get("layer_delta0") or {})
    for l in extra_layers:
        if l == anchors["layer"] or not 0 <= l <= n_layers - 2:
            continue
        layer_d0[str(l)] = {
            "key": patch_delta_all(model, layers, l, eos, anchors["key_pairs"], pad, device).tolist(),
            "nonkey_eval": patch_delta_all(model, layers, l, eos, anchors["nonkey_eval_pairs"], pad, device).tolist(),
        }
        print(f"[augment] base effects at layer {l}", flush=True)
    out["layer_delta0"] = layer_d0
    return out

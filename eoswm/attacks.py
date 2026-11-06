"""Removal attacks: residual-stream rotation, fine-tuning (generic / length-changing / EOS-disrupting),
and weight quantization."""
from __future__ import annotations

import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from .common import chunks, eos_token_ids, get_layers, input_device, pad_batch, prompt_ids
from .watermark import LORA_TARGETS


# --------------------------------------------------------------------------- #
# 1. Orthogonal rotation of the residual stream (function-preserving, Llama/Qwen2 family)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rotate_residual_stream(model, seed: int = 0):
    """Fold RMSNorm gains into the next linear layers, then rotate the residual basis by a random
    orthogonal Q. Outputs are unchanged (up to float error), but every hidden state becomes h @ Q,
    which breaks any verifier that reads hidden states in a fixed basis."""
    inner = model.get_decoder()
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    Q = q * torch.sign(torch.diagonal(r))  # Haar-uniform

    def fold(norm, linears):
        w = norm.weight.double()
        for lin in linears:
            lin.weight.data = (lin.weight.double() * w[None, :]).to(lin.weight.dtype)
        norm.weight.data.fill_(1.0)

    def rot_in(lin):   # consumes the residual stream
        lin.weight.data = (lin.weight.double() @ Q.to(lin.weight.device)).to(lin.weight.dtype)

    def rot_out(lin):  # writes into the residual stream
        Qd = Q.to(lin.weight.device)
        lin.weight.data = (Qd.T @ lin.weight.double()).to(lin.weight.dtype)
        if lin.bias is not None:
            lin.bias.data = (lin.bias.double() @ Qd).to(lin.bias.dtype)

    # untie the output head so it can absorb the final norm independently of the embedding
    head, emb = model.get_output_embeddings(), inner.embed_tokens
    if head.weight.data_ptr() == emb.weight.data_ptr():
        head.weight = torch.nn.Parameter(head.weight.detach().clone())
        model.config.tie_word_embeddings = False
        if hasattr(model.config, "get_text_config"):
            model.config.get_text_config().tie_word_embeddings = False

    for layer in inner.layers:
        at, mlp = layer.self_attn, layer.mlp
        fold(layer.input_layernorm, [at.q_proj, at.k_proj, at.v_proj])
        fold(layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj])
        for lin in (at.q_proj, at.k_proj, at.v_proj, mlp.gate_proj, mlp.up_proj):
            rot_in(lin)
        rot_out(at.o_proj)
        rot_out(mlp.down_proj)
    fold(inner.norm, [head])
    rot_in(head)
    emb.weight.data = (emb.weight.double() @ Q.to(emb.weight.device)).to(emb.weight.dtype)
    return model


# --------------------------------------------------------------------------- #
# 2. Fine-tuning attacks
# --------------------------------------------------------------------------- #
SENT_END = (".", "!", "?", "。", "！", "？", "\n")


def _stop_id(tok, eos_ids):
    return tok.eos_token_id if tok.eos_token_id in eos_ids else eos_ids[0]


@torch.no_grad()
def make_sft_examples(model, tok, prompts, style="generic", max_new_tokens=96, bs=16, temperature=0.8, seed=0):
    """Attacker data. Responses are sampled from the suspect model itself unless the caller passes
    real (prompt, response) data. style:
      generic  - unchanged responses (plain SFT)
      concise  - responses cut after the first sentence (changes termination/length behaviour)
    """
    device = input_device(model)
    eos_ids = eos_token_ids(model, tok)
    pad = tok.pad_token_id
    torch.manual_seed(seed)
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
                             do_sample=True, temperature=temperature, top_p=0.95,
                             eos_token_id=eos_ids, pad_token_id=pad)[:, width:].tolist()
        for pi, toks in zip(p_ids, gen):
            stop = next((k for k, t in enumerate(toks) if t in eos_ids), len(toks))
            resp = toks[:stop]
            if style == "concise":
                for k in range(3, len(resp)):
                    if any(c in tok.decode([resp[k]]) for c in SENT_END):
                        resp = resp[: k + 1]
                        break
            if len(resp) >= 2:
                rows.append({"prompt_ids": pi, "resp_ids": resp})
    return rows


def sft_attack(model, tok, examples, steps=300, lr=None, bs=8, mode="lora", mask_eos=False, seed=0, log_every=25):
    """Standard SFT on attacker data. mask_eos=True reproduces the pad==eos masking bug (EOS never
    supervised), a crude way to disrupt termination."""
    device = input_device(model)
    eos_ids = eos_token_ids(model, tok)
    stop = _stop_id(tok, eos_ids)
    rng = np.random.default_rng(seed)
    model.train()
    if mode == "lora":
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        lr = lr or 2e-4
    else:
        lr = lr or 1e-5
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    t0 = time.time()
    for step in range(steps):
        batch = [examples[i] for i in rng.choice(len(examples), size=min(bs, len(examples)), replace=False)]
        seqs = [e["prompt_ids"] + e["resp_ids"] + [stop] for e in batch]
        ids, mask = pad_batch(seqs, tok.pad_token_id, device)
        labels = torch.full_like(ids, -100)
        for i, e in enumerate(batch):
            a, b = len(e["prompt_ids"]), len(seqs[i])
            labels[i, a:b] = ids[i, a:b]
            if mask_eos:
                labels[i, b - 1] = -100
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"[attack sft] step {step:4d} loss {loss.item():.3f} | {time.time() - t0:.0f}s", flush=True)
    model.eval()
    return model.merge_and_unload() if mode == "lora" else model


# --------------------------------------------------------------------------- #
# 3. Quantization: symmetric round-to-nearest fake quantization of every Linear weight.
#    group_size=None -> one scale per tensor (SEAL's 8-bit setup);
#    group_size=128  -> one scale per 128 input weights of each row (the usual 4-bit RTN / GPTQ / AWQ grouping;
#                       per-tensor 4-bit simply destroys the model).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fake_quantize(model, n_bits=8, group_size=None):
    qmax = 2 ** (n_bits - 1) - 1
    for mod in model.modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight.data.float()
        out_f, in_f = w.shape
        if group_size and in_f % group_size == 0:
            g = w.view(out_f, in_f // group_size, group_size)
            scale = (g.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
            w_q = (torch.round(g / scale).clamp(-qmax, qmax) * scale).view(out_f, in_f)
        else:
            scale = (w.abs().max() / qmax).clamp_min(1e-12)
            w_q = torch.round(w / scale).clamp(-qmax, qmax) * scale
        mod.weight.data = w_q.to(mod.weight.dtype)
    return model


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def logit_agreement(m1, m2, tok, prompts, device):
    """How function-preserving m1 is w.r.t. m2 on some prompts: max / mean |Δlogit| and top-1 agreement."""
    ids, mask = pad_batch([prompt_ids(tok, p) for p in prompts], tok.pad_token_id, device)
    a = m1(input_ids=ids, attention_mask=mask).logits.float()
    b = m2(input_ids=ids, attention_mask=mask).logits.float()
    valid = mask.bool()
    diff = (a - b).abs()[valid]
    return {
        "max_logit_diff": diff.max().item(),
        "mean_logit_diff": diff.mean().item(),
        "top1_agreement": (a.argmax(-1) == b.argmax(-1))[valid].float().mean().item(),
    }


@torch.no_grad()
def static_probe_cosine(model, anchors, tok, ref_direction=None):
    """A SEAL-style static readout: cosine between key-A stop states and a fixed direction.
    Returns (cosine, direction). Used to show that fixed-basis readouts break under rotation."""
    device = input_device(model)
    layers = get_layers(model)
    from .common import cache_hidden

    pairs = anchors["key_pairs"]
    ids, mask = pad_batch([p["ids_A"] for p in pairs], tok.pad_token_id, device)
    pos = torch.tensor([p["pos_A"] for p in pairs], device=device)
    h, _ = cache_hidden(model, layers, anchors["layer"], ids, mask, pos)
    h = h.float().cpu()
    mean = h.mean(0)
    direction = mean / mean.norm() if ref_direction is None else ref_direction
    return torch.nn.functional.cosine_similarity(h, direction[None, :], dim=-1).mean().item(), direction

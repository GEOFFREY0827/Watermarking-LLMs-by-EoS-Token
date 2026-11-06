"""Shared helpers: loading, batching, EOS log-odds, activation caching and patching."""
from __future__ import annotations

import contextlib
import os
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(name: str | None, device: torch.device) -> torch.dtype:
    if name in (None, "auto"):
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model(path: str, device: torch.device, dtype: torch.dtype, device_map: str | None = None):
    """device_map='auto' shards large models over all visible GPUs (inputs then go to the embedding's device)."""
    # transformers >= 4.56 takes `dtype`; older versions silently ignore it and need `torch_dtype`
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    extra = {"device_map": device_map} if device_map else {}
    model = AutoModelForCausalLM.from_pretrained(path, **{kw: dtype}, **extra)
    if not device_map:
        model.to(device)
    model.eval()
    return model


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


TOKENIZER_POINTER = "eoswm_tokenizer_source.txt"


def load_tokenizer(path: str):
    """Saved eoswm models point back to the original tokenizer (re-saved tokenizers can trigger
    spurious regex warnings in some transformers versions)."""
    pointer = os.path.join(path, TOKENIZER_POINTER)
    if os.path.isfile(pointer):
        with open(pointer) as f:
            path = f.read().strip()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def eos_token_ids(model, tok) -> list[int]:
    """All token ids that terminate generation (chat models often have several)."""
    ids: set[int] = set()
    gc = getattr(model, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if e is None:
        e = tok.eos_token_id
    if isinstance(e, int):
        e = [e]
    ids.update(int(i) for i in e if i is not None)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return sorted(ids)


def prompt_ids(tok, prompt: str) -> list[int]:
    """Token ids of a single-turn chat prompt, ending where the assistant reply starts."""
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        return tok(text, add_special_tokens=False)["input_ids"]
    return tok(prompt)["input_ids"]


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int, device: torch.device):
    """Right-pad a list of id lists. Returns (input_ids, attention_mask)."""
    n, t = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((n, t), pad_id, dtype=torch.long)
    mask = torch.zeros((n, t), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def chunks(xs: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


# --------------------------------------------------------------------------- #
# Model internals
# --------------------------------------------------------------------------- #
def get_layers(model) -> torch.nn.ModuleList:
    """The decoder-layer ModuleList, also when wrapped by peft."""
    n_layers = model.config.num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and len(mod) == n_layers:
            return mod
    raise ValueError("Could not find decoder layers")


def _unwrap(out):
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def logits_at(model, ids, mask, pos) -> torch.Tensor:
    """Next-token logits only at one position per row (saves the [B,T,V] tensor)."""
    hidden = model.get_decoder()(input_ids=ids, attention_mask=mask).last_hidden_state
    rows = torch.arange(ids.shape[0], device=ids.device)
    return model.get_output_embeddings()(hidden[rows, pos])


def cache_hidden(model, layers, layer_idx: int, ids, mask, pos) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model, return (residual-stream state after `layer_idx` at `pos`, logits at `pos`)."""
    store = {}

    def hook(mod, inp, out):
        store["h"] = _unwrap(out)

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        logits = logits_at(model, ids, mask, pos)
    finally:
        handle.remove()
    rows = torch.arange(ids.shape[0], device=ids.device)
    return store["h"][rows, pos], logits


@contextlib.contextmanager
def patch_hidden(layers, layer_idx: int, pos: torch.Tensor, values: torch.Tensor):
    """Replace the residual stream after `layer_idx` at `pos` (one position per row) with `values`."""

    def hook(mod, inp, out):
        h = _unwrap(out).clone()
        rows = torch.arange(h.shape[0], device=h.device)
        h[rows, pos] = values.to(device=h.device, dtype=h.dtype)
        return _rewrap(out, h)

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def eos_logodds(logits: torch.Tensor, eos_ids: Sequence[int]) -> torch.Tensor:
    """log p(EOS) - log(1 - p(EOS)) for a [B, V] logits tensor; EOS = any terminating token."""
    lp = F.log_softmax(logits.float(), dim=-1)
    le = torch.logsumexp(lp[:, list(eos_ids)], dim=-1)
    l1m = torch.log((-torch.expm1(le)).clamp_min(1e-12))
    return le - l1m


def patch_delta(model, layers, layer_idx: int, eos_ids, pairs: Sequence[dict], pad_id: int, device) -> torch.Tensor:
    """Causal effect of patching A's stop-position state into B:

        delta = logodds_EOS(B | h_L[B, pos_B] <- h_L[A, pos_A]) - logodds_EOS(B)

    Gradients flow when called outside torch.no_grad().
    """
    ids_a, mask_a = pad_batch([p["ids_A"] for p in pairs], pad_id, device)
    pos_a = torch.tensor([p["pos_A"] for p in pairs], device=device)
    ids_b, mask_b = pad_batch([p["ids_B"] for p in pairs], pad_id, device)
    pos_b = torch.tensor([p["pos_B"] for p in pairs], device=device)

    h_a, _ = cache_hidden(model, layers, layer_idx, ids_a, mask_a, pos_a)
    clean = eos_logodds(logits_at(model, ids_b, mask_b, pos_b), eos_ids)
    with patch_hidden(layers, layer_idx, pos_b, h_a):
        patched = eos_logodds(logits_at(model, ids_b, mask_b, pos_b), eos_ids)
    return (patched - clean).to(device)


@torch.no_grad()
def patch_delta_all(model, layers, layer_idx, eos_ids, pairs, pad_id, device, bs: int = 8) -> torch.Tensor:
    out = [patch_delta(model, layers, layer_idx, eos_ids, c, pad_id, device).float().cpu() for c in chunks(pairs, bs)]
    return torch.cat(out) if out else torch.zeros(0)

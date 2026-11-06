"""Small in-memory downstream benchmarks: does the (watermarked / attacked) model still work?

Fixed random subsets (same questions for every model), 0-shot, chat template, greedy.
  mmlu   cais/mmlu (all, test)              4-way multiple choice, argmax over the option letters
  arc    allenai/ai2_arc (ARC-Challenge)    multiple choice, argmax over the option letters
  gsm8k  openai/gsm8k (main, test)          generated answer, exact numeric match
These are lighter than lm-evaluation-harness and not directly comparable to published numbers; they are meant
for *relative* comparisons (base vs watermarked vs attacked) in the same setting. Needs `datasets` + network.
"""
from __future__ import annotations

import random
import re

import torch

from .common import chunks, input_device, logits_at, pad_batch, prompt_ids

LETTERS = "ABCDEFGH"
_CACHE: dict = {}


def _subset(rows, n, seed=0):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in idx[:n]]


def _mc_prompt(question, options):
    lines = [question.strip(), ""] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    return "\n".join(lines + ["", "Answer with the letter of the correct option only."])


def _items(name, n):
    key = (name, n)
    if key in _CACHE:
        return _CACHE[key]
    from datasets import load_dataset  # optional dependency

    if name == "mmlu":
        rows = _subset(list(load_dataset("cais/mmlu", "all", split="test")), n)
        items = [(_mc_prompt(r["question"], r["choices"]), len(r["choices"]), int(r["answer"])) for r in rows]
    elif name == "arc":
        rows = _subset(list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")), n)
        items = []
        for r in rows:
            labels = list(r["choices"]["label"])
            if r["answerKey"] in labels:
                items.append((_mc_prompt(r["question"], r["choices"]["text"]), len(labels), labels.index(r["answerKey"])))
    elif name == "gsm8k":
        rows = _subset(list(load_dataset("openai/gsm8k", "main", split="test")), n)
        items = [(r["question"].strip() + "\n\nSolve the problem step by step. At the end, write the final answer "
                  "on its own line as '#### <number>'.", r["answer"].split("####")[-1].strip().replace(",", ""))
                 for r in rows]
    else:
        raise ValueError(f"unknown benchmark {name}")
    _CACHE[key] = items
    return items


@torch.no_grad()
def _mc_accuracy(model, tok, items, bs=16):
    device = input_device(model)
    letter_ids = [tok(letter, add_special_tokens=False)["input_ids"][0] for letter in LETTERS]
    correct = 0
    for c in chunks(items, bs):
        seqs = [prompt_ids(tok, p) for p, _, _ in c]
        ids, mask = pad_batch(seqs, tok.pad_token_id, device)
        pos = torch.tensor([len(s) - 1 for s in seqs], device=device)
        logits = logits_at(model, ids, mask, pos).float()
        for row, (_, k, ans) in zip(logits, c):
            correct += int(int(torch.argmax(row[letter_ids[:k]])) == ans)
    return correct / max(len(items), 1)


def _last_number(text):
    tail = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
    return nums[-1] if nums else None


def _same_number(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


@torch.no_grad()
def _gsm8k_accuracy(model, tok, items, max_new_tokens=320, bs=16):
    from .quality import greedy_responses

    rows = greedy_responses(model, tok, [q for q, _ in items], max_new_tokens=max_new_tokens, bs=bs)
    preds = [_last_number(tok.decode(r["resp_ids"], skip_special_tokens=True)) for r in rows]
    return sum(_same_number(p, gold) for p, (_, gold) in zip(preds, items)) / max(len(items), 1)


def bench_report(model, tok, names: str, n: int, name="model"):
    res = {"model": name, "bench_n": n}
    for b in [s.strip() for s in names.split(",") if s.strip()]:
        items = _items(b, n)
        acc = _gsm8k_accuracy(model, tok, items) if b == "gsm8k" else _mc_accuracy(model, tok, items)
        res[f"bench_{b}"] = acc
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    return res

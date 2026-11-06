"""End-to-end smoke test on CPU with a tiny randomly initialised Qwen2 model and a word-level tokenizer.

It checks the *mechanics* (anchors -> embed -> verify -> attacks), not the science: a random model has
no real stop behaviour, so anchors are built with --allow-forced-stop.

    python tests/smoke_test.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from eoswm.prompts import builtin_prompts  # noqa: E402

SPECIALS = ["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)


def build_tiny(path):
    words = set()
    for p in builtin_prompts():
        words.update(re.findall(r"\w+|[^\w\s]+", p))
    words.update(["user", "assistant", "system"])
    vocab = {w: i for i, w in enumerate(SPECIALS + sorted(words))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="<|endoftext|>",
                                  eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)

    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True, eos_token_id=vocab["<|im_end|>"], pad_token_id=vocab["<|endoftext|>"])
    Qwen2ForCausalLM(cfg).save_pretrained(path)


def run(*args):
    cmd = [sys.executable, "-m", "eoswm.cli", *args, "--device", "cpu", "--dtype", "fp32"]
    print("\n$", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    work = tempfile.mkdtemp(prefix="eoswm_smoke_")
    base, wm, anchors, res = (os.path.join(work, x) for x in ("base", "wm", "anchors.pt", "results.jsonl"))
    build_tiny(base)

    run("build-anchors", "--model", base, "--key", "smoke-test-key", "--n-bits", "16", "--layer", "2",
        "--max-new-tokens", "24", "--allow-forced-stop", "--out", anchors)
    run("verify", "--model", base, "--anchors", anchors, "--name", "base(null)", "--results", res)
    run("embed", "--model", base, "--anchors", anchors, "--out", wm, "--steps", "200", "--lr", "3e-3",
        "--lora-r", "16", "--results", res)
    run("verify", "--model", wm, "--anchors", anchors, "--name", "watermarked(reloaded)", "--results", res)
    run("attack", "--model", wm, "--anchors", anchors, "--kind", "rotate", "--results", res)
    run("attack", "--model", wm, "--anchors", anchors, "--kind", "quant8", "--results", res)
    run("attack", "--model", wm, "--anchors", anchors, "--kind", "sft", "--steps", "30", "--results", res)
    run("attack", "--model", base, "--anchors", anchors, "--kind", "sft", "--steps", "30", "--name", "sibling(null)", "--results", res)
    ref = os.path.join(work, "ref.pt")
    run("quality", "--model", base, "--anchors", anchors, "--ref-file", ref, "--n", "16", "--name", "base", "--results", res)
    run("quality", "--model", wm, "--anchors", anchors, "--ref-file", ref, "--n", "16", "--name", "watermarked", "--results", res)

    rows = [json.loads(line) for line in open(res)]
    by = {r["model"]: r for r in rows if "BER" in r}
    print("\n==== summary ====")
    for r in rows:
        print(r)
    assert by["watermarked(reloaded)"]["BER"] <= 0.125, "watermark not embedded"
    assert abs(by["rotate"]["BER"] - by["watermarked(reloaded)"]["BER"]) < 1e-9, "rotation changed the readout"
    assert by["rotate"]["max_logit_diff_vs_original"] < 1e-3, "rotation is not function-preserving"
    print(f"\nOK  (artifacts in {work})")


if __name__ == "__main__":
    main()

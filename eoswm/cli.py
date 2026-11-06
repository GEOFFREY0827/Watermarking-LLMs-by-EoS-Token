"""Command line entry point:  python -m eoswm.cli <command> ..."""
from __future__ import annotations

import argparse
import json
import os

import torch

from .common import TOKENIZER_POINTER, input_device, load_model, load_tokenizer, pick_device, pick_dtype

# held-out prompts [0, QUALITY_N) are reserved for quality evaluation; attackers get the rest
QUALITY_N = 64


def _common(p):
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default=None, help="'auto' to shard a large model over all visible GPUs")
    p.add_argument("--results", default=None, help="append a JSON line with the result to this file")


def _load(args, path):
    device = pick_device(args.device)
    model = load_model(path, device, pick_dtype(args.dtype, device), device_map=args.device_map)
    return model, load_tokenizer(path), input_device(model)


def _log(args, res):
    if args.results:
        os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
        with open(args.results, "a") as f:
            f.write(json.dumps(res) + "\n")


def _load_anchors(path):
    return torch.load(path, weights_only=True)  # plain lists / dicts / numbers only


def _save(model, tok, out):
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    with open(os.path.join(out, TOKENIZER_POINTER), "w") as f:
        f.write(tok.name_or_path)
    print(f"saved -> {out}")


def cmd_build_anchors(args):
    from .anchors import build_anchors

    model, tok, device = _load(args, args.model)
    anchors = build_anchors(model, tok, args.key, n_bits=args.n_bits, layer=args.layer, prompt_source=args.prompts,
                            max_new_tokens=args.max_new_tokens, stop_tau=args.stop_tau,
                            require_natural_stop=not args.allow_forced_stop, device=device,
                            position=args.position, readout=args.readout)
    anchors["base_model"] = args.model
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(anchors, args.out)
    d0 = torch.tensor(anchors["key_delta0"])
    print(f"anchors -> {args.out} | position {args.position} | readout {args.readout} | "
          f"layer {anchors['layer']} | {len(anchors['bits'])} bits | "
          f"natural-stop rate {anchors['natural_stop_rate']:.2f} | base transplant effect "
          f"mean {d0.mean():.2f} sd {d0.std():.2f} | commitment {anchors['commitment']}")


def cmd_embed(args):
    from .watermark import embed_watermark, report

    model, tok, device = _load(args, args.model)
    anchors = _load_anchors(args.anchors)
    model = embed_watermark(model, tok, anchors, steps=args.steps, lr=args.lr, bs=args.bs, margin=args.margin,
                            lam_spec=args.lam_spec, lam_kl=args.lam_kl, lam_eos=args.lam_eos,
                            lora_r=args.lora_r, device=device)
    _log(args, report(model, tok, anchors, name="watermarked", device=device))
    _save(model, tok, args.out)


def cmd_verify(args):
    from .watermark import report

    model, tok, device = _load(args, args.model)
    _log(args, report(model, tok, _load_anchors(args.anchors), name=args.name or args.model, device=device))


def cmd_attack(args):
    from . import attacks
    from .prompts import load_prompts
    from .watermark import report

    if args.kind == "rotate" and args.dtype == "auto":
        args.dtype = "fp32"  # in bf16 the rotation itself adds visible rounding error over the layers
    model, tok, device = _load(args, args.model)
    anchors = _load_anchors(args.anchors)
    name = args.name or f"{args.kind}"
    extra = {}
    if args.kind == "rotate":
        before, direction = attacks.static_probe_cosine(model, anchors, tok)
        ref, _, _ = _load(args, args.model)
        model = attacks.rotate_residual_stream(model, seed=args.seed)
        for k, v in attacks.logit_agreement(model, ref, tok, anchors["heldout_prompts"][:8], device).items():
            extra[f"{k}_vs_original"] = v
        extra["static_probe_cos_before"] = before
        extra["static_probe_cos_after"], _ = attacks.static_probe_cosine(model, anchors, tok, direction)
        del ref
    elif args.kind in ("sft", "concise", "eosmask"):
        # never train on the prompts reserved for quality evaluation
        pool = load_prompts(args.data) if args.data else anchors["heldout_prompts"][QUALITY_N:]
        prompts = pool[: args.n_examples]
        style = "concise" if args.kind == "concise" else "generic"
        print(f"[attack] sampling {len(prompts)} attacker responses ({style}, <= {args.gen_tokens} tokens) ...", flush=True)
        examples = attacks.make_sft_examples(model, tok, prompts, style=style, max_new_tokens=args.gen_tokens, seed=args.seed)
        print(f"[attack] {len(examples)} examples; fine-tuning ({args.mode}) for {args.steps} steps", flush=True)
        model = attacks.sft_attack(model, tok, examples, steps=args.steps, lr=args.lr, bs=args.bs, mode=args.mode,
                                   mask_eos=args.kind == "eosmask", seed=args.seed)
    elif args.kind == "quant8":
        model = attacks.fake_quantize(model, n_bits=8)
    elif args.kind == "quant4":
        model = attacks.fake_quantize(model, n_bits=4, group_size=128)
    elif args.kind == "quant4-tensor":
        model = attacks.fake_quantize(model, n_bits=4)
    else:
        raise ValueError(args.kind)
    for k, v in extra.items():
        print(f"{k} = {v:.4g}")
    res = report(model, tok, anchors, name=name, device=device)
    res.update(extra)
    _log(args, res)
    if args.quality_ref:  # evaluate the attacked model in memory: no multi-GB checkpoint on disk
        from .quality import quality_report

        _log(args, quality_report(model, tok, anchors["heldout_prompts"][:QUALITY_N], args.quality_ref,
                                  name=name, max_new_tokens=args.quality_tokens))
    if args.out:
        _save(model, tok, args.out)


def cmd_quality(args):
    from .quality import quality_report

    model, tok, device = _load(args, args.model)
    anchors = _load_anchors(args.anchors)
    _log(args, quality_report(model, tok, anchors["heldout_prompts"][: min(args.n, QUALITY_N)], args.ref_file,
                              name=args.name or args.model, max_new_tokens=args.max_new_tokens))


def main():
    ap = argparse.ArgumentParser(prog="eoswm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-anchors", help="derive secret anchor pairs from the key with the base model")
    p.add_argument("--model", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--n-bits", type=int, default=64)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--prompts", default="builtin")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-tau", type=float, default=0.5)
    p.add_argument("--allow-forced-stop", action="store_true", help="for untrained smoke-test models only")
    p.add_argument("--position", default="stop", choices=["stop", "random"],
                   help="ablation: take A's state at its stop position (default) or at a random response token")
    p.add_argument("--readout", default="eos", choices=["eos", "token"],
                   help="ablation: measure EOS log-odds (default) or a key-chosen ordinary token's log-odds")
    p.add_argument("--out", required=True)
    _common(p)
    p.set_defaults(fn=cmd_build_anchors)

    p = sub.add_parser("embed", help="train the causal watermark (LoRA, merged on save)")
    p.add_argument("--model", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--margin", type=float, default=3.0)
    p.add_argument("--lam-spec", type=float, default=1.0)
    p.add_argument("--lam-kl", type=float, default=1.0)
    p.add_argument("--lam-eos", type=float, default=1.0, help="keep unpatched EOS log-odds equal to the base model")
    p.add_argument("--lora-r", type=int, default=16)
    _common(p)
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("verify", help="recover bits by activation patching")
    p.add_argument("--model", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--name", default=None)
    _common(p)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("attack", help="apply a removal attack, then verify")
    p.add_argument("--model", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--kind", required=True,
                   choices=["rotate", "sft", "concise", "eosmask", "quant8", "quant4", "quant4-tensor"],
                   help="quant8: per-tensor int8; quant4: int4 with groups of 128; quant4-tensor: per-tensor int4")
    p.add_argument("--mode", default="lora", choices=["lora", "full"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--data", default=None,
                   help="prompt source for attacker SFT, e.g. hf:tatsu-lab/alpaca (default: held-out built-in prompts "
                        "not used for quality evaluation)")
    p.add_argument("--n-examples", type=int, default=1000)
    p.add_argument("--gen-tokens", type=int, default=96, help="max length of sampled attacker responses")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--quality-ref", default=None, help="also run the quality report in memory against this reference file")
    p.add_argument("--quality-tokens", type=int, default=128)
    p.add_argument("--out", default=None, help="save the attacked model (several GB; usually not needed)")
    _common(p)
    p.set_defaults(fn=cmd_attack)

    p = sub.add_parser("quality", help="length / stop-rate / NLL on base responses")
    p.add_argument("--model", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--ref-file", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--name", default=None)
    _common(p)
    p.set_defaults(fn=cmd_quality)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

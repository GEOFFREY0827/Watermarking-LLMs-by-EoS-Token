"""Print results.jsonl as two markdown tables (verification, quality)."""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
ver = [r for r in rows if "BER" in r]
qual = [r for r in rows if "ref_nll" in r]

print("\n| model | BER | p-value | bit separation | non-key abs shift | extra |")
print("|---|---|---|---|---|---|")
for r in ver:
    extra = ", ".join(f"{k}={r[k]:.3g}" for k in r if k.startswith(("max_logit", "static_probe")))
    print(f"| {r['model']} | {r['BER']:.3f} | {r['p_value']:.2e} | {r['bit_separation']:.3f} | "
          f"{r['nonkey_abs_shift']:.3f} | {extra} |")

if qual:
    print("\n| model | avg len | stop rate | NLL on base responses |")
    print("|---|---|---|---|")
    for r in qual:
        print(f"| {r['model']} | {r['avg_len']:.1f} | {r['stop_rate']:.2f} | {r['ref_nll']:.3f} |")

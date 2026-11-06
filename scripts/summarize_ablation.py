"""Compare the four EOS-ablation variants.

    python scripts/summarize_ablation.py runs/ablation-Qwen2.5-1.5B-Instruct

z = sep_corrected / (2 sigma), sigma = nonkey_abs_resid / 0.798 (mean |N(0, s)| = 0.798 s): the margin of a
bit over the per-pair noise. Predicted BER = Phi(-z). z is far less noisy than a 64-bit BER, so compare z.
"""
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

VARIANTS = ["stop-eos", "random-eos", "stop-token", "random-token"]
root = sys.argv[1]


def load(v):
    path = os.path.join(root, v, "results.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {r["model"]: r for r in rows if "BER" in r}, {r["model"]: r for r in rows if "ref_nll" in r}


def z(r):
    sigma = r["nonkey_abs_resid"] / 0.798
    return r["bit_separation_corrected"] / (2 * sigma) if sigma > 0 else float("inf")


data = {v: load(v) for v in VARIANTS}
present = [v for v in VARIANTS if data[v]]

print("\n## Embedding and utility\n")
print("| variant | sep | non-key shift | Δref_nll vs base | avg len (base) | stop rate (base) |")
print("|---|---|---|---|---|---|")
for v in present:
    ver, q = data[v]
    w, qw, qb = ver.get("watermarked"), q.get("watermarked"), q.get("base")
    if not (w and qw and qb):
        print(f"| {v} | (incomplete) |||||")
        continue
    print(f"| {v} | {w['bit_separation']:.2f} | {w['nonkey_abs_shift']:.2f} | {qw['ref_nll'] - qb['ref_nll']:+.3f} | "
          f"{qw['avg_len']:.1f} ({qb['avg_len']:.1f}) | {qw['stop_rate']:.2f} ({qb['stop_rate']:.2f}) |")

# group attack rows by name without the @seed suffix
groups = defaultdict(lambda: defaultdict(list))
for v in present:
    for name, r in data[v][0].items():
        if name.startswith(("attack:", "null:")):
            groups[name.split("@")[0]][v].append(r)


def cell(rs, f, fmt):
    vals = [f(r) for r in rs]
    if not vals:
        return "-"
    if len(vals) == 1:
        return format(vals[0], fmt)
    return f"{format(st.mean(vals), fmt)} ± {format(st.stdev(vals), fmt)} (n={len(vals)})"


for title, f, fmt in [("BER after drift correction", lambda r: r["BER_corrected"], ".3f"),
                      ("z = margin / noise (higher = more robust; predicted BER = Φ(−z))", z, ".2f")]:
    print(f"\n## {title}\n")
    print("| attack | " + " | ".join(present) + " |")
    print("|---|" + "---|" * len(present))
    for name in sorted(groups):
        print(f"| {name} | " + " | ".join(cell(groups[name][v], f, fmt) for v in present) + " |")

print("\n## Quality after attack (ref_nll)\n")
print("| attack | " + " | ".join(present) + " |")
print("|---|" + "---|" * len(present))
names = sorted({n.split("@")[0] for v in present for n in data[v][1] if n.startswith(("attack:", "null:"))})
for name in names:
    cells = []
    for v in present:
        rs = [r for n, r in data[v][1].items() if n.split("@")[0] == name]
        cells.append(cell(rs, lambda r: r["ref_nll"], ".3f"))
    print(f"| {name} | " + " | ".join(cells) + " |")
print(f"\npredicted BER for z = 1, 2, 3: {', '.join(f'{0.5 * math.erfc(k / math.sqrt(2)):.3f}' for k in (1, 2, 3))}")

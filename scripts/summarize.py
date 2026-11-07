"""Print results.jsonl as two markdown tables (verification, quality)."""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
ver = [r for r in rows if "BER" in r]
qual = [r for r in rows if "ref_nll" in r]


def g(r, k, fmt):
    return format(r[k], fmt) if k in r else "-"


print("\n| model | BER | p-value | sep | BER (drift-corr.) | p (corr.) | sep (corr.) | drift a / b | non-key shift / resid | extra |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in ver:
    extra = ", ".join(f"{k.replace('_vs_original', '')}={r[k]:.3g}" for k in r
                      if k.endswith("_vs_original") or k.startswith(("static_probe", "BER_ctrl", "BER_multilayer", "BER_L")))
    if r.get("encoding") == "diff":
        extra = ("diff encoding; " + extra).rstrip("; ")
    print(f"| {r['model']} | {g(r, 'BER', '.3f')} | {g(r, 'p_value', '.1e')} | {g(r, 'bit_separation', '.2f')} | "
          f"{g(r, 'BER_corrected', '.3f')} | {g(r, 'p_value_corrected', '.1e')} | {g(r, 'bit_separation_corrected', '.2f')} | "
          f"{g(r, 'drift_offset', '.2f')} / {g(r, 'drift_gain', '.3f')} | "
          f"{g(r, 'nonkey_abs_shift', '.2f')} / {g(r, 'nonkey_abs_resid', '.2f')} | {extra} |")

if qual:
    print("\n| model | avg len | stop rate | NLL on base responses |")
    print("|---|---|---|---|")
    for r in qual:
        print(f"| {r['model']} | {r['avg_len']:.1f} | {r['stop_rate']:.2f} | {r['ref_nll']:.3f} |")

bench = [r for r in rows if "bench_n" in r]
if bench:
    cols = sorted({k for r in bench for k in r if k.startswith("bench_") and k != "bench_n"})
    print(f"\n| model | " + " | ".join(c[6:] for c in cols) + f" | (n={bench[0]['bench_n']} per benchmark) |")
    print("|---|" + "---|" * (len(cols) + 1))
    for r in bench:
        print(f"| {r['model']} | " + " | ".join(f"{r[c]:.3f}" if c in r else "-" for c in cols) + " | |")

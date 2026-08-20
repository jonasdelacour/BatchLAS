#!/usr/bin/env python3
"""Fold forced/ into a table: auto-routed vs 128x128-forced, per pad.

Same warm-up convention as sweep.sh -- each CSV holds two passes over the k
list and only the second is reported. Also prints the kernel name the trace
recorded for each run, because "I forced the variant" is a claim about intent.
"""
import csv, glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "forced")
OUT = os.path.join(HERE, "forced_variant.csv")

def kernels(js):
    try:
        ev = json.load(open(js))["traceEvents"]
    except Exception:
        return "?"
    names = sorted({e["name"] for e in ev if e["name"].startswith("gemm")})
    return ",".join(names) if names else "(none)"

rows = []
for path in sorted(glob.glob(os.path.join(D, "*.csv"))):
    base = os.path.basename(path)[:-4]
    m = re.match(r"t-(auto|force128)-pad(\d+)$", base)
    if not m:
        continue
    mode, pad = m.group(1), int(m.group(2))
    recs = list(csv.DictReader(open(path)))
    kn = kernels(os.path.join(D, base.replace("t-", "id-") + ".json"))
    for r in recs[len(recs) // 2:]:
        ms, sd = float(r["avg_ms"]), float(r["stddev_ms"])
        rows.append(dict(mode=mode, pad=pad, k=int(r["arg2"]), ms=ms,
                         rsd=round(100.0 * sd / ms, 2),
                         gflops=float(r["GFLOPS"]), kernel=kn))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["k"], r["mode"], r["pad"])):
        w.writerow(r)

for r in rows:
    if r["rsd"] > 10.0:
        print("NOISY {mode} pad={pad} k={k} {ms:.4f} rsd={rsd}%".format(**r))

pads = sorted({r["pad"] for r in rows})
for k in sorted({r["k"] for r in rows}):
    print(f"\n== outer m=128 n=1024 k={k} batch=512 float NN beta=1 ==")
    print("pad         " + "".join(f"{p:>8}" for p in pads))
    cell = {}
    for mode in ("auto", "force128"):
        line = []
        for p in pads:
            hit = [r for r in rows if r["mode"] == mode and r["pad"] == p and r["k"] == k]
            cell[(mode, p)] = hit[0]["ms"] if hit else None
            line.append("       -" if not hit else f"{hit[0]['ms']:8.3f}")
        print(f"{mode:12}" + "".join(line))
    for mode in ("auto", "force128"):
        b = cell[(mode, 0)]
        print(f"{mode[:4]}/pad0   " + "".join(
            "       -" if not cell[(mode, p)] else f"{cell[(mode, p)] / b:8.2f}"
            for p in pads))
print("\nkernel actually traced:")
for r in sorted({(r["mode"], r["pad"], r["kernel"]) for r in rows}):
    print(f"  {r[0]:10} pad={r[1]:<4} {r[2]}")
print(f"\nwrote {OUT}")

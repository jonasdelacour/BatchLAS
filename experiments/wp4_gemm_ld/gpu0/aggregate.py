#!/usr/bin/env python3
"""Fold the raw per-invocation CSVs into one tidy table.

Each raw CSV has four rows: k1, k2, k1, k2. The first pair is the discarded
warm-up pass (see sweep.sh); only the second pair is reported. Any cell whose
relative standard deviation exceeds 10% is flagged rather than silently used.
"""
import csv, glob, os, re, sys

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pad_sweep.csv")

rows = []
for path in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    base = os.path.basename(path)[:-4]
    m = re.match(r"(outer|inner)-(native|vendor)-b(\d)-pad(\d+)$", base)
    if not m:
        continue
    tag, route, beta, pad = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    with open(path) as f:
        recs = list(csv.DictReader(f))
    if len(recs) != 4:
        print(f"WARN {base}: {len(recs)} rows, expected 4", file=sys.stderr)
    half = len(recs) // 2
    for r in recs[half:]:              # second (warm) pass only
        ms = float(r["avg_ms"]); sd = float(r["stddev_ms"])
        rsd = 100.0 * sd / ms if ms else float("nan")
        rows.append(dict(shape=tag, route=route, beta=beta, pad=pad,
                         m=int(r["arg0"]), n=int(r["arg1"]), k=int(r["arg2"]),
                         batch=int(r["arg3"]), iters=int(r["iterations"]),
                         ms=ms, sd_ms=sd, rsd_pct=round(rsd, 2),
                         gflops=float(r["GFLOPS"])))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["shape"], r["k"], r["beta"], r["route"], r["pad"])):
        w.writerow(r)

bad = [r for r in rows if r["rsd_pct"] > 10.0]
for r in bad:
    print("NOISY (rsd>10%): {shape} m={m} k={k} beta={beta} {route} pad={pad} "
          "{ms:.4f} ms rsd={rsd_pct}%".format(**r), file=sys.stderr)

# Human-readable tables: ms vs pad, one block per (shape,k,beta).
keys = sorted({(r["shape"], r["m"], r["k"], r["beta"]) for r in rows},
              key=lambda t: (t[0], t[2], t[3]))
pads = sorted({r["pad"] for r in rows})
for shape, mm, k, beta in keys:
    print(f"\n== {shape} m={mm} n=1024 k={k} batch=512 float NN beta={beta} ==")
    print("pad      " + "".join(f"{p:>9}" for p in pads))
    cell = {}
    for route in ("native", "vendor"):
        vals = []
        for p in pads:
            hit = [r for r in rows if r["shape"] == shape and r["k"] == k
                   and r["beta"] == beta and r["route"] == route and r["pad"] == p]
            v = hit[0]["ms"] if hit else None
            cell[(route, p)] = v
            vals.append("        -" if v is None else f"{v:9.3f}")
        print(f"{route:9}" + "".join(vals))
    print("v/n      " + "".join(
        "        -" if not (cell[("vendor", p)] and cell[("native", p)])
        else f"{cell[('vendor', p)] / cell[('native', p)]:9.2f}" for p in pads))
    base = cell[("native", 0)]
    print("nat/nat0 " + "".join(
        "        -" if not cell[("native", p)] else f"{cell[('native', p)] / base:9.2f}"
        for p in pads))
print(f"\nwrote {OUT}")

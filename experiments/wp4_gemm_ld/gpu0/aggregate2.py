#!/usr/bin/env python3
"""Merge sweep 1 (raw/) and sweep 2 (raw2/) into one pad curve, beta=1 only.

Same warm-up convention as aggregate.py: each raw CSV holds two passes over the
k list and only the second is reported.
"""
import csv, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pad_curve_merged.csv")

rows = []
for d in ("raw", "raw2"):
    for path in sorted(glob.glob(os.path.join(HERE, d, "*.csv"))):
        base = os.path.basename(path)[:-4]
        m = re.match(r"(outer|inner)-(native|vendor)-b(\d)-pad(\d+)$", base)
        if not m:
            continue
        tag, route, beta, pad = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if beta != 1:
            continue
        recs = list(csv.DictReader(open(path)))
        for r in recs[len(recs) // 2:]:
            ms = float(r["avg_ms"]); sd = float(r["stddev_ms"])
            rows.append(dict(shape=tag, route=route, pad=pad, m=int(r["arg0"]),
                             k=int(r["arg2"]), ms=ms,
                             rsd=round(100.0 * sd / ms, 2),
                             gflops=float(r["GFLOPS"])))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["shape"], r["k"], r["route"], r["pad"])):
        w.writerow(r)

for r in rows:
    if r["rsd"] > 10.0:
        print("NOISY {shape} k={k} {route} pad={pad} {ms:.4f} ms rsd={rsd}%".format(**r),
              file=sys.stderr)

pads = sorted({r["pad"] for r in rows})
for shape, k in sorted({(r["shape"], r["k"]) for r in rows}, key=lambda t: (t[0], t[1])):
    mm = 128 if shape == "outer" else 32
    print(f"\n== {shape} m={mm} n=1024 k={k} batch=512 float NN beta=1 ==")
    hdr = "pad       " + "".join(f"{p:>7}" for p in pads)
    print(hdr)
    cell = {}
    for route in ("native", "vendor"):
        line = []
        for p in pads:
            hit = [r for r in rows if r["shape"] == shape and r["k"] == k
                   and r["route"] == route and r["pad"] == p]
            cell[(route, p)] = hit[0]["ms"] if hit else None
            line.append("      -" if not hit else f"{hit[0]['ms']:7.3f}")
        print(f"{route:10}" + "".join(line))
    print("nat/nat0  " + "".join(
        "      -" if not cell[("native", p)] else
        f"{cell[('native', p)] / cell[('native', 0)]:7.2f}" for p in pads))
    print("v/n       " + "".join(
        "      -" if not (cell[("vendor", p)] and cell[("native", p)]) else
        f"{cell[('vendor', p)] / cell[('native', p)]:7.2f}" for p in pads))
print(f"\nwrote {OUT}")

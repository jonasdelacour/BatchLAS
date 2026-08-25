#!/usr/bin/env python3
"""Render the (m, n) probe grid as GB/s, one grid per (type, transA).

Fixed ~1 GB of A across the whole grid, so the only thing that varies is SHAPE.
Cells at or above 900 GB/s are the roof; anything well below it is a cuBLAS
slow path and therefore WP7 headroom.
"""
import csv, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "slowpath_probe.csv"
SZ = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}
d = collections.defaultdict(dict)
batch = {}
err = 0.0
with open(path) as f:
    for r in csv.DictReader(f):
        if r["type"] not in SZ:
            print("FAILED ROW:", r)
            continue
        k = (r["type"], r["transA"])
        d[k][(int(r["m"]), int(r["n"]))] = float(r["GBs"])
        batch[(int(r["m"]), int(r["n"]), r["type"])] = int(r["batch"])
        err = max(err, float(r["relerr"]))

dims = sorted({m for k in d for (m, n) in d[k]})
print(f"max relerr over the whole probe: {err:.1e}\n")
for k in sorted(d):
    print(f"=== {k[0]}  transA={k[1]}   GB/s, A footprint held at ~1 GB ===")
    print("  m\\n  " + "".join(f"{n:>8d}" for n in dims))
    for m in dims:
        row = f"{m:6d}  "
        for n in dims:
            v = d[k].get((m, n))
            row += f"{v:8.0f}" if v is not None else f"{'-':>8}"
        print(row)
    lo = sorted(((v, mn) for mn, v in d[k].items()))[:5]
    print("  slowest: " + ", ".join(f"{mn[0]}x{mn[1]}={v:.0f}" for v, mn in lo))
    print()

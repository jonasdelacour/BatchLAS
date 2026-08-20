#!/usr/bin/env python3
"""Geomean over refused cells, with and without the small-min_dim tail."""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLS = {"S_256", "S_512", "A_256", "A_512", "A_320"}

cells = []
for path, col in (("ratios.csv", "ratio_auto_over_wide"),
                  ("ratios2.csv", "auto/wide"),
                  ("ratios3.csv", "auto/wide")):
    with open(os.path.join(HERE, path)) as f:
        for r in csv.DictReader(f):
            tag = r["tag"]
            if r["type"] == "cdouble" and tag in ("A_256pad1", "A_512pad1"):
                continue          # VecLen==1: the ld cannot fail, so this is a control
            if tag in CONTROLS:
                continue
            m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
            cells.append((r["type"], tag, m, n, k, int(r["batch"]), int(r["beta"]),
                          min(m, n, k), float(r[col])))


def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


for floor in (0, 24, 32, 64):
    print(f"min_dim >= {floor}:")
    for t in ("cfloat", "cdouble"):
        v = [c[8] for c in cells if c[0] == t and c[7] >= floor]
        if v:
            print(f"   {t:8} n={len(v):3d}  geomean {geo(v):6.3f}  "
                  f"min {min(v):6.3f}  max {max(v):6.3f}")

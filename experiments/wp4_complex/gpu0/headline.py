#!/usr/bin/env python3
"""Geomeans over the cells the routing gate currently REFUSES."""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLS = {"S_256", "S_512", "A_256", "A_512", "A_320"}


def rows(path, col, tagi=1):
    with open(os.path.join(HERE, path)) as f:
        return list(csv.DictReader(f))


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


buckets = {}
for r in rows("ratios.csv", None):
    key = (r["type"], "control" if r["tag"] in CONTROLS else "refused")
    buckets.setdefault(key, []).append(float(r["ratio_auto_over_wide"]))
for r in rows("ratios2.csv", None):
    tag = r["tag"]
    if tag in ("A_256pad1",) and r["type"] == "cdouble":
        tag = "A_256"        # ld cannot fail the %1 test: this is a control
    if tag in ("A_512pad1",) and r["type"] == "cdouble":
        tag = "A_512"
    key = (r["type"], "control" if tag in CONTROLS else "refused")
    buckets.setdefault(key, []).append(float(r["auto/wide"]))
for r in rows("ratios3.csv", None):
    key = (r["type"], "refused")
    buckets.setdefault(key, []).append(float(r["auto/wide"]))

print(f"{'type':9} {'bucket':9} {'cells':6} {'geomean':>8} {'min':>8} {'max':>8}")
for k in sorted(buckets):
    v = buckets[k]
    print(f"{k[0]:9} {k[1]:9} {len(v):6d} {geo(v):8.3f} {min(v):8.3f} {max(v):8.3f}")

print("\nrefused cells where the wide kernel LOSES (auto/wide < 1):")
for path, col, tag, extra in (("ratios.csv", "ratio_auto_over_wide", "tag", ""),
                              ("ratios2.csv", "auto/wide", "tag", ""),
                              ("ratios3.csv", "auto/wide", "tag", "")):
    for r in rows(path, col):
        if r[tag] in CONTROLS:
            continue
        if float(r[col]) < 1.0:
            print(f"  {r['type']:9} {r['tag']:14} "
                  f"{r['m']}x{r['n']}x{r['k']} b{r['batch']} beta{r['beta']}  {r[col]}")

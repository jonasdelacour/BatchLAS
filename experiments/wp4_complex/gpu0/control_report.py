#!/usr/bin/env python3
"""Strict guard (raw/) vs relaxed guard (raw_ctl/) on the same cells."""
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FN = re.compile(r"^(?P<tag>[A-Za-z0-9_]+)-(?P<arm>auto|wide)-b1-pad0-r(?P<rep>\d+)\.csv$")


def scalar_of(n):
    return "cfloat" if "complex<float>" in n else "cdouble"


def load(d):
    out = {}
    for p in glob.glob(os.path.join(HERE, d, "*.csv")):
        m = FN.match(os.path.basename(p))
        if not m or m.group("rep") == "0":
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                out.setdefault((scalar_of(r["name"]), m.group("tag"), m.group("arm")),
                               []).append(float(r["avg_ms"]))
    return {k: min(v) for k, v in out.items()}


a, b = load("raw"), load("raw_ctl")
print(f"{'type':8} {'tag':8} {'arm':5} {'strict_ms':>10} {'relaxed_ms':>10} {'rel/strict':>10}")
for k in sorted(b):
    if k not in a:
        continue
    print(f"{k[0]:8} {k[1]:8} {k[2]:5} {a[k]:10.5f} {b[k]:10.5f} {b[k]/a[k]:10.3f}")

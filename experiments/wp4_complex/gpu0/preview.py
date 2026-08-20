#!/usr/bin/env python3
"""Quick look at whatever raw cells exist, INCLUDING the discarded rep 0."""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "raw")
FNAME = re.compile(r"^(?P<tag>[A-Za-z0-9_]+)-(?P<arm>auto|wide|t16|vendor)-b(?P<beta>\d)"
                   r"(-pad(?P<pad>\d+))?-r(?P<rep>\d+)\.csv$")


def scalar_of(name):
    if "complex<float>" in name:
        return "cfloat"
    if "complex<double>" in name:
        return "cdouble"
    return "double" if "double" in name else "float"


d = {}
meta = {}
for path in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    mm = FNAME.match(os.path.basename(path))
    if not mm:
        continue
    with open(path) as f:
        for r in csv.DictReader(f):
            t = scalar_of(r["name"])
            key = (t, mm.group("tag"), mm.group("beta"), mm.group("pad") or "-",
                   mm.group("rep"))
            d[key + (mm.group("arm"),)] = float(r["avg_ms"])
            meta[key] = (r["arg0"], r["arg1"], r["arg2"], r["arg3"])

seen = sorted({k[:5] for k in d})
print(f"{'type':8} {'tag':10} {'beta':4} {'pad':4} {'rep':3} {'shape':22} "
      f"{'auto':>9} {'wide':>9} {'t16':>9} {'vendor':>9} {'auto/wide':>9}")
for k in seen:
    a = d.get(k + ("auto",))
    w = d.get(k + ("wide",))
    s = d.get(k + ("t16",))
    v = d.get(k + ("vendor",))
    if w is None:
        continue
    m, n, kk, b = meta[k]
    f = lambda x: f"{x:9.4f}" if x is not None else " " * 9
    r = f"{a/w:9.3f}" if a else " " * 9
    print(f"{k[0]:8} {k[1]:10} {k[2]:4} {k[3]:4} {k[4]:3} "
          f"{m}x{n}x{kk} b{b:<6} {f(a)} {f(w)} {f(s)} {f(v)} {r}")

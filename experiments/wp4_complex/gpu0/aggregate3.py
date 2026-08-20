#!/usr/bin/env python3
"""Aggregate experiments/wp4_complex/gpu0/raw3 (sweep3: where does the win stop)."""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw3")
FN = re.compile(r"^(?P<tag>[A-Za-z0-9_]+)-(?P<arm>auto|wide)-b(?P<beta>\d)-r(?P<rep>\d+)\.csv$")


def scalar_of(n):
    return "cfloat" if "complex<float>" in n else "cdouble"


g = {}
meta = {}
for p in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    m = FN.match(os.path.basename(p))
    if not m or m.group("rep") == "0":
        continue
    with open(p) as f:
        for r in csv.DictReader(f):
            k = (scalar_of(r["name"]), m.group("tag"), int(m.group("beta")))
            g.setdefault(k + (m.group("arm"),), []).append(float(r["avg_ms"]))
            meta[k] = (int(r["arg0"]), int(r["arg1"]), int(r["arg2"]), int(r["arg3"]))

hdr = ["type", "tag", "m", "n", "k", "batch", "beta", "min_dim",
       "auto_ms", "wide_ms", "auto/wide", "spread_max"]
out = []
for k in sorted(meta):
    a = g.get(k + ("auto",))
    w = g.get(k + ("wide",))
    if not (a and w):
        continue
    am, wm = min(a), min(w)
    sp = max((max(a) - min(a)) / am, (max(w) - min(w)) / wm)
    m, n, kk, b = meta[k]
    out.append([k[0], k[1], m, n, kk, b, k[2], min(m, n, kk),
                f"{am:.5f}", f"{wm:.5f}", f"{am/wm:.3f}", f"{sp:.3f}"])

out.sort(key=lambda r: (r[0], float(r[10])))
with open(os.path.join(HERE, "ratios3.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(out)

widths = [max(len(str(r[i])) for r in [hdr] + out) for i in range(len(hdr))]
for r in [hdr] + out:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)).rstrip())

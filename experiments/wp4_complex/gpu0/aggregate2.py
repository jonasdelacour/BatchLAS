#!/usr/bin/env python3
"""Aggregate experiments/wp4_complex/gpu0/raw2 (sweep2: leg cost + Tiled16 arm)."""
import csv
import glob
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw2")
FNAME = re.compile(r"^(?P<tag>[A-Za-z0-9_]+)-(?P<arm>auto|wide|t16|vendor)-b(?P<beta>\d)-r(?P<rep>\d+)\.csv$")


def scalar_of(name):
    if "complex<float>" in name:
        return "cfloat"
    if "complex<double>" in name:
        return "cdouble"
    if "double" in name:
        return "double"
    return "float"


rows = []
for path in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    m = FNAME.match(os.path.basename(path))
    if not m or m.group("rep") == "0":
        continue
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "tag": m.group("tag"), "arm": m.group("arm"),
                "beta": int(m.group("beta")), "rep": int(m.group("rep")),
                "type": scalar_of(r["name"]),
                "m": int(r["arg0"]), "n": int(r["arg1"]),
                "k": int(r["arg2"]), "batch": int(r["arg3"]),
                "avg_ms": float(r["avg_ms"]), "sd_ms": float(r["stddev_ms"]),
            })

if not rows:
    print("no data", file=sys.stderr)
    sys.exit(1)

groups = {}
for r in rows:
    groups.setdefault((r["type"], r["tag"], r["beta"], r["arm"]), []).append(r)

summary = {}
for kk, g in groups.items():
    ms = sorted(x["avg_ms"] for x in g)
    med = min(ms)   # see aggregate.py: interference only adds time
    spread = (max(ms) - min(ms)) / med if med else 0.0
    rsd = max((x["sd_ms"] / x["avg_ms"]) if x["avg_ms"] else 0.0 for x in g)
    summary[kk] = (med, spread, rsd, g[0], len(g))

hdr = ["type", "tag", "m", "n", "k", "batch", "beta",
       "auto_ms", "wide_ms", "t16_ms", "vendor_ms",
       "auto/wide", "t16/wide", "vendor/wide", "flag"]
out = []
for (t, tag, beta, arm) in list(summary):
    if arm != "wide":
        continue
    wv = summary[(t, tag, beta, "wide")]
    a = summary.get((t, tag, beta, "auto"))
    s16 = summary.get((t, tag, beta, "t16"))
    vv = summary.get((t, tag, beta, "vendor"))
    g0 = wv[3]
    flags = [f"{nm}:sp{s[1]:.2f}/rsd{s[2]:.2f}"
             for nm, s in (("auto", a), ("wide", wv), ("t16", s16), ("vendor", vv))
             if s and (s[1] > 0.10 or s[2] > 0.10)]
    fmt = lambda s: f"{s[0]:.5f}" if s else ""
    rat = lambda s: f"{s[0]/wv[0]:.3f}" if s else ""
    out.append([t, tag, g0["m"], g0["n"], g0["k"], g0["batch"], beta,
                fmt(a), fmt(wv), fmt(s16), fmt(vv),
                rat(a), rat(s16), rat(vv), ";".join(flags)])

order = {"cfloat": 0, "cdouble": 1}
out.sort(key=lambda r: (order.get(r[0], 9), r[1], r[6]))
with open(os.path.join(HERE, "ratios2.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(out)

widths = [max(len(str(r[i])) for r in [hdr] + out) for i in range(len(hdr))]
for r in [hdr] + out:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)).rstrip())

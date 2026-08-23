#!/usr/bin/env python3
"""Geomeans of tail_summary.txt, per op and per type, on BOTH readings.

The two readings are never averaged together and never reported as one number:
  common  -- the batch schedule an A/B grid would use (what the implementer's
             grid reports), which at n >= 1024 divides by a vendor that is not
             using the machine
  ceiling -- each arm at its own best measured batch
The gap between them IS the finding.
"""
import csv
import math
import os

D = os.path.dirname(os.path.abspath(__file__))
rows = []
with open(os.path.join(D, "tail_summary.txt")) as f:
    for r in csv.DictReader(f):
        rows.append(r)


def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def block(title, keyf):
    print(title)
    print("  key           n_cells  geo_common  geo_ceiling  wins_common  wins_ceiling")
    groups = {}
    for r in rows:
        groups.setdefault(keyf(r), []).append(r)
    for k in sorted(groups):
        g = groups[k]
        c = [float(r["ratio_at_common_batch"]) for r in g
             if r["ratio_at_common_batch"] != "n/a"]
        p = [float(r["ratio_at_own_ceilings"]) for r in g]
        print("  %-13s %7d  %10.3f  %11.3f  %11d  %12d"
              % (k, len(g), geo(c), geo(p),
                 sum(1 for x in c if x > 1), sum(1 for x in p if x > 1)))
    print()


block("PER OP (all four types, all seven orders)", lambda r: r["op"])
block("PER TYPE (both ops)", lambda r: r["type"])
block("PER OP x TYPE", lambda r: r["op"] + " " + r["type"])
block("PER ORDER (both ops, all types)", lambda r: "n=" + r["n"].rjust(4))

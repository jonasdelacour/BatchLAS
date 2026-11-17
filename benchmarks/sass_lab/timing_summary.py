#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pivot sass_lab timing.csv: one row per (op, type, N, n), each variant as median time / hand median."""
import collections
import csv
import sys

VARIANTS = {0: "hand", 1: "dblas", 2: "dblas_ws", 3: "lean", 4: "lean_rtwg", 5: "lean_bv",
            6: "dblas_reqd", 7: "dblas_static", 8: "hand_reqd"}


def main(path):
    groups = collections.OrderedDict()
    seen = set()
    for r in csv.DictReader(open(path)):
        v = int(r["variant"])
        seen.add(v)
        groups.setdefault((r["op"], r["type"][0], r["N"], r["n"]), {})[v] = (float(r["median_us"]),
                                                                           float(r["max_rel_err"]))
    vs = sorted(x for x in seen if x != 0)
    print("op t N n | hand_us | " + " ".join(VARIANTS.get(v, str(v)) for v in vs) + " | max_rel_err")
    for k, d in groups.items():
        h = d.get(0, (0.0, 0.0))[0]
        cells = [("%.2f" % (d[v][0] / h) if v in d and h > 0 else "-") for v in vs]
        err = max(e for _, e in d.values())
        print(" ".join(k), "| %.1f |" % h, " ".join(cells), "| %.1e" % err)


if __name__ == "__main__":
    main(sys.argv[1])

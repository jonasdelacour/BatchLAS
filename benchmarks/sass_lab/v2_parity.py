#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SASS-parity gate for device LA v2: v2 vs the lab's lean variant, per (op@WG, type, N).

  v2_parity.py SASS_CSV [--md]     (SASS_CSV from sass_stats.py --csv)
PASS when REG within +-2, BAR/SHFL/LDS/STS/LDG/STG equal, total instructions within +-3%.
Exit status 1 when any row fails.
"""
import csv
import sys

EQUAL = ["BAR", "SHFL", "LDS", "STS", "LDG", "STG"]


def cell(r):
    return "/".join(str(r[k]) for k in ["REG", "total"] + EQUAL)


def main():
    path = sys.argv[1]
    md = "--md" in sys.argv
    rows = {}
    for r in csv.DictReader(open(path)):
        if "@" not in r["op"]:
            continue
        for k in ["REG", "total"] + EQUAL:
            r[k] = int(r[k])
        rows.setdefault((r["op"], r["type"], int(r["N"])), {})[r["variant"]] = r
    fails = 0
    head = ["op@WG", "type", "N", "hand", "lean", "v2", "dREG", "dtotal", "verdict"]
    print("cells: REG/total/" + "/".join(EQUAL))
    print(("| " + " | ".join(head) + " |") if md else ",".join(head))
    if md:
        print("|" + "---|" * len(head))
    for key in sorted(rows):
        g = rows[key]
        if "lean" not in g or "v2" not in g:
            continue
        lean, v2 = g["lean"], g["v2"]
        dreg = v2["REG"] - lean["REG"]
        dtot = (v2["total"] - lean["total"]) / max(1, lean["total"])
        bad = [k for k in EQUAL if v2[k] != lean[k]]
        ok = abs(dreg) <= 2 and abs(dtot) <= 0.03 and not bad
        fails += not ok
        verdict = "PASS" if ok else "FAIL " + ",".join(bad + (["REG"] if abs(dreg) > 2 else []) +
                                                     (["total"] if abs(dtot) > 0.03 else []))
        hand = cell(g["hand"]) if "hand" in g else "-"
        vals = [key[0], key[1], str(key[2]), hand, cell(lean), cell(v2), f"{dreg:+d}", f"{dtot:+.1%}", verdict]
        print(("| " + " | ".join(vals) + " |") if md else ",".join(vals))
    print(f"\n{fails} failing rows", file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

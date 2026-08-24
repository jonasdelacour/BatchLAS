#!/usr/bin/env python3
"""getrf / getri BEFORE vs AFTER, cell by cell, against the recorded WP6 medians.

BEFORE is experiments/wp6_lu/bench/order32_*.csv and order1024_*.csv -- the
numbers WP6 published, produced by the pre-change library on the same cells with
the same WARM_S and REPS. AFTER is this directory's lu_*.csv.

The verdict rule is fixed here rather than argued downstream: a cell is UNCHANGED
when |after/before - 1| is at most 5 %, which is wp6_lu/bench/README.md section 1's
own reproduction band (it accepted 2.2 % across a rebuild and a different day, and
the largest relative sd recorded across its 982 rows was 7.2 %). Anything outside
5 % is listed individually, with its ratio, and is a finding rather than noise.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WP6 = os.path.join(HERE, "..", "..", "wp6_lu", "bench")


def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("op"):
                continue
            try:
                out[(r["op"], r["type"], int(r["n"]), int(r["batch"]))] = r
            except ValueError:
                continue
    return out


def med(r):
    try:
        return float(r["med_ms"])
    except (TypeError, ValueError):
        return None


def main():
    before = {}
    for f in ("order32_vendor.csv", "order1024_vendor.csv"):
        before.setdefault("vendor", {}).update(load(os.path.join(WP6, f)))
    for f in ("order32_native.csv", "order1024_native.csv"):
        before.setdefault("native", {}).update(load(os.path.join(WP6, f)))
    after = {"vendor": load(os.path.join(HERE, "lu_vendor.csv")),
             "native": load(os.path.join(HERE, "lu_native.csv"))}

    for arm in ("vendor", "native"):
        rs, moved, missing = [], [], []
        print("== %s arm: AFTER / BEFORE, getrf and getri" % arm)
        print("%-6s %-8s %-6s %-7s %10s %10s %8s %s"
              % ("op", "type", "n", "batch", "before", "after", "ratio", "route(after)"))
        for k in sorted(after[arm]):
            b = before[arm].get(k)
            if b is None:
                missing.append(k)
                continue
            x, y = med(b), med(after[arm][k])
            if not x or not y:
                missing.append(k)
                continue
            r = y / x
            rs.append(r)
            flag = "" if abs(r - 1.0) <= 0.05 else "   <-- MOVED"
            print("%-6s %-8s %-6d %-7d %10.4f %10.4f %8.3f %s%s"
                  % (k[0], k[1], k[2], k[3], x, y, r, after[arm][k]["route"], flag))
            if abs(r - 1.0) > 0.05:
                moved.append((k, r))
        if rs:
            g = math.exp(sum(math.log(v) for v in rs) / len(rs))
            print("   %d cells compared, GEOMEAN after/before = %.4f, "
                  "worst %.3f, best %.3f, %d outside +/-5%%"
                  % (len(rs), g, min(rs), max(rs), len(moved)))
        if missing:
            print("   %d cells had no BEFORE row: %s" % (len(missing), missing[:8]))
        print()


if __name__ == "__main__":
    main()

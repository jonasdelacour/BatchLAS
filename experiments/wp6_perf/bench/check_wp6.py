#!/usr/bin/env python3
"""DOES THE VENDOR ARM HERE REPRODUCE WP6's OWN VENDOR ARM?

Before any BEFORE/AFTER claim is made, the two directories have to agree about
what cuBLAS costs, cell by cell, on the cells they share -- otherwise a "BEFORE"
column is a different machine on a different day wearing the same label. Same
check wp6_lu/bench/README.md section 1 ran against the orchestrator's baseline
before it quoted a ratio.

Compared: every getrs cell present in BOTH wp6_lu/bench/getrs_vendor.csv and this
directory's grid/nrhs/w8/flat/flat2/flat3 vendor CSVs, and separately the
composition arm (wp6_lu getrs_native.csv against this directory's *_blocked.csv),
which is the actual BEFORE.
"""
import os

from analyse import load, med

HERE = os.path.dirname(os.path.abspath(__file__))
WP6 = os.path.join(HERE, "..", "..", "wp6_lu", "bench")
TAGS = ["grid", "nrhs", "w8", "flat", "flat2", "flat3"]


def mine(kind):
    out = {}
    for t in TAGS:
        p = os.path.join(HERE, "%s_%s.csv" % (t, kind))
        if os.path.exists(p):
            for k, r in load(p).items():
                out.setdefault(k, r)
    return out


def compare(label, wp6_path, kind):
    a = load(os.path.join(WP6, wp6_path))
    b = mine(kind)
    rs = []
    print("== %s : wp6_lu %s against this directory's %s arm" % (label, wp6_path, kind))
    print("%-8s %-6s %-6s %-7s %10s %10s %8s" % ("type", "n", "nrhs", "batch", "wp6", "here", "here/wp6"))
    for k in sorted(a):
        if k not in b:
            continue
        if a[k]["flag"] != "ok" or b[k]["flag"] != "ok":
            continue
        x, y = med(a[k]), med(b[k])
        if not x or not y:
            continue
        rs.append(y / x)
        print("%-8s %-6d %-6d %-7d %10.4f %10.4f %8.4f" % (k[1], k[2], k[3], k[4], x, y, y / x))
    if rs:
        worst = max(abs(r - 1.0) for r in rs)
        print("   %d shared cells, worst disagreement %.2f %%, %d cells above 5 %%"
              % (len(rs), 100 * worst, sum(1 for r in rs if abs(r - 1.0) > 0.05)))
    else:
        print("   no shared cells")
    print()


if __name__ == "__main__":
    compare("cuBLAS", "getrs_vendor.csv", "vendor")
    compare("the composition (BEFORE)", "getrs_native.csv", "blocked")

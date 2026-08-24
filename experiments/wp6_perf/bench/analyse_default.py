#!/usr/bin/env python3
"""THE SHIPPED DEFAULT, vendor-present, nothing set in the environment.

default_auto.csv (no pin) against grid_vendor.csv (vendor pinned, the same cells)
= what the flip is worth to a user who sets nothing. The route column decides
which side of the window each row is on -- it is READ, never assumed, and a row
whose route disagrees with the window predicate is reported as a MISMATCH rather
than quietly scored.
"""
import math
import os
import sys

from analyse import load, med, getrs_route

HERE = os.path.dirname(os.path.abspath(__file__))


def in_window(t, nrhs):
    return nrhs <= 2 or (t == "float" and nrhs <= 4)


def main():
    auto = load(os.path.join(HERE, "default_auto.csv"))
    vend = load(os.path.join(HERE, "grid_vendor.csv"))
    inside, outside, mismatch = [], [], []
    for k, ra in sorted(auto.items()):
        rv = vend.get(k)
        if rv is None or ra["flag"] != "ok" or rv["flag"] != "ok":
            continue
        t, n, nrhs, b = k[1], k[2], k[3], k[4]
        route = getrs_route(ra)
        want_native = in_window(t, nrhs)
        if want_native != (route == "native:cta"):
            mismatch.append((t, n, nrhs, b, route))
            continue
        a, v = med(ra), med(rv)
        if not a or not v:
            continue
        (inside if want_native else outside).append((v / a, t, n, nrhs, b))

    print("== THE SHIPPED DEFAULT (no pin, cuBLAS present) vs the vendor, same cells")
    if mismatch:
        print("   ROUTE MISMATCHES -- the window predicate and the resolver disagree:")
        for m in mismatch:
            print("      %-8s n=%-5d nrhs=%-3d b=%-6d route=%s" % m)
    else:
        print("   route column agrees with the window predicate on ALL %d cells"
              % (len(inside) + len(outside)))
    for name, rows in (("INSIDE the window (routes native:cta)", inside),
                       ("OUTSIDE it (routes vendor:auto)", outside)):
        if not rows:
            continue
        xs = [r[0] for r in rows]
        g = math.exp(sum(math.log(x) for x in xs) / len(xs))
        print("\n   %-38s %3d cells  geomean %.3f  min %.3f  max %.3f  losses %d"
              % (name, len(rows), g, min(xs), max(xs), sum(1 for x in xs if x < 1.0)))
        for x, t, n, nrhs, b in sorted(rows)[:5]:
            print("      worst  %-8s n=%-5d nrhs=%-3d b=%-6d  %.3f" % (t, n, nrhs, b, x))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""BEFORE / AFTER against cuBLAS, grouped the way WP6 grouped it.

WP6's headline getrs table (experiments/wp6_lu/bench/README.md section 6) is a
geomean of vendor_ms / native_ms by nrhs, over every (type, n) cell measured at
that nrhs. This reproduces that grouping exactly, for BOTH native arms, so the
BEFORE column here and WP6's published column are the same statistic and not
merely the same word:

    BEFORE  = blocked  (the composition, which is what WP6 measured)
    AFTER   = cta      (the fused narrow-RHS tier)

Cells dropped by analyse.py's rule are dropped here too, by the same code path,
and a cell where only the cta arm fell through the capacity ceiling is reported
in its own line rather than folded into either column -- because at those shapes
the shipped vendor-free route IS the composition, and pretending otherwise would
overstate the AFTER column.
"""
import math
import sys
from collections import OrderedDict

from analyse import ARMROUTE, geo, getrs_route, load, med, triage


def main(tag):
    names = ["vendor", "blocked", "cta"]
    arms = OrderedDict((a, load("%s_%s.csv" % (tag, a))) for a in names)
    keys = list(arms["vendor"])

    # Triage each native arm against the vendor SEPARATELY, so a cta capacity
    # fallback does not also delete the blocked cell that is perfectly good.
    kept_b, drop_b = triage(["vendor", "blocked"], arms, keys)
    kept_c, drop_c = triage(["vendor", "cta"], arms, keys)
    kb = set(k for k, _ in kept_b)
    kc = set(k for k, _ in kept_c)

    print("cells: %d ; vendor-vs-blocked kept %d ; vendor-vs-cta kept %d" % (len(keys), len(kb), len(kc)))
    print()
    print("DROPPED, vendor-vs-cta (the cta arm's own capacity ceiling shows up here):")
    for k, why in drop_c:
        print("   %-8s n=%-5d nrhs=%-4d b=%-6d : %s" % (k[1], k[2], k[3], k[4], "; ".join(why)))
    print()
    print("DROPPED, vendor-vs-blocked:")
    for k, why in drop_b:
        print("   %-8s n=%-5d nrhs=%-4d b=%-6d : %s" % (k[1], k[2], k[3], k[4], "; ".join(why)))
    print()

    nrhs_vals = sorted(set(k[3] for k in keys))
    types = ["float", "double", "cfloat", "cdouble"]
    orders = sorted(set(k[2] for k in keys))

    def ratio(k, arm):
        v, a = med(arms["vendor"][k]), med(arms[arm].get(k) or {})
        return (v / a) if (v and a) else None

    print("== GEOMEAN of cuBLAS_ms / native_ms, by nrhs, over every (type, n) cell")
    print("   above 1.0 means NATIVE is faster.  WP6's published BEFORE row is")
    print("   nrhs 1 -> 0.323, 2 -> 0.586, 8 -> 0.484, 32 -> 0.848, 64 -> 1.088, 128 -> 1.362")
    print()
    hdr = "%-10s" % "nrhs" + "".join("%9d" % r for r in nrhs_vals)
    print(hdr)
    for arm, label in (("blocked", "BEFORE"), ("cta", "AFTER")):
        keep = kb if arm == "blocked" else kc
        row = "%-10s" % label
        cnt = "%-10s" % ("  cells")
        win = "%-10s" % ("  wins")
        for r in nrhs_vals:
            xs = [ratio(k, arm) for k in keys if k[3] == r and k in keep]
            xs = [x for x in xs if x]
            row += ("%9.3f" % geo(xs)) if xs else ("%9s" % "-")
            cnt += "%9d" % len(xs)
            win += "%9s" % ("%d/%d" % (sum(1 for x in xs if x > 1.0), len(xs)))
        print(row)
        print(cnt)
        print(win)
    print()

    for r in nrhs_vals:
        print("== nrhs = %d : cuBLAS_ms / native_ms per (type, n).  b = BEFORE (composition), a = AFTER (fused)" % r)
        print("%-9s" % "type" + "".join("%14d" % n for n in orders))
        for t in types:
            rb = "%-9s" % (t + " b")
            rc = "%-9s" % (t + " a")
            for n in orders:
                k = ("getrs", t, n, r, None)
                match = [kk for kk in keys if kk[1] == t and kk[2] == n and kk[3] == r]
                if not match:
                    rb += "%14s" % "-"
                    rc += "%14s" % "-"
                    continue
                kk = match[0]
                xb = ratio(kk, "blocked") if kk in kb else None
                xc = ratio(kk, "cta") if kk in kc else None
                rb += "%14s" % (("%.3f" % xb) if xb else "-")
                rc += "%14s" % (("%.3f" % xc) if xc else "-")
            print(rb)
            print(rc)
        print()

    print("== AFTER / BEFORE, i.e. blocked_ms / cta_ms : how much the fused tier bought over the composition")
    for r in nrhs_vals:
        xs = []
        for k in keys:
            if k[3] != r or k not in kc or k not in kb:
                continue
            b, c = med(arms["blocked"][k]), med(arms["cta"][k])
            if b and c:
                xs.append(b / c)
        if xs:
            print("   nrhs %-4d geomean %7.3f over %d cells, min %.3f, max %.3f"
                  % (r, geo(xs), len(xs), min(xs), max(xs)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid")

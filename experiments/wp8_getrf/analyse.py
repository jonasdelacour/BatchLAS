#!/usr/bin/env python3
"""Pair two lubench6 CSVs on (op,type,n,nrhs,batch) and print ratio = A/B.

READ BY POSITION, never by DictReader: lubench6 prints 16 columns for
getrf/getri and 15 for getrs under one 16-column header, so a dict reader
returns flag=None for getrs rows. The flag is the LAST field before the
runner's appended foreign-process count.

THE DISCARD RULE is experiments/wp6_perf/bench/analyse.py's, verbatim: drop and
NAME a cell when any arm is BAD, any arm relsd > 0.10, an arm is missing, the
printed route disagrees with the expectation, or a foreign compute process was
seen on the device.
"""
import sys, math, csv


def load(path, want_route=None):
    rows = {}
    bad = []
    with open(path) as f:
        for line in f:
            fs = line.rstrip("\n").split(",")
            if not fs or fs[0] in ("op", ""):
                continue
            foreign = fs[-1]
            flag = fs[-2]
            key = (fs[0], fs[1], int(fs[2]), int(fs[3]), int(fs[4]))
            if flag != "ok":
                bad.append((key, "flag=%s" % flag)); continue
            try:
                med = float(fs[5]); relsd = float(fs[7])
            except ValueError:
                bad.append((key, "unparseable")); continue
            route = fs[11]
            if relsd > 0.10:
                bad.append((key, "relsd=%.3f" % relsd)); continue
            if foreign not in ("0",):
                bad.append((key, "foreign=%s" % foreign)); continue
            if want_route is not None and want_route not in route:
                bad.append((key, "route=%s" % route)); continue
            rows[key] = (med, route, relsd)
    return rows, bad


def main(a_path, b_path, a_route=None, b_route=None, label_a="A", label_b="B"):
    A, abad = load(a_path, a_route)
    B, bbad = load(b_path, b_route)
    keys = sorted(set(A) & set(B), key=lambda k: (k[1], k[2], k[4]))
    print("# %s vs %s   ratio = %s_ms / %s_ms  (>1 means %s faster)" %
          (a_path, b_path, label_a, label_b, label_b))
    print("type,n,batch,%s_ms,%s_ms,ratio,%s_route,%s_route" % (label_a, label_b, label_a, label_b))
    ratios = []
    for k in keys:
        am, ar, _ = A[k]
        bm, br, _ = B[k]
        r = am / bm
        ratios.append((k, r))
        print("%s,%d,%d,%.4f,%.4f,%.4f,%s,%s" % (k[1], k[2], k[4], am, bm, r, ar, br))
    if ratios:
        g = math.exp(sum(math.log(r) for _, r in ratios) / len(ratios))
        wins = sum(1 for _, r in ratios if r > 1.0)
        print("# GEOMEAN %.4f over %d cells, %d above 1.0, min %.4f, max %.4f" %
              (g, len(ratios), wins, min(r for _, r in ratios), max(r for _, r in ratios)))
        for t in ("float", "double", "cfloat", "cdouble"):
            sub = [r for k, r in ratios if k[1] == t]
            if sub:
                gt = math.exp(sum(math.log(r) for r in sub) / len(sub))
                print("#   %-8s geomean %.4f over %2d cells, min %.4f max %.4f" %
                      (t, gt, len(sub), min(sub), max(sub)))
    for k, why in abad:
        print("# DISCARD %s in %s: %s" % (str(k), a_path, why))
    for k, why in bbad:
        print("# DISCARD %s in %s: %s" % (str(k), b_path, why))
    missing = sorted(set(A) ^ set(B))
    for k in missing:
        print("# DISCARD %s: present in only one arm" % str(k))


if __name__ == "__main__":
    main(*sys.argv[1:])

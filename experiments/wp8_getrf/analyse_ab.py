#!/usr/bin/env python3
"""Score one or two passes of ab.sh.

DISCARD AND NAME a row when: flag != ok, either relsd > 0.10, bitdiff != 0,
foreign != 0, or the two arms resolved the SAME mode (which would mean the
environment read had latched and the A/B was vacuous).

With two passes, also report the CROSS-PASS MEDIAN SPREAD per cell:
max(med_p1, med_p2)/min(med_p1, med_p2) on EACH arm, and quote the worse pass's
ratio (the conservative direction).
"""
import sys, math


def load(path):
    rows, bad = {}, []
    with open(path) as f:
        for line in f:
            fs = line.rstrip("\n").split(",")
            if not fs or fs[0] in ("type", ""):
                continue
            key = (fs[0], int(fs[1]), int(fs[2]))
            flag = fs[17]
            foreign = fs[18] if len(fs) > 18 else "0"
            if flag != "ok":
                bad.append((key, "flag=%s" % flag)); continue
            try:
                mA, mB = float(fs[7]), float(fs[8])
                rsA, rsB = float(fs[10]), float(fs[11])
                bitdiff = int(fs[14])
            except ValueError:
                bad.append((key, "unparseable")); continue
            if fs[5] == fs[6]:
                bad.append((key, "modeA==modeB (vacuous A/B)")); continue
            if bitdiff != 0:
                bad.append((key, "bitdiff=%d" % bitdiff)); continue
            if max(rsA, rsB) > 0.10:
                bad.append((key, "relsd=%.3f/%.3f" % (rsA, rsB))); continue
            if foreign != "0":
                bad.append((key, "foreign=%s" % foreign)); continue
            rows[key] = (mA, mB, fs[15])
    return rows, bad


def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main(p1, p2=None):
    A, abad = load(p1)
    keys = sorted(A, key=lambda k: (k[0], k[1], k[2]))
    B = None
    if p2:
        B, bbad = load(p2)
        keys = [k for k in keys if k in B]
        abad += bbad
    hdr = "type,n,batch,route,A_p1,B_p1,ratio_p1"
    if B: hdr += ",A_p2,B_p2,ratio_p2,ratio_quoted,spreadA,spreadB"
    print(hdr)
    quoted, spreads = [], []
    for k in keys:
        a1, b1, rt = A[k]
        r1 = a1 / b1
        line = "%s,%d,%d,%s,%.4f,%.4f,%.4f" % (k[0], k[1], k[2], rt, a1, b1, r1)
        if B:
            a2, b2, _ = B[k]
            r2 = a2 / b2
            q = min(r1, r2)
            sa = max(a1, a2) / min(a1, a2)
            sb = max(b1, b2) / min(b1, b2)
            spreads += [sa, sb]
            line += ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (a2, b2, r2, q, sa, sb)
            quoted.append((k, q, rt))
        else:
            quoted.append((k, r1, rt))
        print(line)
    blocked = [(k, r) for k, r, rt in quoted if "blocked" in rt]
    other = [(k, r) for k, r, rt in quoted if "blocked" not in rt]
    print("# ALL          geomean %.4f over %d cells" % (geo([r for _, r, _ in quoted]), len(quoted)))
    if blocked:
        v = [r for _, r in blocked]
        print("# native:blocked geomean %.4f over %d cells, min %.4f, max %.4f, %d below 1.00"
              % (geo(v), len(v), min(v), max(v), sum(1 for x in v if x < 1.0)))
        for t in ("float", "double", "cfloat", "cdouble"):
            sub = [r for k, r in blocked if k[0] == t]
            if sub:
                print("#   %-8s geomean %.4f over %2d cells, min %.4f, max %.4f"
                      % (t, geo(sub), len(sub), min(sub), max(sub)))
    if other:
        v = [r for _, r in other]
        print("# NOT blocked  geomean %.4f over %d cells, min %.4f, max %.4f  (change must be inert here)"
              % (geo(v), len(v), min(v), max(v)))
    if spreads:
        spreads.sort()
        print("# CROSS-PASS median spread %.4f, worst %.4f, %d of %d above 1.10"
              % (spreads[len(spreads) // 2], spreads[-1],
                 sum(1 for s in spreads if s > 1.10), len(spreads)))
    for k, why in abad:
        print("# DISCARD %s: %s" % (str(k), why))


if __name__ == "__main__":
    main(*sys.argv[1:])

#!/usr/bin/env python3
"""Search the clause family against EVERY cdouble transposed cell measured here.

The lead's rule (B6, and item 5 of the audit brief): a preferred() clause may
ship only if EVERY cell it admits shows >= 1.15x in BOTH passes. So a clause is
scored by its WORST admitted cell, not by its average, and the fitted grid and
the out-of-sample grid are pooled -- a clause that needs one of them excluded to
survive has not survived.

Also prints the vendor's own GB/s distribution, because the whole win is a
cuBLAS dip: in every winning cell cuBLAS reads ~310-370 GB/s and in every
parity cell it reads ~810-940. Predicting the win is predicting that dip.
"""
import csv, sys, itertools

def load(paths):
    out = {}
    for p in paths:
        d = {}
        for r in csv.DictReader(open(p)):
            if r["route"] == "FAILED":
                continue
            d[(r["transA"], int(r["m"]), int(r["n"]), int(r["batch"]), r["arm"])] = (
                float(r["median_ms"]), float(r["GBs"]), int(r["MB"]))
        out[p] = d
    return out

A = "experiments/wp7_gemv/audit/"
fit = load([A + "prize_p1.csv", A + "prize_p2.csv"])
oos = load([A + "oos_p1.csv", A + "oos_p2.csv"])

def pairs(dd, tag):
    p1, p2 = list(dd.values())
    res = []
    for k in sorted({k[:4] for k in p1}):
        v1, n1 = p1.get(k + ("vendor",)), p1.get(k + ("native:cta",))
        v2, n2 = p2.get(k + ("vendor",)), p2.get(k + ("native:cta",))
        if not (v1 and n1 and v2 and n2):
            continue
        res.append((k, v1[0] / n1[0], v2[0] / n2[0], v1[2], v1[1], n1[1], tag))
    return res

cells = pairs(fit, "fit") + pairs(oos, "oos")
print("pooled cdouble transposed cells: %d  (fit %d, oos %d)"
      % (len(cells), sum(1 for c in cells if c[6] == "fit"),
         sum(1 for c in cells if c[6] == "oos")))

wins = [c for c in cells if min(c[1], c[2]) >= 1.15]
rest = [c for c in cells if min(c[1], c[2]) < 1.15]
print("vendor GB/s where native WINS >=1.15x : min %.0f  median %.0f  max %.0f  (n=%d)"
      % (min(c[4] for c in wins), sorted(c[4] for c in wins)[len(wins) // 2],
         max(c[4] for c in wins), len(wins)))
print("vendor GB/s elsewhere                 : min %.0f  median %.0f  max %.0f  (n=%d)"
      % (min(c[4] for c in rest), sorted(c[4] for c in rest)[len(rest) // 2],
         max(c[4] for c in rest), len(rest)))
print()

print("%-58s %5s %5s %6s %6s  %s" % ("clause", "adm", "wins", "worst", "missed", "verdict"))
best = []
for mlo, mhi in [(64, 320), (64, 384), (48, 320), (64, 512), (96, 320)]:
    for nmin in [128, 256, 320, 384, 512, 768]:
        for mbmin in [0, 128, 256, 512, 1024]:
            def pred(k, mb, mlo=mlo, mhi=mhi, nmin=nmin, mbmin=mbmin):
                return mlo <= k[1] <= mhi and k[2] >= nmin and mb >= mbmin
            adm = [c for c in cells if pred(c[0], c[3])]
            if not adm:
                continue
            worst = min(min(c[1], c[2]) for c in adm)
            nw = sum(1 for c in adm if min(c[1], c[2]) >= 1.15)
            missed = len(wins) - nw
            ok = worst >= 1.15
            name = "m in [%d,%d] and n >= %d%s" % (
                mlo, mhi, nmin, "" if not mbmin else " and A >= %d MB" % mbmin)
            if ok:
                best.append((nw, name, len(adm), worst, missed))
            print("%-58s %5d %5d %6.2f %6d  %s"
                  % (name, len(adm), nw, worst, missed, "PASSES" if ok else "fails"))

print()
print("=" * 92)
if best:
    best.sort(reverse=True)
    nw, name, adm, worst, missed = best[0]
    print("TIGHTEST PASSING CLAUSE, by wins captured:")
    print("   %s" % name)
    print("   admits %d cells, all >= %.2fx in both passes, captures %d of the %d measured wins"
          % (adm, worst, nw, len(wins)))
else:
    print("NO CLAUSE IN THE FAMILY PASSES. preferred() must ship ALL-FALSE.")
print("=" * 92)

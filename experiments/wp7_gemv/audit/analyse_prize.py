#!/usr/bin/env python3
"""WP7 AUDIT -- the complex<double> transposed region, and the preferred() gate.

Ratio convention: vendor_ms / native_ms. > 1 means the native CTA body is faster.

Prints, per transA, an m x n grid at each batch, the cross-pass agreement, and
then applies the lead's acceptance rule for a preferred() clause:
  a clause may cover a cell only if BOTH passes measure >= 1.15x.

Usage: analyse_prize.py p1.csv p2.csv
"""
import csv, sys, collections

def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["route"] == "FAILED" or not r["median_ms"]:
                continue
            d[(r["transA"], int(r["m"]), int(r["n"]), int(r["batch"]), r["arm"])] = (
                float(r["median_ms"]), float(r["GBs"]), r["route"], float(r["relerr"]),
                int(r["MB"]), int(r.get("foreign", 0) or 0))
    return d

p1 = load(sys.argv[1])
p2 = load(sys.argv[2]) if len(sys.argv) > 2 else {}

bad = [k for k, v in p1.items() if not v[2].startswith(k[4].split(":")[0])]
print("route-column audit: %d rows, %d disagree with the pin" % (len(p1), len(bad)))
print("correctness audit : %d rows with relerr != 0" % sum(1 for v in p1.values() if v[3]))
print("contention audit  : %d rows with a foreign compute process on the device"
      % sum(1 for v in p1.values() if v[5]))
print()

MS = sorted({k[1] for k in p1})
NS = sorted({k[2] for k in p1})
BS = sorted({k[3] for k in p1})

def ratio(p, tr, m, n, b):
    v = p.get((tr, m, n, b, "vendor")); c = p.get((tr, m, n, b, "native:cta"))
    return (v[0] / c[0], v[1], c[1], v[4]) if (v and c) else None

for tr in sorted({k[0] for k in p1}):
    name = {"T": "Trans", "C": "ConjTrans"}[tr]
    for b in BS:
        print("=" * 96)
        print("transA = %s   batch = %d      ratio = vendor/native  (p1 / p2)   [MB = footprint of A]"
              % (name, b))
        print("=" * 96)
        print("  m\\n  " + "".join("%22s" % ("n=%d" % n) for n in NS))
        for m in MS:
            cells = []
            for n in NS:
                a = ratio(p1, tr, m, n, b)
                c = ratio(p2, tr, m, n, b) if p2 else None
                if not a:
                    cells.append("%22s" % "-")
                else:
                    cells.append("%22s" % ("%.2f/%s %4dMB" % (
                        a[0], ("%.2f" % c[0]) if c else "-", a[3])))
            print("%5d  " % m + "".join(cells))
        print()

# The GB/s detail for the region the lead named.
print("=" * 96)
print("GB/s DETAIL -- the region recon named: 64 <= m <= 320, n >= 128")
print("=" * 96)
print("%-2s %5s %5s %6s %7s %11s %11s %8s %8s" %
      ("tr", "m", "n", "batch", "MB", "vendor GB/s", "native GB/s", "p1", "p2"))
for tr in sorted({k[0] for k in p1}):
    for m in MS:
        for n in NS:
            for b in BS:
                a = ratio(p1, tr, m, n, b)
                if not a:
                    continue
                c = ratio(p2, tr, m, n, b) if p2 else None
                print("%-2s %5d %5d %6d %7d %11.1f %11.1f %8.2f %8s" % (
                    tr, m, n, b, a[3], a[1], a[2], a[0],
                    ("%.2f" % c[0]) if c else "-"))

# ---- the acceptance rule -------------------------------------------------
print()
print("=" * 96)
print("preferred() ACCEPTANCE RULE: a cell qualifies only if BOTH passes are >= 1.15x")
print("=" * 96)
qual, fail, near = [], [], []
for k in sorted({k[:4] for k in p1}):
    tr, m, n, b = k
    a = ratio(p1, tr, m, n, b); c = ratio(p2, tr, m, n, b) if p2 else None
    if not (a and c):
        continue
    lo = min(a[0], c[0])
    (qual if lo >= 1.15 else (near if lo >= 0.95 else fail)).append((k, a[0], c[0], a[3]))
print("qualifying cells (both passes >= 1.15x): %d" % len(qual))
print("cells at/near parity  [0.95, 1.15)     : %d" % len(near))
print("cells BELOW 0.95x                      : %d" % len(fail))
print()
print("--- LOSING cells, worst first (these are what a clause must not admit) ---")
for k, r1, r2, mb in sorted(fail, key=lambda t: min(t[1], t[2]))[:40]:
    print("  transA=%s m=%-4d n=%-5d batch=%-5d %6dMB  %.2f / %.2f" % (k + (mb, r1, r2)))

# Test candidate predicates against the whole grid.
print()
print("=" * 96)
print("CANDIDATE PREDICATES, scored over every measured cell")
print("=" * 96)
def score(name, pred):
    adm = [(k, r1, r2, mb) for k, r1, r2, mb in qual + near + fail if pred(*k, mb)]
    if not adm:
        print("%-52s admits 0 cells" % name); return
    worst = min(min(r1, r2) for _, r1, r2, _ in adm)
    nwin = sum(1 for _, r1, r2, _ in adm if min(r1, r2) >= 1.15)
    nbad = sum(1 for _, r1, r2, _ in adm if min(r1, r2) < 1.0)
    missed = sum(1 for k, r1, r2, mb in qual if not pred(*k, mb))
    print("%-52s admits %3d  win>=1.15x %3d  BELOW 1.00x %3d  worst %.2f  misses %d wins"
          % (name, len(adm), nwin, nbad, worst, missed))

score("m in [64,320]", lambda tr, m, n, b, mb: 64 <= m <= 320)
score("m in [64,320] and n >= 128", lambda tr, m, n, b, mb: 64 <= m <= 320 and n >= 128)
score("m in [64,320] and n >= 256", lambda tr, m, n, b, mb: 64 <= m <= 320 and n >= 256)
score("m in [64,320] and n*batch >= 131072",
      lambda tr, m, n, b, mb: 64 <= m <= 320 and n * b >= 131072)
score("m in [64,320] and MB >= 256", lambda tr, m, n, b, mb: 64 <= m <= 320 and mb >= 256)
score("m in [64,320] and MB >= 512", lambda tr, m, n, b, mb: 64 <= m <= 320 and mb >= 512)
score("m in [64,384] and n >= 256 and MB >= 512",
      lambda tr, m, n, b, mb: 64 <= m <= 384 and n >= 256 and mb >= 512)
score("m in [64,320] and n >= 256 and MB >= 512",
      lambda tr, m, n, b, mb: 64 <= m <= 320 and n >= 256 and mb >= 512)
score("m in [64,320] and n >= 256 and MB >= 1024",
      lambda tr, m, n, b, mb: 64 <= m <= 320 and n >= 256 and mb >= 1024)

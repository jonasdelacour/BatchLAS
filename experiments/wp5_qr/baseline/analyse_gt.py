#!/usr/bin/env python3
"""Pair gemmtrail.csv's vendor / vendor-free rows into ratios."""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.reader(open(os.path.join(HERE, 'gemmtrail.csv'))))
d = {}
for r in rows[1:]:
    if len(r) < 13 or r[7] == 'THREW':
        continue
    bld, which, t, N, j0, b = r[0], r[1], r[2], int(r[3]), int(r[5]), int(r[6])
    d[(bld, which, t, N, j0)] = dict(m=int(r[7]), n=int(r[8]), k=int(r[9]),
                                     ms=float(r[10]), sd=float(r[11]), gf=float(r[12]), batch=b)
print(f"{'which':>5} {'type':>8} {'N':>5} {'j0':>5} {'batch':>6} {'m':>5} {'n':>5} {'k':>5} "
      f"{'vend_ms':>9} {'free_ms':>9} {'free/vend':>10} {'vend_GF':>9} {'free_GF':>9}")
for t in ('float', 'double', 'cfloat', 'cdouble'):
    for N in (256, 512, 1024, 2048):
        for j0 in (0, N // 2):
            for w in ('G1', 'G3'):
                a = d.get(('vendor', w, t, N, j0))
                b = d.get(('vendorfree', w, t, N, j0))
                if a and b:
                    print(f"{w:>5} {t:>8} {N:>5} {j0:>5} {a['batch']:>6} {a['m']:>5} {a['n']:>5} "
                          f"{a['k']:>5} {a['ms']:>9.4f} {b['ms']:>9.4f} {b['ms']/a['ms']:>10.2f} "
                          f"{a['gf']:>9.1f} {b['gf']:>9.1f}")

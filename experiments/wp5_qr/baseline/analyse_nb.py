#!/usr/bin/env python3
"""Block-width probe: PART 1 (trailing GEMM pair) and PART 2 (end-to-end WY)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
lines = [l.strip() for l in open(os.path.join(HERE, 'nb.csv')) if l.strip()]

part = 0
p1 = {}   # (type, nb, j0) -> {G1,G3}
p2 = {}   # (type, n, nb) -> ms
for l in lines:
    if l.startswith('== PART 1'):
        part = 1; continue
    if l.startswith('== PART 2'):
        part = 2; continue
    if l.startswith('build,') or l.startswith('nb_forced,'):
        continue
    p = l.split(',')
    if part == 1:
        if len(p) < 13 or p[7] in ('THREW', 'SKIP_n2_le_0'):
            continue
        t, nb, j0 = p[2], int(p[4]), int(p[5])
        m, n, k = int(p[7]), int(p[8]), int(p[9])
        e = p1.setdefault((t, nb, j0), dict(G1=0.0, G3=0.0, fl=0.0))
        e[p[1]] = float(p[10])
        e['fl'] += 128 * (8.0 if t.startswith('c') else 2.0) * m * n * k
    elif part == 2:
        if len(p) < 9 or p[5] == 'THREW':
            continue
        p2[(p[2], int(p[3]), int(p[0]))] = float(p[5])

print("PART 1 -- trailing GEMM pair, vendor-free, N=1024 batch=128.")
print("Effective GFLOP/s of G1+G3 together (higher is better). Total trailing")
print("flops barely move with nb, so the peak here is the width the BLAS-3 core wants.")
for j0 in (0, 512):
    print(f"\n  panel j0={j0}")
    hdr = "    nb  " + "".join(f"{t:>12}" for t in ('float', 'double', 'cfloat', 'cdouble'))
    print(hdr)
    for nb in (8, 16, 24, 32, 48, 56, 64, 96, 128):
        row = f"  {nb:>4}  "
        for t in ('float', 'double', 'cfloat', 'cdouble'):
            e = p1.get((t, nb, j0))
            row += f"{e['fl']/((e['G1']+e['G3'])*1e6):>12.0f}" if e and (e['G1'] + e['G3']) else f"{'-':>12}"
        print(row)

print()
print("PART 2 -- end-to-end WY apply (ormqr on an identity), vendor-free build.")
print("median ms, LOWER is better; * marks the best nb, [] the SHIPPED width.")
shipped = {256: 24, 1024: 56}
for n, b in ((256, 512), (1024, 64)):
    print(f"\n  n={n} batch={b}   (shipped ormqr width for this n: {shipped[n]})")
    print("    nb  " + "".join(f"{t:>12}" for t in ('float', 'double', 'cfloat', 'cdouble')))
    best = {t: min((v, k[2]) for k, v in p2.items() if k[0] == t and k[1] == n)
            for t in ('float', 'double', 'cfloat', 'cdouble')}
    for nb in (8, 16, 24, 32, 48, 56, 64, 96):
        row = f"  {nb:>4}  "
        for t in ('float', 'double', 'cfloat', 'cdouble'):
            v = p2.get((t, n, nb))
            if v is None:
                row += f"{'-':>12}"; continue
            mark = '*' if best[t][1] == nb else (']' if nb == shipped[n] else ' ')
            row += f"{v:>11.2f}{mark}"
        print(row)
    print("    best: " + "  ".join(f"{t}={best[t][1]} ({best[t][0]:.2f} ms)"
                                   for t in ('float', 'double', 'cfloat', 'cdouble')))
    print("    cost of the shipped width vs best: " + "  ".join(
        f"{t}={p2[(t,n,shipped[n])]/best[t][0]:.2f}x" for t in ('float', 'double', 'cfloat', 'cdouble')
        if (t, n, shipped[n]) in p2))

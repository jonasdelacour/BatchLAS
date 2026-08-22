#!/usr/bin/env python3
"""Sum panelsum.csv over every panel: the BLAS-3 lower bound for a blocked geqrf.

Compare against the measured cuSOLVER geqrf of the same (type, N, batch) in
sweep_raw.txt, which is the number WP5 is judged against.
"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))

tot = {}
for r in csv.reader(open(os.path.join(HERE, 'panelsum.csv'))):
    if len(r) < 13 or r[0] == 'build' or r[7] in ('THREW', 'SKIP_n2_le_0'):
        continue
    bld, which, t, N, nb, j0, b = r[0], r[1], r[2], int(r[3]), int(r[4]), int(r[5]), int(r[6])
    key = (bld, t, N, b)
    e = tot.setdefault(key, dict(G1=0.0, G3=0.0, n=0, flops=0.0))
    e[which] += float(r[10])
    e['n'] += 1
    m, n, k = int(r[7]), int(r[8]), int(r[9])
    e['flops'] += b * (8.0 if t.startswith('c') else 2.0) * m * n * k

# the shipped cuSOLVER geqrf, same cell
vend = {}
sec = None
for l in open(os.path.join(HERE, 'sweep_raw.txt')):
    l = l.strip()
    if l.startswith('##'):
        sec = l.split()[1]; continue
    p = l.split(',')
    if sec == 'A' and len(p) > 7 and p[0] == 'geqrf':
        vend[(p[1], int(p[2]), int(p[3]))] = float(p[4])

print("BLAS-3 LOWER BOUND for a blocked geqrf, N=1024 nb=56 batch=128")
print("(sum of the two trailing GEMMs over all 18 panels; excludes panel factorisation)")
print()
print(f"{'type':>8} {'build':>11} {'G1_ms':>9} {'G3_ms':>9} {'sum_ms':>9} {'eff_GF/s':>9} "
      f"{'cuSOLVER_geqrf_ms':>18} {'headroom_x':>11}")
for t in ('float', 'double', 'cfloat', 'cdouble'):
    for bld in ('vendor', 'vendorfree'):
        e = tot.get((bld, t, 1024, 128))
        if not e:
            continue
        s = e['G1'] + e['G3']
        v = vend.get((t, 1024, 128))
        print(f"{t:>8} {bld:>11} {e['G1']:>9.2f} {e['G3']:>9.2f} {s:>9.2f} "
              f"{e['flops']/(s*1e6):>9.1f} {v:>18.1f} {v/s:>11.1f}")
print()
print("free/vendor ratio of the BLAS-3 core:")
for t in ('float', 'double', 'cfloat', 'cdouble'):
    a = tot.get(('vendor', t, 1024, 128)); b = tot.get(('vendorfree', t, 1024, 128))
    if a and b:
        sa, sb = a['G1'] + a['G3'], b['G1'] + b['G3']
        print(f"  {t:>8}  {sb/sa:.2f}x   (G1 {b['G1']/a['G1']:.2f}x, G3 {b['G3']/a['G3']:.2f}x)")

#!/usr/bin/env python3
"""E5: on the panel shapes the library actually issues, is native faster than the vendor?"""
import csv, sys, collections, statistics
TT = {(0, 0): 'NN', (0, 1): 'NT', (1, 0): 'TN'}
v = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r['gflops'] in ('NA', ''):
        continue
    key = (int(r['m']), int(r['n']), int(r['k']), int(r['beta']),
           int(r['tA']), int(r['tB']), r['arm'])
    v[key].append(float(r['gflops']))

shapes = sorted({(k[0], k[1], k[2]) for k in v}, key=lambda x: -x[0])
print("%s, batch 128, PANEL shapes taken from the real-demand table.\n" % sys.argv[2])
print("  %-16s %-4s %-5s %10s %10s %8s %6s  %s"
      % ('shape', 'form', 'beta', 'cuBLAS', 'native', 'ratio', 'sprd', 'verdict'))
wins = losses = noise = 0
for (m, n, k) in shapes:
    for form in ((0, 0), (0, 1), (1, 0)):
        for beta in (0, 1):
            a = v.get((m, n, k, beta, form[0], form[1], 'vendor'))
            s = v.get((m, n, k, beta, form[0], form[1], 'sycl'))
            if not a or not s:
                continue
            ma, ms = statistics.median(a), statistics.median(s)
            sp = max((max(a) - min(a)) / ma, (max(s) - min(s)) / ms)
            ratio = ms / ma
            if abs(ratio - 1) <= sp:
                verdict, noise = 'noise', noise + 1
            elif ratio > 1:
                verdict, wins = 'native WINS', wins + 1
            else:
                verdict, losses = 'native loses', losses + 1
            print("  %-16s %-4s %-5d %10.1f %10.1f %7.2fx %5.1f%%  %s"
                  % ("%dx%dx%d" % (m, n, k), TT[form], beta, ma, ms, ratio, 100 * sp, verdict))
    print()
print("native wins %d, loses %d, noise %d" % (wins, losses, noise))

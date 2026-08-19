#!/usr/bin/env python3
import csv, sys, collections, statistics
TT = {(1, 0): 'TN', (0, 1): 'NT', (1, 1): 'TT', (2, 0): 'CN'}
v = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r['gflops'] in ('NA', ''):
        continue
    v[(int(r['n']), int(r['beta']), int(r['tA']), int(r['tB']), r['arm'])].append(float(r['gflops']))
print("double TRANSPOSED, square, batch 512. preferred() accepts ALL of these today")
print("(its double branch has no transpose test), so the flip would route them native.\n")
print("  %-6s %-4s %-5s %9s %9s %8s %6s  %s" % ('n', 'form', 'beta', 'cuBLAS', 'native', 'ratio', 'sprd', 'verdict'))
losses = []
for form in ((1, 0), (0, 1), (1, 1), (2, 0)):
    for n in (32, 64, 128, 256, 512):
        for beta in (0, 1):
            a = v.get((n, beta, form[0], form[1], 'vendor'))
            s = v.get((n, beta, form[0], form[1], 'sycl'))
            if not a or not s:
                continue
            ma, ms = statistics.median(a), statistics.median(s)
            sp = max((max(a) - min(a)) / ma, (max(s) - min(s)) / ms)
            ratio = ms / ma
            if abs(ratio - 1) <= sp:
                verdict = 'noise'
            elif ratio > 1:
                verdict = 'native wins'
            else:
                verdict = '*** native LOSES'
                losses.append((TT[form], n, beta, round(ratio, 2)))
            print("  %-6d %-4s %-5d %9.1f %9.1f %7.2fx %5.1f%%  %s"
                  % (n, TT[form], beta, ma, ms, ratio, 100 * sp, verdict))
    print()
print("cells where the flip would make double transposed SLOWER: %s" % (losses or 'none'))

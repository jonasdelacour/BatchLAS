#!/usr/bin/env python3
"""E4 float TRANSPOSED: is preferred()'s transposed window in the right place?

preferred() accepts transposed float only at batch >= 128 and 128 <= max_dim <= 512.
"""
import csv, sys, collections, statistics

TT = {(1, 0): 'TN', (0, 1): 'NT', (1, 1): 'TT'}

vals = collections.defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r['gflops'] in ('NA', ''):
            continue
        vals[(int(r['n']), int(r['beta']), int(r['tA']), int(r['tB']), r['arm'])].append(float(r['gflops']))


def accepted(n, batch):
    return batch >= 128 and 128 <= n <= 512


print("float transposed, square, batch 256. ratio = native / cuBLAS\n")
print("  %-6s %-4s %-5s %9s %9s %8s %6s  %-8s %s"
      % ('n', 'form', 'beta', 'cuBLAS', 'native', 'ratio', 'sprd', 'preferred', 'verdict'))
narrow, widen = [], []
for form in ((1, 0), (0, 1), (1, 1)):
    for n in sorted({k[0] for k in vals}):
        for beta in (0, 1):
            v = vals.get((n, beta, form[0], form[1], 'vendor'))
            s = vals.get((n, beta, form[0], form[1], 'sycl'))
            if not v or not s:
                continue
            mv, ms = statistics.median(v), statistics.median(s)
            sp = max((max(v) - min(v)) / mv, (max(s) - min(s)) / ms)
            ratio = ms / mv
            acc = accepted(n, 256)
            if abs(ratio - 1) <= sp:
                verdict = 'noise'
            elif ratio < 1 and acc:
                verdict = '*** NARROW: accepted but LOSES'
                narrow.append((form, n, beta, round(ratio, 2)))
            elif ratio > 1 and not acc:
                verdict = '*** WIDEN: rejected but WINS'
                widen.append((form, n, beta, round(ratio, 2)))
            else:
                verdict = 'correct'
            print("  %-6d %-4s %-5d %9.0f %9.0f %7.2fx %5.1f%%  %-8s %s"
                  % (n, TT[form], beta, mv, ms, ratio, 100 * sp, 'accept' if acc else 'reject', verdict))
    print()
print("cells arguing to NARROW: %s" % (narrow or 'none'))
print("cells arguing to WIDEN : %s" % (widen or 'none'))

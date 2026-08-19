#!/usr/bin/env python3
"""E4 float NN: is preferred()'s window in the right place?

The verdict per row is not "who is faster" but "does preferred()'s CURRENT answer agree
with the measurement". Four outcomes matter:

  accepted + native wins  -> correct
  accepted + native loses -> the window should NARROW (a live regression at the flip)
  rejected + native wins  -> the window should WIDEN (money left on the table)
  rejected + native loses -> correct
"""
import csv, sys, collections, statistics

KERN = {8: 'direct', 16: 'direct', 32: 'direct', 33: 'direct', 48: 'direct',
        64: 'reg32x32', 96: 'reg32x32', 127: 'reg32x32', 128: 'reg128x128k8',
        192: 'reg128x32k32gen', 256: 'reg128x128k8', 384: 'reg128x128k8',
        512: 'reg128x128k8', 640: 'reg128x128k8', 768: 'reg128x128k8',
        1024: 'reg128x128k8'}


def accepted(n, batch):
    # RouteTable<Op::gemm,float>::preferred(), NN, square
    if batch < 64:
        return False
    return n <= 32 or (128 <= n <= 512)


vals = collections.defaultdict(list)
batch_of = {}
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r['gflops'] in ('NA', ''):
            continue
        n, b = int(r['n']), int(r['batch'])
        batch_of[n] = b
        vals[(n, int(r['beta']), r['arm'])].append(float(r['gflops']))

print("float NN, square, RTX 4090. GFLOP/s = median of 3. Vendor SGEMM ceiling ~47000")
print("(anything near 80000 would be TF32, not FP32).\n")
print("  %-6s %-6s %-17s %-5s %9s %9s %8s %6s  %-9s %s"
      % ('n', 'batch', 'native kernel', 'beta', 'cuBLAS', 'native', 'ratio', 'sprd',
         'preferred', 'verdict'))

narrow, widen, ok = [], [], 0
peak_vendor = 0.0
for n in sorted(vals and {k[0] for k in vals}):
    for beta in (0, 1):
        v = vals.get((n, beta, 'vendor'))
        s = vals.get((n, beta, 'sycl'))
        if not v or not s:
            continue
        mv, ms = statistics.median(v), statistics.median(s)
        peak_vendor = max(peak_vendor, mv)
        spread = max((max(v) - min(v)) / mv, (max(s) - min(s)) / ms)
        ratio = ms / mv
        acc = accepted(n, batch_of[n])
        if abs(ratio - 1.0) <= spread:
            verdict = 'noise'
            ok += 1
        elif ratio > 1 and acc:
            verdict = 'correct (accepted, wins)'
            ok += 1
        elif ratio < 1 and not acc:
            verdict = 'correct (rejected, loses)'
            ok += 1
        elif ratio < 1 and acc:
            verdict = '*** NARROW: accepted but LOSES'
            narrow.append((n, beta, ratio))
        else:
            verdict = '*** WIDEN: rejected but WINS'
            widen.append((n, beta, ratio))
        print("  %-6d %-6d %-17s %-5d %9.0f %9.0f %7.2fx %5.1f%%  %-9s %s"
              % (n, batch_of[n], KERN.get(n, '?'), beta, mv, ms, ratio,
                 100 * spread, 'accept' if acc else 'reject', verdict))

print("\npeak vendor SGEMM observed: %.0f GFLOP/s %s"
      % (peak_vendor, "(FP32, plausible)" if peak_vendor < 60000 else "(TOO HIGH -- TF32?)"))
print("cells agreeing with preferred(): %d" % ok)
print("cells arguing to NARROW: %s" % (narrow or 'none'))
print("cells arguing to WIDEN : %s" % (widen or 'none'))

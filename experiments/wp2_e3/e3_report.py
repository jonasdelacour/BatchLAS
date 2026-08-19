#!/usr/bin/env python3
"""E3: is the native route actually faster than cuBLAS in the window preferred() accepts?

Reports the ratio per (n, beta) AND the run-to-run spread, because a win inside the
spread is not a win -- WP1 nearly reported a 10.9% syr2k improvement that turned out to
be noise at a shape whose spread was 13%.
"""
import csv, sys, collections, statistics

# Which kernel select_kernel_variant lands double on, measured by BATCHLAS_KERNEL_TRACE.
LANDS = {32: 'Direct', 48: 'Tiled16', 64: 'Tiled16', 96: 'Tiled16', 136: 'Tiled16',
         200: 'Tiled16', 256: 'wide-64x64', 320: 'wide-64x64', 512: 'wide-64x64'}

vals = collections.defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r['gflops'] in ('NA', '', None):
            continue
        try:
            vals[(int(r['n']), int(r['batch']), int(r['beta']), r['arm'])].append(float(r['gflops']))
        except ValueError:
            continue

ns = sorted({k[0] for k in vals})
print("double, batch 512, RTX 4090. GFLOP/s = median of 3; spread = (max-min)/median.")
print("Vendor sanity anchor: a 4090 is 1/64 FP64, so DGEMM must not exceed ~1450 GFLOP/s.\n")
print("  %-6s %-12s %-5s %10s %10s %8s   %s" %
      ('n', 'native kern', 'beta', 'cuBLAS', 'native', 'ratio', 'verdict'))

wins = losses = pushes = 0
for n in ns:
    for beta in (0, 1):
        v = vals.get((n, 512, beta, 'vendor'), [])
        s = vals.get((n, 512, beta, 'sycl'), [])
        if not v or not s:
            continue
        mv, ms = statistics.median(v), statistics.median(s)
        spread = max((max(v) - min(v)) / mv if mv else 0,
                     (max(s) - min(s)) / ms if ms else 0)
        ratio = ms / mv if mv else 0
        # A difference smaller than the observed spread is not a result.
        if abs(ratio - 1.0) <= spread:
            verdict = "NOISE (spread %.0f%%)" % (100 * spread)
            pushes += 1
        elif ratio > 1.0:
            verdict = "native WINS %.2fx" % ratio
            wins += 1
        else:
            verdict = "native LOSES %.2fx" % ratio
            losses += 1
        print("  %-6d %-12s %-5d %10.1f %10.1f %8.3f   %s" %
              (n, LANDS.get(n, '?'), beta, mv, ms, ratio, verdict))

print("\ncells: %d native wins, %d native losses, %d inside noise" % (wins, losses, pushes))

anchor = max((statistics.median(v) for k, v in vals.items() if k[3] == 'vendor'), default=0)
print("peak vendor DGEMM observed: %.1f GFLOP/s %s"
      % (anchor, "(plausible)" if anchor <= 1500 else "(IMPLAUSIBLE -- not FP64)"))

#!/usr/bin/env python3
"""Cross-pass merge for the WP7 R4 baseline.

Campaign rule: a heavy-tailed rep distribution can fail a 10% relative-sd rule
while the MEDIAN reproduces to four significant figures, so quote CROSS-PASS
medians, not one pass. This prints, per cell, both pass medians and the ratio,
and separates:

  * cells where the two passes AGREE and the cell is below the roof
    -> a real cuBLAS slow path, i.e. genuine WP7 headroom;
  * cells where the two passes DISAGREE -> measurement noise, not headroom.
"""
import csv, sys

L2 = 72 * 1024**2
SZ = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}
ROOF = 950.0    # GB/s: the level the best cells in this sweep actually reach.


def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["type"] not in SZ:
                continue
            k = (r["type"], int(r["m"]), int(r["n"]), int(r["batch"]), r["transA"])
            d[k] = (float(r["median_ms"]), float(r["GBs"]), float(r["rel_sd"]),
                    float(r["relerr"]))
    return d


a = load(sys.argv[1])
b = load(sys.argv[2])
keys = [k for k in a if k in b]

rows = []
for k in keys:
    ty, m, n, bt, tr = k
    fp = m * n * bt * SZ[ty]
    ma, ga, sa, ea = a[k]
    mb, gb, sb, eb = b[k]
    g = min(ga, gb)                    # the conservative cross-pass number
    rows.append(dict(k=k, ty=ty, m=m, n=n, b=bt, tr=tr, fp=fp, l2=fp <= L2,
                     ga=ga, gb=gb, gmin=g, ratio=max(ga, gb) / min(ga, gb),
                     sa=sa, sb=sb, err=max(ea, eb)))

print(f"cells compared: {len(rows)};  max relerr over both passes: "
      f"{max(r['err'] for r in rows):.1e}")

hdr = (f"{'type':8} {'m':>5} {'n':>5} {'batch':>6} {'tr':>7} {'A_MB':>8} "
       f"{'p1 GB/s':>9} {'p2 GB/s':>9} {'spread':>7} {'%roof(min)':>11}")

dram = [r for r in rows if not r["l2"]]
rep  = [r for r in dram if r["ratio"] < 1.10]     # reproduced across passes
noisy = [r for r in dram if r["ratio"] >= 1.10]

print("\n### REPRODUCED cells below 85% of the 950 GB/s roof "
      "(= real cuBLAS slow paths = WP7 headroom)")
print(hdr)
hits = sorted((r for r in rep if r["gmin"] / ROOF < 0.85), key=lambda r: r["gmin"])
for r in hits:
    print(f"{r['ty']:8} {r['m']:5d} {r['n']:5d} {r['b']:6d} {r['tr']:>7} "
          f"{r['fp']/1024**2:8.1f} {r['ga']:9.1f} {r['gb']:9.1f} "
          f"{r['ratio']:7.2f} {100*r['gmin']/ROOF:10.0f}%")
if not hits:
    print("  (none)")

print("\n### cells that did NOT reproduce across passes (noise, not headroom)")
print(hdr)
for r in sorted(noisy, key=lambda r: -r["ratio"]):
    print(f"{r['ty']:8} {r['m']:5d} {r['n']:5d} {r['b']:6d} {r['tr']:>7} "
          f"{r['fp']/1024**2:8.1f} {r['ga']:9.1f} {r['gb']:9.1f} "
          f"{r['ratio']:7.2f} {100*r['gmin']/ROOF:10.0f}%")

print("\n### full cross-pass table, DRAM-resident, worst min-of-passes first")
print(hdr)
for r in sorted(dram, key=lambda r: r["gmin"]):
    print(f"{r['ty']:8} {r['m']:5d} {r['n']:5d} {r['b']:6d} {r['tr']:>7} "
          f"{r['fp']/1024**2:8.1f} {r['ga']:9.1f} {r['gb']:9.1f} "
          f"{r['ratio']:7.2f} {100*r['gmin']/ROOF:10.0f}%")

print("\n### L2-resident cells (A fits in the 4090's 72 MB L2 -- NOT a DRAM number)")
print(hdr)
for r in sorted((r for r in rows if r["l2"]), key=lambda r: -r["gmin"]):
    print(f"{r['ty']:8} {r['m']:5d} {r['n']:5d} {r['b']:6d} {r['tr']:>7} "
          f"{r['fp']/1024**2:8.1f} {r['ga']:9.1f} {r['gb']:9.1f} "
          f"{r['ratio']:7.2f} {100*r['gmin']/ROOF:10.0f}%")

ok = [r for r in rep if r["gmin"] / ROOF >= 0.85]
print(f"\nDRAM cells: {len(dram)}   reproduced: {len(rep)}   noisy: {len(noisy)}")
print(f"reproduced AND >=85% of {ROOF:.0f} GB/s: {len(ok)}   "
      f"reproduced AND <85%: {len(hits)}")

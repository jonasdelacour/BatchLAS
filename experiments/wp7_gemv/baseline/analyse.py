#!/usr/bin/env python3
"""Render the WP7 R4 vendor-gemv baseline CSV as a table, and flag the cells that
are far from the DRAM roof.

The extra column the CSV cannot carry is FOOTPRINT: A is re-read every rep, so a
cell whose A fits in the 4090's 72 MB L2 is measuring L2 bandwidth, not DRAM, and
its "fraction of 900 GB/s" is meaningless (it goes over 100%). Those rows are
marked L2 and excluded from the headroom verdict.
"""
import csv, sys, collections

ROOF = 900.0            # GB/s achievable on an RTX 4090
L2   = 72 * 1024**2     # bytes

SZ = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}

path = sys.argv[1] if len(sys.argv) > 1 else "vendor_baseline.csv"
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        if r["type"] == "FAILED" or r.get("m") is None:
            continue
        m, n, b = int(r["m"]), int(r["n"]), int(r["batch"])
        fp = m * n * b * SZ[r["type"]]
        rows.append(dict(
            ty=r["type"], m=m, n=n, b=b, tr=r["transA"],
            ms=float(r["median_ms"]), rsd=float(r["rel_sd"]),
            gbs=float(r["GBs"]), frac=float(r["frac_of_900"]),
            err=float(r["relerr"]), fp=fp, l2=(fp <= L2)))

hdr = f"{'type':8} {'m':>5} {'n':>5} {'batch':>6} {'tr':>7} {'A_MB':>8} {'med_ms':>9} {'GB/s':>8} {'%roof':>6} {'rsd':>6} {'relerr':>9}"
for tr in ("NoTrans", "Trans"):
    print(f"\n=== transA = {tr} ===")
    print(hdr)
    for r in rows:
        if r["tr"] != tr:
            continue
        tag = " L2" if r["l2"] else ""
        print(f"{r['ty']:8} {r['m']:5d} {r['n']:5d} {r['b']:6d} {r['tr']:>7} "
              f"{r['fp']/1024**2:8.1f} {r['ms']:9.4f} {r['gbs']:8.1f} "
              f"{100*r['frac']:5.0f}%{tag} {r['rsd']:6.3f} {r['err']:9.1e}")

print("\n=== DRAM-resident cells sorted by %roof (worst first) ===")
dram = sorted((r for r in rows if not r["l2"]), key=lambda r: r["frac"])
print(hdr)
for r in dram:
    print(f"{r['ty']:8} {r['m']:5d} {r['n']:5d} {r['b']:6d} {r['tr']:>7} "
          f"{r['fp']/1024**2:8.1f} {r['ms']:9.4f} {r['gbs']:8.1f} "
          f"{100*r['frac']:5.0f}% {r['rsd']:6.3f} {r['err']:9.1e}")

bad = [r for r in dram if r["frac"] < 0.85]
print(f"\nDRAM-resident cells: {len(dram)}; below 85% of {ROOF:.0f} GB/s: {len(bad)}")
worst = [r for r in rows if r["err"] > 1e-4]
print(f"cells with relerr > 1e-4: {len(worst)}")

print("\n=== mean %roof by (type, transA) over DRAM-resident cells ===")
g = collections.defaultdict(list)
for r in dram:
    g[(r["ty"], r["tr"])].append(r["frac"])
for k in sorted(g):
    v = g[k]
    print(f"{k[0]:8} {k[1]:>7}  n={len(v):2d}  mean {100*sum(v)/len(v):5.1f}%  "
          f"min {100*min(v):5.1f}%  max {100*max(v):5.1f}%")

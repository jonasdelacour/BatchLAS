#!/usr/bin/env python3
"""THE ORDER AXIS, WITH BATCH HELD FIXED -- twice, at batch 32 and at batch 1024.

Two fixed-batch sweeps rather than one, because a single one cannot tell an order
crossover from a batch crossover wearing its clothes. That mistake is on this
campaign's record (WP5 published order crossovers from a sweep whose batch
schedule varied with n), and for LU it would have been a large error: the two
sweeps below differ by more than a factor of three in geomean at the SAME orders.

Also prints the getrs axis tables, whose three axes (n, nrhs, batch) are each
varied with the other two fixed.
"""
import csv
import os

D = os.path.dirname(os.path.abspath(__file__))


def load(p, keyf):
    r = {}
    p = os.path.join(D, p)
    if not os.path.exists(p):
        return r
    for row in csv.reader(open(p)):
        if not row or row[0] == "op" or len(row) < 12:
            continue
        try:
            r[keyf(row)] = float(row[5])
        except ValueError:
            pass
    return r


K4 = lambda row: (row[0], row[1], int(row[2]), int(row[4]))
K5 = lambda row: (row[0], row[1], int(row[2]), int(row[3]), int(row[4]))
TYPES = ("float", "double", "cfloat", "cdouble")


def table(title, cols, colhdr, cellf):
    print("=== %s" % title)
    print("  %-9s %s" % ("type", " ".join((colhdr % c).ljust(9) for c in cols)))
    for t in TYPES:
        row = []
        for c in cols:
            v = cellf(t, c)
            row.append(("%-9.3f" % v) if v else "%-9s" % "-")
        print("  %-9s %s" % (t, " ".join(row)))
    print()


for tag, vf, nf, batch, orders in (
        ("batch FIXED = 32", "order32_vendor.csv", "order32_native.csv", 32,
         [32, 64, 128, 256, 512, 1024, 2048]),
        ("batch FIXED = 1024", "order1024_vendor.csv", "order1024_native.csv", 1024,
         [32, 64, 128, 256, 512])):
    V, N = load(vf, K4), load(nf, K4)
    for op in ("getrf", "getri"):
        table("%s -- ORDER axis, %s (speedup native/vendor)" % (op, tag),
              orders, "n=%d",
              lambda t, n, op=op, V=V, N=N, b=batch:
                  (V[(op, t, n, b)] / N[(op, t, n, b)])
                  if (op, t, n, b) in V and (op, t, n, b) in N else None)

V, N = load("getrs_vendor.csv", K5), load("getrs_native.csv", K5)
if V and N:
    table("getrs -- NRHS axis, n=512 and batch=256 BOTH FIXED",
          [1, 2, 8, 32, 128, 512], "nrhs=%d",
          lambda t, r: (V[("getrs", t, 512, r, 256)] / N[("getrs", t, 512, r, 256)])
          if ("getrs", t, 512, r, 256) in V and ("getrs", t, 512, r, 256) in N else None)
    table("getrs -- BATCH axis, n=512 and nrhs=8 BOTH FIXED",
          [64, 128, 256, 512, 1024], "b=%d",
          lambda t, b: (V[("getrs", t, 512, 8, b)] / N[("getrs", t, 512, 8, b)])
          if ("getrs", t, 512, 8, b) in V and ("getrs", t, 512, 8, b) in N else None)
    SCHED = {64: 8192, 256: 2048, 512: 512, 1024: 128, 2048: 32}
    for nr in (1, 64):
        table("getrs -- ORDER axis at nrhs=%d (batch on the saturating schedule, "
              "so this table alone cannot separate order from batch)" % nr,
              sorted(SCHED), "n=%d",
              lambda t, n, nr=nr: (V[("getrs", t, n, nr, SCHED[n])] /
                                   N[("getrs", t, n, nr, SCHED[n])])
              if ("getrs", t, n, nr, SCHED[n]) in V and
                 ("getrs", t, n, nr, SCHED[n]) in N else None)

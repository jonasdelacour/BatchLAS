#!/usr/bin/env python3
"""Join a vendor CSV and a native CSV into an A/B table.

DISCARD RULE, stated before it is applied and applied to every table here:
  1. either arm's row flagged BAD (a residual over Tol<T>, a non-zero info, a
     zero non-trivial-pivot count, a pivot mismatch against the host LAPACKE) ->
     DISCARDED and NAMED. A fast wrong answer is not a result.
  2. either arm's relative sd > 10% -> DISCARDED and NAMED.
  3. a missing arm (timeout or throw) -> DISCARDED and NAMED.
Surviving cells only enter the geomean, and the surviving count is printed.

The RESOLVED ROUTE of each arm is carried into every row, so a "native" cell that
silently resolved to the vendor shows up as a route string rather than as parity.

usage: analyse_ab.py <vendor.csv> <native.csv> [label]
"""
import csv
import math
import os
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0] == "op" or len(r) < 12:
                continue
            key = (r[0], r[1], int(r[2]), int(r[3]), int(r[4]))
            if r[5] in ("TIMEOUT_OR_THROW", "THREW"):
                rows[key] = None
                continue
            try:
                rows[key] = {
                    "med": float(r[5]), "relsd": float(r[7]), "gf": float(r[8]),
                    "res": float(r[9]), "ws": int(r[10]), "route": r[11],
                    "flag": r[-1],
                }
            except ValueError:
                rows[key] = None
    return rows


def main():
    vpath, npath = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(vpath)
    V, N = load(vpath), load(npath)
    out, discards, per_op = [], [], {}
    for k in sorted(set(V) | set(N), key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        op, t, n, nrhs, b = k
        v, m = V.get(k), N.get(k)
        if v is None or m is None:
            discards.append("%s %s n=%d nrhs=%d b=%d: missing/threw arm" % k)
            continue
        if v["flag"] != "ok" or m["flag"] != "ok":
            discards.append("%s %s n=%d nrhs=%d b=%d: flag vendor=%s native=%s"
                            % (op, t, n, nrhs, b, v["flag"], m["flag"]))
            continue
        if v["relsd"] > 0.10 or m["relsd"] > 0.10:
            discards.append("%s %s n=%d nrhs=%d b=%d: relsd vendor=%.3f native=%.3f"
                            % (op, t, n, nrhs, b, v["relsd"], m["relsd"]))
            continue
        sp = v["med"] / m["med"]
        out.append((op, t, n, nrhs, b, v["med"], m["med"], sp, v["gf"], m["gf"],
                    v["route"], m["route"], v["res"], m["res"], v["ws"], m["ws"]))
        per_op.setdefault(op, []).append(sp)

    print("# %s" % label)
    print("op,type,n,nrhs,batch,vendor_ms,native_ms,speedup,vendor_GFLOPs,"
          "native_GFLOPs,vendor_route,native_route,vendor_resid,native_resid,"
          "vendor_ws,native_ws")
    for r in out:
        print("%s,%s,%d,%d,%d,%.4f,%.4f,%.3f,%.1f,%.1f,%s,%s,%.2e,%.2e,%d,%d" % r)
    print()
    for op, sps in sorted(per_op.items()):
        g = math.exp(sum(math.log(s) for s in sps) / len(sps))
        wins = sum(1 for s in sps if s > 1.0)
        print("GEOMEAN %s: %.3fx over %d cells, %d wins, worst %.3fx, best %.3fx"
              % (op, g, len(sps), wins, min(sps), max(sps)))
    if discards:
        print("\nDISCARDED / NAMED:")
        for d in discards:
            print("  " + d)
    else:
        print("\nDISCARDED: none")


if __name__ == "__main__":
    main()

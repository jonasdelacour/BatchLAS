#!/usr/bin/env python3
"""Turn grid_vendor.csv and grid_native.csv into the A/B tables.

Rules applied, all of them recorded rather than assumed:
  * a cell is DISCARDED and NAMED when its relative sd exceeds 10% in either arm;
  * a cell is DISCARDED and NAMED when either arm's row is flagged BAD -- a fast
    wrong answer is not a result;
  * the route each arm resolved to is carried into the table, so a row where the
    "native" arm silently resolved to the vendor is visible instead of being
    reported as parity;
  * geomeans are over the SURVIVING cells only, and the count is printed.
"""
import csv
import math
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0] in ("op", "") or len(r) < 15:
                continue
            if r[5] in ("TIMEOUT_OR_THROW", "THREW"):
                rows[(r[0], r[1], int(r[2]), int(r[4]))] = None
                continue
            try:
                rows[(r[0], r[1], int(r[2]), int(r[4]))] = {
                    "med": float(r[5]), "relsd": float(r[7]), "gf": float(r[8]),
                    "res": float(r[9]), "ws": int(r[10]), "route": r[11],
                    "flag": r[-1],
                }
            except ValueError:
                rows[(r[0], r[1], int(r[2]), int(r[4]))] = None
    return rows


def main():
    V = load(os.path.join(D, "grid_vendor.csv"))
    N = load(os.path.join(D, "grid_native.csv"))
    out = []
    discards = []
    per_op = {}
    keys = sorted(set(V) | set(N), key=lambda k: (k[0], k[1], k[2]))
    for k in keys:
        op, t, n, b = k
        v, m = V.get(k), N.get(k)
        if v is None or m is None:
            discards.append(f"{op} {t} n={n} b={b}: missing/threw arm")
            continue
        if v["flag"] != "ok" or m["flag"] != "ok":
            discards.append(f"{op} {t} n={n} b={b}: flag vendor={v['flag']} native={m['flag']}")
            continue
        if v["relsd"] > 0.10 or m["relsd"] > 0.10:
            discards.append(f"{op} {t} n={n} b={b}: relsd vendor={v['relsd']:.3f} "
                            f"native={m['relsd']:.3f}")
            continue
        sp = v["med"] / m["med"]
        out.append((op, t, n, b, v["med"], m["med"], sp, v["gf"], m["gf"],
                    v["route"], m["route"], v["res"], m["res"], v["ws"], m["ws"]))
        per_op.setdefault(op, []).append(sp)

    print("op,type,n,batch,vendor_ms,native_ms,speedup,vendor_GFLOPs,native_GFLOPs,"
          "vendor_route,native_route,vendor_resid,native_resid,vendor_ws,native_ws")
    for r in out:
        print("%s,%s,%d,%d,%.4f,%.4f,%.3f,%.1f,%.1f,%s,%s,%.2e,%.2e,%d,%d" % r)

    print()
    for op, sps in sorted(per_op.items()):
        g = math.exp(sum(math.log(s) for s in sps) / len(sps))
        wins = sum(1 for s in sps if s > 1.0)
        print(f"GEOMEAN {op}: {g:.3f}x over {len(sps)} cells, {wins} wins, "
              f"worst {min(sps):.3f}x, best {max(sps):.3f}x")
    if discards:
        print("\nDISCARDED / NAMED:")
        for d in discards:
            print("  " + d)
    else:
        print("\nDISCARDED: none")


if __name__ == "__main__":
    main()

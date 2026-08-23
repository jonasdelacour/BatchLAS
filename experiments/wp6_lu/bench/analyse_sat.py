#!/usr/bin/env python3
"""The saturation map: us PER BATCH ITEM against batch, at fixed n.

THE RULE, written down before it is applied. For each (op, type, n, arm) let
u(b) be the median us/item. The ladder SATURATES at batch b* when

    u(b*) <= 1.05 * min_b u(b)      -- b* is within 5% of the best point seen
    and u is flat across the top     -- |u(b_last)/u(b_prev) - 1| <= 0.05

The FIRST b meeting the first condition is reported as b_sat. If no point on the
ladder meets it, or the top of the ladder is still falling by more than 5%, the
rung is reported NOT SATURATED and every ratio taken there must be labelled.

A rung is also flagged when u RISES at the top: the best batch is then interior,
so the grid's batch schedule is PESSIMISTIC for that arm, and by how much is
printed rather than silently corrected.
"""
import csv
import os

D = os.path.dirname(os.path.abspath(__file__))


def load(*paths):
    rows = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        _load_into(rows, path)
    return rows


def _load_into(rows, path):
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0] == "op" or len(r) < 12:
                continue
            if r[5] in ("TIMEOUT_OR_THROW", "THREW"):
                continue
            try:
                med = float(r[5])
                relsd = float(r[7])
            except ValueError:
                continue
            b = int(r[4])
            rows[(r[0], r[1], int(r[2]), b)] = {
                "us_item": med * 1000.0 / b, "ms": med, "relsd": relsd,
                "route": r[11], "flag": r[-1],
            }


def report(name, R):
    print("===== %s =====" % name)
    keys = sorted({(k[0], k[1], k[2]) for k in R})
    verdict = {}
    for op, t, n in keys:
        pts = sorted([(k[3], R[k]) for k in R if k[:3] == (op, t, n)])
        if not pts:
            continue
        bad = [str(b) for b, d in pts if d["flag"] != "ok"]
        noisy = ["%d(%.0f%%)" % (b, d["relsd"] * 100) for b, d in pts if d["relsd"] > 0.10]
        us = [d["us_item"] for _, d in pts]
        best = min(us)
        bbest = pts[us.index(best)][0]
        bsat = None
        for b, d in pts:
            if d["us_item"] <= 1.05 * best:
                bsat = b
                break
        tail_flat = len(us) >= 2 and abs(us[-1] / us[-2] - 1.0) <= 0.05
        sat = bsat is not None and tail_flat
        rises = us[-1] > 1.05 * best
        line = "  ".join("%d:%.3f" % (b, d["us_item"]) for b, d in pts)
        v = ("SAT@%d" % bsat) if sat else "NOT-SAT"
        if rises:
            v += " (best interior @%d, top is %.2fx worse)" % (bbest, us[-1] / best)
        verdict[(op, t, n)] = (v, bsat if sat else None, bbest, best)
        print("%-6s %-8s n=%-5d %s" % (op, t, n, v))
        print("        us/item  %s" % line)
        if bad:
            print("        BAD rows at batch %s" % ",".join(bad))
        if noisy:
            print("        relsd>10%% at batch %s" % ",".join(noisy))
    print()
    return verdict


V = load(os.path.join(D, "sat_vendor.csv"), os.path.join(D, "sat2_vendor.csv"),
         os.path.join(D, "tail_vendor.csv"))
N = load(os.path.join(D, "sat_native.csv"), os.path.join(D, "sat2_native.csv"),
         os.path.join(D, "tail_native.csv"))
vv = report("VENDOR (cuBLAS, build/, routes pinned vendor)", V)
nn = report("NATIVE (build-novendor/, no pin)", N)

print("===== CEILING-TO-CEILING: each arm at ITS OWN best batch =====")
print("op,type,n,vendor_best_batch,vendor_us_item,native_best_batch,"
      "native_us_item,ceiling_ratio,vendor_saturates")
for k in sorted(set(vv) & set(nn)):
    vs, vsat, vb, vu = vv[k]
    ns, nsat, nb, nu = nn[k]
    print("%s,%s,%d,%d,%.4f,%d,%.4f,%.3f,%s" %
          (k[0], k[1], k[2], vb, vu, nb, nu, vu / nu, "yes" if vsat else "NO"))

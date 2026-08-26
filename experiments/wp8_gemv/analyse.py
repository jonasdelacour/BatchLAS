#!/usr/bin/env python3
"""WP8/I3 -- score a sweep.sh CSV: vendor vs a native arm, two passes.

Rows are REFUSED and NAMED when relerr != 0, the printed route != the pinned
arm, rel_sd > 0.10, or foreign != 0. Nothing is silently dropped.

usage: analyse.py <p1.csv> <p2.csv> [--base ARM] [--test ARM] [--group KEYS]
"""
import csv, sys, math
from collections import OrderedDict

COLS = ["arm","type","m","n","batch","transA","route","median_ms","mean_ms",
        "rel_sd","GBs","frac_of_950","relerr","ld","out_len","red_len","MB","foreign"]

def load(path):
    rows = {}
    refused = []
    with open(path) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        for r in rd:
            if len(r) < len(COLS):
                refused.append((tuple(r), "short row")); continue
            d = dict(zip(COLS, r))
            key = (d["type"], d["out_len"], d["red_len"], d["batch"], d["transA"], d["arm"])
            if d["route"] == "FAILED":
                refused.append((key, "FAILED")); continue
            try:
                med = float(d["median_ms"]); rsd = float(d["rel_sd"])
                rel = float(d["relerr"]); gbs = float(d["GBs"])
                fgn = int(d["foreign"])
            except ValueError:
                refused.append((key, "unparseable")); continue
            if d["route"] != d["arm"] and not (d["arm"] == "vendor" and d["route"].startswith("vendor")):
                refused.append((key, "route %s != pin %s" % (d["route"], d["arm"]))); continue
            if rel != 0.0:
                refused.append((key, "relerr %.2e" % rel)); continue
            if rsd > 0.10:
                refused.append((key, "rel_sd %.3f" % rsd)); continue
            if fgn != 0:
                refused.append((key, "foreign %d" % fgn)); continue
            rows[key] = dict(med=med, gbs=gbs, mb=d["MB"], m=d["m"], n=d["n"])
    return rows, refused

def main():
    a = sys.argv[1]; b = sys.argv[2]
    base = "vendor"; test = "native:cta"
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--base": base = args[i+1]; i += 2
        elif args[i] == "--test": test = args[i+1]; i += 2
        else: i += 1
    r1, ref1 = load(a); r2, ref2 = load(b)
    for name, ref in (("p1", ref1), ("p2", ref2)):
        if ref:
            print("REFUSED in %s: %d" % (name, len(ref)))
            for k, why in ref: print("   ", k, why)
    cells = OrderedDict()
    for k in r1:
        ty, ol, rl, bt, tr, arm = k
        if arm != test: continue
        cells.setdefault((ty, int(ol), int(rl), int(bt), tr), None)
    print("type      out_len red_len batch tr   %-9s %-9s   %-9s %-9s   r_p1  r_p2  spread  MB" %
          (base+"GB/s", test+"GB/s", base+"ms", test+"ms"))
    ratios1 = []; ratios2 = []; rows = []
    for (ty, ol, rl, bt, tr) in cells:
        kb1 = (ty, str(ol), str(rl), str(bt), tr, base); kt1 = (ty, str(ol), str(rl), str(bt), tr, test)
        if kb1 not in r1 or kt1 not in r1 or kb1 not in r2 or kt1 not in r2:
            print("  MISSING", ty, ol, rl, bt, tr); continue
        v1 = r1[kb1]; t1 = r1[kt1]; v2 = r2[kb1]; t2 = r2[kt1]
        # ratio = base_ms / test_ms  (>1 means the TEST arm is faster)
        rr1 = v1["med"] / t1["med"]; rr2 = v2["med"] / t2["med"]
        sp = max(rr1, rr2) / min(rr1, rr2)
        ratios1.append(rr1); ratios2.append(rr2)
        rows.append((ty, ol, rl, bt, tr, v1["gbs"], t1["gbs"], v1["med"], t1["med"], rr1, rr2, sp, t1["mb"]))
        print("%-9s %7d %7d %5d %2s   %9.1f %9.1f   %9.5f %9.5f   %5.2f %5.2f  %6.4f  %s" %
              (ty, ol, rl, bt, tr, v1["gbs"], t1["gbs"], v1["med"], t1["med"], rr1, rr2, sp, t1["mb"]))
    if ratios1:
        def geo(v): return math.exp(sum(math.log(x) for x in v)/len(v))
        print("\ncells %d   geomean p1 %.4f  p2 %.4f   min p1 %.4f  p2 %.4f   max p1 %.4f"
              % (len(ratios1), geo(ratios1), geo(ratios2), min(ratios1), min(ratios2), max(ratios1)))
        sp = [max(x,y)/min(x,y) for x,y in zip(ratios1, ratios2)]
        sp.sort()
        print("cross-pass spread: median %.4f  worst %.4f  count>1.10 %d"
              % (sp[len(sp)//2], sp[-1], sum(1 for x in sp if x > 1.10)))

main()

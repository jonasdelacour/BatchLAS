#!/usr/bin/env python3
"""Score audit/parity.sh output BEFORE (repair/parity_r{1,2}) vs AFTER
(wp8_gemv/parity_w8_p{1,2}) -- native:cta against the vendor, per cell.

The point of this file is the B6 line: every cell below 0.50x must be fixed or
STATED. It prints the before/after ratio for every (type, out_len, red_len,
transA) cell the parity grid holds, and lists what is still below 0.50x.
"""
import csv, sys, math
from collections import defaultdict

COLS = ["arm","type","m","n","batch","transA","route","median_ms","mean_ms","rel_sd",
        "GBs","frac_of_950","relerr","ld","out_len","red_len","foreign"]

def load(paths):
    got = defaultdict(dict)
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r['route'] == 'FAILED' or not r['median_ms']:
                continue
            if float(r['relerr']) != 0 or int(r['foreign']) != 0:
                continue
            k = (r['type'], int(r['out_len']), int(r['red_len']), int(r['batch']), r['transA'])
            got[k].setdefault(r['arm'], []).append((float(r['median_ms']), float(r['GBs'])))
    return got

before = load(sys.argv[1:3])
after = load(sys.argv[3:5])

def med(v):
    v = sorted(v); return v[len(v)//2]

print("%-8s %6s %6s %5s %2s | %-22s | %-22s" %
      ("type","out","red","batch","tr","BEFORE vend/nat  ratio","AFTER  vend/nat  ratio"))
worse, blockers_before, blockers_after = [], [], []
for k in sorted(set(before) & set(after), key=lambda x: (x[0], x[1], x[2], x[4])):
    b, a = before[k], after[k]
    if 'vendor' not in b or 'native:cta' not in b: continue
    if 'vendor' not in a or 'native:cta' not in a: continue
    bv, bn = med([x[1] for x in b['vendor']]), med([x[1] for x in b['native:cta']])
    av, an = med([x[1] for x in a['vendor']]), med([x[1] for x in a['native:cta']])
    br = med([x[0] for x in b['vendor']]) / med([x[0] for x in b['native:cta']])
    ar = med([x[0] for x in a['vendor']]) / med([x[0] for x in a['native:cta']])
    flag = ""
    if br < 0.50: blockers_before.append((k, br, ar))
    if ar < 0.50: blockers_after.append((k, br, ar)); flag = "  <- STILL BELOW 0.50x"
    if an < bn * 0.97: worse.append((k, bn, an)); flag += "  <- NATIVE GOT SLOWER"
    print("%-8s %6d %6d %5d %2s | %7.1f %7.1f %6.3f | %7.1f %7.1f %6.3f%s" %
          (k[0], k[1], k[2], k[3], k[4], bv, bn, br, av, an, ar, flag))

print("\ncells below 0.50x BEFORE: %d" % len(blockers_before))
for k, br, ar in blockers_before: print("   %s  %.3f -> %.3f" % (str(k), br, ar))
print("cells below 0.50x AFTER: %d" % len(blockers_after))
for k, br, ar in blockers_after: print("   %s  %.3f -> %.3f" % (str(k), br, ar))
print("cells where the NATIVE arm got slower by >3%%: %d" % len(worse))
for k, bn, an in worse: print("   %s  %.1f -> %.1f GB/s" % (str(k), bn, an))

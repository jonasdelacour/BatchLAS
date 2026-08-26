#!/usr/bin/env python3
"""The (out_len, red_len) plane for the SHIPPED `auto` decision, two passes.

ratio = body3_median / body5_median, so > 1 means BODY 5 IS FASTER.
A cell where the gate declines (wA == 1) is NOT refused here -- it is the point:
it says body 3 ran on both arms, which is what the gate is for. Such rows are
printed with W = "-" and a ratio of 1.00 by construction, and they are excluded
from the geomean of the ADMITTED set.
"""
import csv, sys, math
from collections import defaultdict

files = sys.argv[1:]
d = defaultdict(dict)
refused = []
for path in files:
    tag = 'p1' if path.endswith('p1.csv') else 'p2'
    for r in csv.DictReader(open(path)):
        k = (r['type'], int(r['out_len']), int(r['red_len']), int(r['batch']), r['transA'])
        if not r['ratio']:
            refused.append((tag, k, 'FAILED')); continue
        if float(r['relerr_a']) != 0 or float(r['relerr_b']) != 0:
            refused.append((tag, k, 'relerr %s/%s' % (r['relerr_a'], r['relerr_b']))); continue
        if int(r['foreign']) != 0:
            refused.append((tag, k, 'foreign %s' % r['foreign'])); continue
        d[k][tag] = r

if refused:
    print("REFUSED %d rows (relerr / FAILED / foreign only; rel_sd is handled by the two-pass spread):" % len(refused))
    for t, k, why in refused: print("   %s %s : %s" % (t, k, why))
    print()

print("%-8s %6s %6s %5s %3s | %8s %8s | %6s %6s | %6s | %6s" %
      ("type", "out", "red", "batch", "W", "b3 GB/s", "b5 GB/s", "r_p1", "r_p2", "spread", "relsd"))
adm, dec = [], []
worst = None
for k in sorted(d, key=lambda x: (x[0], x[1], x[2], x[3])):
    v = d[k]
    if 'p1' not in v or 'p2' not in v:
        print("  INCOMPLETE", k); continue
    r1, r2 = float(v['p1']['ratio']), float(v['p2']['ratio'])
    wA, wB = int(v['p1']['wA']), int(v['p1']['wB'])
    assert wB == 1, "arm B must always be body 3"
    sp = max(r1, r2) / min(r1, r2)
    rsd = max(float(v['p1']['relsd_a']), float(v['p1']['relsd_b']),
              float(v['p2']['relsd_a']), float(v['p2']['relsd_b']))
    tag = str(wA) if wA > 1 else "-"
    print("%-8s %6d %6d %5d %3s | %8.1f %8.1f | %6.3f %6.3f | %6.4f | %6.3f%s" %
          (k[0], k[1], k[2], k[3], tag,
           float(v['p1']['GBs_b']), float(v['p1']['GBs_a']), r1, r2, sp, rsd,
           "" if wA > 1 else "   <- gate declined: body 3 on BOTH arms"))
    if wA > 1:
        adm.append((min(r1, r2), k))
    else:
        dec.append((min(r1, r2), k))

def geo(v): return math.exp(sum(math.log(x) for x in v) / len(v))
if adm:
    rs = [x[0] for x in adm]
    adm.sort()
    print("\nADMITTED (body 5 ran): %d cells   geomean(worse pass) %.4f   MIN %.4f %s   MAX %.4f"
          % (len(rs), geo(rs), adm[0][0], adm[0][1], adm[-1][0]))
    print("   cells below 1.00: %d    below 1.05: %d" %
          (sum(1 for x in rs if x < 1.00), sum(1 for x in rs if x < 1.05)))
    for r, k in adm[:6]: print("      worst: %.4f %s" % (r, k))
if dec:
    rs = [x[0] for x in dec]
    dec.sort()
    print("\nDECLINED (gate sent both arms to body 3): %d cells, ratios %.4f..%.4f (must be ~1.00)"
          % (len(rs), min(rs), max(rs)))

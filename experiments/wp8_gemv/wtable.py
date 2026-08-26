#!/usr/bin/env python3
"""Tabulate a segab.sh CSV as ratio-vs-W per (type, out_len, red_len).

REFUSES and NAMES a row when wA == wB (the gate declined and the row compares an
arm with itself -- campaign trap 4 in miniature), relerr != 0 on either arm,
rel_sd > 0.10 on either arm, or foreign != 0.
"""
import csv, sys, math
from collections import defaultdict

files = [a for a in sys.argv[1:] if not a.startswith('--')]
data = defaultdict(dict)   # (type,out,red,batch,tr) -> W -> [ratio per file]
gbsb = defaultdict(dict)
refused = []
for fi, path in enumerate(files):
    for r in csv.DictReader(open(path)):
        k = (r['type'], int(r['out_len']), int(r['red_len']), int(r['batch']), r['transA'])
        w = r['arm']
        if not r['ratio']:
            refused.append((path, k, w, 'FAILED')); continue
        if r['wA'] == r['wB']:
            refused.append((path, k, w, 'gate declined: wA==wB==%s' % r['wA'])); continue
        if float(r['relerr_a']) != 0 or float(r['relerr_b']) != 0:
            refused.append((path, k, w, 'relerr %s/%s' % (r['relerr_a'], r['relerr_b']))); continue
        if float(r['relsd_a']) > 0.10 or float(r['relsd_b']) > 0.10:
            refused.append((path, k, w, 'rel_sd %s/%s' % (r['relsd_a'], r['relsd_b']))); continue
        if int(r['foreign']) != 0:
            refused.append((path, k, w, 'foreign %s' % r['foreign'])); continue
        data[k].setdefault(w, []).append(float(r['ratio']))
        gbsb[k].setdefault(w, []).append((float(r['GBs_b']), float(r['GBs_a'])))

if refused:
    print("REFUSED %d rows:" % len(refused))
    for p, k, w, why in refused:
        print("   %s %s W=%s : %s" % (p.split('/')[-1], k, w, why))
    print()

WS = ['2', '4', '8']
print("%-8s %6s %6s %5s %2s  %8s | %-22s | %-22s | best" %
      ("type", "out", "red", "batch", "tr", "b3 GB/s", "body5 GB/s  W=2/4/8", "ratio      W=2/4/8"))
for k in sorted(data, key=lambda x: (x[0], x[1], x[2], x[3], x[4])):
    d = data[k]
    g = gbsb[k]
    if not d: continue
    b3 = g[list(g)[0]][0][0]
    def med(v):
        v = sorted(v); return v[len(v)//2]
    rs = {w: (med(d[w]) if w in d else None) for w in WS}
    gs = {w: (med([x[1] for x in g[w]]) if w in g else None) for w in WS}
    best = max((v, w) for w, v in rs.items() if v is not None)[1]
    print("%-8s %6d %6d %5d %2s  %8.1f | %7s %7s %7s | %6s %6s %6s | W=%s" %
          (k[0], k[1], k[2], k[3], k[4], b3,
           *["%.1f" % gs[w] if gs[w] is not None else "-" for w in WS],
           *["%.3f" % rs[w] if rs[w] is not None else "-" for w in WS],
           best))

#!/usr/bin/env python3
# Where does cuBLAS getrf / getri stop scaling with batch?
#
# The number that matters is us PER BATCH ITEM. A routine that saturates has a
# flat per-item curve above some batch; a routine whose wall time is nearly
# independent of batch (WP5's geqrfBatched) has a per-item curve that keeps
# FALLING, and every wall-clock ratio taken against it flatters the other side.
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else 'sat.csv')))
by = {}
for r in rows:
    if r['flag'] != 'ok': continue
    by.setdefault((r['op'], r['type'], int(r['n'])), []).append(
        (int(r['batch']), float(r['med_ms']), float(r['relsd']), float(r['GFLOPs'])))

for k in sorted(by, key=lambda k: (k[0], k[1], k[2])):
    op, t, n = k
    xs = sorted(by[k])
    print("\n== %s %s n=%d ==" % (op, t, n))
    print("%8s %12s %14s %12s %10s %s" % ("batch", "med_ms", "us/item", "GFLOP/s", "relsd", "note"))
    base = None
    for b, ms, sd, gf in xs:
        us = ms * 1000.0 / b
        if base is None: base = us
        note = "DISCARD relsd>10%" if sd > 0.10 else ""
        print("%8d %12.4f %14.4f %12.2f %10.4f %s" % (b, ms, us, gf, sd, note))
    # saturation: the smallest batch whose us/item is within 5% of the minimum
    clean = [(b, ms * 1000.0 / b) for b, ms, sd, gf in xs if sd <= 0.10]
    if clean:
        best = min(u for _, u in clean)
        sat = min(b for b, u in clean if u <= best * 1.05)
        print("   -> flat (within 5%% of the best us/item) from batch %d; best %.4f us/item" % (sat, best))

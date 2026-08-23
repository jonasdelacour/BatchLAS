#!/usr/bin/env python3
# What does pivoting cost, split into swap and search, against the unpivoted
# lower bound. Every cell is printed, including the ones that disagree.
import csv, sys, math
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else 'pivot.csv')))
DISCARD = 0.10
key, disc, fail = {}, [], []
for r in rows:
    if r['flag'] is None or 'LAUNCH_FAIL' in r['flag']:
        fail.append(r); continue
    if r['flag'] != 'ok':
        fail.append(r); continue
    if float(r['relsd']) > DISCARD:
        disc.append(r); continue
    key[(r['section'], r['variant'], r['type'], int(r['n']), int(r['batch']))] = r

print("== LAUNCH_FAIL / BAD ==")
for r in fail:
    print("   %s %s %s n=%s batch=%s wg=%s slm=%s : %s" %
          (r['section'], r['variant'], r['type'], r['n'], r['batch'], r['wg'], r['slm_bytes'], r['flag']))
if not fail: print("   (none)")
print("== discarded, relative sd > 10% ==")
for r in disc:
    print("   %s %s %s n=%s batch=%s relsd=%s" %
          (r['section'], r['variant'], r['type'], r['n'], r['batch'], r['relsd']))
if not disc: print("   (none)")

def table(section, title, groupby):
    print("\n== %s ==" % title)
    print("%-8s %5s %7s %10s %10s %10s %10s %8s %8s %8s %12s %12s" %
          ("type", "n", "batch", "nopiv_ms", "swap_ms", "pivman_ms", "pivgrp_ms",
           "swap/np", "man/np", "grp/np", "slm_nopiv", "slm_pivman"))
    seen = set()
    for k in key:
        if k[0] != section: continue
        seen.add((k[2], k[3], k[4]))
    for t, n, b in sorted(seen, key=lambda x: (x[0], x[1], x[2])):
        g = {v: key.get((section, v, t, n, b)) for v in ('nopiv', 'swaponly', 'pivman', 'pivgrp')}
        if not g['nopiv']: continue
        np_ = float(g['nopiv']['med_ms'])
        def ms(v): return float(g[v]['med_ms']) if g[v] else float('nan')
        print("%-8s %5d %7d %10.4f %10.4f %10.4f %10.4f %8.2f %8.2f %8.2f %12s %12s" %
              (t, n, b, np_, ms('swaponly'), ms('pivman'), ms('pivgrp'),
               ms('swaponly')/np_, ms('pivman')/np_, ms('pivgrp')/np_,
               g['nopiv']['slm_bytes'], g['pivman']['slm_bytes'] if g['pivman'] else '-'))

table('n', 'pivot cost vs the unpivoted lower bound, by n (batch 4096, wg 256)', 'n')
table('batch', 'pivot cost vs the unpivoted lower bound, by batch (float n=64, wg 256)', 'batch')
table('hole', 'the n ladder across the 48 KB band (float, batch 1024, wg 256)', 'n')

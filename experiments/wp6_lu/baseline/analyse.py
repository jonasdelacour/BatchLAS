#!/usr/bin/env python3
# Summarise grid_norm.csv. Prints EVERY cell, including the ones that disagree
# with the conclusion, and names every cell discarded for relative sd > 10%.
import csv, sys, math

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else 'grid_norm.csv')))
DISCARD = 0.10

def f(r, k):
    try: return float(r[k])
    except: return float('nan')

key = {}
disc = []
bad = []
for r in rows:
    if r['flag'] != 'ok':
        bad.append(r); continue
    if f(r, 'relsd') > DISCARD:
        disc.append(r); continue
    key[(r['laswp'], r['op'], r['type'], int(r['n']), int(r['nrhs']), int(r['batch']))] = r

print("== discarded, relative sd > %.0f%% ==" % (DISCARD*100))
for r in disc:
    print("   %s %s %s n=%s nrhs=%s b=%s relsd=%s" %
          (r['laswp'], r['op'], r['type'], r['n'], r['nrhs'], r['batch'], r['relsd']))
if not disc: print("   (none)")
print("== flagged BAD/THREW ==")
for r in bad: print("   ", r)
if not bad: print("   (none)")

types = ['float', 'double', 'cfloat', 'cdouble']
ns = sorted({int(r['n']) for r in rows})

print("\n== 1. cuBLAS getrf: ms and GFLOP/s at saturating batch (2n^3/3 per item) ==")
print("%-8s %6s %7s %12s %12s %10s" % ("type", "n", "batch", "med_ms", "GFLOP/s", "resid"))
for t in types:
    for n in ns:
        for k, r in key.items():
            if k[0] == '-' and k[1] == 'getrf' and k[2] == t and k[3] == n:
                print("%-8s %6d %7d %12.4f %12.2f %10s" %
                      (t, n, k[5], f(r, 'med_ms'), f(r, 'GFLOPs'), r['resid']))

def ratio(op_v, op_c, laswp, nrhs):
    print("\n== %s: cuBLAS vs routed-trsm composition (laswp=%s, nrhs=%s) ==" % (op_v, laswp, nrhs))
    print("%-8s %6s %7s %12s %12s %8s  %s" %
          ("type", "n", "batch", "vendor_ms", "comp_ms", "speedup", "trsm route"))
    for t in types:
        for n in ns:
            v = c = None
            for k, r in key.items():
                if k[2] == t and k[3] == n and k[4] == nrhs:
                    if k[0] == '-' and k[1] == op_v: v = r
                    if k[0] == laswp and k[1] == op_c: c = r
            if v and c:
                print("%-8s %6d %7s %12.4f %12.4f %8.2fx  %s" %
                      (t, n, v['batch'], f(v, 'med_ms'), f(c, 'med_ms'),
                       f(v, 'med_ms') / f(c, 'med_ms'), c['route']))
            elif v or c:
                print("%-8s %6d  INCOMPLETE (vendor=%s comp=%s)" % (t, n, bool(v), bool(c)))

ratio('getri', 'getri_trsm', 'list', 1)
ratio('getri', 'getri_trsm', 'gather', 1)
ratio('getrs', 'getrs_trsm', 'list', 1)
ratio('getrs', 'getrs_trsm', 'gather', 1)
ratio('getrs', 'getrs_trsm', 'list', 64)
ratio('getrs', 'getrs_trsm', 'gather', 64)

print("\n== workspace, bytes ==")
print("%-8s %6s %7s %5s %14s %14s %14s" %
      ("type", "n", "batch", "nrhs", "vendor", "comp(list)", "comp(gather)"))
for op_v, op_c, nrhs in (('getri', 'getri_trsm', 1), ('getrs', 'getrs_trsm', 1), ('getrs', 'getrs_trsm', 64)):
    for t in types:
        for n in ns:
            v = key.get(('-', op_v, t, n, nrhs, 0))
            vs = [r for k, r in key.items() if k[0] == '-' and k[1] == op_v and k[2] == t and k[3] == n and k[4] == nrhs]
            ls = [r for k, r in key.items() if k[0] == 'list' and k[1] == op_c and k[2] == t and k[3] == n and k[4] == nrhs]
            gs = [r for k, r in key.items() if k[0] == 'gather' and k[1] == op_c and k[2] == t and k[3] == n and k[4] == nrhs]
            if vs and ls and gs:
                print("%-8s %6d %7s %5d %14s %14s %14s" %
                      (t, n, vs[0]['batch'], nrhs, vs[0]['ws_bytes'], ls[0]['ws_bytes'], gs[0]['ws_bytes']))

def geo(xs):
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')

print("\n== geomean speedups (composition over cuBLAS), and win counts ==")
for op_v, op_c, laswp, nrhs in (('getri','getri_trsm','list',1), ('getri','getri_trsm','gather',1),
                                ('getrs','getrs_trsm','list',1), ('getrs','getrs_trsm','gather',1),
                                ('getrs','getrs_trsm','list',64), ('getrs','getrs_trsm','gather',64)):
    rs = []
    for t in types:
        for n in ns:
            v = [r for k, r in key.items() if k[0]=='-' and k[1]==op_v and k[2]==t and k[3]==n and k[4]==nrhs]
            c = [r for k, r in key.items() if k[0]==laswp and k[1]==op_c and k[2]==t and k[3]==n and k[4]==nrhs]
            if v and c: rs.append(f(v[0],'med_ms')/f(c[0],'med_ms'))
    print("  %-12s laswp=%-7s nrhs=%-4d cells=%2d  geomean %.2fx  wins %d  worst %.2fx  best %.2fx" %
          (op_v, laswp, nrhs, len(rs), geo(rs), sum(1 for x in rs if x > 1),
           min(rs) if rs else float('nan'), max(rs) if rs else float('nan')))

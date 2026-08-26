#!/usr/bin/env python3
"""Score candidate gemv preferred() clauses against the G6 grid.

TWO THINGS THIS DOES THAT audit/clause_search.py COULD NOT.

  1. batch IS A CLAUSE TERM. The WP7 search enumerated (m band) x (n threshold)
     x (A threshold) only, so every 'REFUTED' verdict it produced was a verdict
     about clauses that cannot see the axis the effect lives on.
  2. EVERY REJECTION NAMES ITS CELL. A clause is refuted by a cell, not by a
     geomean, and 'no predicate exists' is only a result if the cells that refute
     each candidate are on the record.

Axes are out_len and red_len -- NEVER m and n. Under Trans/ConjTrans out_len is
A.cols() and red_len is A.rows(); a predicate written on the wrong extent inverts
the window, and that exact error was caught twice in WP7.
"""
import sys, math, itertools, collections

def load(path, relsd_cap=0.10):
    """-> {(ty,out,red,batch,tr): (ratio, vendor_gbs, native_gbs, MB)}, refusals"""
    cells, refused = {}, []
    arms = collections.defaultdict(dict)
    for i, line in enumerate(open(path)):
        if i == 0: continue
        f = line.rstrip('\n').split(',')
        if len(f) < 19: continue
        arm, ty, m, n, b, tr, route = f[0], f[1], int(f[2]), int(f[3]), int(f[4]), f[5], f[6]
        out, red, gpu, foreign = int(f[14]), int(f[15]), f[17], f[18]
        # THE SWEEP'S OWN MB COLUMN IS WRONG FOR EVERY TYPE BUT complex<double>:
        # g6_sweep.sh computes m*n*b*16 unconditionally. Recompute it here rather
        # than edit a script that sweeps were already running under -- a
        # footprint column that silently reads 4x high for float would make an
        # L2-resident control look DRAM-resident, which is the one thing the
        # column exists to prevent.
        esz = {'float': 4, 'double': 8, 'cfloat': 8, 'cdouble': 16}[ty]
        mb = m * n * b * esz // 1048576
        key = (ty, out, red, b, tr)
        if route == 'FAILED':
            refused.append((key, arm, 'FAILED')); continue
        med, relsd, gbs, relerr = float(f[7]), float(f[9]), float(f[10]), float(f[12])
        bad = None
        if foreign != '0':            bad = f'foreign={foreign}'
        elif relsd > relsd_cap:       bad = f'relsd={relsd:.4f}'
        elif relerr != 0.0:           bad = f'relerr={relerr:g}'
        elif arm == 'vendor'     and not route.startswith('vendor'): bad = f'route={route}'
        elif arm == 'native:cta' and route != 'native:cta':          bad = f'route={route}'
        if bad: refused.append((key, arm, bad)); continue
        arms[key][arm] = (med, gbs, mb)
    for key, a in arms.items():
        if 'vendor' not in a or 'native:cta' not in a:
            refused.append((key, '-', 'missing arm')); continue
        cells[key] = (a['vendor'][0] / a['native:cta'][0], a['vendor'][1],
                      a['native:cta'][1], a['native:cta'][2])
    return cells, refused

def merge(paths):
    """Pool passes; a cell survives only if EVERY pass measured it cleanly."""
    per = [load(p) for p in paths]
    common = set(per[0][0])
    for c, _ in per[1:]: common &= set(c)
    out = {}
    for k in common:
        rs = [p[0][k][0] for p in per]
        out[k] = dict(rs=rs, q=min(rs), spread=max(rs)/min(rs),
                      ven=per[0][0][k][1], nat=per[0][0][k][2], mb=per[0][0][k][3])
    refused = [(p, r) for p, per_p in zip(paths, per) for r in per_p[1]]
    return out, refused

def score(cells, pred):
    adm = {k: v for k, v in cells.items() if pred(k)}
    if not adm: return None
    qs = [v['q'] for v in adm.values()]
    g = math.exp(sum(math.log(q) for q in qs) / len(qs))
    worst = min(adm.items(), key=lambda kv: kv[1]['q'])
    return dict(n=len(adm), geo=g, min=min(qs), worst=worst,
                loss=sum(1 for q in qs if q < 1.0),
                sub=sum(1 for q in qs if q < 1.15), adm=adm)

def fmt(k):
    ty, out, red, b, tr = k
    return f"{ty} out={out} red={red} b={b} {tr}"

def report(name, cells, pred):
    s = score(cells, pred)
    if s is None:
        print(f"  {name:52s} EMPTY -- no measured cell"); return None
    verdict = "PASS" if (s['sub'] == 0 and s['min'] >= 1.15) else "FAIL"
    print(f"  {name:52s} n={s['n']:3d} geo={s['geo']:5.3f} min={s['min']:.4f} "
          f"loss={s['loss']:2d} sub1.15={s['sub']:2d}  {verdict}")
    if verdict == "FAIL":
        wk, wv = s['worst']
        print(f"      REFUTED BY  {fmt(wk):40s} ratio {wv['q']:.4f} "
              f"(vendor {wv['ven']:.1f} GB/s, native {wv['nat']:.1f}, {wv['mb']} MB)")
    return s

if __name__ == '__main__':
    cells, refused = merge(sys.argv[1:])
    print(f"# {len(cells)} cells paired across {len(sys.argv)-1} pass(es), "
          f"{len(refused)} arm-rows refused")
    for p, (k, arm, why) in refused:
        print(f"# REFUSED {fmt(k)} [{arm}] {why}")
    sp = sorted(v['spread'] for v in cells.values())
    if sp and len(sys.argv) > 2:
        print(f"# cross-pass median spread {sp[len(sp)//2]:.4f} worst {sp[-1]:.4f} "
              f"{sum(1 for s in sp if s>1.10)} above 1.10")
    print("\n=== FULL CELL TABLE (out_len, red_len axes) ===")
    print("type,out_len,red_len,batch,transA,MB,vendor_GBs,native_GBs," +
          ",".join(f"r_p{i+1}" for i in range(len(sys.argv)-1)) + ",QUOTED,spread")
    for k in sorted(cells):
        v = cells[k]
        print(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{v['mb']},{v['ven']:.1f},{v['nat']:.1f}," +
              ",".join(f"{x:.4f}" for x in v['rs']) + f",{v['q']:.4f},{v['spread']:.4f}")

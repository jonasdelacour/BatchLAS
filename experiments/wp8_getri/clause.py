#!/usr/bin/env python3
"""Score candidate LU clauses over one or more passes, and NAME the refuting cell.

usage: clause.py <csv> [<csv> ...]

The QUOTED ratio for a cell is the MINIMUM over passes -- the conservative
direction, which is what GATE-B and GATE-C ask a table to show. A clause PASSES
only if every admitted cell is at or above 1.15 on its QUOTED value.
"""
import sys, math
from analyse import pair

def load(paths):
    per, refused = pair(paths)
    common = set(per[0])
    for p in per[1:]: common &= set(p)
    return ({c: dict(rs=[p[c][0] for p in per],
                     q=min(p[c][0] for p in per),
                     spread=max(p[c][0] for p in per) / min(p[c][0] for p in per),
                     route=per[0][c][3]) for c in common},
            refused)

def score(cells, pred):
    a = {c: v for c, v in cells.items() if pred(c)}
    if not a: return None
    qs = [v['q'] for v in a.values()]
    return dict(n=len(a), geo=math.exp(sum(map(math.log, qs)) / len(qs)),
                min=min(qs), loss=sum(1 for q in qs if q < 1.0),
                sub=sum(1 for q in qs if q < 1.15),
                worst=min(a.items(), key=lambda kv: kv[1]['q']), adm=a)

def report(name, cells, pred):
    s = score(cells, pred)
    if s is None:
        print(f"  {name:46s} EMPTY"); return None
    ok = s['sub'] == 0
    print(f"  {name:46s} n={s['n']:3d} geo={s['geo']:6.3f} min={s['min']:.4f} "
          f"loss={s['loss']:2d} sub1.15={s['sub']:2d}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        (op, t, n, q, b), v = s['worst']
        print(f"      REFUTED BY {t} n={n} nrhs={q} batch={b}  ratio {v['q']:.4f}  "
              f"(passes {', '.join(f'{x:.4f}' for x in v['rs'])})")
    return s

if __name__ == '__main__':
    cells, refused = load(sys.argv[1:])
    print(f"# {len(cells)} cells paired across {len(sys.argv)-1} pass(es), "
          f"{len(refused)} refusals")
    for p, c, why in refused: print(f"# REFUSED {c}: {why}")
    sp = sorted(v['spread'] for v in cells.values())
    if sp and len(sys.argv) > 2:
        print(f"# cross-pass median spread {sp[len(sp)//2]:.4f} worst {sp[-1]:.4f} "
              f"{sum(1 for s in sp if s > 1.10)} above 1.10")

    G = lambda op: (lambda c: c[0] == op)
    print("\n=== getri ===")
    for t, th in (('float', 128), ('float', 64), ('float', 256),
                  ('cfloat', 256), ('cfloat', 128), ('cfloat', 512),
                  ('double', 512), ('double', 256), ('double', 1024),
                  ('cdouble', 512), ('cdouble', 1024)):
        report(f"getri {t} order>={th}",
               cells, lambda c, t=t, th=th: c[0]=='getri' and c[1]==t and c[2]>=th)
    report("getri SHIPPED float>=128 | cfloat>=256", cells,
           lambda c: c[0]=='getri' and ((c[1]=='float' and c[2]>=128) or
                                        (c[1]=='cfloat' and c[2]>=256)))

    print("\n=== getrs (batch >= 128 unless stated) ===")
    for t, th in (('float', 128), ('float', 64), ('double', 128), ('double', 64),
                  ('cfloat', 128), ('cdouble', 128)):
        report(f"getrs {t} nrhs>={th} batch>=128", cells,
               lambda c, t=t, th=th: c[0]=='getrs' and c[1]==t and c[3]>=th and c[4]>=128)
    report("getrs SHIPPED float|double nrhs>=128 batch>=128", cells,
           lambda c: c[0]=='getrs' and c[1] in ('float','double') and c[3]>=128 and c[4]>=128)
    report("getrs SAME BUT NO BATCH FLOOR", cells,
           lambda c: c[0]=='getrs' and c[1] in ('float','double') and c[3]>=128)

    print("\n=== getrf ===")
    for t, th in (('float', 256), ('float', 128), ('float', 512),
                  ('cfloat', 512), ('cfloat', 256), ('cfloat', 128),
                  ('double', 512), ('cdouble', 1024)):
        report(f"getrf {t} order>={th}", cells,
               lambda c, t=t, th=th: c[0]=='getrf' and c[1]==t and c[2]>=th)
    report("getrf SHIPPED float>=256 | cfloat>=512", cells,
           lambda c: c[0]=='getrf' and ((c[1]=='float' and c[2]>=256) or
                                        (c[1]=='cfloat' and c[2]>=512)))
    report("getrf WIDER   float>=256 | cfloat>=256", cells,
           lambda c: c[0]=='getrf' and ((c[1]=='float' and c[2]>=256) or
                                        (c[1]=='cfloat' and c[2]>=256)))

    print("\n=== FULL TABLE ===")
    print("op,type,order,nrhs,batch,route," +
          ",".join(f"p{i+1}" for i in range(len(sys.argv)-1)) + ",QUOTED")
    for c in sorted(cells):
        v = cells[c]
        print(f"{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{v['route']}," +
              ",".join(f"{x:.4f}" for x in v['rs']) + f",{v['q']:.4f}")

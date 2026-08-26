#!/usr/bin/env python3
"""Enumerate candidate gemv preferred() clauses and NAME THE CELL that kills each.

usage: g6_clauses.py <csv> [<csv> ...]

The family searched is the one WP7's clause_search.py could not express:

    red_len in [RLO, RHI]  AND  out_len >= OT  AND  batch >= BT   [AND MB >= AT]

with batch a first-class term. Every candidate is scored against EVERY measured
cell it admits; a candidate is REFUTED by its worst cell, printed with the
vendor and native GB/s so the reader can see whether the vendor was at the roof
(a cell we cannot win) or dipped (a cell we lost on our own merits).
"""
import sys, math, itertools
from g6_score import merge, score, fmt

def main():
    cells, refused = merge(sys.argv[1:])
    tr_all = sorted({k[4] for k in cells})
    print(f"# {len(cells)} cells, transA spellings {tr_all}, "
          f"{len(refused)} arm-rows refused")

    RLO = [1, 16, 32, 48, 56, 64, 80, 96, 128]
    RHI = [64, 96, 128, 192, 256, 320, 352, 384, 448, 512, 10**9]
    OT  = [1, 64, 128, 192, 256, 384, 512, 768]
    BT  = [1, 64, 128, 192, 256, 320, 384, 448, 512, 640]

    total_wins = sum(1 for v in cells.values() if v['q'] >= 1.15)
    print(f"# {total_wins} cells win >= 1.15x somewhere in the grid\n")

    best = []
    for rlo, rhi, ot, bt in itertools.product(RLO, RHI, OT, BT):
        if rhi < rlo: continue
        pred = lambda k, rlo=rlo, rhi=rhi, ot=ot, bt=bt: (
            k[0] == 'cdouble' and rlo <= k[2] <= rhi and k[1] >= ot and k[3] >= bt)
        s = score(cells, pred)
        if s is None: continue
        if s['sub'] == 0 and s['min'] >= 1.15:
            best.append((s['n'], rlo, rhi, ot, bt, s))
    best.sort(key=lambda x: (-x[0], x[1], -x[2], x[3], x[4]))
    print("=== CLAUSES THAT PASS GATE-C (zero cells below 1.15x), by capture ===")
    seen = set()
    for n, rlo, rhi, ot, bt, s in best[:25]:
        sig = (n, round(s['min'], 4))
        print(f"  red_len in [{rlo},{'inf' if rhi>10**8 else rhi}] & out_len>={ot} "
              f"& batch>={bt:4d} : n={n:3d} geo={s['geo']:5.3f} min={s['min']:.4f}")
    if not best: print("  (none)")

    print("\n=== NAMED CANDIDATES, EACH WITH ITS REFUTING CELL ===")
    named = [
        ("D3's clause  red[64,352] out>=256 batch>=320",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=256 and k[3]>=320),
        ("D3 conservative  red[64,352] out>=256 batch>=512",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=256 and k[3]>=512),
        ("widened low  red[48,352] out>=256 batch>=320",
         lambda k: k[0]=='cdouble' and 48<=k[2]<=352 and k[1]>=256 and k[3]>=320),
        ("widened batch  red[64,352] out>=256 batch>=128",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=256 and k[3]>=128),
        ("widened batch  red[64,352] out>=256 batch>=192",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=256 and k[3]>=192),
        ("widened batch  red[64,352] out>=256 batch>=256",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=256 and k[3]>=256),
        ("no upper red bound  red>=64 out>=256 batch>=320",
         lambda k: k[0]=='cdouble' and k[2]>=64 and k[1]>=256 and k[3]>=320),
        ("no out bound  red[64,352] batch>=320",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[3]>=320),
        ("out>=128  red[64,352] out>=128 batch>=320",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=128 and k[3]>=320),
        ("out>=192  red[64,352] out>=192 batch>=320",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=192 and k[3]>=320),
        ("no batch term at all  red[64,352] out>=256",
         lambda k: k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=256),
        ("footprint instead of batch  red[64,352] out>=256 MB>=256",
         None),
        ("all four scalar types  red[64,352] out>=256 batch>=320",
         lambda k: 64<=k[2]<=352 and k[1]>=256 and k[3]>=320),
    ]
    for name, pred in named:
        if pred is None:
            pred = lambda k: (k[0]=='cdouble' and 64<=k[2]<=352 and k[1]>=256
                              and cells[k]['mb'] >= 256)
        s = score(cells, pred)
        if s is None:
            print(f"  {name:52s} EMPTY"); continue
        ok = s['sub'] == 0 and s['min'] >= 1.15
        print(f"  {name:52s} n={s['n']:3d} geo={s['geo']:5.3f} min={s['min']:.4f} "
              f"loss={s['loss']:2d} sub={s['sub']:2d} {'PASS' if ok else 'FAIL'}")
        if not ok:
            wk, wv = s['worst']
            print(f"      REFUTED BY {fmt(wk):38s} {wv['q']:.4f}  "
                  f"vendor {wv['ven']:7.1f} GB/s  native {wv['nat']:7.1f}  {wv['mb']} MB")

main()

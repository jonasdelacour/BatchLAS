#!/usr/bin/env python3
"""WP3 step 9 -- rank native TRSM against the vendor on the ortho grid.

Two rules this script exists to enforce, both of which have been got wrong in
this repo before:

  * A ratio is only quoted where the shape is SATURATED. An unsaturated ratio
    measures launch overhead, not the algorithm. Saturation here is judged per
    (type, side, n, q) by whether GFLOP/s is still climbing with batch: if the
    largest batch is more than 1.15x the one below it, the cell is still
    scaling and its ratio is reported as UNSAT rather than ranked.

  * The kill criterion is applied as it was WRITTEN, in advance (spec S10):
    if native real TRSM exceeds 1.10 x vendor at the saturated ortho shape,
    real stays vendor-first and only complex flips. Deciding the threshold
    after seeing the numbers is how a measurement becomes a justification.
"""
import csv, os, re, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SAT_TOL = 1.15     # GFLOP/s growth below this at the top batch == saturated
KILL    = 1.10     # spec S10, stated in advance

NAME_RE = re.compile(r'BM_TRSM_(\w+?)<([^,]+(?:<[^>]*>)?),')


def load(path):
    """(type, n, q, batch) -> (avg_ms, gflops)"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            m = NAME_RE.search(row['name'])
            if not m:
                continue
            typ = m.group(2).strip()
            key = (typ, int(row['arg0']), int(row['arg1']), int(row['arg2']))
            out[key] = (float(row['avg_ms']), float(row.get('GFLOPS', 'nan')))
    return out


def saturated(gflops_by_batch):
    """Is the top batch's GFLOP/s within SAT_TOL of the batch below it?"""
    bs = sorted(gflops_by_batch)
    if len(bs) < 2:
        return False
    top, prev = gflops_by_batch[bs[-1]], gflops_by_batch[bs[-2]]
    return prev > 0 and (top / prev) < SAT_TOL


def report(side):
    ven = load(os.path.join(HERE, f'{side}-vendor.csv'))
    nat = load(os.path.join(HERE, f'{side}-native.csv'))
    common = sorted(set(ven) & set(nat))
    if not common:
        print(f'  ({side}: no overlapping rows -- did both runs complete?)')
        return {}

    # Saturation is a property of the (type, n, q) family across batch.
    fam = defaultdict(dict)
    for (t, n, q, b) in common:
        fam[(t, n, q)][b] = ven[(t, n, q, b)][1]
    sat = {k: saturated(v) for k, v in fam.items()}

    print(f'\n=== Side::{side.capitalize()} ===')
    print(f'{"type":16} {"n":>4} {"q":>5} {"batch":>6} '
          f'{"vendor ms":>10} {"native ms":>10} {"speedup":>8}  verdict')
    verdicts = {}
    for key in common:
        t, n, q, b = key
        vm, _ = ven[key]
        nm, _ = nat[key]
        sp = vm / nm if nm > 0 else float('nan')
        if not sat[(t, n, q)] or b != max(fam[(t, n, q)]):
            tag = 'unsat' if not sat[(t, n, q)] else 'sub-top'
        else:
            tag = 'WIN' if sp >= 1.0 / KILL else 'loss'
            verdicts[key] = sp
        print(f'{t:16} {n:>4} {q:>5} {b:>6} {vm:>10.4f} {nm:>10.4f} {sp:>8.2f}x  {tag}')
    return verdicts


def main():
    allv = {}
    for side in ('right', 'left'):
        allv[side] = report(side)

    print('\n=== Ranked cells only (saturated, top batch) ===')
    bytype = defaultdict(list)
    for side, v in allv.items():
        for (t, n, q, b), sp in v.items():
            bytype[t].append((sp, side, n, q, b))
    if not bytype:
        print('  none -- nothing reached saturation, so nothing may be flipped.')
        return
    for t in sorted(bytype):
        rows = bytype[t]
        wins = [r for r in rows if r[0] >= 1.0 / KILL]
        print(f'\n{t}: {len(wins)}/{len(rows)} cells at or above the 1/{KILL:.2f} kill line')
        for sp, side, n, q, b in sorted(rows, reverse=True):
            mark = 'native' if sp >= 1.0 / KILL else 'VENDOR'
            print(f'    {side:5} n={n:<4} q={q:<5} batch={b:<5} {sp:5.2f}x -> {mark}')


if __name__ == '__main__':
    main()

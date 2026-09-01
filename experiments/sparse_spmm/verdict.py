#!/usr/bin/env python3
"""The clause test.

Takes the joined CSVs of pairs of INDEPENDENT passes over the same grid, keeps
the rows that are saturated, batch >= 128 and chk-agreeing in BOTH, and reports,
for each candidate `preferred()` predicate:

  * how many rows it would move, and whether EVERY one of them meets the gate
    t_native <= 1.10 * t_vendor in BOTH passes (worst-of-two is the number
    reported -- a clause is only as good as its worse pass);
  * whether the rows it does NOT move contain a measured non-winner, i.e.
    whether its boundary is bracketed rather than sitting at the edge of the
    sampled grid.

NOISE RULE, AND WHY IT IS NOT JUST rel_sd. A row is admitted when either arm's
rel_sd is <= 0.02 in both passes, OR the two passes' ratios agree to within 5%.
The second clause exists because the first one alone SILENTLY DROPPED the single
most important negative result in this sweep: (cfloat, transA=NoTrans,
transB=Trans, banded, m=2048, nnz/row=16, nrhs=25, batch=128) measures ratio
1.934 in pass1 and 1.872 in pass2 -- reproducing to 3% -- but its pass-2 rel_sd
is 0.033, so an rel_sd-only filter threw the row away and the unconditional
gather clause then "passed" with a worst ratio of 1.019. Cross-pass reproduction
is stronger evidence about a row than its within-pass spread, and a hygiene
filter that can delete a reproducible non-winner is a filter that manufactures
wins.

usage: verdict.py <p1-joined.csv> <p2-joined.csv> [more pairs...]
"""
import csv, sys, collections

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from report import load, mark_saturation, KEY, GATE

NOISE = 0.02
REPRO = 1.05


def paired(p1, p2):
    a = load(p1); mark_saturation(a)
    b = load(p2); mark_saturation(b)
    idx = {tuple(r[k] for k in KEY) + (r['batch'],): r for r in b}
    out, noisy = [], 0
    for r in a:
        o = idx.get(tuple(r[k] for k in KEY) + (r['batch'],))
        if o is None:
            continue
        if not (r['agree'] and o['agree']):
            continue
        if not (r['sat'] and o['sat'] and r['batch'] >= 128):
            continue
        spread = max(r['ratio'], o['ratio']) / min(r['ratio'], o['ratio'])
        quiet = r['relsd'] <= NOISE and o['relsd'] <= NOISE
        if not quiet and spread > REPRO:
            noisy += 1
            continue
        r['ratio2'] = o['ratio']
        r['spread'] = spread
        r['worst'] = max(r['ratio'], o['ratio'])
        r['quiet'] = quiet
        out.append(r)
    return out, noisy


CANDIDATES = [
    ("transA == NoTrans  (the gather, unconditional)",
     lambda r: r['transA'] == 0),
    ("transA == NoTrans  AND NOT (cfloat && transB != NoTrans)",
     lambda r: r['transA'] == 0 and not (r['typ'] == 'cfloat' and r['transB'] != 0)),
    ("transA == NoTrans  AND NOT (cfloat && transB != NoTrans && nrhs >= 16)",
     lambda r: r['transA'] == 0 and not (r['typ'] == 'cfloat' and r['transB'] != 0
                                         and r['nrhs'] >= 16)),
    ("transA == NoTrans  AND NOT (cfloat && transB != NoTrans && nrhs >= 13)",
     lambda r: r['transA'] == 0 and not (r['typ'] == 'cfloat' and r['transB'] != 0
                                         and r['nrhs'] >= 13)),
    ("transA != NoTrans  (the scatter, unconditional)",
     lambda r: r['transA'] != 0),
    ("transA != NoTrans  AND nrhs <= 1",
     lambda r: r['transA'] != 0 and r['nrhs'] <= 1),
    ("transA != NoTrans  AND nrhs <= 2",
     lambda r: r['transA'] != 0 and r['nrhs'] <= 2),
    ("transA != NoTrans  AND nrhs <= 4",
     lambda r: r['transA'] != 0 and r['nrhs'] <= 4),
    ("transA != NoTrans  AND nrhs <= 2 AND type != cdouble",
     lambda r: r['transA'] != 0 and r['nrhs'] <= 2 and r['typ'] != 'cdouble'),
    ("transA != NoTrans  AND nrhs <= 4 AND type != cdouble",
     lambda r: r['transA'] != 0 and r['nrhs'] <= 4 and r['typ'] != 'cdouble'),
]


def main():
    rows, dropped = [], 0
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        r, n = paired(args[i], args[i + 1])
        rows += r
        dropped += n
    quiet = sum(1 for r in rows if r['quiet'])
    print(f"== {len(rows)} rows survive: saturated, batch >= 128 and chk-agreeing "
          f"in BOTH passes ({quiet} quiet, {len(rows)-quiet} admitted on "
          f"cross-pass reproduction alone); {dropped} dropped as noisy AND "
          f"non-reproducing")
    by = collections.Counter((r['typ'], r['transA']) for r in rows)
    print("   " + ", ".join(f"{t}/tA{a}:{n}" for (t, a), n in sorted(by.items())))

    for label, pred in CANDIDATES:
        sel = [r for r in rows if pred(r)]
        rej = [r for r in rows if not pred(r)]
        if not sel:
            continue
        bad = [r for r in sel if r['worst'] > GATE]
        rej_bad = [r for r in rej if r['worst'] > GATE]
        ok = "PASSES" if not bad else f"FAILS ({len(bad)} of {len(sel)} over the gate)"
        ws = sorted(r['worst'] for r in sel)
        print(f"\n-- {label}")
        print(f"   moves {len(sel)} rows; {ok}; worst-of-two-passes ratio "
              f"{ws[-1]:.3f}, median {ws[len(ws)//2]:.3f}, best {ws[0]:.3f}")
        print(f"   the {len(rej)} rows it does NOT move contain {len(rej_bad)} "
              f"measured non-winners "
              f"({'bracketed' if rej_bad else 'UNBRACKETED -- wider than the evidence'})")
        for r in sorted(bad, key=lambda r: -r['worst'])[:8]:
            print(f"     OVER GATE {r['typ']:8s} tA={r['transA']} m={r['m']} "
                  f"nnz/row={r['nnzrow']} nrhs={r['nrhs']} b={r['batch']} "
                  f"tB={r['transB']} beta={r['beta']} pat={r['pattern']} "
                  f"p1={r['ratio']:.3f} p2={r['ratio2']:.3f}")


if __name__ == '__main__':
    main()

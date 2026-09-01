#!/usr/bin/env python3
"""Turn one or two joined passes into the tables the README quotes.

Everything here obeys three rules that are not negotiable on this box:

  SATURATION. A ratio is only quoted where both arms are saturated, and
  saturation is measured off the batch ladder rather than assumed. See
  mark_saturation for the exact rule and for the L2 cliff that forced it.

  L2 RESIDENCY. The RTX 4090 has 72 MB of L2. A cell whose whole batch footprint
  fits inside it is NOT a DRAM measurement and its GB/s must never be compared
  against the 1008 GB/s pin. Every table prints the footprint and flags it.

  REPRODUCTION. A ratio quoted for the acceptance gate must appear in BOTH
  passes. The cross-pass spread is printed next to it.

usage: report.py <pass1-joined.csv> [<pass2-joined.csv>]
"""
import csv, sys, collections

GATE = 1.10          # t_native <= 1.10 * t_vendor
SAT_TOL = 1.10       # a wider batch buying < 10% per item means this one is saturated
DRAM_ROOF = 1008.0

KEY = ('typ', 'transA', 'm', 'nnzrow', 'nrhs', 'transB', 'beta', 'pattern')


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        d = dict(r)
        for k in ('m', 'nnzrow', 'nrhs', 'batch', 'transB', 'beta', 'pattern', 'transA'):
            d[k] = int(r[k])
        for k in ('t_vendor', 't_native', 'ratio', 'relsd', 'gbs_v', 'gbs_n',
                  'fp_mb', 'roof_v', 'roof_n'):
            d[k] = float(r[k])
        d['agree'] = r['agree'] == 'True'
        d['l2res'] = r['l2res'] == 'True'
        rows.append(d)
    return rows


def mark_saturation(rows):
    """Mark each row saturated / not, from the BATCH LADDERS only.

    SATURATION IS "GOING WIDER BUYS LESS THAN 10%", NOT "WITHIN 10% OF THE
    FASTEST BATCH IN THE LADDER". The naive definition breaks on exactly the
    cells this sweep is about: the native gather at (float, m=1024, 16 nnz/row,
    nrhs=12) runs 2.369, 0.788, 0.309, 0.485 us per item over batch 8/32/128/512
    -- it RISES at 512 because the footprint (119 MB) leaves the 72 MB L2. That
    rise is an L2 cliff, not a return to launch-latency territory, and a rule
    anchored on the ladder minimum would declare the LARGEST batch unsaturated
    and throw away the rows the acceptance gate is about. So:

        sat(b)  <=>  per_item(b) <= 1.10 * min{ per_item(b') : b' >= b }

    A family with fewer than three batches (the `cells` files, which fix one
    batch and sweep transB/beta/pattern instead) carries no ladder of its own.
    It inherits the verdict of the ladder family at the same
    (type, transA, m, nnz/row, nrhs, pattern), or failing that the same tuple at
    the scattered pattern: the knee is a property of the launch geometry, which
    transB and beta do not change. A row that can inherit nothing is sat=False
    and never enters the gate population.
    """
    fam = collections.defaultdict(list)
    for r in rows:
        fam[tuple(r[k] for k in KEY)].append(r)

    ladders = {}
    for k, rs in fam.items():
        by_b = {}
        for r in rs:
            by_b.setdefault(r['batch'], r)
        if len(by_b) < 3:
            continue
        typ, transA, m, nnzrow, nrhs, transB, beta, pattern = k
        lk = (typ, transA, m, nnzrow, nrhs, pattern)
        if lk in ladders and len(ladders[lk]) >= len(by_b):
            continue
        ladders[lk] = by_b

    def thresholds(by_b, arm):
        bs = sorted(by_b)
        per = {b: by_b[b]['t_' + arm] / b for b in bs}
        out = {}
        for i, b in enumerate(bs):
            tail = min(per[c] for c in bs[i:])
            out[b] = per[b] <= SAT_TOL * tail
        sat_b = [b for b in bs if out[b]]
        return out, (min(sat_b) if sat_b else None)

    for k, rs in fam.items():
        typ, transA, m, nnzrow, nrhs, transB, beta, pattern = k
        by_b = (ladders.get((typ, transA, m, nnzrow, nrhs, pattern))
                or ladders.get((typ, transA, m, nnzrow, nrhs, 1)))
        own_batches = {r['batch'] for r in rs}
        for arm in ('vendor', 'native'):
            for r in rs:
                r['peritem_' + arm] = r['t_' + arm] / r['batch']
            if by_b is None:
                for r in rs:
                    r['sat_' + arm] = False
                continue
            tab, thr = thresholds(by_b, arm)
            for r in rs:
                r['sat_' + arm] = (tab[r['batch']] if r['batch'] in tab
                                   else (thr is not None and r['batch'] >= thr))
        for r in rs:
            r['sat'] = r['sat_vendor'] and r['sat_native']
            r['inherited'] = len(own_batches) < 3
            r['ladder'] = len(own_batches)
    return rows


def usable(rows):
    return [r for r in rows if r['agree'] and r['relsd'] <= 0.02]


def fmt(r, extra=''):
    return (f"{r['typ']:8s} tA={r['transA']} m={r['m']:5d} nnz/row={r['nnzrow']:3d} "
            f"nrhs={r['nrhs']:3d} b={r['batch']:5d} tB={r['transB']} beta={r['beta']} "
            f"pat={r['pattern']} | tv={r['t_vendor']:10.5f} tn={r['t_native']:10.5f} "
            f"ratio={r['ratio']:6.3f} | GB/s v={r['gbs_v']:7.1f} n={r['gbs_n']:7.1f} "
            f"| fp={r['fp_mb']:8.1f}MB {'L2' if r['l2res'] else '  '} "
            f"sat={'Y' if r['sat'] else 'n'} relsd={r['relsd']:.4f}{extra}")


def main():
    p1 = load(sys.argv[1]); mark_saturation(p1)
    two = len(sys.argv) > 2
    if two:
        p2 = load(sys.argv[2]); mark_saturation(p2)
        idx = {tuple(r[k] for k in KEY) + (r['batch'],): r for r in p2}
        for r in p1:
            o = idx.get(tuple(r[k] for k in KEY) + (r['batch'],))
            r['ratio2'] = o['ratio'] if o and o['agree'] else None
            r['spread'] = (max(r['ratio'], o['ratio']) / min(r['ratio'], o['ratio'])
                           if o and o['agree'] else None)

    good = usable(p1)
    print(f"== {len(p1)} joined rows, {len(good)} usable "
          f"(chk agrees + rel_sd <= 0.02), {len(p1)-len(good)} dropped")

    dis = [r for r in p1 if not r['agree']]
    print(f"== {len(dis)} rows dropped for chk disagreement / no-op")
    for r in dis:
        print("   DISAGREE " + fmt(r))

    gate = [r for r in good if r['sat'] and r['batch'] >= 128]
    print(f"\n== ACCEPTANCE-GATE POPULATION: saturated AND batch >= 128 -> "
          f"{len(gate)} rows")
    win = [r for r in gate if r['ratio'] <= GATE]
    loss = [r for r in gate if r['ratio'] > GATE]
    print(f"   {len(win)} meet t_native <= {GATE}*t_vendor, {len(loss)} do NOT")
    if loss:
        print("   THE NON-WINNERS (these are the boundary brackets):")
        for r in sorted(loss, key=lambda r: -r['ratio']):
            print("     " + fmt(r))
    if win:
        w = sorted(win, key=lambda r: r['ratio'])
        print(f"   best  {fmt(w[0])}")
        print(f"   worst {fmt(w[-1])}")

    if two:
        rep = [r for r in gate if r.get('spread') is not None]
        bad = [r for r in rep if r['spread'] > 1.10]
        print(f"\n== CROSS-PASS: {len(rep)}/{len(gate)} gate rows present in both "
              f"passes; {len(bad)} with ratio spread > 1.10")
        for r in sorted(bad, key=lambda r: -r['spread'])[:15]:
            print("   " + fmt(r, f" ratio2={r['ratio2']:.3f} spread={r['spread']:.3f}"))
        if rep:
            sp = sorted(r['spread'] for r in rep)
            print(f"   worst spread {sp[-1]:.4f}; median {sp[len(sp)//2]:.4f}")
        # did any gate row FLIP side of the gate between passes?
        flip = [r for r in rep
                if (r['ratio'] <= GATE) != (r['ratio2'] <= GATE)]
        print(f"   {len(flip)} gate rows FLIPPED side of the {GATE} line between passes")
        for r in flip:
            print("     FLIP " + fmt(r, f" ratio2={r['ratio2']:.3f}"))

    print("\n== BY REGIME (saturated rows only)")
    for label, pred in [
        ('lanczos  nnz/row=3, nrhs<=2', lambda r: r['nnzrow'] == 3 and r['nrhs'] <= 2),
        ('LOBPCG   nnz/row=16, nrhs 12-50', lambda r: r['nnzrow'] == 16 and r['nrhs'] >= 12),
    ]:
        for ta in (0, 1):
            sub = [r for r in good if pred(r) and r['sat'] and r['transA'] == ta]
            if not sub:
                continue
            rs = sorted(r['ratio'] for r in sub)
            print(f"  {label:32s} transA={ta}: n={len(sub):4d} "
                  f"min={rs[0]:.3f} p25={rs[len(rs)//4]:.3f} med={rs[len(rs)//2]:.3f} "
                  f"p75={rs[3*len(rs)//4]:.3f} max={rs[-1]:.3f}")
            for typ in ('float', 'double', 'cfloat', 'cdouble'):
                st = sorted(r['ratio'] for r in sub if r['typ'] == typ)
                if st:
                    print(f"      {typ:8s} n={len(st):3d} min={st[0]:.3f} "
                          f"med={st[len(st)//2]:.3f} max={st[-1]:.3f} "
                          f"(>{GATE}: {sum(1 for x in st if x > GATE)})")

    print("\n== DISTANCE FROM THE DRAM ROOF (1008 GB/s), saturated non-L2 rows only")
    nl = [r for r in good if r['sat'] and not r['l2res']]
    for typ in ('float', 'double', 'cfloat', 'cdouble'):
        st = [r for r in nl if r['typ'] == typ]
        if st:
            print(f"  {typ:8s} n={len(st):3d} vendor {min(r['roof_v'] for r in st):.2f}"
                  f"-{max(r['roof_v'] for r in st):.2f} x roof   "
                  f"native {min(r['roof_n'] for r in st):.2f}"
                  f"-{max(r['roof_n'] for r in st):.2f} x roof")
    l2 = [r for r in good if r['sat'] and r['l2res']]
    print(f"  ({len(l2)} further saturated rows are L2-RESIDENT and are excluded "
          f"from the roof comparison entirely; their ideal-model GB/s exceeds the "
          f"DRAM pin because the traffic never reaches DRAM.)")

    print("\n== SATURATION LADDERS (per-item us; '*' = saturated)")
    fam = collections.defaultdict(list)
    for r in good:
        fam[tuple(r[k] for k in KEY)].append(r)
    for k in sorted(fam):
        rs = sorted({r['batch']: r for r in fam[k]}.values(), key=lambda r: r['batch'])
        if len(rs) < 4:
            continue
        v = " ".join(f"{r['batch']}:{1000*r['peritem_vendor']:.3f}"
                     f"{'*' if r['sat_vendor'] else ''}" for r in rs)
        n = " ".join(f"{r['batch']}:{1000*r['peritem_native']:.3f}"
                     f"{'*' if r['sat_native'] else ''}" for r in rs)
        print(f"  {k}\n     vendor {v}\n     native {n}")


if __name__ == '__main__':
    main()

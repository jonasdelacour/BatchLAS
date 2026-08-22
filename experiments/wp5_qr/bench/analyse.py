#!/usr/bin/env python3
"""Turn a bench CSV into the ratio / geomean / crossover tables.

THE DISCARD RULE, applied here and nowhere else so it cannot be applied
inconsistently between tables:

  * relsd > 0.10 on EITHER arm  -> the cell is DISCARDED and listed by name.
  * flag != 'ok' (a non-finite probe) -> DISCARDED and listed. A fast wrong
    answer must never enter a ratio.
  * a residual above the type's threshold -> DISCARDED and listed.
    float/cfloat 1e-4, double/cdouble 1e-11. These are LOOSE on purpose: the
    probe is a random-vector estimate, and the job of the bound is to catch a
    garbage answer, not to grade accuracy.

Ratio convention: vendor_ms / native_ms, so >1 means NATIVE IS AHEAD.
"""
import sys, csv, math
from collections import defaultdict

RESID = {'float': 1e-4, 'cfloat': 1e-4, 'double': 1e-11, 'cdouble': 1e-11}

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0] == 'bin' or r[0].startswith('#'):
                continue
            if len(r) < 17 or r[6] in ('THREW', 'TIMEOUT_OR_CRASH'):
                rows.append({'bad': ' '.join(r)})
                continue
            d = dict(bin=r[0], op=r[1], type=r[2], m=int(r[3]), n=int(r[4]),
                     batch=int(r[5]), med=float(r[6]), mean=float(r[7]),
                     relsd=float(r[8]), gflops=float(r[9]), res=float(r[10]),
                     ortho=float(r[11]), recon=float(r[12]), ws=int(r[13]),
                     route=r[14], cta=int(r[15]), flag=r[16].strip(),
                     pin=(r[17].strip() if len(r) > 17 else ''))
            rows.append(d)
    return rows

def key(d):
    return (d['op'], d['type'], d['m'], d['n'], d['batch'], d.get('pin', ''))

def main(path):
    rows = load(path)
    bad = [r['bad'] for r in rows if 'bad' in r]
    rows = [r for r in rows if 'bad' not in r]
    by = defaultdict(dict)
    for r in rows:
        by[key(r)][r['bin']] = r

    print('# discarded / failed cells')
    for b in bad:
        print('  FAILED_CELL  ' + b)

    ratios = defaultdict(list)
    print()
    print('op,type,m,n,batch,vendor_ms,native_ms,ratio,v_relsd,n_relsd,'
          'v_route,n_route,v_res,n_res,v_ws_MB,n_ws_MB')
    for k in sorted(by, key=lambda x: (x[0], x[1], x[3], x[2], x[4])):
        p = by[k]
        v, nvv = p.get('qrbench_v'), p.get('qrbench_nv')
        if not v or not nvv:
            only = v or nvv
            print('  SINGLE_ARM  ' + ','.join(str(only[c]) for c in
                  ('op', 'type', 'm', 'n', 'batch', 'med', 'route', 'flag')))
            continue
        why = []
        for tag, r in (('v', v), ('nv', nvv)):
            if r['relsd'] > 0.10: why.append(f'{tag}_relsd={r["relsd"]:.3f}')
            if r['flag'] != 'ok': why.append(f'{tag}_flag={r["flag"]}')
            probes = [r['res']] + ([r['ortho'], r['recon']] if r['op'] == 'orgqr' else [])
            for pv in probes:
                if pv >= 0 and pv > RESID[r['type']]:
                    why.append(f'{tag}_residual={pv:.2e}')
        if why:
            print(f'  DISCARDED  {k}  ({"; ".join(why)})')
            continue
        ratio = v['med'] / nvv['med']
        ratios[(k[0], k[1])].append((k[3], k[4], ratio))
        print(','.join([k[0], k[1], str(k[2]), str(k[3]), str(k[4]),
                        f'{v["med"]:.4f}', f'{nvv["med"]:.4f}', f'{ratio:.3f}',
                        f'{v["relsd"]:.4f}', f'{nvv["relsd"]:.4f}',
                        v['route'], nvv['route'],
                        f'{v["res"]:.2e}', f'{nvv["res"]:.2e}',
                        f'{v["ws"]/1e6:.1f}', f'{nvv["ws"]/1e6:.1f}']))

    print()
    print('# geomean of vendor_ms/native_ms  (>1 = native ahead)')
    allr = []
    for (op, t), lst in sorted(ratios.items()):
        g = math.exp(sum(math.log(r) for _, _, r in lst) / len(lst))
        wins = sum(1 for _, _, r in lst if r > 1.0)
        print(f'  {op:6s} {t:8s} n={len(lst):3d}  geomean={g:7.3f}  '
              f'wins={wins}/{len(lst)}  min={min(r for _,_,r in lst):.3f}  '
              f'max={max(r for _,_,r in lst):.3f}')
        allr += [r for _, _, r in lst]
    for op in sorted({k[0] for k in ratios}):
        sub = [r for (o, t), lst in ratios.items() if o == op for _, _, r in lst]
        if sub:
            g = math.exp(sum(math.log(r) for r in sub) / len(sub))
            print(f'  {op:6s} ALL      n={len(sub):3d}  geomean={g:7.3f}  '
                  f'wins={sum(1 for r in sub if r>1)}/{len(sub)}')
    if allr:
        g = math.exp(sum(math.log(r) for r in allr) / len(allr))
        print(f'  OVERALL          n={len(allr):3d}  geomean={g:7.3f}  '
              f'wins={sum(1 for r in allr if r>1)}/{len(allr)}')

    # CROSSOVERS ON THE TWO AXES SEPARATELY. WP4 found its potrf crossover was
    # in ORDER and not in BATCH, which is not the intuition, so walking one
    # sorted list of (n, batch) pairs and calling every sign change a "crossover
    # in n" is wrong -- it reports a batch effect as an order effect. Each walk
    # below holds the OTHER axis fixed.
    print()
    print('# crossovers in ORDER (batch held fixed)')
    for (op, t), lst in sorted(ratios.items()):
        for b in sorted({b for _, b, _ in lst}):
            seq = sorted((n, r) for n, bb, r in lst if bb == b)
            if len(seq) < 2:
                continue
            prev = None
            for n, r in seq:
                if prev is not None and (prev[1] < 1.0) != (r < 1.0):
                    print(f'  {op} {t} batch={b}: crosses between n={prev[0]} '
                          f'({prev[1]:.2f}x) and n={n} ({r:.2f}x)')
                prev = (n, r)
    print()
    print('# crossovers in BATCH (order held fixed)')
    for (op, t), lst in sorted(ratios.items()):
        for n in sorted({n for n, _, _ in lst}):
            seq = sorted((b, r) for nn, b, r in lst if nn == n)
            if len(seq) < 2:
                continue
            prev = None
            for b, r in seq:
                if prev is not None and (prev[1] < 1.0) != (r < 1.0):
                    print(f'  {op} {t} n={n}: crosses between batch={prev[0]} '
                          f'({prev[1]:.2f}x) and batch={b} ({r:.2f}x)')
                prev = (b, r)

if __name__ == '__main__':
    main(sys.argv[1])

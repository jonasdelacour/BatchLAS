#!/usr/bin/env python3
"""Join the vendor and native CSVs of one pass on the full argument tuple and
report the ratio, plus the two hygiene columns that decide whether a ratio may
be quoted at all.

  ratio  = t_native / t_vendor.  < 1 means native is FASTER. The campaign gate
           is t_native <= 1.10 * t_vendor, i.e. ratio <= 1.10.
  relsd  = the larger of the two arms' rel_sd. > 0.02 disqualifies a row.
  chk    = the L1 norm of C item 0. A row where the two arms disagree by more
           than 1e-3 relative, or where either is 0 on a beta=0 row, is a
           NO-OP or a WRONG ANSWER and is dropped, not ranked.
  L2res  = whether the whole batch footprint fits the 72 MB L2 of an RTX 4090.
           An L2-resident cell is NOT a DRAM-bandwidth measurement and its
           GB/s must not be compared against the 1008 GB/s roof.

usage: analyse.py <pass-dir> [<pass-dir-2>]   (two dirs => cross-pass spread)
"""
import csv, glob, os, sys, collections

L2_BYTES = 72 * 1024 * 1024      # RTX 4090 (AD102) L2
DRAM_ROOF = 1008.0               # GB/s, 384-bit @ 21 Gbps

ARGS = ['arg0','arg1','arg2','arg3','arg4','arg5','arg6','arg7']
NAMES = dict(zip(ARGS, ['m','nnzrow','nrhs','batch','transB','beta','pattern','transA']))

ELEM = {'float':4, 'double':8, 'cfloat':8, 'cdouble':16}


def load(passdir):
    """-> {(type, sweep, argtuple): {route: row}}"""
    out = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(passdir, '*.csv'))):
        base = os.path.basename(f)
        if base.endswith('.routes.csv'):
            continue
        stem = base[:-4]
        # <tag>_<type>_<route>
        for route in ('vendor', 'native_direct'):
            if stem.endswith('_' + route):
                tag_type = stem[:-(len(route) + 1)]
                break
        else:
            continue
        tag, _, typ = tag_type.rpartition('_')
        for r in csv.DictReader(open(f)):
            key = (typ, tag, tuple(int(r[a]) for a in ARGS))
            out[key][route] = r
    return out


def footprint(typ, m, nnzrow, nrhs, batch):
    s = ELEM[typ]
    nnz = m * nnzrow
    return batch * (nnz * (s + 4) + (m + 1) * 4 + 2 * m * nrhs * s)


def rows(passdir):
    data = load(passdir)
    res = []
    for (typ, tag, args), arms in sorted(data.items()):
        if 'vendor' not in arms or 'native_direct' not in arms:
            continue
        v, n = arms['vendor'], arms['native_direct']
        tv, tn = float(v['avg_ms']), float(n['avg_ms'])
        sdv = float(v['stddev_ms']) / tv
        sdn = float(n['stddev_ms']) / tn
        cv, cn = float(v['chk']), float(n['chk'])
        m, nnzrow, nrhs, batch, transB, beta, pattern, transA = args
        fp = footprint(typ, m, nnzrow, nrhs, batch)
        # BETA = 1 ACCUMULATES ACROSS THE WHOLE RUN, so chk on such a row is a
        # function of how many timed calls the arm happened to make and the two
        # arms are NOT comparable by value. On those rows the check degrades to
        # "not a no-op". On beta = 0 rows, where every call overwrites C, the two
        # arms must agree to 1e-3 relative or one of them is wrong.
        if beta == 0:
            agree = (cv != 0 and cn != 0 and
                     abs(cv - cn) <= 1e-3 * max(abs(cv), abs(cn)))
        else:
            agree = (cv != 0 and cn != 0)
        res.append(dict(
            typ=typ, tag=tag, m=m, nnzrow=nnzrow, nrhs=nrhs, batch=batch,
            transB=transB, beta=beta, pattern=pattern, transA=transA,
            t_vendor=tv, t_native=tn, ratio=tn / tv,
            relsd=max(sdv, sdn), sd_v=sdv, sd_n=sdn,
            chk_v=cv, chk_n=cn, agree=agree,
            gbs_v=float(v['GB/s']), gbs_n=float(n['GB/s']),
            fp_mb=fp / 1e6, l2res=fp <= L2_BYTES,
            roof_v=float(v['GB/s']) / DRAM_ROOF, roof_n=float(n['GB/s']) / DRAM_ROOF,
        ))
    return res


def emit(res, path):
    cols = ['typ','tag','transA','m','nnzrow','nrhs','batch','transB','beta','pattern',
            't_vendor','t_native','ratio','relsd','sd_v','sd_n','chk_v','chk_n','agree',
            'gbs_v','gbs_n','roof_v','roof_n','fp_mb','l2res']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in res:
            w.writerow({c: r[c] for c in cols})


def main():
    p1 = sys.argv[1]
    r1 = rows(p1)
    emit(r1, os.path.join(p1, 'joined.csv'))
    print(f"{len(r1)} joined rows -> {os.path.join(p1,'joined.csv')}")

    bad = [r for r in r1 if not r['agree']]
    if bad:
        print(f"\n!! {len(bad)} rows where the two arms DISAGREE or a chk is zero "
              f"(dropped from every ranking):")
        for r in bad[:20]:
            print(f"   {r['typ']:8s} transA={r['transA']} m={r['m']} nnz/row={r['nnzrow']} "
                  f"nrhs={r['nrhs']} b={r['batch']} tB={r['transB']} beta={r['beta']} "
                  f"pat={r['pattern']}  chk_v={r['chk_v']:.6g} chk_n={r['chk_n']:.6g}")

    good = [r for r in r1 if r['agree'] and r['relsd'] <= 0.02]
    noisy = [r for r in r1 if r['agree'] and r['relsd'] > 0.02]
    print(f"\n{len(good)} usable rows, {len(noisy)} dropped for rel_sd > 0.02")

    if len(sys.argv) > 2:
        p2 = sys.argv[2]
        r2 = {(r['typ'], r['tag'], r['transA'], r['m'], r['nnzrow'], r['nrhs'],
               r['batch'], r['transB'], r['beta'], r['pattern']): r for r in rows(p2)}
        emit(list(r2.values()), os.path.join(p2, 'joined.csv'))
        worst = 0.0
        n = 0
        for r in good:
            k = (r['typ'], r['tag'], r['transA'], r['m'], r['nnzrow'], r['nrhs'],
                 r['batch'], r['transB'], r['beta'], r['pattern'])
            if k in r2:
                o = r2[k]
                if o['agree']:
                    sp = max(r['ratio'], o['ratio']) / min(r['ratio'], o['ratio'])
                    worst = max(worst, sp)
                    n += 1
                    r['ratio2'] = o['ratio']
                    r['spread'] = sp
        print(f"cross-pass: {n} rows compared, worst ratio spread {worst:.4f}")
    return r1, good


if __name__ == '__main__':
    main()

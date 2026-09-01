#!/usr/bin/env python3
"""The small-batch corner of the GATHER arm, reported as a HARM CHECK.

usage: sb_report.py <sb-pass-1> <sb-pass-2>

WHAT THIS SCRIPT WILL NOT DO
----------------------------
It will not call a batch <= 16 ratio an algorithmic result. Below saturation the
timed region is launch latency, route dispatch and (on the vendor arm) a
cusparseSpMM_bufferSize re-query; the ratio there is a property of the two host
chains, not of the two kernels. So every table below carries the ABSOLUTE
per-call microseconds of both arms next to the ratio, and the verdict is phrased
as "is native ever materially slower", never as "native is N x faster".

ADMISSION RULE -- deliberately the same one verdict.py uses, and for the reason
recorded in README.md ("the filter that manufactured a win"): a row is admitted
when either arm's rel_sd <= 0.02 in BOTH passes, OR the two passes' ratios agree
to within 5 %. An rel_sd-only filter has already been measured deleting the one
real loss in this campaign. Cross-pass reproduction outranks within-pass spread.

HARM RULE
---------
A row is a MEASURED HARM only if BOTH passes put it over the campaign's 1.10
gate. A row over the gate in one pass only is reported as UNREPRODUCED and named
explicitly rather than dropped silently.
"""
import csv, os, sys, collections, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import rows as pass_rows

GATE = 1.10

KEY = ('typ', 'tag', 'transA', 'm', 'nnzrow', 'nrhs', 'batch', 'transB', 'beta', 'pattern')

FAMILY = {
    'sbL': 'lanczos   m=1024 nnz/row=3  nrhs=2  tB=N',
    'sbM': 'LOBPCG M  m=1024 nnz/row=16 nrhs=12 tB=N scat',
    'sbS': 'LOBPCG S  m=2048 nnz/row=16 nrhs=25 tB=N scat',
    'sbB': 'LOBPCG XL m=4096 nnz/row=16 nrhs=50 tB=N scat',
    'sbT': 'transB=T  m=2048 nnz/row=16 nrhs=25 tB=T banded',
}
TYPES = ['float', 'double', 'cfloat', 'cdouble']
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


def key(r):
    return tuple(r[k] for k in KEY)


def load(d1, d2):
    a = {key(r): r for r in pass_rows(d1)}
    b = {key(r): r for r in pass_rows(d2)}
    out = []
    for k in sorted(set(a) & set(b)):
        r1, r2 = a[k], b[k]
        if not (r1['agree'] and r2['agree']):
            continue
        lo, hi = sorted((r1['ratio'], r2['ratio']))
        spread = hi / lo if lo > 0 else 9e9
        quiet = (r1['relsd'] <= 0.02) and (r2['relsd'] <= 0.02)
        admitted = quiet or spread <= 1.05
        out.append(dict(
            typ=r1['typ'], tag=r1['tag'], m=r1['m'], nnzrow=r1['nnzrow'],
            nrhs=r1['nrhs'], batch=r1['batch'], transB=r1['transB'],
            beta=r1['beta'], pattern=r1['pattern'],
            r1=r1['ratio'], r2=r2['ratio'], worst=hi, best=lo, spread=spread,
            relsd1=r1['relsd'], relsd2=r2['relsd'],
            quiet=quiet, admitted=admitted,
            # per-call microseconds, worst (largest) of the two passes for native,
            # best (smallest) for vendor -- i.e. the most native-unfavourable read.
            tv_us=min(r1['t_vendor'], r2['t_vendor']) * 1000.0,
            tn_us=max(r1['t_native'], r2['t_native']) * 1000.0,
            tv_us_mean=(r1['t_vendor'] + r2['t_vendor']) * 500.0,
            tn_us_mean=(r1['t_native'] + r2['t_native']) * 500.0,
            fp_mb=r1['fp_mb'], l2res=r1['l2res'],
        ))
    return out


def ladder_tables(rs):
    print("=" * 100)
    print("THE LADDERS -- ratio = t_native / t_vendor, worst of two independent passes.")
    print("us columns are PER CALL, mean of the two passes. d us = native - vendor,")
    print("i.e. the absolute cost of routing native on this cell, per spmm call.")
    print("A ratio at batch <= 16 is a launch-latency ratio, not a kernel ratio.")
    print("=" * 100)
    for tag in ['sbL', 'sbM', 'sbS', 'sbB', 'sbT']:
        for pat in (0, 1):
            for beta in (0, 1):
                sel = [r for r in rs if r['tag'] == tag and r['pattern'] == pat
                       and r['beta'] == beta]
                if not sel:
                    continue
                print()
                print(f"--- {tag}: {FAMILY[tag]}  pattern={'banded' if pat==0 else 'scattered'}"
                      f"  beta={beta}")
                hdr = f"{'type':9s}{'':2s}" + "".join(f"{('b=%d'%b):>19s}" for b in BATCHES)
                print(hdr)
                for typ in TYPES:
                    line = f"{typ:9s}  "
                    for b in BATCHES:
                        m = [r for r in sel if r['typ'] == typ and r['batch'] == b]
                        if not m:
                            line += f"{'-':>19s}"
                            continue
                        r = m[0]
                        flag = '' if r['admitted'] else '?'
                        mark = '*' if (r['r1'] > GATE and r['r2'] > GATE) else (
                               '~' if (r['r1'] > GATE or r['r2'] > GATE) else ' ')
                        line += f"{r['worst']:8.3f}{mark}{flag:1s}{r['tn_us_mean']-r['tv_us_mean']:+8.1f}"
                        line += "" if len(line) % 1 else ""
                    print(line)
                    line2 = f"{'  us v/n':9s}  "
                    for b in BATCHES:
                        m = [r for r in sel if r['typ'] == typ and r['batch'] == b]
                        if not m:
                            line2 += f"{'-':>19s}"
                            continue
                        r = m[0]
                        line2 += f"{r['tv_us_mean']:9.2f}{r['tn_us_mean']:10.2f}"
                    print(line2)
    print()
    print("  * = over the 1.10 gate in BOTH passes (a MEASURED HARM)")
    print("  ~ = over the gate in ONE pass only (unreproduced; named individually below)")
    print("  ? = not admitted by the rel_sd/reproduction rule")


def clause(rs, name, pred):
    sel = [r for r in rs if pred(r)]
    adm = [r for r in sel if r['admitted']]
    harms = [r for r in adm if r['r1'] > GATE and r['r2'] > GATE]
    unrep = [r for r in adm if (r['r1'] > GATE) != (r['r2'] > GATE)]
    print()
    print(f"{name}")
    print(f"   moves {len(sel)} rows ({len(adm)} admitted, {len(sel)-len(adm)} not admitted)")
    if not adm:
        print("   NO ADMITTED ROWS")
        return
    ws = sorted(r['worst'] for r in adm)
    print(f"   worst-of-two {ws[-1]:.3f}   median {statistics.median(ws):.3f}   best {ws[0]:.3f}")
    print(f"   {'PASSES' if not harms else 'FAILS'}: {len(harms)} of {len(adm)} over the "
          f"{GATE} gate in both passes")
    for r in sorted(harms, key=lambda r: -r['worst']):
        print(f"     HARM  {r['typ']:8s} {r['tag']} m={r['m']} nnz/row={r['nnzrow']} "
              f"nrhs={r['nrhs']} b={r['batch']} tB={r['transB']} beta={r['beta']} "
              f"pat={r['pattern']}  p1={r['r1']:.3f} p2={r['r2']:.3f}  "
              f"vendor {r['tv_us_mean']:.2f}us native {r['tn_us_mean']:.2f}us  "
              f"d={r['tn_us_mean']-r['tv_us_mean']:+.2f}us")
    for r in sorted(unrep, key=lambda r: -r['worst']):
        print(f"     UNREPRODUCED  {r['typ']:8s} {r['tag']} b={r['batch']} tB={r['transB']} "
              f"beta={r['beta']} pat={r['pattern']}  p1={r['r1']:.3f} p2={r['r2']:.3f}  "
              f"d={r['tn_us_mean']-r['tv_us_mean']:+.2f}us")


def main():
    d1, d2 = sys.argv[1], sys.argv[2]
    rs = load(d1, d2)
    print(f"{len(rs)} cells present and chk-agreeing in BOTH passes "
          f"({sum(1 for r in rs if r['admitted'])} admitted)")

    ladder_tables(rs)

    print()
    print("=" * 100)
    print("CLAUSE ARITHMETIC over the small-batch corner (batch <= 64 only, i.e. the")
    print("region NO existing sweep covers; batch 128 is the overlap anchor and is")
    print("scored separately against pass1/pass2).")
    print("=" * 100)
    lo = [r for r in rs if r['batch'] <= 64]
    clause(lo, "batch <= 64, transA=NoTrans, unconditional", lambda r: True)
    clause(lo, "batch <= 64, WITH the cfloat/transB!=NoTrans exclusion",
           lambda r: not (r['typ'] == 'cfloat' and r['transB'] != 0))
    for b in BATCHES:
        clause([r for r in rs if r['batch'] == b],
               f"batch == {b} (all types, all families, incl. cfloat tB=T)",
               lambda r: True)

    print()
    print("=" * 100)
    print("PER-BATCH SUMMARY under the RECOMMENDED clause "
          "(transA=NoTrans, minus cfloat+transB!=NoTrans)")
    print("=" * 100)
    print(f"{'batch':>6s}{'n':>5s}{'worst':>9s}{'median':>9s}{'best':>9s}"
          f"{'harms':>7s}{'max d us/call':>16s}")
    for b in BATCHES:
        sel = [r for r in rs if r['batch'] == b and r['admitted']
               and not (r['typ'] == 'cfloat' and r['transB'] != 0)]
        if not sel:
            continue
        ws = sorted(r['worst'] for r in sel)
        h = sum(1 for r in sel if r['r1'] > GATE and r['r2'] > GATE)
        dmax = max(r['tn_us_mean'] - r['tv_us_mean'] for r in sel)
        print(f"{b:6d}{len(ws):5d}{ws[-1]:9.3f}{statistics.median(ws):9.3f}{ws[0]:9.3f}"
              f"{h:7d}{dmax:+16.2f}")

    print()
    print("=" * 100)
    print("THE cfloat + transB=Trans + BANDED FAMILY ACROSS THE WHOLE BATCH LADDER")
    print("(the known large-batch loser: 1.71-1.73 at batch 512, cfedge1/cfedge2).")
    print("float on the identical cells is the control.")
    print("=" * 100)
    print(f"{'type':10s}" + "".join(f"{('b=%d'%b):>10s}" for b in BATCHES))
    for typ in TYPES:
        line = f"{typ:10s}"
        for b in BATCHES:
            m = [r for r in rs if r['tag'] == 'sbT' and r['typ'] == typ and r['batch'] == b]
            line += f"{m[0]['worst']:10.3f}" if m else f"{'-':>10s}"
        print(line)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'smallbatch.csv'), 'w', newline='') as fh:
        cols = ['typ', 'tag', 'm', 'nnzrow', 'nrhs', 'batch', 'transB', 'beta',
                'pattern', 'r1', 'r2', 'worst', 'best', 'spread', 'relsd1',
                'relsd2', 'quiet', 'admitted', 'tv_us_mean', 'tn_us_mean',
                'fp_mb', 'l2res']
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in sorted(rs, key=lambda r: (r['tag'], r['typ'], r['pattern'],
                                           r['beta'], r['batch'])):
            w.writerow({c: r[c] for c in cols})
    print()
    print("wrote smallbatch.csv")


if __name__ == '__main__':
    main()

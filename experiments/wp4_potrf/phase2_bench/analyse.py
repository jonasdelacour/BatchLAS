#!/usr/bin/env python3
"""Summarise main.csv.

DISCARD RULE, applied here and nowhere else: any row with rel_sd > 0.10 is
dropped and NAMED in the output, so a discarded cell is visible rather than
silently missing.  A cell whose vendor arm or blocked arm was discarded produces
no ratio.

CORRECTNESS FLAG: a blocked row with info_nonzero > 0, upper_changed > 0,
nonfinite > 0, or a residual worse than 1e-4 (float/cfloat) / 1e-12
(double/cdouble) is marked WRONG.  Its timing is still printed -- a
right-looking Cholesky has no pivoting and no early exit, so the wrong run does
the same launches and the same flops as a right one -- but no ratio from a WRONG
row is admissible as a performance claim.
"""
import csv, collections, math, os, sys

D = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else 'main.csv'

# OVERRIDES. main.csv is one run; recheck.csv and wins.csv are three passes of
# 7 and 9 reps on the cells that were either discarded by the sd rule or looked
# like a win.  Where an override exists it REPLACES the main-grid ratio, is
# marked `+` in the table, and is what the geomeans use -- main.csv itself is
# left exactly as the run produced it so the difference stays auditable.
# Built by make_overrides.py; each row names its source and pass count.
OVR = {}
_op = os.path.join(D, 'overrides.csv')
if os.path.exists(_op):
    for _r in csv.DictReader(open(_op)):
        OVR[(_r['cfg'], _r['type'], int(_r['n']), int(_r['batch']))] = (
            float(_r['ratio']), float(_r['med_ms']),
            int(_r['worst_info_nonzero']), _r['source'])
SD_MAX = 0.10
RES_MAX = {'float': 1e-4, 'cfloat': 1e-4, 'double': 1e-12, 'cdouble': 1e-12}

rows = list(csv.DictReader(open(PATH)))
for r in rows:
    r['n'] = int(r['n']); r['batch'] = int(r['batch'])
    r['med'] = float(r['med_ms']); r['sd'] = float(r['rel_sd'])
    r['res'] = float(r['residual']); r['inz'] = int(r['info_nonzero'])
    r['upch'] = int(r['upper_changed']); r['nf'] = int(r['nonfinite'])
    r['wrong'] = (r['inz'] > 0 or r['upch'] > 0 or r['nf'] > 0
                  or r['res'] > RES_MAX[r['type']])

discarded = [r for r in rows if r['sd'] > SD_MAX]
good = [r for r in rows if r['sd'] <= SD_MAX]

by = {}
for r in good:
    by[(r['cfg'], r['variant'], r['type'], r['n'], r['batch'])] = r

cells = sorted({(r['type'], r['n'], r['batch']) for r in good}
               | {(k[1], k[2], k[3]) for k in OVR},
               key=lambda c: (c[0], c[1], c[2]))
cfgs = ['def', 'nn', 'VV']

print(f"discarded (rel_sd > {SD_MAX}): {len(discarded)} of {len(rows)}")
for r in discarded:
    print(f"   DISCARD {r['cfg']:4} {r['variant']:8} {r['type']:8} n={r['n']:5} "
          f"batch={r['batch']:5} med={r['med']:.4f} rel_sd={r['sd']:.3f}")
print()

hdr = f"{'type':8}{'n':>6}{'batch':>7}{'nb':>5}{'W':>4}{'vendor_ms':>11}"
for c in cfgs:
    hdr += f"{c+'_ms':>11}{c+'_x':>8}"
hdr += f"{'cta_ms':>10}{'cta_x':>8}  flags   (+ = value from overrides.csv)"
print(hdr)
ratios = collections.defaultdict(list)
ratios_all = collections.defaultdict(list)
for (t, n, b) in cells:
    v = (by.get(('def', 'vendor', t, n, b)) or by.get(('nn', 'vendor', t, n, b))
         or by.get(('VV', 'vendor', t, n, b)))
    if v is None:
        print(f"{t:8}{n:6}{b:7}   -- no vendor reference (discarded or missing)")
        continue
    line = f"{t:8}{n:6}{b:7}{v['nb']:>5}{v['W']:>4}{v['med']:11.4f}"
    flags = []
    # The Phase 1 CTA tier, on the same row, whenever the order fits it -- so
    # "what does vendor freedom cost at this order" has one answer and not two.
    ct = by.get(('def', 'cta', t, n, b))
    ctastr = (f"{ct['med']:10.4f}{v['med'] / ct['med']:8.3f}" if ct
              else f"{'--':>10}{'--':>8}")
    for c in cfgs:
        r = by.get((c, 'blocked', t, n, b))
        if r is None:
            # DISCARDED by the sd rule -- but if it was re-measured, the
            # re-measurement stands on its own.  Dropping the cell entirely
            # would let the discard rule quietly delete evidence rather than
            # replace it, and one of these five cells is where a false 1.055x
            # came from.
            o0 = OVR.get((c, t, n, b))
            if o0 is None:
                line += f"{'--':>10} {'--':>8}"; continue
            r = {'med': o0[1], 'inz': o0[2], 'upch': 0, 'nf': 0, 'res': 0.0,
                 'wrong': o0[2] > 0}
        # ratio against the vendor arm measured in THAT SAME PROCESS, not the
        # def one -- interleaving is only meaningful within a process.
        vv = by.get((c, 'vendor', t, n, b)) or v
        x = vv['med'] / r['med']
        o = OVR.get((c, t, n, b))
        mark = ' '
        if o:
            # the override also decides WRONG: a cell clean in main.csv but
            # failing in 1 of 3 rechecks is not a correct cell.
            x, med, nz, _src = o
            r = dict(r)
            r['med'], r['inz'] = med, nz
            r['wrong'] = (nz > 0 or r['upch'] > 0 or r['nf'] > 0
                          or r['res'] > RES_MAX[t])
            mark = '+'
        line += f"{r['med']:10.4f}{mark}{x:8.3f}"
        if r['wrong']:
            flags.append(f"{c}=WRONG(info={r['inz']},res={r['res']:.1e})")
        else:
            ratios[(c, t)].append(x)
            ratios[(c, 'ALL')].append(x)
        # The SAME ratio, kept a second time with no correctness filter.  A
        # right-looking Cholesky has no pivoting and no early exit, so a wrong
        # run issues the same kernels on the same shapes and its wall time is a
        # valid timing of the same work; and filtering to correct cells alone
        # biases the summary toward SMALL BATCH, which is precisely where the
        # failures stop.  Both numbers, so neither bias can hide.
        ratios_all[(c, t)].append(x)
        ratios_all[(c, 'ALL')].append(x)
    print(line + ctastr + ("  " + "; ".join(flags) if flags else ""))

def geo(tab, title, note):
    print()
    print("GEOMEAN of vendor_ms/blocked_ms  (>1 = blocked native FASTER than cuSOLVER)")
    print(title)
    print(note)
    print(f"{'cfg':5}{'type':10}{'cells':>7}{'geomean':>10}{'min':>9}{'max':>9}")
    for c in cfgs:
        for t in ['float', 'double', 'cfloat', 'cdouble', 'ALL']:
            v = tab.get((c, t))
            if not v:
                print(f"{c:5}{t:10}{0:7}      (no cell)")
                continue
            g = math.exp(sum(math.log(x) for x in v) / len(v))
            print(f"{c:5}{t:10}{len(v):7}{g:10.3f}{min(v):9.3f}{max(v):9.3f}")


geo(ratios, "CORRECT cells only.",
    "Biased toward SMALL BATCH -- that is exactly where the native trsm defect stops firing.")
geo(ratios_all, "ALL cells, correct or not.",
    "The timing is valid even where the answer is not: same launches, same flops.")

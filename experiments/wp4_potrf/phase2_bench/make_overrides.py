#!/usr/bin/env python3
"""Build overrides.csv from recheck.csv and wins.csv.

WHY AN OVERRIDE FILE RATHER THAN EDITING main.csv: main.csv is the raw output of
one run and stays that way, so anyone can see what the first pass actually said
and what was changed. Every override names its source file and carries the
number of independent passes behind it, and `analyse.py` marks the cell.

Both source files ran 3 passes; the override is the MEDIAN of the per-pass
ratios, each ratio formed against the vendor arm interleaved in that same
process. A cell present in both files takes the one with more reps (wins.csv,
9 reps, over recheck.csv's 7).
"""
import csv, collections, os, statistics

D = os.path.dirname(os.path.abspath(__file__))


def ratios(path, reps):
    d = {}
    for r in csv.DictReader(open(os.path.join(D, path))):
        d[(r['pass'], r['cfg'], r['type'], r['n'], r['batch'], r['variant'])] = r
    out = collections.defaultdict(list)
    for k, r in d.items():
        if k[5] != 'blocked':
            continue
        v = d.get((k[0], k[1], k[2], k[3], k[4], 'vendor'))
        if not v:
            continue
        out[(k[1], k[2], int(k[3]), int(k[4]))].append(
            (float(v['med_ms']) / float(r['med_ms']), float(r['med_ms']),
             int(r['info_nonzero'])))
    return {k: (statistics.median(x[0] for x in v),
                statistics.median(x[1] for x in v),
                max(x[2] for x in v), len(v), path, reps)
            for k, v in out.items()}


best = {}
for path, reps in (('recheck.csv', 7), ('wins.csv', 9)):
    for k, v in ratios(path, reps).items():
        if k not in best or v[5] > best[k][5]:
            best[k] = v

with open(os.path.join(D, 'overrides.csv'), 'w') as fh:
    w = csv.writer(fh)
    w.writerow(['cfg', 'type', 'n', 'batch', 'ratio', 'med_ms',
                'worst_info_nonzero', 'passes', 'reps_per_pass', 'source'])
    for k in sorted(best, key=lambda k: (k[1], k[2], k[3], k[0])):
        x = best[k]
        w.writerow([k[0], k[1], k[2], k[3], f'{x[0]:.4f}', f'{x[1]:.4f}',
                    x[2], x[3], x[5], x[4]])
print(f'wrote overrides.csv with {len(best)} cells')

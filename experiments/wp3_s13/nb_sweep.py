#!/usr/bin/env python3
"""Sweep V2's outer block width on the cells that were losing.

float Side::Left at order 256 and 512 is the only real gap in the baseline
(0.58-0.93x, relative sd 0.1-2.2%). The two-level driver makes the outer width a
knob; this finds where it should sit, and checks the orders that were ALREADY
winning for regressions -- widening the trailing update changes the schedule for
every order above the width, so a win at 256 that costs 128 is not a win.

Serial by construction, one card, guard on every invocation, and every leg is
noise-gated: the step-12 sweep produced 22 of 180 cells at 10-103% relative sd
because two copies of my own script ran at once, and gpu_guard cannot see that.
"""
import csv, os, subprocess, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BIN = os.path.join(ROOT, 'build/benchmarks/trsm_benchmark')
GUARD = os.path.join(ROOT, 'experiments/gpu_guard.sh')
GPU = os.environ.get('GPU', '1')
CAP = 6.0e9
SIZEOF = {'float': 4, 'complex<float>': 8}
BENCH = {'Left': 'BM_TRSM_OrthoLeft', 'Right': 'BM_TRSM_OrthoRight'}

# Orders that were losing, plus the ones just below that must not regress.
NS = [64, 128, 256, 512]
QS = [256, 1024, 4096]
BATCHES = [128, 512]
NBS = ['32', '64', '128', '256']      # 32 == the old single-level schedule


def fits(t, n, q, b):
    return SIZEOF[t] * b * (2.0 * q * n + n * n) <= CAP


def run(t, side, route, n, batch, qs, nb, out):
    env = dict(os.environ, BATCHLAS_TRSM_ROUTE=route, GPU_GUARD_MAX_WAIT='5400')
    if nb is not None:
        env['BATCHLAS_TRSM_OUTER_NB'] = nb
    cmd = [GUARD, GPU, BIN, '--backend=CUDA', f'--type={t}', f'--name={BENCH[side]}',
           '--min_time=200', '--min_iters=10', '--max_iters=200', f'--csv={out}',
           str(n), ','.join(map(str, qs)), str(batch)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if 'WARNING -- foreign process' in r.stderr or r.returncode != 0:
        return None
    return out


def main():
    types = sys.argv[1:] or ['float']
    outdir = os.path.join(HERE, 'nb_raw')
    os.makedirs(outdir, exist_ok=True)
    rows = []
    jobs = [(t, side, n, b) for t in types for side in ('Left', 'Right')
            for n in NS for b in BATCHES]
    for i, (t, side, n, batch) in enumerate(jobs, 1):
        qs = [q for q in QS if fits(t, n, q, batch)]
        if not qs:
            continue
        legs = [('vendor', None)] + [('native', nb) for nb in NBS]
        for route, nb in legs:
            tag = f"{t.replace('<','_').replace('>','')}-{side}-n{n}-b{batch}-{route}{nb or ''}"
            print(f'[{i}/{len(jobs)}] {tag} q={qs}', flush=True)
            got = run(t, side, route, n, batch, qs, nb, os.path.join(outdir, tag + '.csv'))
            if not got:
                print('    FAILED', flush=True)
                continue
            for r in csv.DictReader(open(got)):
                ms, sd = float(r['avg_ms']), float(r['stddev_ms'])
                rows.append(dict(type=t, side=side, n=int(r['arg0']), q=int(r['arg1']),
                                 batch=int(r['arg2']), route=route, nb=(nb or '-'),
                                 ms=ms, sd_pct=100.0 * sd / ms if ms else 0.0))
    with open(os.path.join(HERE, 'nb_sweep.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    noisy = sum(1 for r in rows if r['sd_pct'] > 10)
    print(f'\n{len(rows)} cells, {noisy} noisy (>10% sd)')


if __name__ == '__main__':
    main()

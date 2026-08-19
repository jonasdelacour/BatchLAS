#!/usr/bin/env python3
"""WP3 step 13 -- baseline for "win at every N from 8 to 512, both sides,
float and complex<float>".

N=512 HAS NEVER BEEN MEASURED. Every grid so far (step 9, step 12) stopped at
256, and TrsmOrthoSizes caps cells at 6 GB computed for complex<double>, so it
would have dropped most of the 512 row anyway. This driver sizes each cell for
the ACTUAL type and emits one benchmark invocation per feasible
(type, side, route, n, batch), passing the q values that fit.

Memory per cell = sizeof(T) * batch * (2*q*n + n*n): B, its pristine copy that
the harness keeps so an in-place solve can be re-run, and A.

Serial by construction. Two benchmark processes on one card is how the step-12
sweep produced 22 of 180 cells at 10-103% relative sd while gpu_guard reported
it clean -- the guard samples FOREIGN processes, and a second copy of your own
sweep is not foreign.
"""
import csv, itertools, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
BIN = os.path.join(ROOT, 'build/benchmarks/trsm_benchmark')
GUARD = os.path.join(ROOT, 'experiments/gpu_guard.sh')
GPU = os.environ.get('GPU', '1')
CAP = 6.0e9

SIZEOF = {'float': 4, 'complex<float>': 8}
BENCH = {'Left': 'BM_TRSM_OrthoLeft', 'Right': 'BM_TRSM_OrthoRight'}
NS = [8, 16, 32, 64, 128, 256, 512]
QS = [256, 1024, 4096]
BATCHES = [128, 512, 2048]


def fits(t, n, q, b):
    return SIZEOF[t] * b * (2.0 * q * n + n * n) <= CAP


def run(t, side, route, n, batch, qs, out_csv):
    env = dict(os.environ, BATCHLAS_TRSM_ROUTE=route, GPU_GUARD_MAX_WAIT='5400')
    cmd = [GUARD, GPU, BIN, '--backend=CUDA', f'--type={t}',
           f'--name={BENCH[side]}', '--min_time=200', '--min_iters=10',
           '--max_iters=200', f'--csv={out_csv}',
           str(n), ','.join(map(str, qs)), str(batch)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if 'WARNING -- foreign process' in r.stderr:
        return None, 'foreign process'
    if r.returncode != 0:
        return None, f'exit {r.returncode}: {r.stderr.strip().splitlines()[-1:]}'
    return out_csv, None


def main():
    outdir = os.path.join(HERE, 'raw')
    os.makedirs(outdir, exist_ok=True)
    rows, skipped = [], []
    jobs = list(itertools.product(SIZEOF, ['Left', 'Right'], ['vendor', 'native'],
                                  NS, BATCHES))
    for i, (t, side, route, n, batch) in enumerate(jobs, 1):
        qs = [q for q in QS if fits(t, n, q, batch)]
        if not qs:
            skipped.append((t, side, route, n, batch))
            continue
        tag = f"{t.replace('<','_').replace('>','')}-{side}-{route}-n{n}-b{batch}"
        path = os.path.join(outdir, tag + '.csv')
        print(f'[{i}/{len(jobs)}] {tag} q={qs}', flush=True)
        got, err = run(t, side, route, n, batch, qs, path)
        if err:
            print(f'    FAILED: {err}', flush=True)
            continue
        for r in csv.DictReader(open(got)):
            ms, sd = float(r['avg_ms']), float(r['stddev_ms'])
            rows.append(dict(type=t, side=side, route=route,
                             n=int(r['arg0']), q=int(r['arg1']), batch=int(r['arg2']),
                             ms=ms, sd_pct=(100.0 * sd / ms if ms else 0.0)))
    with open(os.path.join(HERE, 'baseline.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n{len(rows)} cells written; {len(skipped)} (type,side,route,n,batch) '
          f'groups had no q fitting the {CAP/1e9:.0f} GB cap:')
    for s in skipped:
        print('   ', s)


if __name__ == '__main__':
    main()

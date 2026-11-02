#!/usr/bin/env python3
"""A/B timing: same benchmark binary (val-base build), libraries swapped via LD_LIBRARY_PATH.

ab.py <family> <reps> <spec.json>   spec: list of {"label", "argv" (list, may contain {csv}), "env" {..}, "kind": mini|factor}
Rounds alternate order (AB, BA, ...); round 0 discarded. Before every block (one spec) GPU 0 must be idle
(util <= 5% on 3 samples) and GPU 1 free of other compute apps; otherwise the block is reported DEFERRED.
"""
import csv, fcntl, json, os, statistics, subprocess, sys, time

S = os.path.dirname(os.path.abspath(__file__))
WT = os.path.expanduser('~/BatchLAS/.claude/worktrees')
BENCH = WT + '/val-base/build/benchmarks'


def gpu_util(idx):
    out = subprocess.run(['nvidia-smi', '-i', str(idx), '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    return int(out)


def foreign_gpu1():
    ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    return any('gpu_guard.sh 1 ' in l for l in ps.splitlines())


def gpu1_free(wait_min):
    # GPU 0 is not gated (orchestrator, 2026-11-02); GPU 1 must be free of other measuring runs.
    deadline = time.time() + wait_min * 60
    while True:
        samples = []
        for _ in range(3):
            samples.append(gpu_util(1)); time.sleep(1)
        if max(samples) <= 5 and not foreign_gpu1():
            return True, samples
        if time.time() > deadline:
            return False, samples
        time.sleep(20)


def run_one(tree, spec, csvpath):
    env = {k: v for k, v in os.environ.items() if not k.startswith('BATCHLAS_')}
    env['LD_LIBRARY_PATH'] = f'{tree}/build/src:/opt/dpcpp-cuda/lib'
    env['CUDA_VISIBLE_DEVICES'] = '1'
    env['ONEAPI_DEVICE_SELECTOR'] = 'cuda:*'
    env.update(spec.get('env', {}))
    argv = [a.replace('{csv}', csvpath) for a in spec['argv']]
    argv[0] = os.path.join(BENCH, argv[0])
    if os.path.exists(csvpath):
        os.remove(csvpath)
    p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=1800)
    if spec.get('kind', 'mini') == 'factor':
        out = []
        for line in p.stdout.splitlines():
            f = line.split(',')
            if len(f) > 10 and f[0] in ('potrf', 'geqrf', 'posv', 'orgqr', 'getrf'):
                try:
                    out.append((f'{f[0]} {f[1]} n={f[3]} batch={f[5]} {f[6]}:{f[7]}', float(f[9])))
                except ValueError:
                    pass
        return out, p
    rows = list(csv.DictReader(open(csvpath))) if os.path.exists(csvpath) else []
    out = []
    for r in rows:
        key = r['name'].strip('(").').split('<')[0] + ' ' + r['name'].split('(')[-1].split(')')[0].split('<',1)[1].rsplit(', batchlas',1)[0].split(',')[0].replace('std::','') + ' ' + ' '.join(r[k] for k in r if k.startswith('arg'))
        out.append((key, float(r['avg_ms'])))
    return out, p


def main():
    fam, reps, specfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    specs = json.load(open(specfile))
    trees = {'base': WT + '/val-base', 'cand': WT + '/val-' + fam}
    report = []
    for spec in specs:
        lk = open('/tmp/batchlas-measure.lock', 'a')
        fcntl.flock(lk, fcntl.LOCK_EX)
        ok, samples = (True, []) if os.environ.get('AB_NOGATE') else gpu1_free(float(os.environ.get('AB_WAIT_MIN', '60')))
        if not ok:
            report.append(f"{spec['label']}: DEFERRED (gpu1 busy: {samples})")
            print(report[-1], flush=True)
            lk.close()
            continue
        g0, contended = [gpu_util(0)], False
        data = {'base': {}, 'cand': {}}
        for rnd in range(reps):
            order = ['base', 'cand'] if rnd % 2 == 0 else ['cand', 'base']
            for side in order:
                sp = spec
                if 'envB' in spec:  # env A/B inside the candidate tree
                    sp = dict(spec, env=dict(spec.get('env', {}), **(spec['envB'] if side == 'cand' else {})))
                g0.append(gpu_util(0)); contended |= foreign_gpu1()
                res, p = run_one(trees['cand'] if 'envB' in spec else trees[side], sp, f'{S}/ab_tmp.csv')
                if not res:
                    print('NO RESULT', spec['label'], side, p.returncode, p.stderr[-800:], flush=True)
                if rnd == 0 and reps > 1:
                    continue
                for k, v in res:
                    data[side].setdefault(k, []).append(v)
        lk.close()
        hdr = (f"{spec['label']}: GPU0 util median {statistics.median(g0)}% [{min(g0)}-{max(g0)}]"
               f"{' | GPU1 CONTENDED by another measuring run' if contended else ''}")
        report.append(hdr); print(hdr, flush=True)
        for k in data['base']:
            b, c = data['base'][k], data['cand'].get(k, [])
            if not c:
                continue
            mb, mc = statistics.median(b), statistics.median(c)
            if mb <= 0 or mc <= 0:
                print('BAD ZERO', spec['label'], k, flush=True); continue
            line = (f"{spec['label']} | {k} | base {mb:.4f} ms [{min(b):.4f}-{max(b):.4f}] | cand {mc:.4f} ms "
                    f"[{min(c):.4f}-{max(c):.4f}] | cand/base {mc / mb:.3f} | n={len(b)}")
            report.append(line)
            print(line, flush=True)
    if os.environ.get('AB_NOGATE'):
        return
    with open(f'{S}/fam/{fam}.timing.txt', 'a') as f:
        f.write('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()

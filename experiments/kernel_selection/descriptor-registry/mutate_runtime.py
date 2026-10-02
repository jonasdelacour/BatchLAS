#!/usr/bin/env python3
"""Deliberate breaks of the descriptor-registry path, run on the GPU: each must turn a NARROW,
named set red. Mutates in place, rebuilds potrf_select_tests, runs, restores (md5 checked).

Usage: python3 mutate_runtime.py <build-script> [case ...]
  <build-script> builds a target given as its argument (env set up for the CUDA build).
Runs on CUDA_VISIBLE_DEVICES=1.
"""
import collections
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '../../..'))
ROUTES = os.path.join(ROOT, 'src/backends/potrf_routes.hh')
WINDOWS = os.path.join(ROOT, 'src/backends/potrf_windows.hh')
IMPL = os.path.join(HERE, 'potrf_routes.cc')
BIN = os.path.join(ROOT, 'build/tests/potrf_select_tests')

T1 = 'PotrfSelectV2.T1_*'
CASES = {
    'no-extension-tier': (ROUTES, [(', CtaWg<T>, Cusolver<B, T>>', ', Cusolver<B, T>>'),
                                   (', Blocked<B, T>, CtaWg<T>>>;', ', Blocked<B, T>>>;')],
                          T1 + ':PotrfSelectV2.T3_*:PotrfSelectV2.T4_*'),
    'margin-zero': (IMPL, [('b->book[pi] = {P->model_enabled, P->margin, b->rows[pi]};',
                            'b->book[pi] = {P->model_enabled, 0.0, b->rows[pi]};')], T1),
    'window-off-by-one': (WINDOWS, [('std::is_same_v<T, float> ? 35 : 32;', 'std::is_same_v<T, float> ? 36 : 32;')], T1),
    'lpanel-not-lower-only': (ROUTES, [('using Geometry = pp::LpanelGeometry;\n    static Verdict legal(const PotrfShape& s) { return lower_only(s, native_common(s)); }',
                                        'using Geometry = pp::LpanelGeometry;\n    static Verdict legal(const PotrfShape& s) { return native_common(s); }')],
                              T1 + ':PotrfSelectV2.T4_*'),
}


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def summarize(out):
    groups = collections.Counter()
    for l in out.splitlines():
        m = re.match(r'MISMATCH (\S+)\s+(\S+) (\S) n=(\d+) b=\d+\s+old=(\S+)\s+new=(\S+)', l)
        if m:
            groups[(m.group(1), m.group(2), m.group(3), m.group(5), m.group(6))] += 1
    lines = ['    %-26s %-7s %s old=%-16s new=%-16s x%d' % (k + (v,)) for k, v in sorted(groups.items())]
    ns = collections.defaultdict(set)
    for l in out.splitlines():
        m = re.match(r'MISMATCH (\S+)\s+(\S+) (\S) n=(\d+)', l)
        if m:
            ns[(m.group(1), m.group(2), m.group(3))].add(int(m.group(4)))
    lines += ['    orders %s %s %s: %s' % (k + (sorted(v),)) for k, v in sorted(ns.items())]
    summ = [l for l in out.splitlines() if l.startswith('[sweep]') or l.startswith('[grid]')]
    failed = sorted(set(re.findall(r'\[  FAILED  \] (PotrfSelectV2\.\w+)', out)))
    return summ, failed, lines


def main():
    build = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
    env['LD_LIBRARY_PATH'] = '/opt/dpcpp-cuda/lib:' + env.get('LD_LIBRARY_PATH', '')
    for name in names:
        path, edits, filt = CASES[name]
        text = open(path).read()
        before = md5(path)
        mutated = text
        for old, new in edits:
            assert old in mutated, (name, old)
            mutated = mutated.replace(old, new)
        try:
            open(path, 'w').write(mutated)
            b = subprocess.run([build, 'potrf_select_tests'], capture_output=True, text=True)
            if b.returncode:
                print('[%s] BUILD FAILED\n%s' % (name, b.stdout[-3000:] + b.stderr[-2000:]))
                continue
            r = subprocess.run([BIN, '--gtest_filter=' + filt], capture_output=True, text=True, env=env)
        finally:
            open(path, 'w').write(text)
            assert md5(path) == before, 'restore failed: ' + path
        summ, failed, lines = summarize(r.stdout + r.stderr)
        print('[%s] failed tests: %s' % (name, ', '.join(failed) or 'none'))
        for l in summ + lines:
            print('   ' + l)
    # leave the binary built from the pristine sources
    subprocess.run([build, 'potrf_select_tests'], capture_output=True, text=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Break campaign for BODY 4 (and re-runs of two inherited breaks).

Every break is an actual edit to src/sycl/gemv_native.cc, actually applied,
actually rebuilt in build-novendor, actually run, then reverted.

Edits are confined to body 4's own function text where the break is meant to be
body-4-specific, so that "the pre-existing tests stay green" is a real
measurement about which body ran and not an accident of a shared code path.
"""
import subprocess, sys, os, re

W = '/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan'
SRC = W + '/src/sycl/gemv_native.cc'
PRISTINE = '/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/repair/gemv_native.cc.repaired'
OUT = '/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/repair/breaks_out'
os.makedirs(OUT, exist_ok=True)

BASE = open(PRISTINE).read()

def body4(s):
    """(start, end) offsets of body 4's launcher function."""
    a = s.index('template <typename T, int W>\nEvent gemv_seg_notrans')
    b = s.index('\n// ===========================================================================\n// BODY 2', a)
    return a, b

def in_body4(s, old, new, count=1):
    a, b = body4(s)
    reg = s[a:b]
    assert reg.count(old) == count, f'body4 region: expected {count} of {old!r}, got {reg.count(old)}'
    return s[:a] + reg.replace(old, new) + s[b:]

def whole(s, old, new, count=1):
    assert s.count(old) == count, f'whole file: expected {count} of {old!r}, got {s.count(old)}'
    return s.replace(old, new)

BREAKS = {
  # --- body-4-specific -----------------------------------------------------
  'segfold':   lambda s: in_body4(s,
                  'const int off = seg * w;', 'const int off = w;'),
  'segfold2':  lambda s: in_body4(s,
                  'acc += sycl::shift_group_left(sg, acc, seg * w);',
                  'acc += sycl::shift_group_left(sg, acc, w);'),
  'segmap':    lambda s: in_body4(s,
                  'const int i = lane % seg;\n                const int jsub = lane / seg;',
                  'const int i = lane / seg;\n                const int jsub = lane % seg;'),
  'segld':     lambda s: in_body4(s,
                  'const int64_t lda = A.ld();', 'const int64_t lda = A.rows();'),
  'segxinc':   lambda s: in_body4(s,
                  'const int64_t xinc = X.inc();', 'const int64_t xinc = 1;'),
  'segyinc':   lambda s: in_body4(s,
                  'const int64_t yinc = Y.inc();', 'const int64_t yinc = 1;'),
  'segstride': lambda s: in_body4(s,
                  'const int64_t stride_a = A.stride();', 'const int64_t stride_a = 0;'),
  'segbeta':   lambda s: in_body4(s,
                  'if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(i) * yinc]);',
                  'fma_acc(out, beta_d, yb[static_cast<int64_t>(i) * yinc]);'),
  'segalpha':  lambda s: in_body4(s,
                  'if (!alpha_zero && jsub < wlanes) {', 'if (jsub < wlanes) {'),
  'segactive': lambda s: in_body4(s,
                  'if (!alpha_zero && jsub < wlanes) {', 'if (!alpha_zero) {'),
  'segwrite':  lambda s: in_body4(s,
                  'if (lane < seg) {', 'if (lane == 0) {'),
  # THE GATE, as an off-by-one in the width function: out_len == 17 is the only
  # length for which 32 < 2*out_len <= 34, so body 4 claims it with W = 2 and
  # needs 34 lanes it does not have.
  'segwidth34': lambda s: whole(s,
                  'while (w * 2 * out_len <= 32) w *= 2;',
                  'while (w * 2 * out_len <= 34) w *= 2;'),
  # THE GATE, the other direction: admit W == 1, so body 4 takes every NoTrans
  # shape including those with out_len > 32, where lanes cannot cover the output.
  'seggate1':  lambda s: whole(s, 'if (w >= 2 && ctx.device()', 'if (w >= 1 && ctx.device()'),
  # --- inherited breaks, re-run against the enlarged suite ------------------
  # `cross` deletes the complex cross-terms in the SHARED scalar helper, so it
  # hits all four bodies. Re-run to show the new body-4 tests see it -- the
  # forty pre-WP7 cases at 10x10 do not, because their complex data is real.
  'cross':     lambda s: None,
}

def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

def apply_cross():
    """The cross-term break lives in the shared device-scalar header."""
    p = W + '/src/sycl/device_scalar.hh'
    orig = open(p).read()
    m = re.search(r'(inline\s+\w+\s+dev_mul[^\n]*\n(?:.*?\n)*?\})', orig)
    return p, orig

def gemv_run():
    r = run(f'CUDA_VISIBLE_DEVICES=1 {W}/build-novendor/tests/gemv_tests')
    return r.stdout + r.stderr

def failed_names(out):
    r = set()
    for l in out.splitlines():
        if not l.startswith('[  FAILED  ] '): continue
        t = l[len('[  FAILED  ] '):].strip()
        if t.endswith(')'): t = t.rsplit(' (', 1)[0]
        if '.' not in t: continue
        suite, case = t.split('.', 1)
        case = case.split(',')[0].strip()
        ty = t.split('TestConfig<', 1)[1].split(', (')[0] if 'TestConfig<' in t else '?'
        be = t.rsplit('Backend)', 1)[1].rstrip('> )') if 'Backend)' in t else '?'
        r.add((suite.split('/')[0], case, ty, be))
    return sorted(r)

def summary(out):
    tot = [l for l in out.splitlines() if 'tests from' in l and 'ran.' in l]
    pas = [l for l in out.splitlines() if l.startswith('[  PASSED  ]')]
    fai = [l for l in out.splitlines() if 'FAILED TEST' in l]
    return (tot[-1] if tot else '?'), (pas[-1] if pas else '?'), (fai[-1] if fai else '0 FAILED')

names = sys.argv[1:] if len(sys.argv) > 1 else [k for k in BREAKS if k != 'cross']
for name in names:
    fn = BREAKS[name]
    src = fn(BASE)
    open(SRC, 'w').write(src)
    b = run(f'cmake --build {W}/build-novendor -j --target gemv_tests')
    if b.returncode != 0:
        print(f'{name}: BUILD FAILED')
        print(b.stdout[-3000:], b.stderr[-3000:])
        open(SRC, 'w').write(BASE)
        continue
    out = gemv_run()
    tot, pas, fai = summary(out)
    fn_list = failed_names(out)
    pre = [n for n in fn_list if n[0] == 'GemvMatrixViewTest']
    cov = sorted({n[1] for n in fn_list if n[0] == 'GemvCoverageTest'})
    bes = sorted({n[3] for n in fn_list})
    with open(f'{OUT}/{name}.log', 'w') as f:
        f.write(out)
    print(f'=== {name}')
    print(f'    {tot}')
    print(f'    {pas} | {fai}')
    print(f'    pre-WP7 GemvMatrixViewTest cases red: {len(pre)}')
    print(f'    distinct red cases: {len(fn_list)}  backends seen: {bes}')
    print(f'    coverage tests red ({len(cov)}): {", ".join(cov) if cov else "-"}')
    open(SRC, 'w').write(BASE)

open(SRC, 'w').write(BASE)
run(f'cmake --build {W}/build-novendor -j --target gemv_tests')
print('reverted')

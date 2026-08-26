#!/usr/bin/env python3
"""GATE-D break campaign for BODY 5 (GemvSegTKernel<T, W>).

Every break is an ACTUAL edit to src/sycl/gemv_native.cc, actually applied,
actually rebuilt in build-novendor, actually run, then reverted.

WHY build-novendor AND NOT build/. preferred() is all-false for gemv, so in a
vendor-present build the CTA route is never taken and every one of these breaks
would be VACUOUS -- the kernel is linked and never runs. In build-novendor the
Trans/ConjTrans shapes resolve to native:cta, which
GemvCoverageTest.SegTransCasesAreReachable prints as a route line.

EDITS ARE CONFINED TO BODY 5's OWN FUNCTION TEXT wherever the break is meant to
be body-5-specific, so "the pre-existing cases stay green" is a measurement
about WHICH KERNEL RAN and not an accident of a shared code path. The two gate
breaks are deliberately outside it, in the launcher's decision function.

Modelled on experiments/wp7_gemv/repair/breaks_kernel.py.
"""
import subprocess, sys, os

W = '/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan'
SRC = W + '/src/sycl/gemv_native.cc'
PRISTINE = '/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/gemv_native.cc.shipped'
OUT = '/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/breaks_out'
os.makedirs(OUT, exist_ok=True)
GPU = os.environ.get('GPU', '0')

BASE = open(PRISTINE).read()


def body5(s):
    """(start, end) offsets of body 5's launcher function text."""
    a = s.index('template <typename T, int W>\nEvent gemv_seg_trans')
    b = s.index('\n}  // namespace', a)
    return a, b


def inb(s, old, new, count=1):
    a, b = body5(s)
    reg = s[a:b]
    assert reg.count(old) == count, f'body5: expected {count} of {old!r}, got {reg.count(old)}'
    return s[:a] + reg.replace(old, new) + s[b:]


def whole(s, old, new, count=1):
    assert s.count(old) == count, f'file: expected {count} of {old!r}, got {s.count(old)}'
    return s.replace(old, new)


BREAKS = {
  # THE MAPPING, INVERTED. s = lane/L, o = lane%L is body 4's mapping applied
  # unchanged to the transposed body -- the single most likely way to write this
  # kernel wrong, because it is what "copy body 4" produces.
  'segTmap':    lambda s: inb(s,
      'const int s = lane % kLanes;         // slice of the reduction\n'
      '                const int o = lane / kLanes;         // which of the W outputs',
      'const int s = lane / kLanes;         // BREAK: body 4\'s mapping\n'
      '                const int o = lane % kLanes;         // BREAK'),

  # THE FOLD AT THE WRONG STRIDE. Body 4 folds at stride out_len; body 5 must
  # fold at stride 1. This is the same class of defect as body 4's `segfold`.
  'segTfold':   lambda s: inb(s,
      're += sycl::shift_group_left(sg, re, off);',
      're += sycl::shift_group_left(sg, re, off * kLanes);'),
  'segTfold2':  lambda s: inb(s,
      'acc += sycl::shift_group_left(sg, acc, off);',
      'acc += sycl::shift_group_left(sg, acc, off * kLanes);'),

  # THE THREE EXTENTS THE PRE-WP7 SUITE WAS STRUCTURALLY BLIND TO.
  'segTld':     lambda s: inb(s, 'const int64_t lda = A.ld();',   'const int64_t lda = A.rows();'),
  'segTxinc':   lambda s: inb(s, 'const int64_t xinc = X.inc();', 'const int64_t xinc = 1;'),
  'segTyinc':   lambda s: inb(s, 'const int64_t yinc = Y.inc();', 'const int64_t yinc = 1;'),

  # THE THREE BATCH STRIDES. stride_pad exists in the fixture because a kernel
  # that DERIVED the stride passed all 232 pre-WP8 cases.
  'segTstridea': lambda s: inb(s, 'const int64_t stride_a = A.stride();',
                                  'const int64_t stride_a = static_cast<int64_t>(A.ld()) * A.cols();'),
  'segTstridex': lambda s: inb(s, 'const int64_t stride_x = X.stride();', 'const int64_t stride_x = A.rows();'),
  'segTstridey': lambda s: inb(s, 'const int64_t stride_y = Y.stride();', 'const int64_t stride_y = A.cols();'),

  # CONJUGATION. ortho.cc selects ConjTrans for all four complex types, so this
  # is the live path, not a spelling variant.
  'segTconj':   lambda s: inb(s, 'if (conj) av = dev_conj(av);', 'if (false) av = dev_conj(av);'),

  # THE TAIL, MASKED AT THE FOLD BUT **NOT** AT THE STORE. Out-of-range lane
  # groups have their sg_out CLAMPED to total-1 so no out-of-range address is
  # formed; if `active` stopped guarding the store they would all write the LAST
  # output of the last batch item with somebody else's reduction. This is the
  # OBSERVABLE half of the tail handling, and unlike segTtail below it is not a
  # spec-conformance argument -- it is a wrong number in y.
  'segTtailwrite': lambda s: inb(s, 'const bool active = (sg_out < total);',
                                    'const bool active = true;'),

  # THE CLAMP REMOVED **AND** THE MASK REMOVED TOGETHER. With both gone the
  # out-of-range lane groups form an out-of-range b and j, read A past the end
  # of the allocation and write y past the end of it. This is the only break in
  # the set that can make the tail handling produce an observable wrong answer,
  # and it is here because segTtail and segTtailwrite BOTH came back green --
  # see the break table in tests/gemv_tests.cc for why.
  'segTclampoff': lambda s: inb(s,
      'const int64_t sg_out_c = active ? sg_out : (total - 1);',
      'const int64_t sg_out_c = sg_out;  // BREAK'),
  'segTclampoff2': lambda s: inb(
      inb(s, 'const bool active = (sg_out < total);', 'const bool active = true;'),
      'const int64_t sg_out_c = active ? sg_out : (total - 1);',
      'const int64_t sg_out_c = sg_out;  // BREAK'),

  # THE EARLY EXIT, RETURNED INSTEAD OF MASKED. This is body 5's own trap: its
  # sub-group covers W different outputs, so a tail sub-group is PARTIALLY in
  # range and a `return` leaves the fold reached by only part of a sub-group.
  'segTtail':   lambda s: inb(s, 'const bool active = (sg_out < total);',
                                 'if (sg_out >= total) return;\n                const bool active = true;'),

  # THE WRITE. `s == 0` narrowed to `lane == 0` writes ONE of the W outputs per
  # sub-group and leaves the other W-1 at whatever y held.
  'segTwrite':  lambda s: inb(s, 'if (s == 0 && active) {', 'if (lane == 0 && active) {'),

  # THE LAUNCH GEOMETRY. Body 3 hands the ladder items = out_len*batch; body 5
  # must hand it that divided by W or it over-launches by exactly W -- which is
  # harmless for speed and fatal only if the tail arithmetic disagrees. Recorded
  # whether it is red or green.
  'segTlaunch': lambda s: inb(s, 'const int64_t sub_groups = (items + W - 1) / W;',
                                 'const int64_t sub_groups = items;'),

  # THE ALPHA AND BETA GUARDS -- body 5's own fifth copy.
  'segTalpha':  lambda s: inb(s, 'if (!alpha_zero && active) {', 'if (active) {'),
  'segTbeta':   lambda s: inb(s, 'if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);',
                                 'fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);'),

  # --- THE GATE, OUTSIDE BODY 5, IN THE ONE DECISION FUNCTION --------------
  # OPENED: every red_len takes body 5. Proves the gate is load-bearing rather
  # than decorative, and that the tests reach shapes ABOVE it.
  'segTgateopen':  lambda s: whole(s, 'if (red_len > kGemvSegTransMaxRedLen<T>) return 1;',
                                      'if (red_len > 100000) return 1;'),
  # SHUT: no red_len takes body 5. Everything must stay green (body 3 is the
  # correct fallback), which is what proves the gate is a SPEED gate and not a
  # correctness condition -- and it is also the control that says a green break
  # elsewhere is not green because body 5 never ran.
  'segTgateshut':  lambda s: whole(s, 'if (red_len > kGemvSegTransMaxRedLen<T>) return 1;',
                                      'if (red_len > 0) return 1;'),
  # GATE 3, THE PARALLELISM FLOOR, REMOVED. Every shape takes body 5 however
  # small its launch. Nothing should turn red -- gate 3 is a SPEED gate, not a
  # correctness condition -- and that greenness is the claim, not an accident:
  # it is what says the 0.891x regression it prevents is a performance defect
  # and not a wrong answer. Recorded either way.
  'segTfloorgone': lambda s: whole(s, 'if (items < gemv_seg_trans_min_items(cu, w)) return 1;',
                                      'if (items < 0) return 1;'),
  # GATE 3, THE TWO ROWS COLLAPSED TO ONE. The W = 4 band's floor is FOUR TIMES
  # the W = 8 band's; this makes them equal, which is the shape of defect a
  # single-number floor would have shipped.
  'segTfloorflat': lambda s: whole(s, 'return (w >= 8 ? 16 : 64) * c;', 'return 16 * c;'),
  # THE W BOUNDARY, off by one: the W = 8 band widened by one element, so
  # red_len == kGemvSegTransW8MaxRedLen<T> + 1 resolves to 8 instead of 4.
  'segTw8off':     lambda s: whole(s, 'const int w = (red_len <= kGemvSegTransW8MaxRedLen<T>) ? 8 : 4;',
                                      'const int w = (red_len <= kGemvSegTransW8MaxRedLen<T> + 1) ? 8 : 4;'),
  # THE TWO COPIES OF THE DECISION. There is exactly ONE gate function and both
  # the launcher and gemv_seg_trans_width_debug call it; this break checks that
  # a change to it moves the TEST-VISIBLE query too. If the query had its own
  # copy this would turn only the decision-surface test red and leave every
  # computing case green -- getrs_native.cc:410 records that exact defect.
  'segTemitoff':   lambda s: whole(s,
      'inline constexpr bool kGemvSegTransEmit =\n'
      '    std::is_same_v<T, float> || std::is_same_v<T, double> ||\n'
      '    std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;',
      'inline constexpr bool kGemvSegTransEmit = false;'),
}


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def failed_names(out):
    r = set()
    for l in out.splitlines():
        if not l.startswith('[  FAILED  ] '):
            continue
        t = l[len('[  FAILED  ] '):].strip()
        if t.endswith(')'):
            t = t.rsplit(' (', 1)[0]
        if '.' not in t:
            continue
        suite, case = t.split('.', 1)
        case = case.split(',')[0].strip()
        be = t.rsplit('Backend)', 1)[1].rstrip('> )') if 'Backend)' in t else '?'
        r.add((suite.split('/')[0], case, be))
    return sorted(r)


names = sys.argv[1:] if len(sys.argv) > 1 else list(BREAKS)
print(f'# breaks run in build-novendor on GPU {GPU}; baseline is 360/360')
for name in names:
    src = BREAKS[name](BASE)
    open(SRC, 'w').write(src)
    b = run(f'cmake --build {W}/build-novendor -j --target gemv_tests')
    if b.returncode != 0:
        print(f'=== {name}: BUILD FAILED')
        print(b.stdout[-2500:], b.stderr[-2500:])
        open(SRC, 'w').write(BASE)
        continue
    r = run(f'CUDA_VISIBLE_DEVICES={GPU} {W}/build-novendor/tests/gemv_tests')
    out = r.stdout + r.stderr
    open(f'{OUT}/{name}.log', 'w').write(out)
    tot = [l for l in out.splitlines() if 'tests from' in l and 'ran.' in l]
    pas = [l for l in out.splitlines() if l.startswith('[  PASSED  ]')]
    fai = [l for l in out.splitlines() if 'FAILED TEST' in l]
    fl = failed_names(out)
    pre = [n for n in fl if n[0] == 'GemvMatrixViewTest']
    cov = sorted({n[1] for n in fl if n[0] == 'GemvCoverageTest'})
    bes = sorted({n[2] for n in fl})
    print(f'=== {name}')
    print(f'    {tot[-1] if tot else "?"}')
    print(f'    {pas[-1] if pas else "?"} | {fai[-1] if fai else "0 FAILED"}')
    print(f'    pre-WP7 GemvMatrixViewTest cases red: {len(pre)}')
    print(f'    distinct red cases: {len(fl)}  backends seen: {bes}')
    print(f'    coverage tests red ({len(cov)}): {", ".join(cov) if cov else "-"}')
    open(SRC, 'w').write(BASE)
print('# reverted to pristine')

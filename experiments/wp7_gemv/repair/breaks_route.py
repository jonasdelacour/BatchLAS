#!/usr/bin/env python3
"""Arming proof for the new gemv section of route_vocabulary_tests.cc.

The V1 trap: a helper that quietly describes a device on which the tier under
test cannot run makes every assertion hold vacuously, and the suite reports green
through a flip and its inverse. Each break below is applied, rebuilt, run and
reverted, and each must turn a NAMED case red.
"""
import subprocess, sys, os

W = '/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan'
TST = W + '/tests/route_vocabulary_tests.cc'
TBL = W + '/include/batchlas/blas/dispatch/route_gemv.hh'
OUT = '/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/repair/breaks_out'
os.makedirs(OUT, exist_ok=True)
BASE_T = open(TST).read()
BASE_H = open(TBL).read()

def sub(s, old, new, n=1):
    assert s.count(old) == n, f'expected {n} of {old!r}, got {s.count(old)}'
    return s.replace(old, new)

BREAKS = {
  # THE V1 TRAP ITSELF: the helper stops arming the capability flags, so
  # supports() is false on every shape and the whole section holds vacuously.
  'unarmed': (TST, lambda s: sub(s,
      '    s.direct_available = direct_available;\n    s.cta_available = cta_available;',
      '    (void)direct_available; (void)cta_available;')),
  # The helper stops arming has_sg32.
  'nosg32':  (TST, lambda s: sub(s, '    s.has_sg32 = has_sg32;',
                                     '    (void)has_sg32;')),
  # THE WP7 DELIVERABLE, FLIPPED: a GPU gate on the Direct arm.
  'gpugate': (TBL, lambda s: sub(s,
      '                return s.direct_available;',
      '                if (!s.is_gpu) return false;\n                return s.direct_available;')),
  # kGemvOrder reordered so Direct comes first: CTA would never be reached
  # vendor-free, and the transposed GPU shape would take the uncoalesced body.
  'order':   (TBL, lambda s: sub(s,
      '    {Origin::Native, Algorithm::CTA},\n    {Origin::Native, Algorithm::Direct},',
      '    {Origin::Native, Algorithm::Direct},\n    {Origin::Native, Algorithm::CTA},')),
  # A preferred() clause lands, of exactly the shape the audit refuted.
  'clause':  (TBL, lambda s: sub(s,
      '        static_cast<void>(r);\n        static_cast<void>(s);\n        return false;',
      '        static_cast<void>(r);\n'
      '        return is_native(r) && s.transA != Transpose::NoTrans &&\n'
      '               s.red_len() >= 64 && s.red_len() <= 320;')),
  # CTA stops requiring an enumerated sub-group 32.
  'sg32gate':(TBL, lambda s: sub(s,
      '                return s.cta_available && s.is_gpu && s.has_sg32 &&',
      '                return s.cta_available && s.is_gpu &&')),
  # The zero-extent quick-return contract turned into a supports() refusal.
  'zeroext': (TBL, lambda s: sub(s,
      '        if (s.m < 0 || s.n < 0 || s.batch < 1) return false;',
      '        if (s.m <= 0 || s.n <= 0 || s.batch < 1) return false;')),
}

def run(c): return subprocess.run(c, shell=True, capture_output=True, text=True)

names = sys.argv[1:] or list(BREAKS)
for name in names:
    path, fn = BREAKS[name]
    base = BASE_T if path == TST else BASE_H
    open(path, 'w').write(fn(base))
    b = run(f'cmake --build {W}/build -j --target route_vocabulary_tests')
    if b.returncode != 0:
        print(f'=== {name}: BUILD FAILED'); print(b.stdout[-2000:]);
        open(path, 'w').write(base); continue
    r = run(f'{W}/build/tests/route_vocabulary_tests')
    out = r.stdout + r.stderr
    red = sorted({l[len('[  FAILED  ] '):].split(' (')[0].strip()
                  for l in out.splitlines() if l.startswith('[  FAILED  ] ') and '.' in l})
    tot = [l for l in out.splitlines() if 'tests from' in l and 'ran.' in l]
    fai = [l for l in out.splitlines() if 'FAILED TEST' in l]
    open(f'{OUT}/route_{name}.log', 'w').write(out)
    print(f'=== {name}')
    print(f'    {tot[-1] if tot else "?"}   {fai[-1] if fai else "0 FAILED"}')
    print(f'    red: {", ".join(red) if red else "-- NOTHING (VACUOUS) --"}')
    open(path, 'w').write(base)

open(TST, 'w').write(BASE_T); open(TBL, 'w').write(BASE_H)
run(f'cmake --build {W}/build -j --target route_vocabulary_tests')
print('reverted')

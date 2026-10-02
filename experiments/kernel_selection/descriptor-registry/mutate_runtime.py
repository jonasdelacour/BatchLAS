#!/usr/bin/env python3
"""Deliberate breaks of the descriptor-registry path, run on the GPU. Each must turn a NARROW,
named set of tests red. A case edits one source in place, rebuilds potrf_select_tests (which
relinks the library: the registry is the CUDA facade now), runs the suite on
CUDA_VISIBLE_DEVICES=1, and restores the file from a copy taken before the edit. The restore is
md5-checked against that copy and the whole set is md5-checked again at the end.

Usage: python3 mutate_runtime.py <build-script> [case ...]
  <build-script> builds the target given as its argument (env set up for the CUDA build) and
  leaves the binary at <build-dir>/tests/potrf_select_tests; pass the build dir in BUILD_DIR.
"""
import collections
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '../../..'))
ROUTES = os.path.join(ROOT, 'src/backends/potrf_routes.hh')
WINDOWS = os.path.join(ROOT, 'src/backends/potrf_windows.hh')
IMPL = os.path.join(ROOT, 'src/dispatch/potrf_select.cc')
SELECT = os.path.join(ROOT, 'src/dispatch/selection/select.hh')
GLUE = os.path.join(ROOT, 'src/dispatch/selection/run.hh')
TOUCHED = [ROUTES, WINDOWS, IMPL, SELECT, GLUE]

CTAWG_DECL = ('        g.fits = g.slm_total <= resident::device_slm_budget(s.dev.local_mem_bytes) &&\n'
              '                 g.wg_size <= s.dev.max_wg_size;')
CASES = {
    # the four of the first report
    'no-extension-tier': (ROUTES, [(', CtaWg<T>, Cusolver<B, T>>', ', Cusolver<B, T>>'),
                                   (', Blocked<B, T>, CtaWg<T>>>;', ', Blocked<B, T>>>;')]),
    'margin-zero': (IMPL, [('b->book[pi] = {P->model_enabled, P->margin, b->rows[pi]};',
                            'b->book[pi] = {P->model_enabled, 0.0, b->rows[pi]};')]),
    'window-off-by-one': (WINDOWS, [('std::is_same_v<T, float> ? 35 : 32;', 'std::is_same_v<T, float> ? 36 : 32;')]),
    'lpanel-not-lower-only': (ROUTES, [('using Geometry = pp::LpanelGeometry;\n    static Verdict legal(const PotrfShape& s) { return lower_only(s, native_common(s)); }',
                                        'using Geometry = pp::LpanelGeometry;\n    static Verdict legal(const PotrfShape& s) { return native_common(s); }')]),
    # the breaker's
    'ctawg-no-capacity-gate': (ROUTES, [(CTAWG_DECL, '        g.fits = true;')]),
    'blocked-no-w-clamp': (ROUTES, [('        g.p.W = std::max(1, std::min(g.p.W, s.n - g.p.nb));\n', '')]),
    'cta-unpacked-geometry': (ROUTES, [(
        'static Geometry plan(const PotrfShape& s) { return pp::cta_geometry<T>(s.n, s.batch, s.dev); }',
        'static Geometry plan(const PotrfShape& s) { Geometry g = pp::cta_geometry<T>(s.n, s.batch, s.dev); '
        'if (g.fits && g.G > 1) { g.G = 1; g.num_wg = s.batch; g.wg_size = g.L; '
        'g.slm_total = potrf_native::potrf_hole_padded(g.slm_per_matrix); } return g; }')]),
    'extrapolated-inverted': (SELECT, [('d.extrapolated = static_cast<std::int8_t>(outside(',
                                        'd.extrapolated = static_cast<std::int8_t>(!outside(')]),
    'no-heterogeneous-check': (ROUTES, [('    if (!s.homogeneous) return Verdict::no("heterogeneous batch");\n', '')]),
    'unpriced-vendor-rows-dropped': (IMPL, [('                    if (!e.present) continue;',
                                             '                    if (!e.present || i == 4) continue;')]),
    'tiny-no-cap-check': (ROUTES, [('return (v.ok && s.n > pp::TinyCap<T>::kMaxN) ? Verdict::no("n > tiny cap") : v;',
                                    'return v;')]),
    # one per repair
    'model-ignores-unpriced-vendor': (SELECT, [('            if (!row && is_vendor(r.route)) return std::nullopt;\n', '')]),
    'run-skips-shape-check': (GLUE, [('    if (!Table::OpT::matches(s.shape, a)) {', '    if (false) {')]),
    'run-skips-sized-check': (GLUE, [('    if (!s.sized) {', '    if (false) {')]),
}


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def summarize(out):
    failed = sorted(set(re.findall(r'\[  FAILED  \] (PotrfSelectV2\.\w+)', out)))
    lines = [l for l in out.splitlines() if l.startswith('[sweep]') or l.startswith('[grid]')]
    groups = collections.Counter()
    for l in out.splitlines():
        m = re.match(r'MISMATCH (\S+) (\S+) (\S) n=(\d+) b=\d+: (.*)', l)
        if m:
            groups[(m.group(1), m.group(2), m.group(3), re.sub(r'-?\d+', '#', m.group(5))[:70])] += 1
    lines += ['    %-14s %-7s %s %-70s x%d' % (k + (v,)) for k, v in sorted(groups.items())]
    ns = collections.defaultdict(set)
    for l in out.splitlines():
        m = re.match(r'MISMATCH (\S+) (\S+) (\S) n=(\d+)', l)
        if m:
            ns[(m.group(1), m.group(2), m.group(3))].add(int(m.group(4)))
    lines += ['    orders %s %s %s: %s' % (k + (sorted(v),)) for k, v in sorted(ns.items())]
    return failed, lines


def main():
    build = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    bin_ = os.path.join(os.environ.get('BUILD_DIR', os.path.join(ROOT, 'build')), 'tests/potrf_select_tests')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
    env['LD_LIBRARY_PATH'] = '/opt/dpcpp-cuda/lib:' + env.get('LD_LIBRARY_PATH', '')
    backup = tempfile.mkdtemp(prefix='dr_mutate_')
    pristine = {}
    for p in TOUCHED:
        shutil.copyfile(p, os.path.join(backup, os.path.basename(p)))
        pristine[p] = md5(p)
    try:
        for name in names:
            path, edits = CASES[name]
            src = os.path.join(backup, os.path.basename(path))
            assert md5(path) == pristine[path], 'dirty before ' + name
            text = open(path).read()
            for old, new in edits:
                assert text.count(old) == 1, (name, old[:60])
                text = text.replace(old, new)
            try:
                open(path, 'w').write(text)
                if os.path.exists(bin_):
                    os.remove(bin_)
                b = subprocess.run([build, 'potrf_select_tests'], capture_output=True, text=True)
                if not os.path.exists(bin_):
                    print('[%s] BUILD FAILED\n%s' % (name, (b.stdout + b.stderr)[-2000:]))
                    continue
                # T7/T12 only report timings; everything else runs.
                r = subprocess.run([bin_, '--gtest_filter=-*T7_*:*T12_*'], capture_output=True,
                                   text=True, env=env)
            finally:
                shutil.copyfile(src, path)
                assert md5(path) == pristine[path], 'restore failed: ' + path
            failed, lines = summarize(r.stdout + r.stderr)
            print('[%s] exit=%d failed: %s' % (name, r.returncode, ', '.join(failed) or 'NONE (stays green)'))
            for l in lines:
                print('   ' + l)
            sys.stdout.flush()
    finally:
        for p in TOUCHED:
            if md5(p) != pristine[p]:
                shutil.copyfile(os.path.join(backup, os.path.basename(p)), p)
        bad = [p for p in TOUCHED if md5(p) != pristine[p]]
        print('restore check: %s' % ('all %d sources md5-identical to the pre-run copies' % len(TOUCHED)
                                     if not bad else 'MISMATCH ' + ' '.join(bad)))
        # leave the binary built from the pristine sources
        subprocess.run([build, 'potrf_select_tests'], capture_output=True, text=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

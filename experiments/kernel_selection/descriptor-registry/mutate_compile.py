#!/usr/bin/env python3
"""Compile-time guards of the descriptor registry: each mutation forgets one piece of a tier
and must FAIL to compile (or leave an undefined symbol), with a narrow, named diagnostic.

Host-only: the selection path is SYCL-free, so g++ checks it in about a second per case.
Files are mutated in place and restored byte-for-byte (md5 checked).
Usage: [BUILD_DIR=<configured build dir>] python3 mutate_compile.py
"""
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '../../..'))
ROUTES = os.path.join(ROOT, 'src/backends/potrf_routes.hh')
WINDOWS = os.path.join(ROOT, 'src/backends/potrf_windows.hh')
IMPL = os.path.join(ROOT, 'src/dispatch/potrf_select.cc')
BUILD = os.environ.get('BUILD_DIR', os.path.join(ROOT, 'build'))
GXX = ['g++', '-std=c++20', '-Wall', '-Wextra', '-I' + ROOT + '/include', '-I' + BUILD + '/include',
       '-I' + ROOT + '/src', '-isystem', '/opt/include']

CASES = [
    # (name, file, old, new, mode, expected diagnostic regex)
    ('baseline', None, None, None, 'syntax', None),
    ('forget CtaWg::launch declaration', ROUTES,
     '    static Event launch(Queue&, const PotrfArgs<T>&, const Geometry&);\n};\n// ---- end EXTENSION',
     '};\n// ---- end EXTENSION', 'syntax', r'RouteDescriptor|constraints not satisfied|no member named .launch'),
    ('forget CtaWg::kernel (explain text)', ROUTES,
     '    static std::string kernel(const Geometry& g) { return Cta<T>::kernel(g); }\n', '', 'syntax',
     r'RouteDescriptor|constraints not satisfied'),
    ('copy-paste key: CtaWg reuses "native:cta"', ROUTES,
     'static constexpr std::string_view key = "native:cta_wg";',
     'static constexpr std::string_view key = "native:cta";', 'syntax', r'two descriptors share a key'),
    ('window names a route not in the table', WINDOWS,
     '{"native:blocked", Question::AmongNative,', '{"native:right_looking", Question::AmongNative,', 'syntax',
     r'a potrf window names no route in the table'),
    ('Geometry without .fits', ROUTES,
     'struct Geometry { bool fits = true; };', 'struct Geometry { bool ok = true; };', 'syntax',
     r'RouteDescriptor|constraints not satisfied'),
    ('an op without matches(): run() could not check the call', ROUTES,
     '    static bool matches(const Shape& s, const Args& a) {', '    static bool matches_(const Shape& s, const Args& a) {',
     'syntax', r'SelectableOp|constraints not satisfied'),
    ('forget CtaWg::launch DEFINITION (declared only)', IMPL,
     re.compile(r'template <class T>\nEvent CtaWg<T>::launch\(.*?\n}\n', re.S), '', 'object',
     r'CtaWg.*launch'),
]


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def compile_case(mode):
    src = IMPL
    if mode == 'syntax':
        r = subprocess.run(GXX + ['-fsyntax-only', src], capture_output=True, text=True)
        return r.returncode, r.stderr
    obj = '/tmp/ks_mutate_%d.o' % os.getpid()
    r = subprocess.run(GXX + ['-c', src, '-o', obj], capture_output=True, text=True)
    if r.returncode:
        return r.returncode, r.stderr
    nm = subprocess.run(['nm', '-C', '--undefined-only', obj], capture_output=True, text=True).stdout
    os.unlink(obj)
    undef = [l for l in nm.splitlines() if 'CtaWg' in l]
    return (1 if undef else 0), '\n'.join('undefined: ' + l.strip() for l in undef)


def main():
    ok = True
    for name, path, old, new, mode, want in CASES:
        before = md5(path) if path else None
        text = open(path).read() if path else None
        try:
            if path:
                mutated = old.sub(new, text, count=1) if hasattr(old, 'sub') else text.replace(old, new, 1)
                if mutated == text:
                    print('[%s] MUTATION DID NOT APPLY' % name)
                    ok = False
                    continue
                open(path, 'w').write(mutated)
            rc, err = compile_case(mode)
        finally:
            if path:
                open(path, 'w').write(text)
                assert md5(path) == before, 'restore failed for ' + path
        if want is None:
            verdict = 'PASS' if rc == 0 else 'FAIL (baseline must compile)'
            line = ''
        else:
            hit = re.search(want, err)
            verdict = 'PASS' if rc != 0 and hit else 'FAIL'
            first = [l for l in err.splitlines() if re.search(want, l)]
            line = (first[0] if first else err.splitlines()[0] if err else '')[:200]
        ok &= verdict == 'PASS'
        print('[%s] %s rc=%d %s' % (verdict, name, rc, line))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

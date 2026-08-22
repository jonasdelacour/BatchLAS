#!/usr/bin/env python3
"""Turn sweep_raw.txt into the tables the WP5 baseline README quotes.

Section labels come from the '## X ...' lines sweep.sh emits:
  A geqrf (vendor)   B qcheck = cuSOLVER orgqr AND routed ormqr-on-identity
  C synthI (vendor)  D synthI (vendor-free)

NOTE ON THE ROW LABEL: the ormqr-on-identity row inside section B is printed
with the MODE name, so it reads 'qcheck', not 'ormqrI'. Same code path.
"""
import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ('float', 'double', 'cfloat', 'cdouble')
NS = (64, 128, 256, 512, 1024, 2048)


def load(path):
    sec, data = None, {}
    for l in open(path):
        l = l.strip()
        if not l or l.startswith('# '):
            continue
        if l.startswith('##'):
            sec = l.split()[1]
            continue
        p = l.split(',')
        if len(p) < 8 or p[4] == 'THREW':
            continue
        op, t, n, b = p[0], p[1], int(p[2]), int(p[3])
        data[(sec, op, t, n)] = dict(batch=b, ms=float(p[4]), sd=float(p[6]),
                                     gf=float(p[7]), rest=p[8:])
    return data


def main():
    d = load(os.path.join(HERE, 'sweep_raw.txt'))

    print("== A. VENDOR BASELINE: cuSOLVER/cuBLAS geqrf through the public API ==")
    print(f"{'type':>8} {'n':>5} {'batch':>6} {'med_ms':>10} {'GFLOP/s':>9} {'rel_sd':>7} {'residual':>10}")
    for t in TYPES:
        for n in NS:
            r = d.get(('A', 'geqrf', t, n))
            if r:
                print(f"{t:>8} {n:>5} {r['batch']:>6} {r['ms']:>10.3f} {r['gf']:>9.1f} "
                      f"{r['sd']:>7.4f} {r['rest'][0]:>10}")

    print()
    print("== B. orgqr: cuSOLVER (batch LOOP) vs routed ormqr-on-identity, one process ==")
    print(f"{'type':>8} {'n':>5} {'batch':>6} {'orgqr_ms':>10} {'ormqrI_ms':>10} "
          f"{'ratio':>7} {'orgqr_ws_MB':>12} {'ormqr_ws_MB':>12} {'dQ':>10}")
    for t in TYPES:
        for n in NS:
            a = d.get(('B', 'orgqr', t, n))
            b = d.get(('B', 'qcheck', t, n))
            if a and b:
                dq = [x for x in b['rest'] if x.startswith('dQ=')]
                print(f"{t:>8} {n:>5} {a['batch']:>6} {a['ms']:>10.3f} {b['ms']:>10.3f} "
                      f"{a['ms']/b['ms']:>7.2f} {int(a['rest'][3])/2**20:>12.1f} "
                      f"{int(b['rest'][3])/2**20:>12.1f} {dq[0][3:] if dq else '-':>10}")

    print()
    print("== C vs D. ormqr-on-identity: vendor build vs VENDOR-FREE build, route held at Native:Blocked ==")
    print(f"{'type':>8} {'n':>5} {'batch':>6} {'vendor_ms':>10} {'free_ms':>10} {'free/vendor':>12} {'nb':>4}")
    for t in TYPES:
        for n in NS:
            c = d.get(('C', 'synthI', t, n))
            e = d.get(('D', 'synthI', t, n))
            if c and e:
                nb = [x for x in c['rest'] if x.startswith('nb=')]
                print(f"{t:>8} {n:>5} {c['batch']:>6} {c['ms']:>10.3f} {e['ms']:>10.3f} "
                      f"{e['ms']/c['ms']:>12.2f} {nb[0][3:] if nb else '-':>4}")

    print()
    print("== control: synthI(C) vs ormqrI(B), vendor build -- do synthetic reflectors measure the same work? ==")
    print(f"{'type':>8} {'n':>5} {'ormqrI_ms':>10} {'synthI_ms':>10} {'delta%':>8}")
    for t in TYPES:
        for n in NS:
            b = d.get(('B', 'qcheck', t, n))
            c = d.get(('C', 'synthI', t, n))
            if b and c:
                print(f"{t:>8} {n:>5} {b['ms']:>10.3f} {c['ms']:>10.3f} "
                      f"{100*(c['ms']-b['ms'])/b['ms']:>8.2f}")


if __name__ == '__main__':
    main()

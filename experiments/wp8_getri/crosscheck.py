#!/usr/bin/env python3
"""Cross-check WP8's clean pass against the sweeps it inherited.

For getrf and getrs the two independent passes backing the shipped clause are
NOT this pass's lu_p1 and lu_c1 -- lu_p1 was contaminated by a concurrent sweep
on the other card and is discarded. They are:
    getrf : WP8-I1's two passes (device 0, idle box)  vs  WP8 lu_c1 (device 1)
    getrs : WP8-I2's two-pass clause_summary          vs  WP8 lu_c1 (device 1)
Different card, different session, different binary build. This prints the
per-cell agreement so "reproduced across two passes" is a number.
"""
import sys, os
W = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/"

def read_i1():
    def rd(p):
        d = {}
        for i, l in enumerate(open(W + p)):
            if i == 0: continue
            f = l.strip().split(',')
            if f[-2] != 'ok' or float(f[7]) > 0.10: continue
            d[(f[1], int(f[2]), int(f[4]))] = float(f[5])
        return d
    b = 'experiments/wp8_getrf/'
    nv1, nv2 = rd(b + 'after_nv_p1.csv'), rd(b + 'after_nv_p2.csv')
    v1, v2 = rd(b + 'base_v_p1.csv'), rd(b + 'base_v_p2.csv')
    return {k: min(v1[k] / nv1[k], v2[k] / nv2[k])
            for k in set(nv1) & set(nv2) & set(v1) & set(v2)}

def read_i2():
    out = {}
    for line in open(W + 'experiments/wp8_getrs/clause_summary.txt'):
        if line.startswith('#') or line.startswith('type,'): continue
        p = line.strip().split(',')
        if len(p) < 13: continue
        out[(p[0], int(p[1]), int(p[2]), int(p[3]))] = float(p[10])
    return out

def read_c1(op):
    raw = {}
    for i, l in enumerate(open(W + 'experiments/wp8_getri/lu_c1.csv')):
        if i == 0: continue
        f = l.strip().split(',')
        if f[1] != op or f[-2] != 'ok' or f[-1] != '0': continue
        if f[6] == 'TIMEOUT_OR_THROW' or float(f[8]) > 0.10: continue
        raw[(f[0], f[2], int(f[3]), int(f[4]), int(f[5]))] = float(f[6])
    out = {}
    for (arm, t, n, q, b), v in raw.items():
        if arm != 'nv': continue
        vv = raw.get(('v', t, n, q, b))
        if vv: out[(t, n, q, b)] = vv / v
    return out

def show(name, a, b):
    common = sorted(set(a) & set(b))
    print(f"=== {name}: {len(common)} cells in common ===")
    sp = []
    for k in common:
        r = max(a[k], b[k]) / min(a[k], b[k]); sp.append(r)
        print(f"  {str(k):34s} inherited {a[k]:8.4f}   WP8-clean {b[k]:8.4f}   spread {r:.4f}")
    if sp:
        sp.sort()
        print(f"  median spread {sp[len(sp)//2]:.4f}  worst {sp[-1]:.4f}  "
              f"above 1.10: {sum(1 for s in sp if s > 1.10)}")

c1f = {(t, n, b): v for (t, n, q, b), v in read_c1('getrf').items()}
show("getrf  I1 (device 0, two passes)  vs  WP8 lu_c1 (device 1, alone)",
     read_i1(), c1f)
show("getrs  I2 (two passes each side)  vs  WP8 lu_c1 (device 1, alone)",
     read_i2(), read_c1('getrs'))

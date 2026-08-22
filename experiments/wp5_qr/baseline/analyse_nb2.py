#!/usr/bin/env python3
"""nb2.csv: wide nb in the vendor-free build, and the whole ladder in the vendor
build as the control that separates 'float-only tuning' from 'vendor-present
tuning'. Merged with the nb.csv ladder for a single table per build."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ('float', 'double', 'cfloat', 'cdouble')
SHIPPED = {256: 24, 1024: 56}

data = {}   # (build, type, n, nb) -> ms


def ingest(path):
    part = None
    for l in open(path):
        l = l.strip()
        if l.startswith('== PART 1'):
            part = 'skip'; continue
        if l.startswith('== PART 2') or l.startswith('== (a)'):
            part = 'nv'; continue
        if l.startswith('== (b)'):
            part = 'v'; continue
        if not l or l.startswith('build,') or l.startswith('nb_forced,') or part in (None, 'skip'):
            continue
        p = l.split(',')
        if len(p) < 9 or p[5] == 'THREW':
            continue
        data[(part, p[2], int(p[3]), int(p[0]))] = float(p[5])


ingest(os.path.join(HERE, 'nb.csv'))
ingest(os.path.join(HERE, 'nb2.csv'))

NBS = (8, 16, 24, 32, 48, 56, 64, 96, 112, 128, 160, 192)
for bld, label in (('v', 'VENDOR build (what the shipped table was tuned in)'),
                   ('nv', 'VENDOR-FREE build (what WP5 must be fast in)')):
    for n, b in ((256, 512), (1024, 64)):
        print(f"\n{label} -- ormqr-on-identity, n={n} batch={b}, median ms "
              f"(* best, ] shipped width {SHIPPED[n]})")
        print("    nb  " + "".join(f"{t:>12}" for t in TYPES))
        best = {}
        for t in TYPES:
            c = [(data[(bld, t, n, nb)], nb) for nb in NBS if (bld, t, n, nb) in data]
            if c:
                best[t] = min(c)
        for nb in NBS:
            if not any((bld, t, n, nb) in data for t in TYPES):
                continue
            row = f"  {nb:>4}  "
            for t in TYPES:
                v = data.get((bld, t, n, nb))
                if v is None:
                    row += f"{'-':>12}"; continue
                mark = '*' if best.get(t, (0, -1))[1] == nb else (']' if nb == SHIPPED[n] else ' ')
                row += f"{v:>11.2f}{mark}"
            print(row)
        if best:
            print("    best nb: " + "  ".join(f"{t}={best[t][1]}" for t in TYPES if t in best))
            print("    shipped costs: " + "  ".join(
                f"{t}={data[(bld,t,n,SHIPPED[n])]/best[t][0]:.2f}x"
                for t in TYPES if t in best and (bld, t, n, SHIPPED[n]) in data))

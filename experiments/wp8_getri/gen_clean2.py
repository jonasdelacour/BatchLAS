#!/usr/bin/env python3
"""SECOND clean pass: the cells that DECIDE a boundary, and only those.

GATE-C asks for >= 1.15x reproduced across two passes with zero losing cells in
the admitted set. The first clean pass measured the whole ladder; this one
re-measures the rungs a boundary actually rests on -- the last winner inside and
the first non-winner outside, on every axis every clause names -- plus the
refutation cells for the two types each clause excludes.

Cells whose answer is 3x-9x and nowhere near a boundary are NOT repeated: a
second pass on float getri n=1024 (5.07x) buys a decimal place on a number that
decides nothing.
"""
C = []
def add(op, t, n, q, bs):
    for b in bs: C.append(f"{op}:{t}:{n}:{q}:{b}")

# ---- getri boundaries -------------------------------------------------------
add('getri', 'float',   64, 1, [8192, 16384])   # last order OUTSIDE, must lose
add('getri', 'float',  128, 1, [256, 16384])    # first order INSIDE, both ends
add('getri', 'cfloat', 128, 1, [512])           # last order OUTSIDE, must lose
add('getri', 'cfloat', 256, 1, [128, 4096])     # first order INSIDE
add('getri', 'double',  256, 1, [4096])         # refutation
add('getri', 'double',  512, 1, [128])
add('getri', 'double', 1024, 1, [512])
add('getri', 'double', 2048, 1, [128])
add('getri', 'cdouble', 512, 1, [1024])
add('getri', 'cdouble', 1024, 1, [128])

# ---- getrs boundaries -------------------------------------------------------
for t in ('float', 'double'):
    add('getrs', t,   64, 128, [128, 4096])
    add('getrs', t,  128, 128, [128, 4096])
    add('getrs', t,  256, 128, [128, 4096])
    add('getrs', t,  512, 128, [128, 2048])
    add('getrs', t, 1024, 128, [128, 1024])
    add('getrs', t,  512, 128, [64])            # just below the batch floor
    add('getrs', t, 1024, 128, [64])
add('getrs', 'float', 1024, 64, [512, 1024])    # the cell that refuted nrhs>=64
add('getrs', 'cfloat',  512, 128, [512])        # type refutation
add('getrs', 'cdouble', 128, 128, [1024])

# ---- getrf boundaries -------------------------------------------------------
add('getrf', 'float',   128, 1, [128, 512])     # last order OUTSIDE (native:cta)
add('getrf', 'float',   256, 1, [128, 1024])    # first order INSIDE
add('getrf', 'float',   512, 1, [128])
add('getrf', 'float',  1024, 1, [128])
add('getrf', 'float',  2048, 1, [128])
add('getrf', 'cfloat',  128, 1, [128])
add('getrf', 'cfloat',  256, 1, [128, 512, 1024])
add('getrf', 'cfloat',  512, 1, [128, 1024])
add('getrf', 'cfloat', 1024, 1, [128])
add('getrf', 'cfloat', 2048, 1, [128])
add('getrf', 'double',  512, 1, [128, 1024])    # refutation
add('getrf', 'cdouble',1024, 1, [128])

print("\n".join(C))

#!/usr/bin/env python3
"""Cell lists for the WP8 routing pass's LU sweeps.

Every cell list in one place (D4 part A1's rule). A cell is op:type:n:nrhs:batch.

THE MEMORY GUARD IS PART OF THE GENERATOR, not of the runner: a cell that OOMs
is a row the analysis must drop, and dropping it silently is how a ladder loses
the rung that decides a boundary. Cells over the cap are printed as comments so
they appear in the record as DECLINED-FOR-MEMORY rather than as absent.
"""
import sys

SZ = {'float': 4, 'double': 8, 'cfloat': 8, 'cdouble': 16}
CAP_GB = 13.0            # A + C + pivots + getri workspace, on a 24 GB card

def gb(op, t, n, nrhs, b):
    e = SZ[t]
    if op == 'getri':   return 2.6 * n * n * b * e / 2**30   # A, C, + slack
    if op == 'getrf':   return 1.3 * n * n * b * e / 2**30
    if op == 'getrs':   return (n * n + 2.5 * n * nrhs) * b * e / 2**30
    raise SystemExit(op)

def emit(cells):
    out, declined = [], []
    for c in cells:
        op, t, n, nrhs, b = c
        (out if gb(*c) <= CAP_GB else declined).append(
            (f"{op}:{t}:{n}:{nrhs}:{b}", gb(*c)))
    for s, g in out:      print(s)
    for s, g in declined: print(f"# DECLINED-FOR-MEMORY {s}  ({g:.1f} GB > {CAP_GB} GB)")

# ---------------------------------------------------------------- getri
# Ladders chosen so each (type, n) reaches the batch where the VENDOR stops
# improving -- D4's saturation criterion -- plus the rung above it where
# available, plus the boundary orders just outside every candidate clause.
GETRI = []
for n, bs in [(64,   [2048, 8192, 16384]),
              (128,  [256, 1024, 4096, 8192, 16384]),
              (256,  [128, 512, 2048, 4096]),
              (512,  [128, 512, 1024, 2048]),
              (1024, [128, 256, 512, 1024]),
              (2048, [64, 128, 256])]:
    for b in bs: GETRI.append(('getri', 'float', n, 1, b))
for n, bs in [(128,  [512, 2048, 8192]),
              (256,  [128, 512, 2048, 4096]),
              (512,  [128, 512, 1024]),
              (1024, [128, 256, 512]),
              (2048, [32, 64, 128])]:
    for b in bs: GETRI.append(('getri', 'cfloat', n, 1, b))
for n, bs in [(256,  [512, 2048, 4096]),
              (512,  [128, 512, 1024, 2048]),
              (1024, [64, 128, 256, 512]),
              (2048, [32, 64, 128])]:
    for b in bs: GETRI.append(('getri', 'double', n, 1, b))
for n, bs in [(512,  [128, 512, 1024]),
              (1024, [64, 128, 256]),
              (2048, [16, 32, 64])]:
    for b in bs: GETRI.append(('getri', 'cdouble', n, 1, b))
# THE BATCH FLOOR IS UNMEASURED EVERYWHERE IN THE CAMPAIGN (D4 R6). batch 1..32
# is where inverse_tests and getrf_tests:2118 actually live.
for t in ('float', 'cfloat', 'double', 'cdouble'):
    for n in (128, 512):
        for b in (1, 2, 4, 32):
            GETRI.append(('getri', t, n, 1, b))

# ---------------------------------------------------------------- getrs
# I2's clause is float nrhs>=64 / double nrhs>=128. Its 45 measured cells sit on
# three SATURATED rungs per order; what is missing is (a) the rung above the
# falling double n=1024 ladder and (b) the whole low-batch end, which the clause
# admits and nothing has ever measured.
GETRS = []
for t, n, nrhs, bs in [
        ('double', 1024, 128, [1024]),          # I2's named marginal cell
        ('float',  1024, 128, [1024]),
        ('float',  1024,  64, [1024]),
        ('float',   512,  64, [1, 2, 4, 32, 128]),
        ('float',   512, 128, [1, 2, 4, 32, 128]),
        ('double',  512, 128, [1, 2, 4, 32, 128]),
        ('float',    64,  64, [1, 32, 128, 512]),
        ('float',    64, 128, [1, 32, 128, 512]),
        ('double',   64, 128, [1, 32, 128, 512]),
        ('float',   128,  64, [1, 2, 4, 32, 128]),
        ('double',  128, 128, [1, 2, 4, 32, 128])]:
    for b in bs: GETRS.append(('getrs', t, n, nrhs, b))

# ---------------------------------------------------------------- getrf
# I1's clause is float n>=256 / cfloat n>=512, from its own two passes. These are
# MY two passes of the same cells on a device with no display attached, plus the
# boundary orders and the unmeasured low-batch end.
GETRF = []
for t, n, bs in [('float',   128, [128, 512]),      # native:cta, must NOT be admitted
                 ('float',   256, [128, 512, 1024]),
                 ('float',   512, [128, 512, 1024]),
                 ('float',  1024, [128, 512]),
                 ('float',  2048, [128]),
                 ('cfloat',  256, [128, 512]),      # boundary below, must lose
                 ('cfloat',  512, [128, 512, 1024]),
                 ('cfloat', 1024, [128, 512]),
                 ('cfloat', 2048, [128])]:
    for b in bs: GETRF.append(('getrf', t, n, 1, b))
for t in ('float', 'cfloat'):
    for b in (1, 2, 4, 32):
        GETRF.append(('getrf', t, 512, 1, b))

WHICH = {'getri': GETRI, 'getrs': GETRS, 'getrf': GETRF,
         'all': GETRI + GETRS + GETRF}
emit(WHICH[sys.argv[1] if len(sys.argv) > 1 else 'all'])

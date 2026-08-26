#!/usr/bin/env python3
"""G6 cell lists, defined in (out_len, red_len, batch) -- never (m, n).

Four sub-grids, each walking ONE axis while the others are held, because the
claim under test is a PREDICATE and a predicate is refuted by a cell, not by a
geomean. Trap 8 in this campaign's list is 'a grid that cannot reach a regime is
not evidence about it': the grid that produced 'no predicate exists' sampled
batch at {128, 256, 512} and never walked red_len below 32.

  A  THE BATCH AXIS -- the hole D3 named. Ten batch levels, not three.
  B  THE red_len AXIS at one batch, walked down to 8.
  C  THE out_len AXIS, walked from 32 to 2048.
  D  THE TYPE CONTROL. Body 5 (WP8-I3) now serves short reductions for ALL FOUR
     scalar types, so 'the prize is complex<double>-only' is a claim that has to
     be re-tested rather than inherited.
"""
import sys

SZ = 16   # complex<double>
CAP_GB = 11.0

def emit(cells):
    seen = set()
    for ty, out, red, b, tr in cells:
        e = {'float': 4, 'double': 8, 'cfloat': 8, 'cdouble': 16}[ty]
        gb = out * red * b * e / 2**30
        key = (ty, out, red, b, tr)
        if key in seen: continue
        seen.add(key)
        if gb > CAP_GB:
            print(f"# DECLINED-FOR-MEMORY {ty}:{out}:{red}:{b}:{tr} ({gb:.1f} GB)")
        else:
            print(f"{ty}:{out}:{red}:{b}:{tr}")

A = [('cdouble', o, r, b, 'T')
     for o in (256, 512, 768, 1024)
     for r in (48, 64, 128, 256)
     for b in (128, 192, 256, 320, 384, 448, 512, 640, 768, 1024)]

B = [('cdouble', o, r, 512, 'T')
     for o in (256, 512)
     for r in (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256,
               320, 352, 384, 448, 512)]

C = [('cdouble', o, r, b, 'T')
     for o in (32, 64, 96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048)
     for r in (64, 128)
     for b in (512, 1024)]

D = [(t, o, r, b, 'T')
     for t in ('float', 'double', 'cfloat', 'cdouble')
     for o in (256, 512)
     for r in (48, 64, 128)
     for b in (512, 1024)]

WHICH = {'A': A, 'B': B, 'C': C, 'D': D, 'all': A + B + C + D}
emit(WHICH[sys.argv[1] if len(sys.argv) > 1 else 'all'])

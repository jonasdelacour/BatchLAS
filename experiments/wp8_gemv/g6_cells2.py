#!/usr/bin/env python3
"""G6, second grid: the three things the first grid still could not reach.

  E  THE LOW-BATCH END. The first grid starts at 128 because that is the
     campaign's saturation floor -- but a preferred() clause routes PRODUCTION
     traffic, and production traffic includes batch 1. A clause with a batch
     floor needs the floor bracketed from BELOW by measured non-winners; a clause
     without one needs the whole low end measured. Walk it to 1.
  F  ConjTrans. The clause will say `transA != NoTrans`, which is two spellings.
     ortho.cc selects ConjTrans for every complex type, so C is the LIVE path and
     measuring only T would be routing the live path on the strength of the other
     one.
  G  THE UPPER red_len EDGE. The band has to close somewhere above 256; grid B
     walks it at out_len 256 and 512 only, and the clause admits out_len >= 256.
"""
import sys

E = [('cdouble', o, r, b, 'T')
     for o in (256, 512)
     for r in (64, 128, 256)
     for b in (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 224, 288)]

F = [('cdouble', o, r, b, 'C')
     for o in (256, 512, 1024)
     for r in (48, 64, 96, 128, 192, 256, 320, 352, 384)
     for b in (192, 256, 320, 512, 1024)]

G = [('cdouble', o, r, 512, 'T')
     for o in (768, 1024, 2048)
     for r in (288, 320, 352, 384, 448, 512, 640, 768)]

WHICH = {'E': E, 'F': F, 'G': G, 'all': E + F + G}
cells = WHICH[sys.argv[1] if len(sys.argv) > 1 else 'all']
seen = set()
for ty, out, red, b, tr in cells:
    k = (ty, out, red, b, tr)
    if k in seen: continue
    seen.add(k)
    gb = out * red * b * 16 / 2**30
    if gb > 11.0: print(f"# DECLINED-FOR-MEMORY {ty}:{out}:{red}:{b}:{tr} ({gb:.1f} GB)")
    else:         print(f"{ty}:{out}:{red}:{b}:{tr}")

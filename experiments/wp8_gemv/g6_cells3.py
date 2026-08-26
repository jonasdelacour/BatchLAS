#!/usr/bin/env python3
"""G6, third grid: the ONE boundary the first two grids cannot bracket.

Grid A showed that at out_len >= 768 the vendor is dipped at EVERY batch it
sampled, the lowest being 128 -- so a second disjunct `out_len >= 768 and
batch >= 128` captures ~18 more measured wins at 2.26-2.91x. But a floor at 128
that has never been approached from below is not a bracketed boundary, it is the
edge of the sampled range wearing a boundary's clothes. That is precisely the
criticism WP7's own audit levelled at its `A >= 1024 MB` candidate.

So: walk batch DOWN to 1 at out_len 768 and 1024, and add out_len 640 -- the
rung between 512 (a measured loss at batch 128-256) and 768 (a measured win) --
so the out_len boundary of the second disjunct is bracketed too.
"""
import sys

H = [('cdouble', o, r, b, 'T')
     for o in (768, 1024, 2048)
     for r in (64, 128, 256, 352)
     for b in (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 224)]

I = [('cdouble', 640, r, b, 'T')
     for r in (64, 128, 256, 352)
     for b in (128, 192, 256, 320, 512)]

# The upper batch end of the shipped clause: it admits every batch above its
# floor, and nothing has ever been measured above 1024 in this family.
J = [('cdouble', o, r, b, 'T')
     for o in (256, 512)
     for r in (64, 128, 256)
     for b in (2048, 4096, 8192)]

WHICH = {'H': H, 'I': I, 'J': J, 'all': H + I + J}
cells = WHICH[sys.argv[1] if len(sys.argv) > 1 else 'all']
seen = set()
for ty, out, red, b, tr in cells:
    k = (ty, out, red, b, tr)
    if k in seen: continue
    seen.add(k)
    gb = out * red * b * 16 / 2**30
    if gb > 11.0: print(f"# DECLINED-FOR-MEMORY {ty}:{out}:{red}:{b}:{tr} ({gb:.1f} GB)")
    else:         print(f"{ty}:{out}:{red}:{b}:{tr}")

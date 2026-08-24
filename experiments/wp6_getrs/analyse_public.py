#!/usr/bin/env python3
# Ratio tables from public_ab.sh.
#
# IT VERIFIES EVERY PIN. lubench6 prints "getrf_route|getrs_route" in field 13 of
# the sed-prefixed row; route_resolve.hh:165 falls through to automatic() at :175
# when a forced route is unsupported, so a row whose printed getrs route does not
# equal the pin was NOT measuring what it says. Those rows are REPORTED AND
# DROPPED, never averaged in.
import sys, collections, math

rows = []
mismatch = []
noisy = []
for L in open(sys.argv[1]):
    f = L.strip().split(',')
    #  0 pin, 1 'getrs', 2 type, 3 n, 4 nrhs, 5 batch, 6 med, 7 mean, 8 relsd,
    #  9 gflops, 10 resid, 11 ws, 12 routes, 13 fres, 14 ntp, 15 flag
    if len(f) < 16 or f[1] != 'getrs':
        mismatch.append(('unparsed', L.strip()))
        continue
    pin = f[0]
    got = f[12].split('|')[1] if '|' in f[12] else '?'
    # `vendor` is spelled `vendor:auto` once resolved: Origin::Vendor carries
    # Algorithm::Auto and rstr() prints both halves. The two native pins are
    # printed verbatim.
    if pin == 'vendor' and got == 'vendor:auto':
        got = 'vendor'
    if got != pin:
        mismatch.append((pin, got, f[2], f[3], f[4], f[5]))
        continue
    med, relsd = float(f[6]), float(f[8])
    if relsd > 0.10:
        noisy.append((pin, f[2], f[3], f[4], f[5], relsd))
    rows.append((pin, f[2], int(f[3]), int(f[4]), int(f[5]), med, float(f[10]), f[15]))

cells = collections.defaultdict(dict)
for pin, t, n, r, b, med, res, flag in rows:
    cells[(t, n, r, b)][pin] = (med, res, flag)

arms = sorted({p for p, *_ in rows})
print("PINS PRESENT:", arms)
hdr = "type      n    nrhs batch |" + "".join(f" {a:>16s}" for a in arms)
if 'vendor' in arms:   hdr += " | cta/vendor"
if 'native:blocked' in arms: hdr += " cta/blocked"
print(hdr)
geo = collections.defaultdict(list)
for k in sorted(cells, key=lambda k: (k[0], k[2], k[3])):
    t, n, r, b = k
    c = cells[k]
    line = f"{t:9s} {n:4d} {r:4d} {b:6d} |"
    for a in arms:
        line += f" {c[a][0]:16.4f}" if a in c else f" {'--':>16s}"
    cta = c.get('native:cta')
    if cta:
        if 'vendor' in c:
            g = c['vendor'][0] / cta[0]; line += f" | {g:10.3f}"
            geo[('vendor', t)].append(g); geo[('vendor', 'ALL')].append(g)
            if r == 1: geo[('vendor', 'nrhs=1')].append(g)
        if 'native:blocked' in c:
            g = c['native:blocked'][0] / cta[0]; line += f" {g:11.3f}"
            geo[('blocked', t)].append(g); geo[('blocked', 'ALL')].append(g)
            if r == 1: geo[('blocked', 'nrhs=1')].append(g)
    bad = [a for a in c if c[a][2] != 'ok']
    if bad: line += "   BAD:" + ",".join(bad)
    print(line)

def gm(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else 0
print()
for k in sorted(geo):
    v = geo[k]
    print(f"geomean cta vs {k[0]:8s} {k[1]:9s} : {gm(v):6.3f}  "
          f"min {min(v):6.3f}  max {max(v):6.3f}  over {len(v)} cells, "
          f"{sum(1 for x in v if x < 1.0)} losses")
print()
print(f"ROWS WHOSE PIN DID NOT TAKE (dropped): {len(mismatch)}")
for m in mismatch: print("   ", m)
print(f"CELLS WITH RELATIVE SD > 10%: {len(noisy)}")
for m in noisy: print("   ", m)

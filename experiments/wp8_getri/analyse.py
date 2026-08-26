#!/usr/bin/env python3
"""Score the WP8 routing pass's LU sweeps.

READ BY POSITION, NEVER BY NAME. lubench6.cpp prints 16 fields for getri and
getrs and 17 for getrf under one header; a csv.DictReader silently misaligns the
flag column and every quality filter then passes vacuously. That trap has been
paid for twice in this campaign, so: type=1, n=3, nrhs=4, batch=5, med=6,
relsd=8, route=12 counting the arm column, and flag/foreign are the LAST TWO
fields whatever the row length.

THE DISCARD RULE is experiments/wp6_perf/bench/analyse.py's, verbatim: drop and
NAME a cell when any arm is BAD, any arm's relsd exceeds 0.10, an arm is missing,
a foreign compute process was seen, or the printed route does not match the arm.
"""
import sys, math, collections

def load(path):
    rows = {}
    for i, line in enumerate(open(path)):
        if i == 0: continue
        f = line.rstrip('\n').split(',')
        if len(f) < 10: continue
        arm, op, ty, n, nrhs, b = f[0], f[1], f[2], int(f[3]), int(f[4]), int(f[5])
        if f[6] == 'TIMEOUT_OR_THROW':
            rows[(arm, op, ty, n, nrhs, b)] = ('BAD-THROW', None, None); continue
        med, relsd, route, flag, foreign = float(f[6]), float(f[8]), f[12], f[-2], f[-1]
        bad = None
        if flag != 'ok':          bad = f'flag={flag}'
        elif relsd > 0.10:        bad = f'relsd={relsd:.4f}'
        elif foreign != '0':      bad = f'foreign={foreign}'
        else:
            half = route.split('|')[-1]
            want_native = (arm == 'nv')
            if want_native and not half.startswith('native'): bad = f'route={route}'
            if (not want_native) and not half.startswith('vendor'): bad = f'route={route}'
        rows[(arm, op, ty, n, nrhs, b)] = (bad, med, route)
    return rows

def pair(paths):
    """-> {(op,ty,n,nrhs,b): [ratio_per_pass...]}, plus the refusal log."""
    per_pass, refused = [], []
    for p in paths:
        r = load(p); out = {}
        cells = sorted({k[1:] for k in r})
        for c in cells:
            nv, v = r.get(('nv',) + c), r.get(('v',) + c)
            if nv is None or v is None:
                refused.append((p, c, 'missing arm')); continue
            if nv[0]: refused.append((p, c, 'nv ' + nv[0])); continue
            if v[0]:  refused.append((p, c, 'v ' + v[0]));  continue
            out[c] = (v[1] / nv[1], nv[1], v[1], nv[2])
        per_pass.append(out)
    return per_pass, refused

def geo(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs))

if __name__ == '__main__':
    paths = sys.argv[1:]
    per_pass, refused = pair(paths)
    common = set(per_pass[0])
    for p in per_pass[1:]: common &= set(p)
    print(f"# {len(paths)} passes, {len(common)} cells paired in ALL passes, "
          f"{len(refused)} refusals")
    for p, c, why in refused: print(f"# REFUSED {p} {c}: {why}")
    print("op,type,n,nrhs,batch,route," +
          ",".join(f"r_p{i+1}" for i in range(len(paths))) + ",QUOTED,spread")
    rowsout = []
    for c in sorted(common):
        rs = [per_pass[i][c][0] for i in range(len(paths))]
        q = min(rs); spread = max(rs) / min(rs)
        rowsout.append((c, rs, q, spread, per_pass[0][c][3]))
        print(f"{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{per_pass[0][c][3]}," +
              ",".join(f"{x:.4f}" for x in rs) + f",{q:.4f},{spread:.4f}")
    if rowsout:
        sp = sorted(s for _, _, _, s, _ in rowsout)
        print(f"# cross-pass median spread {sp[len(sp)//2]:.4f}, "
              f"worst {sp[-1]:.4f}, {sum(1 for s in sp if s > 1.10)} above 1.10")

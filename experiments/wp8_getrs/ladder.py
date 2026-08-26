#!/usr/bin/env python3
# THE LADDER, READ AT SATURATION -- and the saturation verdict printed beside
# every rung, because this grid turned out to straddle a regime boundary.
#
# WHY THIS IS NOT JUST A RATIO TABLE. At n = 128 float nrhs = 128 the composition
# costs 9.96 / 2.80 / 1.90 / 1.76 us per item at batch 32 / 128 / 256 / 512 and
# the vendor 38.2 / 10.2 / 5.59 / 3.31. Neither arm is measuring its own speed
# below batch ~256: both are measuring LAUNCH AND SETUP AMORTISATION, and the
# ratio there is an overhead ratio. The campaign's own rule -- compare algorithms
# ONLY at saturation -- makes every rung below that inadmissible as evidence
# about a window, which is exactly the error the recorded 33.9x-146x getri
# figures embody.
#
# SATURATION CRITERION (D4's, transcribed): an arm is SATURATED at batch b when
# its us/item improved by less than 2% over the previous doubling, or the curve
# turned up. A CELL is admissible when BOTH arms are saturated at it.
#
# There is a second regime boundary in this grid and it is NOT saturation: at
# float n=128 nrhs=128 batch=512 the two matrices are 33.5 MB each, so A and X
# together are ~67 MB against this device's 72 MB L2. Every cell below that is
# partly L2-resident and every cell above it streams. Footprint is printed so the
# boundary is visible rather than inferred.
import sys
import math

SZ = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}
TYPES = ["float", "double", "cfloat", "cdouble"]


def load(path, want_route):
    rows, refused = {}, []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 16 or p[0] != "getrs":
                refused.append((line.strip(), "malformed"))
                continue
            key = (p[1], int(p[2]), int(p[3]), int(p[4]))
            if p[14] != "ok":
                refused.append((key, f"flag {p[14]}"))
                continue
            if float(p[7]) > 0.10:
                refused.append((key, f"relsd {float(p[7]):.4f} > 0.10"))
                continue
            if int(p[15]) != 0:
                refused.append((key, f"{p[15]} foreign compute processes"))
                continue
            got = p[11].split("|")[-1]
            if got != want_route:
                refused.append((key, f"route {got} != {want_route}"))
                continue
            rows[key] = float(p[5])
    return rows, refused


def merge(paths, want):
    out, ref = {}, []
    for p in paths:
        r, x = load(p, want)
        out.update(r)
        ref += x
    return out, ref


def saturated(series, b):
    """series: {batch: us_per_item}. True when the previous doubling bought <2%."""
    prev = b // 2
    while prev >= 1 and prev not in series:
        prev //= 2
    if prev not in series or prev == b:
        return None            # no previous rung measured -> unknown
    gain = (series[prev] - series[b]) / series[prev]
    return gain < 0.02


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(nv_paths, v_paths):
    nv, r1 = merge(nv_paths, "native:blocked")
    vd, r2 = merge(v_paths, "vendor:auto")
    for k, why in r1 + r2:
        print(f"# REFUSED {k}: {why}")

    keys = sorted(set(nv) & set(vd))
    print(f"# {len(keys)} paired cells, {len(r1) + len(r2)} refused rows\n")

    print("type,n,nrhs,batch,footprintMB,nat_us_item,ven_us_item,ratio,nat_sat,ven_sat,admissible")
    admissible = {}
    for t in TYPES:
        for n in sorted({k[1] for k in keys if k[0] == t}):
            for nrhs in sorted({k[2] for k in keys if k[0] == t and k[1] == n}):
                bs = sorted(b for (tt, nn, rr, b) in keys
                            if (tt, nn, rr) == (t, n, nrhs))
                sn = {b: nv[(t, n, nrhs, b)] * 1000.0 / b for b in bs}
                sv = {b: vd[(t, n, nrhs, b)] * 1000.0 / b for b in bs}
                for b in bs:
                    fp = SZ[t] * (n * n + n * nrhs) * b / (1 << 20)
                    a, c = saturated(sn, b), saturated(sv, b)
                    ok = bool(a) and bool(c)
                    r = vd[(t, n, nrhs, b)] / nv[(t, n, nrhs, b)]
                    if ok:
                        admissible[(t, n, nrhs, b)] = r
                    print(f"{t},{n},{nrhs},{b},{fp:.1f},{sn[b]:.4f},{sv[b]:.4f},{r:.4f},"
                          f"{'-' if a is None else int(a)},{'-' if c is None else int(c)},"
                          f"{int(ok)}")

    print(f"\n== ADMISSIBLE CELLS (BOTH arms saturated): {len(admissible)} ==")
    print("clause,cells,geomean,min,losses,sub1.15,worst_cell")
    cands = []
    for tset, tname in [(["float"], "float"),
                        (["double"], "double"),
                        (["cfloat"], "cfloat"),
                        (["cdouble"], "cdouble"),
                        (["float", "double"], "float+double"),
                        (["float", "cfloat"], "float+cfloat"),
                        (["float", "double", "cfloat"], "non-cdouble"),
                        (TYPES, "all types")]:
        for minr in (16, 32, 64, 128):
            cands.append((f"{tname} nrhs>={minr}",
                          lambda k, ts=tset, m=minr: k[0] in ts and k[2] >= m))
            cands.append((f"{tname} nrhs=={minr}",
                          lambda k, ts=tset, m=minr: k[0] in ts and k[2] == m))
    for name, pred in cands:
        sel = [k for k in admissible if pred(k)]
        if not sel:
            continue
        rs = [admissible[k] for k in sel]
        worst = min(sel, key=lambda k: admissible[k])
        print(f"{name},{len(sel)},{geo(rs):.4f},{min(rs):.4f},"
              f"{sum(1 for r in rs if r < 1.0)},{sum(1 for r in rs if 1.0 <= r < 1.15)},"
              f"{worst[0]} n={worst[1]} nrhs={worst[2]} b={worst[3]} = {admissible[worst]:.4f}")
    return admissible


if __name__ == "__main__":
    split = sys.argv.index("--vendor")
    main(sys.argv[1:split], sys.argv[split + 1:])

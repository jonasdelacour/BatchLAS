#!/usr/bin/env python3
# THE QUESTION THIS PASS EXISTS TO ANSWER, made mechanical.
#
# D2: "are the 9 and 4 losses CLUSTERED (a window exists) or MID-LADDER (no
# boundary can exclude them, and the honest answer is no window)?" A loss is
# MID-LADDER on an axis when the ladder has a WIN on BOTH SIDES of it along that
# axis -- because then no one-sided boundary in that axis can exclude the loss
# without also excluding a win. That is the C5/C8/C9 failure mode route_getrs.hh
# already records, and the leg-predicate defect this campaign has found twice.
#
# It prints, for every losing cell, whether it is interior in n and interior in
# batch; and then scores candidate clauses by transcription -- every admitted
# cell listed with its ratio, never inferred from an inequality.
import sys
import math
from collections import defaultdict

TYPES = ["float", "double", "cfloat", "cdouble"]


def load(path, want_route):
    rows = {}
    refused = []
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
                refused.append((key, f"relsd {p[7]}"))
                continue
            if int(p[15]) != 0:
                refused.append((key, f"foreign {p[15]}"))
                continue
            got = p[11].split("|")[-1]
            if got != want_route:
                refused.append((key, f"route {got}"))
                continue
            rows[key] = float(p[5])
    return rows, refused


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(argv):
    nv, r1 = load(argv[0], "native:blocked")
    vd, r2 = load(argv[1], "vendor:auto")
    extra = []
    if len(argv) > 2:
        for i in range(2, len(argv), 2):
            a, ra = load(argv[i], "native:blocked")
            b, rb = load(argv[i + 1], "vendor:auto")
            nv.update(a)
            vd.update(b)
            r1 += ra
            r2 += rb

    for k, why in r1 + r2:
        print(f"# REFUSED {k} {why}")

    ratio = {k: vd[k] / nv[k] for k in nv if k in vd}
    print(f"# {len(ratio)} paired cells")

    print("\n== RATIO = vendor_ms / native_ms  (>1 = the COMPOSITION wins) ==")
    ns = sorted({k[1] for k in ratio})
    bs = sorted({k[3] for k in ratio})
    for nrhs in sorted({k[2] for k in ratio}):
        for t in TYPES:
            head = f"{t:8s} nrhs={nrhs:<4d}"
            print(f"\n{head}  " + "".join(f"b={b:<7d}" for b in bs))
            for n in ns:
                cells = []
                for b in bs:
                    r = ratio.get((t, n, nrhs, b))
                    cells.append(f"{r:<9.3f}" if r else f"{'.':<9s}")
                print(f"  n={n:<5d}          " + "".join(cells))

    # ---- interiority -------------------------------------------------------
    print("\n== EVERY LOSING CELL, AND WHETHER A BOUNDARY COULD EXCLUDE IT ==")
    print("type,n,nrhs,batch,ratio,interior_in_n,interior_in_batch")
    losses = defaultdict(list)
    for k in sorted(ratio):
        if ratio[k] >= 1.0:
            continue
        t, n, nrhs, b = k
        # interior in n: a WIN at some smaller n AND some larger n, same (t,nrhs,b)
        lo = any(ratio.get((t, m, nrhs, b), 0) > 1.0 for m in ns if m < n)
        hi = any(ratio.get((t, m, nrhs, b), 0) > 1.0 for m in ns if m > n)
        blo = any(ratio.get((t, n, nrhs, c), 0) > 1.0 for c in bs if c < b)
        bhi = any(ratio.get((t, n, nrhs, c), 0) > 1.0 for c in bs if c > b)
        print(f"{t},{n},{nrhs},{b},{ratio[k]:.4f},{lo and hi},{blo and bhi}")
        losses[t].append(k)
    for t in TYPES:
        tot = sum(1 for k in ratio if k[0] == t)
        print(f"# {t}: {len(losses[t])} losses of {tot}")

    # ---- candidate clauses, by transcription -------------------------------
    print("\n== CANDIDATE CLAUSES (batch >= 128 only: the campaign's saturation rule) ==")
    print("clause,cells,geomean,min,losses,sub1.15")
    cands = []
    for tset, tname in [(["float"], "float"),
                        (["float", "double"], "float+double"),
                        (["float", "cfloat"], "float+cfloat"),
                        (["float", "double", "cfloat"], "non-cdouble"),
                        (TYPES, "all types")]:
        for minr in (16, 32, 64, 128):
            cands.append((f"{tname} nrhs>={minr}",
                          lambda k, ts=tset, m=minr: k[0] in ts and k[2] >= m))
    for name, pred in cands:
        sel = [k for k in ratio if pred(k) and k[3] >= 128]
        if not sel:
            continue
        rs = [ratio[k] for k in sel]
        nloss = sum(1 for r in rs if r < 1.0)
        sub = sum(1 for r in rs if 1.0 <= r < 1.15)
        print(f"{name},{len(sel)},{geo(rs):.4f},{min(rs):.4f},{nloss},{sub}")
    return ratio


if __name__ == "__main__":
    main(sys.argv[1:])

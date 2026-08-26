#!/usr/bin/env python3
# THE CLAUSE, SCORED BY TRANSCRIPTION.
#
# GATE-C: a preferred() clause ships only if it is >= 1.15x median reproduced
# across two passes, has ZERO losing cells inside the admitted set, and a BATCH
# LADDER exists on every axis it names. So this script does not evaluate an
# inequality against a geomean: it lists EVERY cell a candidate admits, with its
# per-pass ratios and its quoted (conservative) ratio, and reports the minimum
# and the loss count over that transcribed list.
#
# THE AXIS IS nrhs -- GetrsShape::nrhs(), i.e. B.cols() -- and never order().
# Writing the predicate on the wrong extent inverts the window, and that exact
# error was caught twice in WP7. Both axes are therefore printed per cell so the
# reader can see which one the losses live on.
#
# usage: clause.py nv_p1... -- nv_p2... -- v_p1... -- v_p2...
import sys
import math

TYPES = ["float", "double", "cfloat", "cdouble"]


def load(path, want):
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
            if got != want:
                refused.append((key, f"route {got} != {want}"))
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


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(argv):
    # FOUR groups separated by "--": nv_p1 -- nv_p2 -- v_p1 -- v_p2. Each group
    # may hold several CSVs, because a pass was assembled from more than one
    # sweep (the walk ladder's two halves, plus the five gap cells).
    g, cur = [], []
    for x in argv:
        if x == "--":
            g.append(cur); cur = []
        else:
            cur.append(x)
    g.append(cur)
    assert len(g) == 4, f"want four '--'-separated groups, got {len(g)}"
    nv1, r1 = merge(g[0], "native:blocked")
    nv2, r2 = merge(g[1], "native:blocked")
    v1, r3 = merge(g[2], "vendor:auto")
    v2, r4 = merge(g[3], "vendor:auto")
    for k, why in r1 + r2 + r3 + r4:
        print(f"# REFUSED {k}: {why}")

    keys = sorted(set(nv1) & set(nv2) & set(v1) & set(v2))
    print(f"# {len(keys)} cells paired across all four sweeps")

    print("type,n,nrhs,batch,nat_p1,ven_p1,r_p1,nat_p2,ven_p2,r_p2,QUOTED,spread_nat,spread_ven")
    q, sp = {}, []
    for k in keys:
        r_1, r_2 = v1[k] / nv1[k], v2[k] / nv2[k]
        q[k] = min(r_1, r_2)
        sn = max(nv1[k], nv2[k]) / min(nv1[k], nv2[k])
        sv = max(v1[k], v2[k]) / min(v1[k], v2[k])
        sp += [sn, sv]
        print(f"{k[0]},{k[1]},{k[2]},{k[3]},{nv1[k]:.4f},{v1[k]:.4f},{r_1:.4f},"
              f"{nv2[k]:.4f},{v2[k]:.4f},{r_2:.4f},{q[k]:.4f},{sn:.4f},{sv:.4f}")
    sp.sort()
    if sp:
        print(f"# cross-pass median spread {sp[len(sp)//2]:.4f}  worst {sp[-1]:.4f}  "
              f"above 1.10: {sum(1 for s in sp if s > 1.10)} of {len(sp)}")

    print("\n== CANDIDATE CLAUSES, GATE-C SCORED (quoted = the WORSE pass) ==")
    print("clause,cells,geomean,min,losses,sub1.15,GATE-C,worst_cell")
    cands = []
    for tset, tname in [(["float"], "float"), (["double"], "double"),
                        (["cfloat"], "cfloat"), (["cdouble"], "cdouble"),
                        (["float", "double"], "float+double"),
                        (["float", "cfloat"], "float+cfloat"),
                        (["float", "double", "cfloat"], "non-cdouble"),
                        (TYPES, "all types")]:
        for minr in (32, 64, 128):
            cands.append((f"{tname} nrhs>={minr}",
                          lambda k, ts=tset, m=minr: k[0] in ts and k[2] >= m))
            cands.append((f"{tname} nrhs=={minr}",
                          lambda k, ts=tset, m=minr: k[0] in ts and k[2] == m))
    # THE COMPOSITE, named explicitly rather than assembled by the reader: the
    # per-type boundaries differ, and a single scalar boundary cannot express
    # that. This is the recommendation.
    cands.append((
        "RECOMMENDED: float nrhs>=64 OR (double|cfloat) nrhs>=128",
        lambda k: (k[0] == "float" and k[2] >= 64) or
                  (k[0] in ("double", "cfloat") and k[2] >= 128)))
    cands.append((
        "RECOMMENDED + cdouble nrhs>=128 (the rejected widening)",
        lambda k: (k[0] == "float" and k[2] >= 64) or
                  (k[0] in ("double", "cfloat", "cdouble") and k[2] >= 128)))
    for name, pred in cands:
        sel = [k for k in q if pred(k)]
        if not sel:
            continue
        rs = [q[k] for k in sel]
        worst = min(sel, key=lambda k: q[k])
        gate = "PASS" if min(rs) >= 1.15 else "FAIL"
        print(f"{name},{len(sel)},{geo(rs):.4f},{min(rs):.4f},"
              f"{sum(1 for r in rs if r < 1.0)},{sum(1 for r in rs if 1.0 <= r < 1.15)},"
              f"{gate},{worst[0]} n={worst[1]} nrhs={worst[2]} b={worst[3]} = {q[worst]:.4f}")

    print("\n== EVERY CELL BELOW 1.15, WITH ITS INTERIORITY ==")
    print("# a loss/marginal is INTERIOR on an axis when the ladder wins on BOTH")
    print("# sides of it along that axis -- then no one-sided boundary can exclude it.")
    ns = sorted({k[1] for k in q})
    bs = sorted({k[3] for k in q})
    print("type,n,nrhs,batch,ratio,interior_in_n,interior_in_batch")
    for k in sorted(q, key=lambda k: q[k]):
        if q[k] >= 1.15:
            continue
        t, n, nrhs, bb = k
        inn = (any(q.get((t, m, nrhs, bb), 0) >= 1.15 for m in ns if m < n) and
               any(q.get((t, m, nrhs, bb), 0) >= 1.15 for m in ns if m > n))
        inb = (any(q.get((t, n, nrhs, c), 0) >= 1.15 for c in bs if c < bb) and
               any(q.get((t, n, nrhs, c), 0) >= 1.15 for c in bs if c > bb))
        print(f"{t},{n},{nrhs},{bb},{q[k]:.4f},{inn},{inb}")
    return q


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
# GATE-B analyser for the walk-vs-gather A/B.
#
# A row enters a table only if: flag == ok (which already folds in the host
# oracle, the BIT-IDENTITY of the two arms' solutions, and spellA != spellB),
# both relsd <= 0.10, and foreign == 0. Everything dropped is NAMED.
#
# The QUOTED ratio is the WORSE (conservative) of the two passes, and the
# cross-pass median spread is reported over BOTH arms' medians -- because a
# heavy-tailed rep distribution here has failed a 10% relative-sd rule while the
# median reproduced to four significant figures.
import sys
import math

COL = {n: i for i, n in enumerate(
    "type n nrhs batch armA armB spellA spellB medA medB ratio relsdA relsdB "
    "resA resB bitdiff ws route ntpiv flag foreign".split())}


def load(path):
    rows, refused = {}, []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < len(COL):
                refused.append((line.strip(), "malformed"))
                continue
            key = (p[0], int(p[1]), int(p[2]), int(p[3]))
            if p[COL["flag"]] != "ok":
                refused.append((key, f"flag {p[COL['flag']]} "
                                     f"(spell {p[COL['spellA']]}/{p[COL['spellB']]}, "
                                     f"bitdiff {p[COL['bitdiff']]})"))
                continue
            if p[COL["spellA"]] == p[COL["spellB"]]:
                refused.append((key, "both arms resolved the SAME spelling"))
                continue
            if int(p[COL["foreign"]]) != 0:
                refused.append((key, f"{p[COL['foreign']]} foreign compute processes"))
                continue
            ra, rb = float(p[COL["relsdA"]]), float(p[COL["relsdB"]])
            if max(ra, rb) > 0.10:
                refused.append((key, f"relsd {max(ra, rb):.3f} > 0.10"))
                continue
            rows[key] = (float(p[COL["medA"]]), float(p[COL["medB"]]))
    return rows, refused


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(p1, p2):
    a, ra = load(p1)
    b, rb = load(p2)
    for k, why in ra + rb:
        print(f"# REFUSED {k}: {why}")
    keys = sorted(set(a) & set(b))
    print(f"# {len(keys)} paired cells")
    print("type,n,nrhs,batch,walk_p1,gather_p1,r_p1,walk_p2,gather_p2,r_p2,"
          "QUOTED,spread_walk,spread_gather")
    sp, quoted = [], {}
    for k in keys:
        w1, g1 = a[k]
        w2, g2 = b[k]
        r1, r2 = w1 / g1, w2 / g2
        q = min(r1, r2)
        quoted[k] = q
        sw, sg = max(w1, w2) / min(w1, w2), max(g1, g2) / min(g1, g2)
        sp += [sw, sg]
        print(f"{k[0]},{k[1]},{k[2]},{k[3]},{w1:.4f},{g1:.4f},{r1:.4f},"
              f"{w2:.4f},{g2:.4f},{r2:.4f},{q:.4f},{sw:.4f},{sg:.4f}")
    if quoted:
        rs = list(quoted.values())
        worst = min(quoted, key=quoted.get)
        print(f"\n# ALL CELLS: {len(rs)}  geomean {geo(rs):.4f}  min {min(rs):.4f}  "
              f"max {max(rs):.4f}  below 1.00: {sum(1 for r in rs if r < 1.0)}")
        print(f"# worst cell: {worst} = {quoted[worst]:.4f}")
        for t in ("float", "double", "cfloat", "cdouble"):
            s = [quoted[k] for k in quoted if k[0] == t]
            if s:
                print(f"#   {t:8s} {len(s):3d} cells  geomean {geo(s):.4f}  "
                      f"({min(s):.4f} - {max(s):.4f})")
        for nr in sorted({k[2] for k in quoted}):
            s = [quoted[k] for k in quoted if k[2] == nr]
            print(f"#   nrhs={nr:<4d} {len(s):3d} cells  geomean {geo(s):.4f}  "
                  f"({min(s):.4f} - {max(s):.4f})")
    sp.sort()
    if sp:
        print(f"# cross-pass median spread {sp[len(sp)//2]:.4f}  worst {sp[-1]:.4f}  "
              f"above 1.10: {sum(1 for s in sp if s > 1.10)} of {len(sp)}")
    return quoted


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

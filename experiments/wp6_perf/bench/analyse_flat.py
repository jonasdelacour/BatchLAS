#!/usr/bin/env python3
"""Is the routing window FLAT IN BATCH?

One table per (type, order): cuBLAS_ms / cta_ms across the whole wp6_lu batch
ladder, at nrhs = 1, 4 and 8. The verdict rule is stated before it is applied:

  FLAT      -- the ratio stays on ONE side of 1.0 across every rung of the
               ladder. The window's clause for that cell is safe at any batch.
  CROSSES   -- the ratio passes through 1.0 somewhere on the ladder. That cell's
               clause is a one-batch result and must either gain a batch term or
               be dropped from the window.

`spread` is max/min across the ladder and is reported for every row, because a
clause that stays on one side of 1.0 while moving 3x is still a warning about how
much of the headline number is the schedule.
"""
import sys

from analyse import geo, load, med, triage


def main(tag="flat"):
    names = ["vendor", "cta"]
    arms = {a: load("%s_%s.csv" % (tag, a)) for a in names}
    keys = list(arms["vendor"])
    kept, dropped = triage(names, arms, keys)
    kept = set(k for k, _ in kept)

    print("== %s : %d cells, %d kept, %d dropped" % (tag, len(keys), len(kept), len(dropped)))
    for k, why in dropped:
        print("   DROP %-8s n=%-5d nrhs=%-4d b=%-6d : %s" % (k[1], k[2], k[3], k[4], "; ".join(why)))
    print()

    types = ["float", "double", "cfloat", "cdouble"]
    orders = sorted(set(k[2] for k in keys))
    nrhss = sorted(set(k[3] for k in keys))
    verdicts = []
    for t in types:
        for n in orders:
            ladder = sorted(set(k[4] for k in keys if k[1] == t and k[2] == n))
            print("-- %s n=%d : cuBLAS_ms / cta_ms across batch" % (t, n))
            print("%-8s" % "nrhs" + "".join("%10d" % b for b in ladder) + "%10s %9s" % ("spread", "verdict"))
            for r in nrhss:
                row, xs = "%-8d" % r, []
                for b in ladder:
                    k = ("getrs", t, n, r, b)
                    if k not in kept:
                        row += "%10s" % "-"
                        continue
                    v, c = med(arms["vendor"][k]), med(arms["cta"][k])
                    if not v or not c:
                        row += "%10s" % "-"
                        continue
                    xs.append(v / c)
                    row += "%10.3f" % (v / c)
                if xs:
                    above = all(x > 1.0 for x in xs)
                    below = all(x < 1.0 for x in xs)
                    verdict = "FLAT-WIN" if above else ("FLAT-LOSS" if below else "CROSSES")
                    row += "%10.2f %9s" % (max(xs) / min(xs), verdict)
                    verdicts.append((t, n, r, verdict, min(xs), max(xs)))
                else:
                    row += "%10s %9s" % ("-", "no-cells")
                print(row)
            print()

    print("== every row that CROSSES 1.0 on its ladder (these cannot go in the window unguarded)")
    any_cross = False
    for t, n, r, v, lo, hi in verdicts:
        if v == "CROSSES":
            any_cross = True
            print("   %-8s n=%-5d nrhs=%-3d  min %.3f  max %.3f" % (t, n, r, lo, hi))
    if not any_cross:
        print("   none")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "flat")

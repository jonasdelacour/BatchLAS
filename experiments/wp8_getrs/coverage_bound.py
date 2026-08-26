#!/usr/bin/env python3
# EXTENDING THE CLAUSE'S BATCH COVERAGE WITHOUT RE-MEASURING IT, and the argument
# that makes it sound.
#
# THE CLAUSE LADDER measures three SATURATED batch rungs per order. The WALK
# ladder measured seven rungs per order (32 .. 8192) on the same axes, but it
# measured the OLD arm. Those two facts combine into a lower bound:
#
#   * the VENDOR arm is untouched by this pass, so its time at any cell is the
#     same number in both sweeps;
#   * the NATIVE arm only got FASTER: the walk-vs-gather A/B measured 80 cells
#     inside the shipped default set (nrhs >= 16) with MINIMUM 1.0004 and ZERO
#     cells below 1.00, across two passes, bit-identical answers.
#
# Therefore post_ratio(cell) >= pre_ratio(cell) at every cell with nrhs >= 16,
# and every rung of the WALK ladder whose ratio already clears 1.15 is a rung the
# clause clears too. This script reports which admitted cells are covered that
# way and, more importantly, WHICH ARE NOT -- the rungs where the walk ladder was
# below 1.15 and only a direct post-gather measurement can settle it.
#
# A bound is not a measurement. The point of printing the uncovered list is that
# it is the honest statement of what the clause rests on.
import sys
import math
sys.path.insert(0, "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp8_getrs")
from clause import load, merge


def admits(k):
    # THE NARROWED CLAUSE. cfloat nrhs>=128 was in this predicate until the five
    # cells the bound could NOT cover were measured: cfloat n=64 nrhs=128
    # batch=1024 is 0.9944, with 1.2901 at batch 512 and 1.4824 at batch 2048 on
    # either side of it -- a dip in the MIDDLE of its own ladder, which no
    # boundary in batch can exclude. That cell was invisible to every candidate
    # scored before the gap sweep existed.
    return ((k[0] == "float" and k[2] >= 64) or
            (k[0] == "double" and k[2] >= 128))


def main(pre_nv, pre_v, meas_path):
    nv, _ = merge(pre_nv, "native:blocked")
    vd, _ = merge(pre_v, "vendor:auto")
    pre = {k: vd[k] / nv[k] for k in nv if k in vd}
    measured = set()
    with open(meas_path) as f:
        for line in f:
            p = line.split(",")
            if len(p) > 4 and p[0] in ("float", "double", "cfloat", "cdouble"):
                try:
                    measured.add((p[0], int(p[1]), int(p[2]), int(p[3])))
                except ValueError:
                    pass

    cov, unc = [], []
    for k, r in sorted(pre.items()):
        if not admits(k) or k in measured:
            continue
        (cov if r >= 1.15 else unc).append((k, r))
    print(f"# WALK-ladder cells the clause admits and the post-gather sweep did NOT measure: "
          f"{len(cov) + len(unc)}")
    print(f"#   COVERED BY THE BOUND (walk ratio already >= 1.15, and the gather is "
          f"never slower): {len(cov)}")
    if cov:
        rs = [r for _, r in cov]
        print(f"#   their walk ratios: min {min(rs):.4f} geomean "
              f"{math.exp(sum(map(math.log, rs)) / len(rs)):.4f}")
    print(f"#   NOT COVERED (walk ratio below 1.15 -- the bound says nothing): {len(unc)}")
    for k, r in sorted(unc, key=lambda x: x[1]):
        print(f"    UNCOVERED {k[0]} n={k[1]} nrhs={k[2]} b={k[3]}  walk ratio {r:.4f}")


if __name__ == "__main__":
    W = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp8_getrs/"
    main([W + "lad_nv_p1.csv", W + "hi_nv_p1.csv"],
         [W + "lad_v_p1.csv", W + "hi_v_p1.csv"],
         sys.argv[1])

#!/usr/bin/env bash
# THE MISSING LADDER, both arms, two independent passes.
#
# ARM SELECTION IS THE BINARY, NOT THE PIN (D4's protocol B2). The vendor arm is
# lubench6_v with NO pin -- preferred() is all-false at nrhs >= 16, so it resolves
# vendor:auto. The native arm is lubench6_nv with NO pin -- no vendor is linked,
# so it resolves native:blocked. A pin is only for choosing between two NATIVE
# tiers, and at nrhs >= 16 there is only one (kGetrsFusedMaxRhs = 8).
#
# Passes are SEQUENTIAL on one card: two RTX 4090s in this box but only device 0
# is pinned, and two of these running at once would contaminate both.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
export GPU=0 REPS=11 WARM_S=1.0 NPROBE=1 NTRANS=1
export CELLFILE="$D/cells.txt"
for p in p1 p2; do
  bash "$D/run_cells.sh" "$D/lad_nv_$p.csv" lubench6_nv none
  bash "$D/run_cells.sh" "$D/lad_v_$p.csv"  lubench6_v  none
done
echo "LADDER DONE"

#!/usr/bin/env bash
# The high-batch half of the ladder, both arms, one pass.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
export GPU=0 REPS=11 WARM_S=1.0 NPROBE=1 NTRANS=1
export CELLFILE="$D/cells_hi.txt"
P="${1:-p1}"
bash "$D/run_cells.sh" "$D/hi_nv_$P.csv" lubench6_nv none
bash "$D/run_cells.sh" "$D/hi_v_$P.csv"  lubench6_v  none
echo "LADDER-HI $P DONE"

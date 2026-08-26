#!/usr/bin/env bash
# THE FIVE CELLS THE BOUND CANNOT COVER.
#
# coverage_bound.py extends the clause's batch coverage from the 3 saturated rungs
# per order that were re-measured after the gather to the 7 rungs the WALK ladder
# measured, using the fact that the vendor arm did not move and the native arm
# only got faster (A/B minimum 1.0004 over 80 cells inside the default set, zero
# below 1.00). That covers 72 of the 77 unmeasured admitted cells. FIVE are left:
# cfloat nrhs=128 at low batch, where the WALK ladder was itself below 1.15 and
# the bound therefore says nothing. Measuring them is what turns "the clause has
# a batch ladder" from a claim into a table.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
export GPU=0 REPS=11 WARM_S=1.0 NPROBE=1 NTRANS=1
export CELLFILE="$D/gap_cells.txt"
for p in p1 p2; do
  bash "$D/run_cells.sh" "$D/gap_nv_$p.csv" lubench6_nv none
  bash "$D/run_cells.sh" "$D/gap_v_$p.csv"  lubench6_v  none
done
echo "GAP DONE"

#!/usr/bin/env bash
# THE BOUNDARY, TRANSCRIBED RATHER THAN INFERRED.
#
# The main A/B grid samples nrhs in {1, 4, 16, 32, 64, 128} and the shipped
# default boundary sits at 16. That grid shows the gather neutral at 4 and paying
# at 16, but it does not measure the rungs the boundary actually separates: 8 is
# on the WALK side and 16 on the GATHER side and NEITHER 8 NOR 2 was measured.
# GATE-C says transcribe the boundary from a CSV, do not infer it from an
# inequality; that applies to this constant as much as to a preferred() clause.
# 24 is added on the other side so the boundary is bracketed rather than
# one-sided.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
export GPU=0 REPS=11 WARM_S=1.0 NPROBE=1
export CELLFILE="$D/ab_bnd_cells.txt"
for p in p1 p2; do
  bash "$D/ab.sh" "$D/ab_bnd_$p.csv" getrsab_nv
done
echo "AB-BND DONE"

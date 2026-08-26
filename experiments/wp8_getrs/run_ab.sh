#!/usr/bin/env bash
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
export GPU=0 REPS=11 WARM_S=1.0 NPROBE=1
export CELLFILE="$D/ab_cells.txt"
for p in p1 p2; do
  bash "$D/ab.sh" "$D/ab_$p.csv" getrsab_nv
done
echo "AB DONE"

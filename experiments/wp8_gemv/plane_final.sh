#!/usr/bin/env bash
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
export GPU=0 REPS=11 TRS=T WS=auto TYPES="cdouble double cfloat float"
export CELLS="$D/plane_cells.txt"
OUT="$D/planeF_p1.csv" bash "$D/segab.sh"
OUT="$D/planeF_p2.csv" bash "$D/segab.sh"
export TRS=C CELLS="$D/conj_cells.txt"
OUT="$D/conjF_p1.csv" bash "$D/segab.sh"
OUT="$D/conjF_p2.csv" bash "$D/segab.sh"
echo PLANE_DONE

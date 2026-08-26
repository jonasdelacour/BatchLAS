#!/usr/bin/env bash
# Device-0 queue: grid 2 (low batch, ConjTrans, upper red_len edge) once grid 1
# has finished. One process on the device at a time -- two concurrent harnesses
# on one card is exactly the contamination the foreign() guard exists to catch.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"
while pgrep -f 'gemvab_v' >/dev/null; do sleep 20; done
OUT=$W/experiments/wp8_gemv/g6_fit2_p1.csv \
CELLFILE=$W/experiments/wp8_gemv/g6_cells2.txt \
GPU=0 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo QUEUE0_DONE

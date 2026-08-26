#!/usr/bin/env bash
# DEVICE-1 QUEUE. One harness on the card at a time -- two concurrent harnesses
# on one device is exactly the contamination the foreign() guard exists to catch,
# and a guard that fires on every row is a sweep thrown away, not a sweep saved.
#
# Both LU passes run on device 1 (the card with no display attached), so the
# cross-pass spread measures REPRODUCIBILITY and not a difference between cards.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"
while pgrep -x lubench6_v >/dev/null || pgrep -x lubench6_nv >/dev/null; do sleep 20; done
OUT=$W/experiments/wp8_getri/lu_p2.csv \
CELLFILE=$W/experiments/wp8_getri/cells.txt \
GPU=1 REPS=11 WARM_S=1.0 bash "$W/experiments/wp8_getri/pair_cells.sh"
echo LU_P2_DONE

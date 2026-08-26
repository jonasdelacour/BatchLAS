#!/usr/bin/env bash
# DEVICE-0 QUEUE, part 3: grid 3 (the low-batch end at large out_len, the
# out_len rung between 512 and 768, and the high-batch end), two passes.
# Runs after queue0b has drained.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"
wait_idle () { while pgrep -x gemvab_v >/dev/null; do sleep 15; done; }
while pgrep -f queue0b.sh >/dev/null; do sleep 20; done
wait_idle
OUT=$W/experiments/wp8_gemv/g6_fit3_p1.csv CELLFILE=$W/experiments/wp8_gemv/g6_cells3.txt \
  GPU=0 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo FIT3_P1_DONE
wait_idle
OUT=$W/experiments/wp8_gemv/g6_fit3_p2.csv CELLFILE=$W/experiments/wp8_gemv/g6_cells3.txt \
  GPU=0 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo QUEUE0C_DONE

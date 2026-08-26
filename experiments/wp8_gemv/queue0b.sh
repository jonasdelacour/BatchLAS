#!/usr/bin/env bash
# DEVICE-0 QUEUE, the rest of it: grid 2 (low batch / ConjTrans / upper red_len
# edge), then SECOND PASSES of both grids. Everything gemv stays on device 0 so
# the cross-pass spread measures reproducibility rather than a card difference;
# the shipped clause's admitted set is then re-confirmed on device 1 separately,
# because device 0 drives the display and WP8-I3 measured that as an L2 effect on
# the VENDOR arm.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"
wait_idle () { while pgrep -x gemvab_v >/dev/null; do sleep 15; done; }

wait_idle
OUT=$W/experiments/wp8_gemv/g6_fit2_p1.csv CELLFILE=$W/experiments/wp8_gemv/g6_cells2.txt \
  GPU=0 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo FIT2_P1_DONE
wait_idle
OUT=$W/experiments/wp8_gemv/g6_fit_p2.csv CELLFILE=$W/experiments/wp8_gemv/g6_cells.txt \
  GPU=0 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo FIT_P2_DONE
wait_idle
OUT=$W/experiments/wp8_gemv/g6_fit2_p2.csv CELLFILE=$W/experiments/wp8_gemv/g6_cells2.txt \
  GPU=0 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo QUEUE0B_DONE

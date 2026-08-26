#!/usr/bin/env bash
# THE SERIALISED MEASUREMENT QUEUE.
#
# One harness on the BOX at a time, not one per card. The first attempt at this
# pass ran an LU sweep on device 1 and a gemv sweep on device 0 simultaneously,
# on the reasoning that two cards are two machines. They are not: both are on
# NUMA node 0 with the same CPU affinity mask, and lubench6 runs on managed
# memory, so the UVM driver is shared. The per-row foreign() guard reported 0 on
# every row -- correctly, since --query-compute-apps is per device -- and rel_sd
# stayed at 0.0004-0.017 on the contaminated rows. Only a hand re-run against
# WP8-I1's recorded figure showed it: getrf float n=256 batch=128 read 5.51 ms
# against 1.006 ms idle, and the RATIO moved 1.254 -> 1.764.
#
# Everything below therefore runs one after another, with a hard check that the
# box is idle of compute before each step.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"

box_idle () {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    sleep 10
  done
}

box_idle
OUT=$W/experiments/wp8_getri/lu_c1.csv CELLFILE=$W/experiments/wp8_getri/cells_clean.txt \
  GPU=1 REPS=11 WARM_S=1.0 bash "$W/experiments/wp8_getri/pair_cells.sh"
echo LU_C1_DONE

box_idle
OUT=$W/experiments/wp8_getri/lu_c2.csv CELLFILE=$W/experiments/wp8_getri/cells_clean.txt \
  GPU=1 REPS=11 WARM_S=1.0 bash "$W/experiments/wp8_getri/pair_cells.sh"
echo LU_C2_DONE

box_idle
OUT=$W/experiments/wp8_gemv/g6_conf_p1.csv CELLFILE=$W/experiments/wp8_gemv/g6_confirm_cells.txt \
  GPU=1 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo GEMV_CONF_P1_DONE

box_idle
OUT=$W/experiments/wp8_gemv/g6_conf_p2.csv CELLFILE=$W/experiments/wp8_gemv/g6_confirm_cells.txt \
  GPU=1 REPS=11 bash "$W/experiments/wp8_gemv/g6_sweep.sh"
echo SERIAL_ALL_DONE

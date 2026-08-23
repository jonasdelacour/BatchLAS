#!/usr/bin/env bash
# Both arms of the A/B, SEQUENTIALLY on ONE GPU. Never in parallel: this box has
# two RTX 4090s and running the arms concurrently would put each in the other's
# contention, which is one of the recorded ways a false result gets fabricated
# here.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}"
export WARM_S="${WARM_S:-0.6}"
export REPS="${REPS:-5}"
export NPROBE="${NPROBE:-1}"
export NTRANS=1
bash "$D/run_grid.sh" "$D/grid_vendor.csv" luverify_v vendor
bash "$D/run_grid.sh" "$D/grid_native.csv" luverify_nv none

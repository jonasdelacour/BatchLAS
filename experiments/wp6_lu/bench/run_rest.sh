#!/usr/bin/env bash
# Everything after the first saturation sweep, in one queue so the GPU is never
# shared between two arms: sat2 (double and cfloat batch ladders), then getrs,
# then the two FIXED-BATCH order sweeps.
#
# Order inside each pair is always vendor arm, then native arm, both on the same
# pinned GPU, sequentially. Never concurrently: this box has two RTX 4090s and
# co-running the arms is one of the recorded ways a false result gets fabricated
# here.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" NPROBE=1 NTRANS=1

export WARM_S=0.5 REPS=3
CELLFILE="$D/sat2_cells.txt" bash "$D/run_cells.sh" "$D/sat2_vendor.csv" lubench6_v vendor
CELLFILE="$D/sat2_cells.txt" bash "$D/run_cells.sh" "$D/sat2_native.csv" lubench6_nv none

export WARM_S=0.8 REPS=5
CELLFILE="$D/getrs_cells.txt" bash "$D/run_cells.sh" "$D/getrs_vendor.csv" lubench6_v vendor
CELLFILE="$D/getrs_cells.txt" bash "$D/run_cells.sh" "$D/getrs_native.csv" lubench6_nv none

export WARM_S=0.5 REPS=3
"$D/gen_cells.py" order32 > "$D/order32_cells.txt"
CELLFILE="$D/order32_cells.txt" bash "$D/run_cells.sh" "$D/order32_vendor.csv" lubench6_v vendor
CELLFILE="$D/order32_cells.txt" bash "$D/run_cells.sh" "$D/order32_native.csv" lubench6_nv none

"$D/gen_cells.py" order1024 > "$D/order1024_cells.txt"
CELLFILE="$D/order1024_cells.txt" bash "$D/run_cells.sh" "$D/order1024_vendor.csv" lubench6_v vendor
CELLFILE="$D/order1024_cells.txt" bash "$D/run_cells.sh" "$D/order1024_native.csv" lubench6_nv none
echo ALL-DONE

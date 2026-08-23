#!/usr/bin/env bash
# THE SATURATION MAP, both arms, run SEQUENTIALLY on one GPU.
# us/item at fixed n across a batch ladder. A cell is saturated when us/item has
# gone flat; a ratio taken at an unsaturated batch is a comparison against a
# routine that is not using the machine, and must be labelled as such.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" WARM_S="${WARM_S:-0.5}" REPS="${REPS:-3}" NPROBE=1
CELLFILE="$D/sat_cells.txt" bash "$D/run_cells.sh" "$D/sat_vendor.csv" lubench6_v vendor
CELLFILE="$D/sat_cells.txt" bash "$D/run_cells.sh" "$D/sat_native.csv" lubench6_nv none

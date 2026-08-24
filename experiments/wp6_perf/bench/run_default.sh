#!/usr/bin/env bash
# WHAT AN UNPINNED, VENDOR-PRESENT USER ACTUALLY GETS, after preferred() landed.
#
# Every other sweep here PINS a route, which is right for comparing arms and
# wrong for the only question a routing decision finally has to answer: with
# nothing set in the environment, on a CUDA build with cuBLAS present, which
# route does the library take and how fast is the call? So: no pin at all
# (PIN = none, and run_cells.sh unsets all three LU route variables), and the
# printed route column is the instrument -- a row that says vendor:auto inside
# the window would mean the window did not land.
#
# Scored against grid_vendor.csv, which is the SAME cells with the vendor pinned,
# i.e. what the same user got before the flip.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" NPROBE=1 NTRANS=1 WARM_S="${WARM_S:-0.8}" REPS="${REPS:-5}"
python3 "$D/gen_cells.py" grid | awk -F: '$4 <= 4' > "$D/default_cells.txt"
CELLFILE="$D/default_cells.txt" bash "$D/run_cells.sh" "$D/default_auto.csv" lubench6_v none
echo default-DONE

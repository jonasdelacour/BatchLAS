#!/usr/bin/env bash
# THE VENDOR'S ASYMPTOTE. At n >= 1024 cuBLAS's us/item is still falling at the
# top of the saturation ladder, so the ceiling-to-ceiling ratio computed there
# OVERSTATES native's advantage. This pushes the batch as far as 24 GB allows, to
# separate cuBLAS's fixed per-call cost from its marginal per-item cost, and quote
# the marginal one as the honest headline.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" WARM_S="${WARM_S:-0.3}" REPS="${REPS:-3}" NPROBE=1
CELLFILE="$D/tail_cells.txt" bash "$D/run_cells.sh" "$D/tail_vendor.csv" lubench6_v vendor
CELLFILE="$D/tail_cells.txt" bash "$D/run_cells.sh" "$D/tail_native.csv" lubench6_nv none

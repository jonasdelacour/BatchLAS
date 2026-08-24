#!/usr/bin/env bash
# THE FLATNESS CHECK, and it is the one this campaign keeps paying for skipping.
#
# The routing window proposed from grid_summary.txt is read at ONE batch per
# order -- the saturating schedule. A window written from one batch is a window
# that can be wrong at every other batch, and WP6 measured a 2.2x and a 3.9x swing
# in getrf's geomean from the batch axis alone (wp6_lu/bench/README.md section 7).
# Worse for THIS op: section 2 of the same file measured that cuBLAS does not
# saturate at n >= 1024 on any ladder 24 GB holds, so the n = 2048 column of the
# headline grid is read against a vendor that is still latency-bound -- which
# INFLATES the native ratio there, in native's favour, and has to be shown.
#
# So: the full wp6_lu SAT_LADDER at three orders, one per regime, at the three
# nrhs values that bracket the window's boundary (1 inside, 4 on it, 8 at the
# capability edge), all four types.
#
# TWO ARMS ONLY -- vendor and cta. The composition's batch behaviour is already
# published (wp6_lu/bench/README.md section 6: flat for float, adverse for
# cdouble) and re-running it here would triple the wall clock to re-establish a
# number nothing in the routing decision reads.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" NPROBE=1 NTRANS=1 WARM_S="${WARM_S:-0.8}" REPS="${REPS:-5}"
CELLFILE="$D/${LIST:-flat}_cells.txt" bash "$D/run_cells.sh" "$D/${LIST:-flat}_vendor.csv" lubench6_v  vendor
CELLFILE="$D/${LIST:-flat}_cells.txt" bash "$D/run_cells.sh" "$D/${LIST:-flat}_cta.csv"    lubench6_nv native:cta
echo "${LIST:-flat}-DONE"

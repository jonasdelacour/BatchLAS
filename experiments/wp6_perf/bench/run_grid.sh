#!/usr/bin/env bash
# THE HEADLINE A/B: three arms, one queue, one GPU, never concurrently.
#
#   vendor         lubench6_v  + BATCHLAS_GETRS_ROUTE=vendor   -> cublas?getrsBatched
#   blocked (BEFORE) lubench6_nv + native:blocked              -> the WP6 composition
#   cta     (AFTER)  lubench6_nv + native:cta                  -> the fused narrow-RHS tier
#
# The two native arms are PINNED tiers inside the VENDOR-FREE binary. wp6_lu's
# rule -- "the arm is the binary, never a pin" -- is about vendor-vs-native, and
# it still holds here: the vendor arm is the vendor-present binary. Choosing
# between two NATIVE tiers cannot be done by the binary, so it is done by the pin,
# and the resolved route is printed on every row so a pin that did not take is
# visible rather than assumed (an unsupported forced route falls through to
# automatic(), route_resolve.hh:165 -> :175).
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" NPROBE=1 NTRANS=1 WARM_S="${WARM_S:-0.8}" REPS="${REPS:-5}"
LIST="${LIST:-grid}"
CELLFILE="$D/${LIST}_cells.txt" bash "$D/run_cells.sh" "$D/${LIST}_vendor.csv"  lubench6_v  vendor
CELLFILE="$D/${LIST}_cells.txt" bash "$D/run_cells.sh" "$D/${LIST}_blocked.csv" lubench6_nv native:blocked
CELLFILE="$D/${LIST}_cells.txt" bash "$D/run_cells.sh" "$D/${LIST}_cta.csv"     lubench6_nv native:cta
echo "${LIST}-DONE"

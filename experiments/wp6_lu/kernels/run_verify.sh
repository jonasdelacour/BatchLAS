#!/usr/bin/env bash
# WP6 correctness sweep through the PUBLIC API, with the routes PINNED and the
# RESOLVED route printed on every row -- so a pin that did not take is visible
# rather than assumed (route_resolve.hh:165 falls through to automatic() at :175,
# and an unrecognised BATCHLAS_*_ROUTE value silently means VENDOR).
#
# usage: run_verify.sh <binary> [getrf_pin] [getrs_getri_pin]
#   binary : luverify_v | luverify_nv
#   pins   : native:cta | native:blocked | vendor | "" (the build's Auto)
#
# env: NS   -- order list (default straddles the nb=32 block boundary: an exact
#              multiple AND one that is not, plus a sub-block order)
#      NB   -- batch (default 4; > 1 so the LAST item is checked, which the
#              probes do explicitly)
#      GPU  -- CUDA_VISIBLE_DEVICES (default 1)
set -u
D="$(cd "$(dirname "$0")" && pwd)"
BIN="$D/${1:-luverify_v}"
PIN="${2:-}"
PIN2="${3:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WARM_S="${WARM_S:-0.05}"
unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
[ -n "$PIN" ]  && export BATCHLAS_GETRF_ROUTE="$PIN"
[ -n "$PIN2" ] && export BATCHLAS_GETRS_ROUTE="$PIN2" && export BATCHLAS_GETRI_ROUTE="$PIN2"

echo "# bin=$(basename "$BIN") getrf=${BATCHLAS_GETRF_ROUTE:-<auto>} getrs/getri=${BATCHLAS_GETRS_ROUTE:-<auto>}"
fail=0
for t in float double cfloat cdouble; do
  for n in ${NS:-31 32 40 64 96 100}; do
    "$BIN" getrf   "$t" "$n" 1 "${NB:-4}" 3 || fail=$((fail+1))
    "$BIN" getri   "$t" "$n" 1 "${NB:-4}" 3 || fail=$((fail+1))
    "$BIN" getrs   "$t" "$n" 5 "${NB:-4}" 3 || fail=$((fail+1))
  done
  "$BIN" singular "$t" 6 1 3 1 || fail=$((fail+1))
done
echo "FAILS=$fail"

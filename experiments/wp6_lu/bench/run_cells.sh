#!/usr/bin/env bash
# The one runner every table in this directory uses. It takes an explicit list of
# op:type:n:nrhs:batch cells, so the batch schedule of a sweep is written down
# rather than being a side effect of the n ladder -- WP5 produced confounded
# "order crossovers" from a sweep whose batch varied with n, and every table here
# that varies one axis holds the other FIXED by construction.
#
# THE ARM IS THE BINARY, NEVER A PIN, for the native side:
#   lubench6_v  + PIN=vendor -> cuBLAS through the public API
#   lubench6_nv + PIN=none   -> native, resolved by the vendor-free walk
# A bare BATCHLAS_GETRF_ROUTE=native is {Native, Algorithm::Auto}, which names
# neither tier, so supports() refuses it and route_resolve.hh falls through to
# automatic() -- i.e. silently to the vendor. The resolved route is printed on
# every row so that failure mode is visible and not assumed away.
#
# HYGIENE: one GPU pinned (two RTX 4090s in this box); WARM_S seconds of untimed
# warm-up per cell (a cold SYCL JIT once fabricated a 3.7x loss here); medians of
# REPS with mean and relative sd on every row so a noisy cell can be discarded and
# NAMED rather than averaged away; correctness checked in process against a HOST
# oracle on every timed row; nothing run under BATCHLAS_KERNEL_TRACE.
#
# usage: run_cells.sh <out.csv> <lubench6_v|lubench6_nv> <pin|none> <cells...>
#        cells are op:type:n:nrhs:batch, or read from $CELLFILE
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?out.csv}"; shift
BIN="$D/${1:?binary}"; shift
PIN="${1:-none}"; shift || true
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WARM_S="${WARM_S:-0.8}"
export NPROBE="${NPROBE:-1}"
export NTRANS="${NTRANS:-1}"
REPS="${REPS:-5}"

unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
if [ "$PIN" != none ]; then
  export BATCHLAS_GETRF_ROUTE="$PIN" BATCHLAS_GETRS_ROUTE="$PIN" BATCHLAS_GETRI_ROUTE="$PIN"
fi

CELLS="$*"
if [ -z "$CELLS" ] && [ -n "${CELLFILE:-}" ]; then CELLS="$(grep -v '^#' "$CELLFILE")"; fi

: > "$OUT"
echo "op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,extra,ntpiv,extra2,flag" >> "$OUT"
for c in $CELLS; do
  IFS=: read -r op t n nrhs b <<< "$c"
  timeout "${TMO:-2400}" "$BIN" "$op" "$t" "$n" "$nrhs" "$b" "$REPS" >> "$OUT" 2>>"${OUT%.csv}_err.txt" \
    || echo "$op,$t,$n,$nrhs,$b,TIMEOUT_OR_THROW,-,-,-,-,-,-,-,-,-,BAD" >> "$OUT"
done
echo "wrote $OUT"

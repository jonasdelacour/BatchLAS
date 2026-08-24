#!/usr/bin/env bash
# WP6-PERF runner. experiments/wp6_lu/bench/run_cells.sh with ONE change: the pin
# is applied to the GETRS variable ALONE, not to all three LU variables at once.
#
# WHY THAT CHANGE EXISTS. This pass has to compare TWO NATIVE GETRS TIERS against
# each other and against the vendor, which needs a pin -- but wp6_lu's runner
# exports the same value into BATCHLAS_GETRF_ROUTE, BATCHLAS_GETRS_ROUTE and
# BATCHLAS_GETRI_ROUTE together. A `native:cta` there would ALSO pin getrf, whose
# CTA arm has a capacity ceiling of its own; above it the getrf pin falls through
# to automatic() (route_resolve.hh:165 -> :175) and the untimed factorisation
# feeding every getrs row would silently change arm partway up the n ladder. The
# factorisation is untimed, so that would not move a getrs millisecond -- but it
# would move the `rf` half of the printed route column, which is this
# directory's only instrument for "did the pin take", and an instrument that
# moves for reasons unrelated to the thing measured is not one.
#
# Everything else is wp6_lu's runner verbatim: one GPU pinned, WARM_S seconds of
# untimed warm-up per cell, medians of REPS with mean and relative sd on every
# row, correctness against the HOST oracle in process on every timed row, the
# resolved route printed on every row, and nothing run under
# BATCHLAS_KERNEL_TRACE.
#
# usage: run_cells.sh <out.csv> <lubench6_v|lubench6_nv> <getrs-pin|none> <cells...>
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
if [ "$PIN" != none ]; then export BATCHLAS_GETRS_ROUTE="$PIN"; fi
# PINF/PINI pin the other two ops when a table is ABOUT them (the getrf/getri
# regression check), and are unset otherwise.
if [ -n "${PINF:-}" ]; then export BATCHLAS_GETRF_ROUTE="$PINF"; fi
if [ -n "${PINI:-}" ]; then export BATCHLAS_GETRI_ROUTE="$PINI"; fi

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

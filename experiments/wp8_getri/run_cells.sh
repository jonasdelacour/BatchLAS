#!/usr/bin/env bash
# WP8 routing-pass LU runner. experiments/wp6_perf/bench/run_cells.sh VERBATIM
# except for one addition D4 named as a defect: the wp6 runners have NO
# foreign-GPU-process detection, and this box has two RTX 4090s with other
# agents on it. The wp7 audit's foreign() guard is pasted in and its count is
# appended to EVERY row (sampled before AND after the cell, max taken).
#
# usage: run_cells.sh <out.csv> <bin> <getrs-pin|none> [cells...]   (or $CELLFILE)
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?out.csv}"; shift
BIN="$D/${1:?binary}"; shift
PIN="${1:-none}"; shift || true
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WARM_S="${WARM_S:-1.0}"
export NPROBE="${NPROBE:-1}"
export NTRANS="${NTRANS:-1}"
REPS="${REPS:-11}"

UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
BASE="$(basename "$BIN")"
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "$BASE" || true
}

unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
if [ "$PIN" != none ]; then export BATCHLAS_GETRS_ROUTE="$PIN"; fi
if [ -n "${PINF:-}" ]; then export BATCHLAS_GETRF_ROUTE="$PINF"; fi
if [ -n "${PINI:-}" ]; then export BATCHLAS_GETRI_ROUTE="$PINI"; fi

CELLS="$*"
if [ -z "$CELLS" ] && [ -n "${CELLFILE:-}" ]; then CELLS="$(grep -v '^#' "$CELLFILE")"; fi

: > "$OUT"
echo "op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,extra,ntpiv,extra2,x,flag,foreign" >> "$OUT"
for c in $CELLS; do
  IFS=: read -r op t n nrhs b <<< "$c"
  f0=$(foreign)
  row=$(timeout "${TMO:-2400}" "$BIN" "$op" "$t" "$n" "$nrhs" "$b" "$REPS" 2>>"${OUT%.csv}_err.txt") \
    || row="$op,$t,$n,$nrhs,$b,TIMEOUT_OR_THROW,-,-,-,-,-,-,-,-,-,-,BAD"
  f1=$(foreign)
  fc=$(( f0 > f1 ? f0 : f1 ))
  echo "$row,$fc" >> "$OUT"
done
echo "wrote $OUT"

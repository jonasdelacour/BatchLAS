#!/usr/bin/env bash
# WP8-I1 runner. experiments/wp6_perf/bench/run_cells.sh with ONE addition: a
# per-row FOREIGN COMPUTE PROCESS count on the pinned device, appended as the
# last column. D4's B4: the wp6 runners have no foreign-process detection at all
# and this box has two RTX 4090s with other agents on it, so a wp6 sweep run
# as-is produces rows that cannot be audited for contamination after the fact.
#
# usage: run_cells.sh <out.csv> <binary-name> <getrs-pin|none> [cells...]
#        cells are op:type:n:nrhs:batch, or read from $CELLFILE
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?out.csv}"; shift
BIN="$D/${1:?binary}"; shift
PIN="${1:-none}"; shift || true
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WARM_S="${WARM_S:-1.0}"
export NPROBE="${NPROBE:-1}"
export NTRANS="${NTRANS:-1}"
REPS="${REPS:-11}"

UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
BINNAME="$(basename "$BIN")"
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "$BINNAME"
}

unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
if [ "$PIN" != none ]; then export BATCHLAS_GETRS_ROUTE="$PIN"; fi
if [ -n "${PINF:-}" ]; then export BATCHLAS_GETRF_ROUTE="$PINF"; fi
if [ -n "${PINI:-}" ]; then export BATCHLAS_GETRI_ROUTE="$PINI"; fi

CELLS="$*"
if [ -z "$CELLS" ] && [ -n "${CELLFILE:-}" ]; then CELLS="$(grep -v '^#' "$CELLFILE")"; fi

: > "$OUT"
echo "op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,extra,ntpiv,extra2,flag,foreign" >> "$OUT"
for c in $CELLS; do
  IFS=: read -r op t n nrhs b <<< "$c"
  f0=$(foreign)
  ROW=$(timeout "${TMO:-2400}" "$BIN" "$op" "$t" "$n" "$nrhs" "$b" "$REPS" 2>>"${OUT%.csv}_err.txt") \
    || ROW="$op,$t,$n,$nrhs,$b,TIMEOUT_OR_THROW,-,-,-,-,-,-,-,-,-,BAD"
  f1=$(foreign)
  fc=$(( f0 > f1 ? f0 : f1 ))
  echo "$ROW,$fc" >> "$OUT"
done
echo "wrote $OUT"

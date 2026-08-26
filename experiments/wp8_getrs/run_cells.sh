#!/usr/bin/env bash
# WP8-GETRS runner. experiments/wp6_perf/bench/run_cells.sh (the copy that pins
# the GETRS variable ALONE) with ONE addition: a PER-ROW FOREIGN COMPUTE-PROCESS
# COUNT.
#
# WHY. D4's audit of the instrument found that the wp6 LU runners have NO
# foreign-GPU-process detection at all -- only the wp7 gemv audit scripts and
# experiments/gpu_guard.sh do. This box has two RTX 4090s and another agent has
# been measured on device 0 mid-sweep. A wp6 sweep run as-is produces rows that
# cannot be audited for contamination after the fact. The snippet below is the
# wp7 audit's, sampled before AND after each cell and reported as the max, so a
# process that appears and leaves inside one cell is still counted.
#
# The count is appended as an EXTRA COLUMN after lubench6's own 15/16, so the
# existing analysers (which read by POSITION and take the flag as the LAST field)
# would break -- analyse.py here reads the flag at a fixed index instead and is
# written against this format, not against wp6_perf's.
#
# usage: run_cells.sh <out.csv> <lubench6_v|lubench6_nv> <getrs-pin|none> [cells...]
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

unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
# NEVER a bare `native`: it resolves to the FIRST supported route of that origin
# in the order array, which today means CTA, not the composition (campaign trap 3
# and route_getrs.hh:100-109). Callers pass native:blocked / native:cta in full.
if [ "$PIN" != none ]; then export BATCHLAS_GETRS_ROUTE="$PIN"; fi
if [ -n "${PINF:-}" ]; then export BATCHLAS_GETRF_ROUTE="$PINF"; fi
if [ -n "${LASWP:-}" ]; then export BATCHLAS_GETRS_LASWP="$LASWP"; fi

UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")"
BASE="$(basename "$BIN")"
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "$BASE"
}

CELLS="$*"
if [ -z "$CELLS" ] && [ -n "${CELLFILE:-}" ]; then CELLS="$(grep -v '^#' "$CELLFILE")"; fi

: > "$OUT"
echo "op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,extra,ntpiv,flag,foreign" >> "$OUT"
for c in $CELLS; do
  IFS=: read -r op t n nrhs b <<< "$c"
  f0="$(foreign)"
  row="$(timeout "${TMO:-2400}" "$BIN" "$op" "$t" "$n" "$nrhs" "$b" "$REPS" 2>>"${OUT%.csv}_err.txt")" \
    || row="$op,$t,$n,$nrhs,$b,TIMEOUT_OR_THROW,-,-,-,-,-,-,-,-,BAD"
  f1="$(foreign)"
  fc=$(( f0 > f1 ? f0 : f1 ))
  echo "$row,$fc" >> "$OUT"
done
echo "wrote $OUT"

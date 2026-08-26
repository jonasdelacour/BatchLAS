#!/usr/bin/env bash
# The GATE-B A/B: two spellings of the blocked driver's left-hand interchange,
# interleaved rep by rep INSIDE ONE PROCESS, >= 11 reps, median, warm JIT, GPU
# pinned, foreign compute-process count recorded on every row.
#
# usage: OUT=x.csv A=inloop B=defer_gather GPU=0 REPS=11 bash ab.sh <cellfile>
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:?OUT=}"
A="${A:-inloop}"
B="${B:-defer_gather}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WARM_S="${WARM_S:-1.0}"
REPS="${REPS:-11}"
CELLFILE="${1:-$D/cells.txt}"

UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc getrfab_nv
}

: > "$OUT"
echo "type,n,batch,armA,armB,modeA,modeB,A_ms,B_ms,ratio,relsdA,relsdB,resA,resB,bitdiff,route,ntpiv,flag,foreign" >> "$OUT"
grep -v '^#' "$CELLFILE" | while read -r cell; do
  [ -z "$cell" ] && continue
  IFS=: read -r op t n nrhs b <<< "$cell"
  f0=$(foreign)
  ROW=$(timeout "${TMO:-2400}" "$D/getrfab_nv" "$t" "$n" "$b" "$REPS" "$A" "$B" 2>>"${OUT%.csv}_err.txt") \
    || ROW="$t,$n,$b,$A,$B,-,-,-,-,-,-,-,-,-,-,-,-,BAD"
  f1=$(foreign)
  fc=$(( f0 > f1 ? f0 : f1 ))
  echo "$ROW,$fc" >> "$OUT"
done
echo "wrote $OUT"

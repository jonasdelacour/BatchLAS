#!/usr/bin/env bash
# GATE-B. The two PERMUTATION SPELLINGS, interleaved rep by rep inside ONE
# process, against a host oracle, with the resolved spelling read back per arm.
#
# The binary is the VENDOR-FREE one: at nrhs >= 16 preferred() is all-false, so a
# vendor-present build never reaches the composition at all and every row would
# be cuBLAS wearing a native label. (getrsab_v exists for ncu, which refuses the
# vendor-free binaries on this box -- see WP8-I1's unresolved note.)
#
# usage: ab.sh <out.csv> <bin> [pass-tag]
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?out.csv}"
BIN="$D/${2:-getrsab_nv}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export WARM_S="${WARM_S:-1.0}"
export NPROBE="${NPROBE:-1}"
REPS="${REPS:-11}"
ARMA="${ARMA:-walk}"
ARMB="${ARMB:-gather}"
TR="${TR:-N}"

UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")"
BASE="$(basename "$BIN")"
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "$BASE"
}

CELLS="$*"
if [ -z "${3:-}" ] && [ -n "${CELLFILE:-}" ]; then CELLS="$(grep -v '^#' "$CELLFILE")"; fi

: > "$OUT"
echo "type,n,nrhs,batch,armA,armB,spellA,spellB,medA,medB,ratio,relsdA,relsdB,resA,resB,bitdiff,ws,route,ntpiv,flag,foreign" >> "$OUT"
for c in $CELLS; do
  IFS=: read -r op t n nrhs b <<< "$c"
  f0="$(foreign)"
  row="$(timeout "${TMO:-2400}" "$BIN" "$t" "$n" "$nrhs" "$b" "$REPS" "$ARMA" "$ARMB" "$TR" \
          2>>"${OUT%.csv}_err.txt")" \
    || row="$t,$n,$nrhs,$b,$ARMA,$ARMB,-,-,TIMEOUT_OR_THROW,-,-,-,-,-,-,-,-,-,-,BAD"
  f1="$(foreign)"
  echo "$row,$(( f0 > f1 ? f0 : f1 ))" >> "$OUT"
done
echo "wrote $OUT"

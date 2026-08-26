#!/usr/bin/env bash
# WP8/I3 -- the GATE-B sweep: body 5 (each W) against body 3, interleaved rep by
# rep INSIDE ONE PROCESS by experiments/wp8_gemv/gemvsegab.cpp.
#
# CELLS ARE IN (out_len, red_len), never (m, n) -- campaign trap 8. Under
# Trans/ConjTrans m = red_len and n = out_len, which is what the binary takes.
#
# The red_len axis is walked DOWN TO 1 and UP PAST THE GATE: 96/128/192/256 are
# REGRESSION cells, not controls. Body 4's recorded history is that the
# segmented decomposition LOSES on the long end, and body 5's gate is the only
# thing standing between this work package and that outcome.
set -uo pipefail
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "gemvsegab" | head -1
}
export OPENBLAS_CORETYPE=SKYLAKEX
export WARM_S="${WARM_S:-1.0}"
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
BIN="${BIN:-$D/gemvsegab_v}"
OUT="${OUT:-$D/segab.csv}"
REPS="${REPS:-11}"
CELLS="${CELLS:-$D/segab_cells.txt}"
TYPES="${TYPES:-cdouble double cfloat float}"
WS="${WS:-2 4 8}"
TRS="${TRS:-T C}"

echo "type,m,n,batch,transA,arm,wA,wB,med_a_ms,med_b_ms,relsd_a,relsd_b,GBs_a,GBs_b,ratio,relerr_a,relerr_b,ld,out_len,red_len,MB,foreign" > "$OUT"
while read -r ol rl b; do
  case "$ol" in ""|\#*) continue;; esac
  for ty in $TYPES; do
    case "$ty" in cdouble) es=16;; double|cfloat) es=8;; *) es=4;; esac
    mb=$(( ol * rl * b * es / 1048576 ))
    for tr in $TRS; do
      for w in $WS; do
        f0=$(foreign)
        row=$("$BIN" "$ty" "$rl" "$ol" "$b" "$tr" "$REPS" "$w" 2>>"$D/segab_err.txt")
        f1=$(foreign); fc=$(( f0 > f1 ? f0 : f1 ))
        if [ -z "$row" ]; then
          echo "$ty,$rl,$ol,$b,$tr,$w,,,,,,,,,,,,,$ol,$rl,$mb,$fc" >> "$OUT"
        else
          echo "$row,$ol,$rl,$mb,$fc" >> "$OUT"
        fi
      done
    done
  done
done < "$CELLS"
echo "wrote $OUT"

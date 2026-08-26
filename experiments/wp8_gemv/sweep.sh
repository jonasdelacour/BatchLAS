#!/usr/bin/env bash
# WP8/I3 -- the generic (out_len, red_len, batch) sweep for gemv.
#
# Cells are defined in the OPERATION'S OWN AXES (out_len, red_len) and mapped
# onto (m, n) per transA, exactly as experiments/wp7_gemv/audit/parity.sh does:
#     NoTrans          m = out_len   n = red_len
#     Trans/ConjTrans  m = red_len   n = out_len
# so a skinny cell stays skinny when the operation is transposed (campaign
# trap 8).
#
# Every arm is pinned EXPLICITLY (trap 3), the RESOLVED ROUTE is printed as a
# column (trap 4), and the number of FOREIGN compute processes on the target
# device is sampled before and after every cell and recorded as the max.
set -uo pipefail
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "gemvab" | head -1
}
export OPENBLAS_CORETYPE=SKYLAKEX
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
OUT="${OUT:-$D/sweep.csv}"
REPS="${REPS:-11}"
CELLS="${CELLS:-$D/cells.txt}"
TYPES="${TYPES:-cdouble}"
ARMS="${ARMS:-vendor native:cta}"
TRS="${TRS:-T C}"
export WARM_S="${WARM_S:-1.0}"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,out_len,red_len,MB,foreign" > "$OUT"
while read -r ol rl b; do
  case "$ol" in ""|\#*) continue;; esac
  for tr in $TRS; do
    if [ "$tr" = "N" ]; then m=$ol; n=$rl; else m=$rl; n=$ol; fi
    for ty in $TYPES; do
      case "$ty" in cdouble) es=16;; double|cfloat) es=8;; *) es=4;; esac
      mb=$(( ol * rl * b * es / 1048576 ))
      for arm in $ARMS; do
        if [ "$tr" = "N" ] && [ "$arm" = "native:cta" ]; then continue; fi
        f0=$(foreign)
        row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" 2>>"$D/sweep_err.txt")
        f1=$(foreign); fc=$(( f0 > f1 ? f0 : f1 ))
        if [ -z "$row" ]; then
          echo "$arm,$ty,$m,$n,$b,$tr,FAILED,,,,,,,,$ol,$rl,$mb,$fc" >> "$OUT"
        else
          echo "$arm,$row,$ol,$rl,$mb,$fc" >> "$OUT"
        fi
      done
    done
  done
done < "$CELLS"
echo "wrote $OUT"

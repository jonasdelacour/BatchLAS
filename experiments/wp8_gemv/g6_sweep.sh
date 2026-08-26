#!/usr/bin/env bash
# WP8 ROUTING PASS, G6: the cdouble transposed prize, re-fitted AFTER body 5.
#
# WHY A NEW GRID RATHER THAN audit/prize.sh. Two reasons, both of them results
# rather than preferences.
#
#  1. THE BAND'S LOWER EDGE WAS OUR KERNEL'S LIMIT, NOT cuBLAS'S. D3 established
#     that at red_len = 48 the vendor is ALREADY dipped (561-648 GB/s) and the
#     native CTA arm lost only because it was stuck at 442-449 GB/s -- the
#     short-reduction defect. Body 5 (WP8-I3) fixed exactly that, so a clause
#     fitted on the old grid is fitted to a band the fix invalidates.
#  2. THE OLD GRID COULD NOT SEE THE AXIS THE EFFECT LIVES ON. prize.sh samples
#     batch at {128, 256, 512} only; oos.sh adds a few more but never on the
#     (n, batch) corner the clause needs. Trap 8: a grid that cannot reach a
#     regime is not evidence about it.
#
# CELLS ARE DEFINED IN (out_len, red_len) AND MAPPED TO (m, n) HERE, once, so a
# skinny cell stays skinny when the operation is transposed. Under Trans and
# ConjTrans out_len == n == A.cols() and red_len == m == A.rows().
#
# Both arms are PINNED and run in ONE binary, so GATE-B's "interleaved within one
# session" is met literally: vendor and native:cta alternate cell by cell in the
# same process image, sharing clock and thermal state.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
foreign () {
  nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null \
    | grep -F "$UUID" | grep -vc "gemvab_v" | head -1
}
export OPENBLAS_CORETYPE=SKYLAKEX
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
OUT="${OUT:?OUT=...csv}"
CELLFILE="${CELLFILE:?CELLFILE=...txt}"
REPS="${REPS:-11}"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,out_len,red_len,MB,gpu,foreign" > "$OUT"
grep -v '^#' "$CELLFILE" | grep -v '^[[:space:]]*$' | while IFS=: read -r ty out red b tr; do
  # transA is T or C here, so m = red_len and n = out_len. NoTrans is not in this
  # grid at all: the CTA tier refuses it (supports()), and a NoTrans row would be
  # the vendor row wearing a native label.
  m=$red; n=$out
  mb=$(( m * n * b * 16 / 1048576 ))
  for arm in vendor native:cta; do
    f0=$(foreign)
    row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" 2>>"${OUT%.csv}_err.txt")
    f1=$(foreign); fc=$(( f0 > f1 ? f0 : f1 ))
    if [ -z "$row" ]; then
      echo "$arm,$ty,$m,$n,$b,$tr,FAILED,,,,,,,,$out,$red,$mb,$GPU,$fc" >> "$OUT"
    else
      echo "$arm,$row,$out,$red,$mb,$GPU,$fc" >> "$OUT"
    fi
  done
done
echo "wrote $OUT"

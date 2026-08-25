#!/usr/bin/env bash
# WP7 AUDIT -- the OUT-OF-SAMPLE test of the surviving preferred() candidates.
#
# Every predicate that survives the fitted grid was fitted ON that grid, so
# scoring it there is circular. These cells use m, n and batch values that
# appear NOWHERE in prize_p{1,2}.csv, and each one is chosen to sit on the far
# side of at least one candidate's boundary from at least one other candidate,
# so the two disagree and the disagreement is decidable:
#
#   P2  64 <= m <= 320  and  n * batch >= 131072
#   P3  64 <= m <= 320  and  m*n*batch*16 >= 512 MB
#
# A candidate is refuted if it ADMITS a cell that measures below ~1.00x, in
# either pass. Being right about a cell it rejects costs nothing; admitting a
# loser is what moves live traffic onto a slower route.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENBLAS_CORETYPE=SKYLAKEX
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/audit"
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
OUT="${OUT:-$D/oos_p1.csv}"
REPS="${REPS:-11}"

# m n batch -- none of these m, n or batch values is in the fitted grid
# (m: 32..512 by the 11-value ladder; n: 128/256/512; batch: 128/256/512).
CELLS="
96 384 384
96 768 320
96 1024 320
160 192 1024
160 384 640
160 768 192
224 192 640
224 384 320
224 768 192
288 192 768
288 384 384
288 1024 192
96 192 1024
160 1024 384
224 1024 384
288 768 320
352 384 640
352 768 320
64 384 384
64 1024 192
320 192 1024
320 768 320
"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld,MB" > "$OUT"
while read -r m n b; do
  [ -z "$m" ] && continue
  mb=$(( m * n * b * 16 / 1048576 ))
  for tr in T C; do
    for arm in vendor native:cta; do
      row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" cdouble "$m" "$n" "$b" "$tr" "$REPS" 2>>"$D/prize_err.txt")
      if [ -z "$row" ]; then
        echo "$arm,cdouble,$m,$n,$b,$tr,FAILED,,,,,,,,$mb" >> "$OUT"
      else
        echo "$arm,$row,$mb" >> "$OUT"
      fi
    done
  done
done <<< "$CELLS"
echo "wrote $OUT"

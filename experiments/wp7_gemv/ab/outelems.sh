#!/usr/bin/env bash
# WP7 -- test the ONE candidate predicate for cuBLAS's complex<double>
# transposed dip, on two shapes it was NOT fitted on.
#
# THE HYPOTHESIS, and where it came from. batchdep_p{1,2}.csv puts the dip's
# onset between two adjacent batches at each of two very different shapes:
#
#     m=256 n=256   no dip at batch 256, dip at batch 512
#     m= 64 n=2048  no dip at batch  32, dip at batch  64
#
# Those are 268 MB -> 537 MB and 67 MB -> 134 MB, so it is NOT a footprint
# threshold and it is NOT an L2 boundary. What the four points DO share is the
# number of OUTPUT ELEMENTS, n*batch (out_len()*batch under a transposed
#     transA): both transitions sit between 65,536 and 131,072.
#
# Fitting a threshold on two transitions and then shipping it is exactly the
# move this campaign has been burned by. So: predict, then test on shapes the
# threshold was not fitted on. m=128,n=512 and m=64,n=256 are both inside the
# measured 1.89-3.06x rectangle at ~1 GB, and the batches below straddle
# n*batch = 65,536 / 131,072 at BOTH of them.
#
# PREDICTION: 65,536 loses or ties, 131,072 and above wins.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/ab"
BIN="${BIN:-$D/gemvab_v}"
OUT="${OUT:-$D/outelems_p1.csv}"
REPS="${REPS:-9}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld" > "$OUT"
run() {
  for arm in vendor native:cta; do
    row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" cdouble "$1" "$2" "$3" C "$REPS" 2>>"$D/run_err.txt")
    [ -z "$row" ] && echo "$arm,FAILED,cdouble,$1,$2,$3,C" >> "$OUT" || echo "$arm,$row" >> "$OUT"
  done
}
# m=128 n=512 : n*batch = 32768 / 65536 / 131072 / 262144
for b in 64 128 256 512; do run 128 512 "$b"; done
# m=64  n=256 : n*batch = 32768 / 65536 / 131072 / 262144
for b in 128 256 512 1024; do run 64 256 "$b"; done
echo "wrote $OUT"

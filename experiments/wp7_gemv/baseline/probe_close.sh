#!/usr/bin/env bash
# WP7 R4 follow-up 3. Two closing checks on the cdouble+Trans slow region:
#   (a) does the SAME shape run at the roof for real double? (type-specificity)
#   (b) does the batch cliff reproduce at a SECOND shape? (one shape is anecdote)
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/baseline"
BIN="$D/gemvbase_v"
OUT="${OUT:-$D/close.csv}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX
export WARM_S=0.6
R=11
echo "tag,type,m,n,batch,transA,median_ms,mean_ms,rel_sd,GBs,frac_of_900,relerr,ld" > "$OUT"
emit() { tag="$1"; shift; "$BIN" "$@" 2>>"$D/close_err.txt" | sed "s/^/$tag,/" >> "$OUT"; }

# (a) the same (m, n, bytes) in the other three types.
emit typecheck double  256 256 2048 T $R
emit typecheck cdouble 256 256 1024 T $R
emit typecheck cfloat  256 256 2048 T $R
emit typecheck float   256 256 4096 T $R
emit typecheck double   64 256 8192 T $R
emit typecheck cdouble  64 256 4096 T $R
emit typecheck cfloat   64 256 8192 T $R
emit typecheck float    64 256 16384 T $R

# (b) the batch cliff at a second shape (m=64, n=512).
for b in 128 256 384 512 768 1024 2048 4096; do
  emit batchdep2 cdouble 64 512 "$b" T $R
done
echo "wrote $OUT"

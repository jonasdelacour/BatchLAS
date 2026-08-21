#!/usr/bin/env bash
# Batch dependence of the info != 0 failures, through the PUBLIC FACADE with
# BATCHLAS_POTRF_ROUTE=blocked (route.txt shows that pin resolves to
# Native:Blocked at every order tested, so this is not silently cuSOLVER).
# Reported per batch so a small-batch verification run that saw nothing is
# distinguishable from a driver that is correct.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
BIN=${BIN:-./bench}
OUT=${OUT:-$D/batchdep.csv}
MODE=${MODE:-facade}
echo "rep,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
export BATCHLAS_POTRF_ROUTE=blocked
for t in float cdouble; do
for n in 512 1024; do
for b in 1 8 32 64 96 128 256; do
for r in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU BENCH_WARM_S=0.2 $BIN "$MODE" "$t" "$n" "$b" 2 2>>"$D/batchdep.err" \
    | sed "s/^/$r,/" >> "$OUT"
done
done
done
done
unset BATCHLAS_POTRF_ROUTE
echo "wrote $OUT"

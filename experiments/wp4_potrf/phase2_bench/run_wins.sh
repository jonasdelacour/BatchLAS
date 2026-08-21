#!/usr/bin/env bash
# EVERY CELL THAT LOOKED LIKE A WIN, re-measured.
#
# main.csv contains five ratios at or above 1.00 and they are the only claim in
# this whole file that would let anyone say the native driver beats cuSOLVER, so
# they get more evidence than the rest, not less.  The recheck already killed
# one of them: double n=512 batch=128 read nn_x = 1.055 in main.csv, but its
# VENDOR arm had been discarded for rel_sd = 0.147 and the re-measurement puts
# the pair at 5.98 / 6.01 ms, i.e. 0.995x.
#
# Three passes, nine reps.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
OUT="$D/wins.csv"
echo "pass,cfg,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
: > "$D/wins.err"
export BENCH_WARM_S=2.0
for pass in 1 2 3; do
for spec in "double 1024 128" "double 1024 256" "float 1024 256" "cfloat 1024 128" "cfloat 1024 256" "double 512 512"; do
  set -- $spec; t=$1; n=$2; b=$3
  for cfg in def nn VV; do
    unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
    case $cfg in
      nn) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native ;;
      VV) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=vendor ;;
    esac
    CUDA_VISIBLE_DEVICES=$GPU ./bench ab "$t" "$n" "$b" 9 2>>"$D/wins.err" \
      | sed "s/^/$pass,$cfg,/" >> "$OUT"
  done
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
done
done
echo "wrote $OUT"

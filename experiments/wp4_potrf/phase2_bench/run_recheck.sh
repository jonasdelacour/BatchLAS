#!/usr/bin/env bash
# Re-measure the cells the first pass either DISCARDED (rel_sd > 10%) or where
# two configurations that should resolve to the SAME routes disagreed by more
# than the spread -- double n=256 batch=256 came out def 2.12 ms against
# nn 9.47 ms, and for double both gemm and trsm already default to native, so
# those two configurations ought to be the same run twice.
# Three passes, seven reps each, so an outlier cannot survive as a reported
# number and a REPRODUCIBLE difference is distinguishable from a fluke.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
OUT="$D/recheck.csv"
echo "pass,cfg,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
: > "$D/recheck.err"
export BENCH_WARM_S=2.0
for pass in 1 2 3; do
for spec in "double 256 256" "double 512 128" "float 128 512" "double 2048 32"; do
  set -- $spec; t=$1; n=$2; b=$3
  for cfg in def nn VV; do
    unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
    case $cfg in
      nn) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native ;;
      VV) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=vendor ;;
    esac
    CUDA_VISIBLE_DEVICES=$GPU ./bench ab "$t" "$n" "$b" 7 2>>"$D/recheck.err" \
      | sed "s/^/$pass,$cfg,/" >> "$OUT"
  done
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
done
done
echo "wrote $OUT"

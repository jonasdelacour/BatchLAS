#!/usr/bin/env bash
# out_len 1..16 at TWO footprints: a large-batch DRAM-resident one and the
# batch=512 one the parity grid used (where body 1 starves for parallelism).
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp7_gemv/ab
export CUDA_VISIBLE_DEVICES=1
TAG=${TAG:-x}
for m in 1 2 3 4 5 6 7 8 10 12 14 16; do
  n=2048; batch=$(( 33554432 / (m*n) )); [ $batch -lt 1 ] && batch=1
  for r in vendor native:direct; do
    echo -n "$TAG,bigbatch,"
    BATCHLAS_GEMV_ROUTE=$r ./gemvab_v float $m $n $batch N 11 2>&1 | tail -1
  done
done
for m in 1 2 3 4 5 6 7 8 10 12 14 16; do
  for r in vendor native:direct; do
    echo -n "$TAG,b512,"
    BATCHLAS_GEMV_ROUTE=$r ./gemvab_v float $m 2048 512 N 11 2>&1 | tail -1
  done
done
for m in 1 4 8 16; do
  for r in vendor native:direct; do
    echo -n "$TAG,cd512,"
    BATCHLAS_GEMV_ROUTE=$r ./gemvab_v cdouble $m 2048 512 N 11 2>&1 | tail -1
  done
done

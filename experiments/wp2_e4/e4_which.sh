#!/usr/bin/env bash
# Which kernel does the float ladder land on at each n? The float branch of
# select_kernel_variant has ~10 exits, so a float ratio is meaningless without
# knowing which kernel produced it.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
T=/home/jonaslacour/.claude/jobs/20812aa0/tmp
BIN=./build/benchmarks/gemm_benchmark
for n in 8 16 32 33 48 64 96 127 128 192 256 384 512 640 768 1024; do
  rm -f "$T/f_$n.log"
  BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$T/f_$n.log" \
    BATCHLAS_GEMM_VARIANT=sycl timeout 900 "$BIN" --backend=CUDA --type=float \
    --name=BM_GEMM_FIXED128 --warmup=1 --min_iters=1 --max_iters=1 \
    "$n" "$n" "$n" 64 >/dev/null 2>&1
  printf 'n=%-5s -> %s\n' "$n" \
    "$(grep -o 'gemm_sycl[a-z0-9_]*' "$T/f_$n.log" 2>/dev/null | sort -u | tr '\n' ' ')"
done

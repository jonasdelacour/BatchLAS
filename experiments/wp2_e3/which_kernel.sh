#!/usr/bin/env bash
# Which kernel does the `sycl` arm actually land on at each n? The E3 comparison
# only means what I say it means if the native arm runs the kernel I claim.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
T=/home/jonaslacour/.claude/jobs/20812aa0/tmp
for n in 32 48 64 96 136 200 256 320 512; do
  rm -f "$T/t_$n.log"
  BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$T/t_$n.log" \
    BATCHLAS_GEMM_VARIANT=sycl \
    timeout 600 ./build/benchmarks/gemm_benchmark --backend=CUDA --type="${1:-double}" \
    --name=BM_GEMM_FIXED128 --warmup=1 --min_iters=1 --max_iters=1 \
    "$n" "$n" "$n" 64 >/dev/null 2>&1
  printf 'n=%-4s -> %s\n' "$n" "$(grep -o 'gemm_sycl[a-z0-9_]*' "$T/t_$n.log" 2>/dev/null | sort -u | tr '\n' ' ')"
done

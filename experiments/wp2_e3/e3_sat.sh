#!/usr/bin/env bash
# Saturation check. An unsaturated ratio is overhead, not algorithm -- so sweep
# batch at the two shapes with the biggest claimed margins and see whether the
# ratio is stable. If native's advantage shrinks as batch grows, the "win" was
# launch overhead in the vendor arm.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,beta,arm,gflops" > "$OUT"
for n in 48 136; do
  for batch in 64 128 512 2048 8192; do
    for arm in vendor sycl; do
      v=$(BATCHLAS_BENCH_BETA=1 BATCHLAS_GEMM_VARIANT=$arm \
          timeout 900 "$BIN" --backend=CUDA --type=double --name=BM_GEMM_FIXED128 \
          --warmup=5 "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
      echo "$n,$batch,1,$arm,${v:-NA}" >> "$OUT"
    done
  done
done
echo "saturation sweep done"

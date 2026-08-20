#!/usr/bin/env bash
# The representative shape, native vs vendor, ld==rows vs real ld, beta 0 and 1.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu1"
GPU="${GPU:-1}"
for beta in 1 0; do
for pad in 0 384; do
for route in native vendor; do
  BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_BENCH_BETA=$beta BATCHLAS_BENCH_LD_PAD=$pad \
  GPU_GUARD_MAX_WAIT=1800 \
    ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
      --backend=CUDA --type=float --name=BM_GEMM \
      --min_time=300 --min_iters=20 --max_iters=300 \
      --csv="$OUT/t-b${beta}-p${pad}-${route}.csv" \
      128 1024 128 512 > "$OUT/t-b${beta}-p${pad}-${route}.log" 2>&1
  echo "beta=$beta pad=$pad route=$route exit=$?"
done; done; done

#!/usr/bin/env bash
# B's leading dimension alone, swept. A and C stay at ld == rows.
#
# ldb = k + pad = 128 + pad, so the byte stride between two columns of B is
# 4*(128+pad). The pairs 128/132, 384/392 and 896/904 sit either side of a
# power-of-two byte stride (1024 / 2048 / 4096 B) with almost the same stride,
# which separates "power-of-two set/partition conflict" from "plain locality".
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu1"
GPU="${GPU:-1}"
for pad in 0 4 8 32 64 128 132 256 384 392 448 896 904; do
  BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA=1 \
  BATCHLAS_BENCH_LD_PAD_A=0 BATCHLAS_BENCH_LD_PAD_B="$pad" BATCHLAS_BENCH_LD_PAD_C=0 \
  GPU_GUARD_MAX_WAIT=1800 \
    ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
      --backend=CUDA --type=float --name=BM_GEMM --min_time=300 --min_iters=20 --max_iters=300 \
      --csv="$OUT/bs-$pad.csv" 128 1024 128 512 > "$OUT/bs-$pad.log" 2>&1
  echo "padB=$pad exit=$?"
done

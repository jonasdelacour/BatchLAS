#!/usr/bin/env bash
# ncu on the B-ONLY stride (A and C at ld == rows), to confirm the counter
# signature of the whole-operand case is attributable to B alone.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu1"
GPU="${GPU:-1}"
BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA=1 \
BATCHLAS_BENCH_LD_PAD_A=0 BATCHLAS_BENCH_LD_PAD_B=384 BATCHLAS_BENCH_LD_PAD_C=0 \
GPU_GUARD_MAX_WAIT=1800 \
  ./experiments/gpu_guard.sh "$GPU" ncu --set full \
    --kernel-name regex:GemmRegister128x128 \
    --launch-skip 3 --launch-count 1 \
    --export "$OUT/full-b1-Bonly384" --force-overwrite \
    ./build/benchmarks/gemm_benchmark --backend=CUDA --type=float \
      --name=BM_GEMM --min_time=1 --min_iters=8 --max_iters=8 \
      128 1024 128 512 > "$OUT/full-b1-Bonly384.txt" 2>&1
echo "exit=$?"

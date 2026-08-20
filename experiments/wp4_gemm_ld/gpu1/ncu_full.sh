#!/usr/bin/env bash
# ncu --set full on ONE warm launch of the native GEMM, at ld==rows and at the
# real (strided) ld. Same kernel, same template instantiation, same shape --
# the ONLY difference is the leading dimension of every operand.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu1"
GPU="${GPU:-1}"
BETA="${BETA:-1}"
for pad in 0 384; do
  BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA=$BETA BATCHLAS_BENCH_LD_PAD=$pad \
  GPU_GUARD_MAX_WAIT=1800 \
    ./experiments/gpu_guard.sh "$GPU" ncu --set full \
      --kernel-name regex:GemmRegister128x128 \
      --launch-skip 3 --launch-count 1 \
      --export "$OUT/full-b${BETA}-p${pad}" --force-overwrite \
      ./build/benchmarks/gemm_benchmark --backend=CUDA --type=float \
        --name=BM_GEMM --min_time=1 --min_iters=8 --max_iters=8 \
        128 1024 128 512 > "$OUT/full-b${BETA}-p${pad}.txt" 2>&1
  echo "beta=$BETA pad=$pad exit=$?"
done

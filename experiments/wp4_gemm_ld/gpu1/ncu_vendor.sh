#!/usr/bin/env bash
# The same two configurations on the VENDOR route. cuBLAS picks
# ampere_sgemm_128x128_nn -- the same 128x128 tile, the same 4096-block grid and
# 118 registers against our 119 -- so any difference in how the two react to the
# stride is a difference in the memory pipeline, not in tile shape or occupancy.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu1"
GPU="${GPU:-1}"
for pad in 0 384; do
  BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD=$pad \
  GPU_GUARD_MAX_WAIT=1800 \
    ./experiments/gpu_guard.sh "$GPU" ncu --set full \
      --kernel-name regex:ampere_sgemm \
      --launch-skip 3 --launch-count 1 \
      --export "$OUT/vfull-b1-p${pad}" --force-overwrite \
      ./build/benchmarks/gemm_benchmark --backend=CUDA --type=float \
        --name=BM_GEMM --min_time=1 --min_iters=8 --max_iters=8 \
        128 1024 128 512 > "$OUT/vfull-b1-p${pad}.txt" 2>&1
  echo "pad=$pad exit=$?"
done

#!/usr/bin/env bash
# Confirm WHICH kernel each E2 cell actually ran. Timing is meaningless here
# (BATCHLAS_KERNEL_TRACE inflates wall time ~60%); this run exists only to name
# the kernel.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/trace"
mkdir -p "$OUT"
GPU="${GPU:-1}"

one() { # tag cfg m n k batch pad
    local tag=$1 cfg=$2 m=$3 n=$4 k=$5 batch=$6 pad=$7
    local kern=""
    [ "$cfg" = f128 ] && kern=register128x128k8
    BATCHLAS_GEMM_ROUTE=native BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
    BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$OUT/${tag}-${cfg}-pad${pad}.json" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=50 --min_iters=3 --max_iters=5 \
            --csv="$OUT/${tag}-${cfg}-pad${pad}.csv" \
            "$m" "$n" "$k" "$batch" > "$OUT/${tag}-${cfg}-pad${pad}.log" 2>&1
    echo "$tag $cfg pad=$pad exit=$?"
}

for pad in 0 384; do
  for cfg in auto f128; do
    one R1 "$cfg" 1000 1024 128 128 "$pad"
    one R2 "$cfg" 1024 1024 64  128 "$pad"
    one R3 "$cfg" 1024 1024 128 64  "$pad"
  done
done

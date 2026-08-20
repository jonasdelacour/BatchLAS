#!/usr/bin/env bash
# R3 at pad 0 is the control: auto and forced trace to the SAME kernel
# (gemm_sycl_register_128x128_k8, aligned leg), so they must time the same.
# The first pass disagreed by 12%. Re-run interleaved to find out whether that
# is process-to-process variance or a real cost of the forcing harness.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/raw"
GPU="${GPU:-1}"

one() { # cfg rep
    local cfg=$1 rep=$2 kern=""
    [ "$cfg" = f128 ] && kern=register128x128k8
    BATCHLAS_GEMM_ROUTE=native BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD=0 \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$OUT/e3-R3-${cfg}-r${rep}.csv" 1024 1024 128,128 64 \
            > "$OUT/e3-R3-${cfg}-r${rep}.log" 2>&1
    echo "  e3 R3 $cfg rep$rep exit=$?"
}

for rep in 1 2 3; do
    one auto "$rep"
    one f128 "$rep"
done

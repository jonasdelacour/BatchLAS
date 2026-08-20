#!/usr/bin/env bash
# Re-run the three E5 cells that gpu_guard flagged exit=5 (a foreign process,
# /home/avery/.../test_sycl_alexandrov pid 1489460, landed on the card
# mid-run). Their first numbers are discarded, not quoted.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/raw"
GPU="${GPU:-1}"

one() { # tag cfg m n k batch pad
    local tag=$1 cfg=$2 m=$3 n=$4 k=$5 batch=$6 pad=$7 kern="" route=native
    [ "$cfg" = f128 ] && kern=register128x128k8
    [ "$cfg" = vendor ] && route=vendor
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
    GPU_GUARD_MAX_WAIT=3600 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$OUT/e5-${tag}-${cfg}-pad${pad}.csv" \
            "$m" "$n" "$k,$k" "$batch" \
            > "$OUT/e5-${tag}-${cfg}-pad${pad}.log" 2>&1
    echo "  rerun $tag $cfg pad=$pad exit=$?"
}

one S5 f128 64 1024 64 512 384
one S6 f128 1024 64 64 512 384
one S5 f128 64 1024 64 512 0

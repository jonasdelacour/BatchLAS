#!/usr/bin/env bash
# The R3 control showed process-to-process spread up to 15% even though the
# within-process RSD is 0.4%. R1/R2's auto-vs-forced gaps are 1.85x and 2.28x,
# far outside that, but confirm them interleaved rather than back-to-back-by-cfg.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/raw"
GPU="${GPU:-1}"

one() { # tag cfg m n k batch pad rep
    local tag=$1 cfg=$2 m=$3 n=$4 k=$5 batch=$6 pad=$7 rep=$8 kern=""
    [ "$cfg" = f128 ] && kern=register128x128k8
    BATCHLAS_GEMM_ROUTE=native BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$OUT/e4-${tag}-${cfg}-pad${pad}-r${rep}.csv" \
            "$m" "$n" "$k,$k" "$batch" \
            > "$OUT/e4-${tag}-${cfg}-pad${pad}-r${rep}.log" 2>&1
    echo "  e4 $tag $cfg pad=$pad rep$rep exit=$?"
}

for rep in 1 2; do
  for pad in 0 384; do
    one R1 auto 1000 1024 128 128 "$pad" "$rep"
    one R1 f128 1000 1024 128 128 "$pad" "$rep"
    one R2 auto 1024 1024 64  128 "$pad" "$rep"
    one R2 f128 1024 1024 64  128 "$pad" "$rep"
  done
done

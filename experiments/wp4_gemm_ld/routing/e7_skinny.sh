#!/usr/bin/env bash
# E7: 512x64x512, the exact shape pinned by
# GemmDispatchPolicyTest.KeepsSkinnyTallNNOnLegacyK16PathUntilBenchmarked
# (tests/gemm_tests.cc:352-355). That test's own name says it is waiting for a
# measurement; this is it. It is the ONLY existing routing assertion the
# proposed rule flips.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/raw"
GPU="${GPU:-1}"

one() { # cfg pad
    local cfg=$1 pad=$2 kern="" route=native
    [ "$cfg" = f128 ] && kern=register128x128k8
    [ "$cfg" = vendor ] && route=vendor
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
    GPU_GUARD_MAX_WAIT=3600 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$OUT/e7-U1-${cfg}-pad${pad}.csv" 512 64 512,512 128 \
            > "$OUT/e7-U1-${cfg}-pad${pad}.log" 2>&1
    echo "  e7 U1 $cfg pad=$pad exit=$?"
}

for pad in 0 384; do
  for cfg in auto f128 vendor; do one "$cfg" "$pad"; done
done
echo "e7 complete"

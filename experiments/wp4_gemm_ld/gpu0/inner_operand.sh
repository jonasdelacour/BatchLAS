#!/usr/bin/env bash
# WHICH operand's stride costs the inner shape its time?
#
# The inner shape (m=32, n=1024, k=32) never changes kernel: the trace says
# gemm_sycl_register_32x32 at pad 0, 1, 4, 16, 128 and 384. So unlike the outer
# shape its curve is not the selector -- it is a real access-pattern cost. But
# the curve is not a slope either: ld=36..44 costs 2.1-2.8x while ld=48,56,64,
# 96,128 cost ~1.0x. This isolates the operand responsible, at one pad from each
# regime.
#
# GPU 1's per_operand.sh covers the OUTER shape (128 1024 128 512); this is the
# inner one only, deliberately not a duplicate.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu0/inner_operand"
mkdir -p "$OUT"
GPU="${GPU:-0}"

run() { # pa pb pc tag
    local f="$OUT/i-$4"
    BATCHLAS_BENCH_LD_PAD_A="$1" BATCHLAS_BENCH_LD_PAD_B="$2" BATCHLAS_BENCH_LD_PAD_C="$3" \
    BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA=1 GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$f.csv" 32 1024 32,96,32,96 512 > "$f.log" 2>&1
    echo "  inner $4 exit=$?"
}

for p in 4 128; do
    run 0 0 0 "p$p-none"
    run "$p" 0 0 "p$p-A"
    run 0 "$p" 0 "p$p-B"
    run 0 0 "$p" "p$p-C"
    run "$p" "$p" "$p" "p$p-ABC"
done
echo "inner_operand complete"

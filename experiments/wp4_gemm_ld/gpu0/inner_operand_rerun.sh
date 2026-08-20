#!/usr/bin/env bash
# Re-run of the two inner_operand cells gpu_guard rejected (exit 5: a foreign
# process, pid 1399028, appeared on GPU 0 mid-run). Their numbers were
# discarded, not used.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu0/inner_operand"
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
run 0 0 0 "p4-none"
run 4 4 4 "p4-ABC"

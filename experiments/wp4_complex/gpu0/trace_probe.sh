#!/usr/bin/env bash
# Which kernel does the native route actually run for complex, and does forcing
# the wide register kernel actually land on it? Trace only -- never timed.
set -uo pipefail
cd "$(dirname "$0")/../../.."
T=/home/jonaslacour/.claude/jobs/20812aa0/tmp
GPU="${GPU:-0}"
TYPE="${TYPE:-cfloat}"
M="${M:-1024}"; N="${N:-1024}"; K="${K:-32}"; B="${B:-8}"; PAD="${PAD:-0}"

for cfg in auto wide; do
    kern=""
    [ "$cfg" = wide ] && kern=reg64x64k16wide
    BATCHLAS_KERNEL_TRACE=1 \
    BATCHLAS_KERNEL_TRACE_PATH="$T/tr-$TYPE-$M-$N-$K-pad$PAD-$cfg.json" \
    BATCHLAS_GEMM_ROUTE=native BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$PAD" \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type="$TYPE" --name=BM_GEMM_FIXED128 \
            --min_time=1 --min_iters=1 --max_iters=1 --warmup=1 \
            "$M" "$N" "$K" "$B" >/dev/null 2>&1
    echo "--- $TYPE ${M}x${N}x${K} pad$PAD cfg=$cfg"
    python3 experiments/wp4_complex/gpu0/trace_names.py \
        "$T/tr-$TYPE-$M-$N-$K-pad$PAD-$cfg.json"
done

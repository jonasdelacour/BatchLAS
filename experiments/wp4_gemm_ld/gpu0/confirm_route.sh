#!/usr/bin/env bash
# Confirm the route AND the kernel variant that actually ran at each pad.
#
# "Route is forced with BATCHLAS_GEMM_ROUTE" is a claim about intent, not about
# what executed. The 128x32 s2u1 family has SEPARATE trace names for its aligned
# and generic instantiations (gemm_kernels.cc:140-143), so the trace shows
# directly whether the pad flipped can_use_aligned_nn_fast_path.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu0/trace"
mkdir -p "$OUT"
GPU="${GPU:-0}"

for route in native vendor; do
for shape in outer inner; do
  if [ "$shape" = outer ]; then M=128; K=128; else M=32; K=32; fi
  for pad in 0 1 4 16 128 384; do
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
    BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$OUT/${shape}-${route}-pad${pad}.json" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=50 --min_iters=3 --max_iters=5 \
            --csv="$OUT/${shape}-${route}-pad${pad}.csv" \
            "$M" 1024 "$K" 512 > "$OUT/${shape}-${route}-pad${pad}.log" 2>&1
    echo "$shape $route pad=$pad exit=$?"
  done
done
done

#!/usr/bin/env bash
# WP4 -- the strided-ld penalty as a FUNCTION of the pad.
#
# The question this sweep exists to answer: is the native register-tiled GEMM's
# loss on strided operands a CLIFF the moment ld != rows (a predicate/fast-path
# flag flipping) or does it SCALE with the pad (a real memory-access cost)?
# Those have different fixes, and two endpoints cannot tell them apart.
#
# Design notes that are not optional:
#  * Each k value is listed TWICE. The first pass is a throwaway that warms the
#    JIT and lets the SM clock ramp -- a 50 ms run on this box measured 1.57 ms
#    for a shape that measures 0.984 ms warm, purely from a 210 MHz cold clock.
#    Only the SECOND occurrence of each k is reported.
#  * --name=BM_GEMM_FIXED128 rather than BM_GEMM: --name is a SUBSTRING match,
#    so "BM_GEMM" runs both registrations and doubles the work for nothing.
#  * pads 1,2,3 matter as much as 4,8,...: supports_aligned_packet_loads
#    (src/sycl/gemm/load_policies.hh:27) requires ld % Width == 0, so for float
#    (Width 4) a pad of 4 keeps vector loads while a pad of 1 does not, while
#    BOTH break is_contiguous_dense_matrix. That splits the two candidate causes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu0/raw"
mkdir -p "$OUT"
GPU="${GPU:-0}"
PADS="${PADS:-0 1 2 3 4 8 16 32 64 128 384}"

run() { # route beta pad tag m n klist batch
    local route=$1 beta=$2 pad=$3 tag=$4 m=$5 n=$6 klist=$7 batch=$8
    local f="$OUT/${tag}-${route}-b${beta}-pad${pad}"
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_BENCH_BETA="$beta" BATCHLAS_BENCH_LD_PAD="$pad" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$f.csv" "$m" "$n" "$klist" "$batch" > "$f.log" 2>&1
    echo "  $tag $route beta=$beta pad=$pad exit=$?"
}

for route in native vendor; do
  for beta in 1 0; do
    for pad in $PADS; do
      run "$route" "$beta" "$pad" outer 128 1024 128,256,128,256 512
      run "$route" "$beta" "$pad" inner 32  1024 32,96,32,96     512
    done
  done
done
echo "sweep complete"

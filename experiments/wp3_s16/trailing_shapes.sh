#!/usr/bin/env bash
# The six GEMM shapes V2's trailing updates actually issue, native vs vendor.
#
# These are read straight off the nsys profile of float / Side::Left / n=512,
# q=1024, batch=512 (OUTER_NB=128, inner nb=32):
#
#   outer, one per panel after the first:  m=128, n=q, k=128 / 256 / 384
#   inner, three per panel:                m=32,  n=q, k=32  /  64 /  96
#
# gemm_benchmark calls the PUBLIC gemm, so BATCHLAS_GEMM_ROUTE selects, which is
# exactly the question: V2 today calls sycl_gemm::gemm_custom and bypasses the
# route table entirely, so it always gets the native kernel whether or not the
# native kernel is the better one.
#
# beta=1 because the trailing update always reads C (it is B's own data being
# updated in place). A beta=0 measurement would be a different kernel path.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s16"
mkdir -p "$OUT"
GPU="${GPU:-1}"

exec 9>"$OUT/.lock"
flock -n 9 || { echo "another sweep holds the lock"; exit 3; }

for route in vendor native; do
    echo "=== BATCHLAS_GEMM_ROUTE=$route ==="
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_BENCH_BETA=1 GPU_GUARD_MAX_WAIT=5400 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM \
            --min_time=200 --min_iters=10 --max_iters=200 \
            --csv="$OUT/outer-$route.csv" \
            128 1024 128,256,384 512 > "$OUT/outer-$route.log" 2>&1
    echo "  outer exit=$?"
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_BENCH_BETA=1 GPU_GUARD_MAX_WAIT=5400 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM \
            --min_time=200 --min_iters=10 --max_iters=200 \
            --csv="$OUT/inner-$route.csv" \
            32 1024 32,64,96 512 > "$OUT/inner-$route.log" 2>&1
    echo "  inner exit=$?"
done
echo "trailing shape A/B complete"

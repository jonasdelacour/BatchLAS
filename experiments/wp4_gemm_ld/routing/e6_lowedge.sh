#!/usr/bin/env bash
# E6: the LOWER edge of the proposed routing rule.
#
# E5 measured wins for 128x128 down to min(m,n)=64 and k=16. Below that the
# 128x128 tile is mostly padding and the rule must stop. The two shapes that
# matter most are the WP3 step-16 "inner" rows (m=32, n=1024, k=32/96), which
# currently take Tiled32x32Register at every pad and are the OTHER half of the
# measured strided-ld collapse.
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
            --csv="$OUT/e6-${tag}-${cfg}-pad${pad}.csv" \
            "$m" "$n" "$k,$k" "$batch" \
            > "$OUT/e6-${tag}-${cfg}-pad${pad}.log" 2>&1
    echo "  e6 $tag $cfg pad=$pad exit=$?"
}

SHAPES="T1:32:1024:32:512 T2:32:1024:96:512 T3:64:64:64:512 T4:1024:1024:8:128 T5:128:128:8:512"

for pad in 0 384; do
  for s in $SHAPES; do
    IFS=: read -r tag m n k batch <<< "$s"
    for cfg in auto f128 vendor; do
      one "$tag" "$cfg" "$m" "$n" "$k" "$batch" "$pad"
    done
  done
done
echo "e6 complete"

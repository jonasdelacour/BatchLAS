#!/usr/bin/env bash
# E5: where does "route it to 128x128" STOP paying?
#
# R1/R2 show the 128x128 kernel beating the router's choice by 1.86-2.41x on
# ragged-m and small-k panel shapes. A routing rule needs the other edge too:
# the 128x128 tile is 128 wide in BOTH m and n, so a shape with m<128 or n<128
# wastes half or more of every CTA, and k<TileK*something wastes the pipeline.
# Everything here is at beta=1, both at ld==rows and at pad 384.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/raw"
GPU="${GPU:-1}"

one() { # tag cfg m n k batch pad
    local tag=$1 cfg=$2 m=$3 n=$4 k=$5 batch=$6 pad=$7 kern=""
    [ "$cfg" = f128 ] && kern=register128x128k8
    [ "$cfg" = vendor ] && kern=""
    local route=native
    [ "$cfg" = vendor ] && route=vendor
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$OUT/e5-${tag}-${cfg}-pad${pad}.csv" \
            "$m" "$n" "$k,$k" "$batch" \
            > "$OUT/e5-${tag}-${cfg}-pad${pad}.log" 2>&1
    echo "  e5 $tag $cfg pad=$pad exit=$?"
}

# tag  m     n     k   batch
SHAPES="S1:1024:1024:32:128 S2:1024:1024:16:128 S3:512:512:32:512 S4:128:1024:32:512 S5:64:1024:64:512 S6:1024:64:64:512 S7:256:256:64:512"

for pad in 0 384; do
  for s in $SHAPES; do
    IFS=: read -r tag m n k batch <<< "$s"
    for cfg in auto f128 vendor; do
      one "$tag" "$cfg" "$m" "$n" "$k" "$batch" "$pad"
    done
  done
done

#!/usr/bin/env bash
# E1: does the B-only strided penalty scale with m?
#
# The pack-B proposal lives or dies on this. Packing B costs 2*|B| bytes of
# DRAM traffic; the gap it buys back is a fraction f of the aligned GEMM time.
# At m=128 (the only m measured so far) |B| == |C| and the pack is a loss by
# arithmetic. A real trailing panel update has m >> k, so |B| shrinks relative
# to the total while the penalty may not.
#
# Grid blocks = batch*(m/128)*(n/128) is held at 4096 and |C| at 268 MB across
# all four rows, so occupancy and C traffic are constant and only m changes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/raw"
GPU="${GPU:-1}"

run() { # tag m n klist batch padB
    local tag=$1 m=$2 n=$3 klist=$4 batch=$5 padB=$6
    local f="$OUT/e1-${tag}-padB${padB}"
    BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA=1 \
    BATCHLAS_BENCH_LD_PAD=0 BATCHLAS_BENCH_LD_PAD_B="$padB" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$f.csv" "$m" "$n" "$klist" "$batch" > "$f.log" 2>&1
    echo "  e1 $tag padB=$padB exit=$?"
}

for pad in 0 384; do
  run m128  128  1024 128,128 512 $pad
  run m256  256  1024 128,128 256 $pad
  run m512  512  1024 128,128 128 $pad
  run m1024 1024 1024 128,128 64  $pad
done
echo "e1 complete"

#!/usr/bin/env bash
# E2: is the 128x128 kernel being ROUTED AWAY FROM on the shapes real panel
# updates produce, and what does forcing it back cost/buy at strided ld?
#
# select_kernel_variant (src/sycl/gemm_kernels.cc:509) gates the 128x128 kernel
# on can_use_128x128_fast_path, which requires m%128==0, n%128==0, k%8==0 AND
# ld%4==0 AND 16-byte base on all three operands. That predicate is the
# ALIGNED-LEG predicate; the kernel itself has a correct predicated leg
# (register_128x128.hh:205-296) that needs none of it.
#
#   R1 m=1000  -> extents ragged, fast path fails, router drops to 128x32 K16
#   R2 k=64    -> the k>=128 gate at :509/:512/:523 fails, router drops to 64x64
#   R3 m=1024 k=128 -> aligned control; auto should already be 128x128
#
# Each at pad 0 (ld==rows) and pad 384 (the sub-view case), auto vs forced vs
# vendor.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/routing/raw"
GPU="${GPU:-1}"

run() { # tag cfg m n klist batch pad
    local tag=$1 cfg=$2 m=$3 n=$4 klist=$5 batch=$6 pad=$7
    local f="$OUT/e2-${tag}-${cfg}-pad${pad}"
    local route=native kern=""
    case "$cfg" in
        auto)    route=native ;;
        f128)    route=native; kern=register128x128k8 ;;
        vendor)  route=vendor ;;
    esac
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
    GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$f.csv" "$m" "$n" "$klist" "$batch" > "$f.log" 2>&1
    echo "  e2 $tag $cfg pad=$pad exit=$?"
}

for pad in 0 384; do
  for cfg in auto f128 vendor; do
    run R1 "$cfg" 1000 1024 128,128 128 "$pad"
    run R2 "$cfg" 1024 1024 64,64   128 "$pad"
    run R3 "$cfg" 1024 1024 128,128 64  "$pad"
  done
done
echo "e2 complete"

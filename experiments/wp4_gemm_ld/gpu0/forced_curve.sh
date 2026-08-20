#!/usr/bin/env bash
# The pad curve with the ROUTER REMOVED: 128x128 forced at every pad.
#
# forced_variant.sh showed that at pad 1 the auto route costs 1.90x while the
# same shape with KernelVariant::Tiled128x128RegisterK8 forced costs 1.02x. So
# the cliff is the selector, not the ld. This run measures what is LEFT once the
# selector cannot move: the honest memory-access cost of a strided ld for one
# fixed kernel, including the ld % 4 != 0 points (129) that the selector would
# otherwise divert.
#
# beta=0 at three pads as a control: the epilogue is a read-modify-write only
# when beta != 0, and a cost that lives in the C write should move with beta.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu0/forced"
mkdir -p "$OUT"
GPU="${GPU:-0}"

run() { # pad beta
    local pad=$1 beta=$2
    local f="$OUT/t-force128-b${beta}-pad${pad}"
    BATCHLAS_GEMM_SYCL_KERNEL=128x128x8 BATCHLAS_GEMM_ROUTE=native \
    BATCHLAS_BENCH_BETA="$beta" BATCHLAS_BENCH_LD_PAD="$pad" GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=20 --max_iters=300 \
            --csv="$f.csv" 128 1024 128,256,128,256 512 > "$f.log" 2>&1
    echo "  force128 beta=$beta pad=$pad exit=$?"
}

for pad in 0 1 2 3 4 8 16 32 64 128 129 192 256 384 512; do run "$pad" 1; done
for pad in 0 1 128; do run "$pad" 0; done
echo "forced_curve complete"

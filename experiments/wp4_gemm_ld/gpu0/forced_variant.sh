#!/usr/bin/env bash
# THE decisive test for the outer shape's cliff, plus a reproducibility check.
#
# Sweep 1+2 showed the outer shape (m=128, n=1024) costs ~1.9x the moment
# ld % 4 != 0, and that the cost is IDENTICAL at pad 1 and pad 129 -- i.e. it
# does not scale with the pad at all. The kernel trace says why:
#
#   trace/outer-native-pad0.json   gemm_sycl_register_128x128_k8
#   trace/outer-native-pad1.json   gemm_sycl_register_128x32_k16     <-- different kernel
#   trace/outer-native-pad4.json   gemm_sycl_register_128x128_k8
#
# can_use_128x128_fast_path (register_128x128.hh:84-90) requires (ld % 4) == 0,
# so pad % 4 != 0 does not slow the 128x128 kernel down -- it ROUTES AWAY from
# it, into Tiled128x32RegisterK16 (gemm_kernels.cc:509,517).
#
# If that is the whole story, then FORCING the 128x128 variant at pad 1 should
# recover most of the 1.9x, because the launcher falls back to
# launch_register_128x128_k8<T,false> -- the predicated, non-vector-load leg of
# the SAME kernel (gemm_kernels.cc:733-742). If forcing recovers nothing, the
# cliff is intrinsic to the misaligned ld and not a routing decision.
#
# TWO TRAPS this script exists past:
#  * The kernel-variant env is BATCHLAS_GEMM_SYCL_KERNEL (gemm_kernels.cc:277).
#    BATCHLAS_GEMM_VARIANT is a different knob -- a ROUTE request parsed in
#    src/backends/gemm_variant.hh:69-92, where an unrecognised value silently
#    means "vendor". A first pass of this script used it and forced nothing.
#  * BATCHLAS_KERNEL_TRACE=1 costs ~60% of wall time (traced 1.562 ms vs a
#    970 us traced kernel duration for the same shape, trace/outer-native-pad0).
#    So timing runs are UNTRACED and identity is confirmed by separate short
#    traced runs.
#
# Same protocol as sweep.sh otherwise: each k listed twice, first pass
# discarded (JIT + clock ramp), --name=BM_GEMM_FIXED128 (substring match).
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu0/forced"
mkdir -p "$OUT"
GPU="${GPU:-0}"

run() { # tag kernel pad trace
    local tag=$1 kernel=$2 pad=$3 trace=$4
    local f="$OUT/${tag}-pad${pad}"
    local pre=(BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad")
    local iters=(--min_time=300 --min_iters=20 --max_iters=300)
    [ -n "$kernel" ] && pre+=(BATCHLAS_GEMM_SYCL_KERNEL="$kernel")
    if [ "$trace" = trace ]; then
        pre+=(BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$f.json")
        iters=(--min_time=50 --min_iters=3 --max_iters=5)
    fi
    env "${pre[@]}" GPU_GUARD_MAX_WAIT=1800 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
            "${iters[@]}" \
            --csv="$f.csv" 128 1024 128,256,128,256 512 > "$f.log" 2>&1
    echo "  $tag pad=$pad trace=$trace exit=$?"
}

# A. reproducibility spot-check of the inherited sweep, unforced, untraced.
for pad in 0 1 4 128; do run t-auto "" "$pad" notrace; done
# B. the same pads with the 128x128 kernel FORCED, untraced.
for pad in 0 1 2 4 128 384; do run t-force128 "128x128x8" "$pad" notrace; done
# C. identity confirmation only (short, traced) for the forced runs.
for pad in 0 1 4; do run id-force128 "128x128x8" "$pad" trace; done
echo "forced_variant complete"

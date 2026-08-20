#!/usr/bin/env bash
# What kernel did each arm ACTUALLY run? Trace only -- never timed (the trace
# inflates wall time ~60%).
set -uo pipefail
cd "$(dirname "$0")/../../.."
T=/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp4c
mkdir -p "$T"
GPU="${GPU:-0}"
# Which guard. experiments/gpu_guard.sh is the default and the strict one;
# GUARD=ctx swaps in gpu_guard_ctx.sh, which tolerates a foreign process that
# holds only an idle context on this card. See gpu_guard_ctx.sh.
GUARD_SH="${GUARD_SH:-./experiments/gpu_guard.sh}"
OUT=experiments/wp4_complex/gpu0/kernels.txt
touch "$OUT"

CASES=(
  "1024 1024 8   8"
  "1024 1024 32  8"
  "1024 1024 64  8"
  "1024 1024 96  8"
  "1024 1024 136 8"
  "992  992  32  8"
  "480  480  32  8"
  "128  128  128 8"
  "256  256  256 8"
  "512  512  512 8"
)

for type in cfloat cdouble; do
  for pad in 0 1 384; do
    for c in "${CASES[@]}"; do
      read -r m n k b <<< "$c"
      for cfg in auto wide; do
        if grep -q "^$type ${m}x${n}x${k} pad$pad $cfg -> " "$OUT"; then continue; fi
        kern=""
        [ "$cfg" = wide ] && kern=reg64x64k16wide
        f="$T/tr-$type-$m-$n-$k-pad$pad-$cfg.json"
        BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$f" \
        BATCHLAS_GEMM_ROUTE=native BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
        BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD="$pad" \
        GPU_GUARD_MAX_WAIT=3600 \
            "$GUARD_SH" "$GPU" ./build/benchmarks/gemm_benchmark \
                --backend=CUDA --type="$type" --name=BM_GEMM_FIXED128 \
                --min_time=1 --min_iters=1 --max_iters=1 --warmup=1 \
                "$m" "$n" "$k" "$b" > /dev/null 2>&1
        name=$(python3 experiments/wp4_complex/gpu0/trace_names.py "$f" \
                 | grep -v sycl_parallel_for | head -1 | awk '{print $2}')
        echo "$type ${m}x${n}x${k} pad$pad $cfg -> $name" | tee -a "$OUT"
      done
    done
  done
done

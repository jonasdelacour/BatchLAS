#!/usr/bin/env bash
# Does the relaxed guard change the numbers?
#
# Re-measures cells that sweep.sh already has under the STRICT gpu_guard.sh,
# this time under gpu_guard_ctx.sh (which tolerates a foreign idle context on
# the card). If the two disagree, the relaxed cells are not usable and this
# says so instead of quietly shipping them.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_complex/gpu0/raw_ctl"
mkdir -p "$OUT"
GPU="${GPU:-0}"

CASES=(
  "S_128   128  128 128 512"
  "P_k96  1024 1024  96 128"
  "S_256   256  256 256 256"
)

for rep in 1 2; do
for c in "${CASES[@]}"; do
  read -r tag m n k b <<< "$c"
  for cfg in auto wide; do
    f="$OUT/${tag}-${cfg}-b1-pad0-r${rep}"
    [ -s "$f.csv" ] && continue
    kern=""
    [ "$cfg" = wide ] && kern=reg64x64k16wide
    BATCHLAS_GEMM_ROUTE=native BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
    BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD=0 GPU_GUARD_MAX_WAIT=600 \
        ./experiments/wp4_complex/gpu0/gpu_guard_ctx.sh "$GPU" \
            ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=cfloat,cdouble --name=BM_GEMM_FIXED128 \
            --min_time=300 --min_iters=10 --max_iters=200 \
            --csv="$f.csv" "$m" "$n" "$k" "$b" > "$f.log" 2>&1
    echo "  ctl $tag $cfg rep=$rep exit=$?"
  done
done
done

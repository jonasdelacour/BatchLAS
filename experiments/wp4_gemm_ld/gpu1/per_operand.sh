#!/usr/bin/env bash
# WHICH operand's leading dimension costs the time?
#
# pad=384 puts ld=512 on a 128-row A and C, and ld=512 on a 128-row B (k=128).
# Applied one operand at a time, then in pairs, then all three, against pad=0.
# Also re-runs the all-zero and all-384 cells so the new binary is compared
# against itself, not against the previous binary's numbers.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu1"
GPU="${GPU:-1}"
run() { # padA padB padC tag
  BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA="${BETA:-1}" \
  BATCHLAS_BENCH_LD_PAD_A="$1" BATCHLAS_BENCH_LD_PAD_B="$2" BATCHLAS_BENCH_LD_PAD_C="$3" \
  GPU_GUARD_MAX_WAIT=1800 \
    ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
      --backend=CUDA --type=float --name=BM_GEMM --min_time=300 --min_iters=20 --max_iters=300 \
      --csv="$OUT/o-$4.csv" 128 1024 128 512 > "$OUT/o-$4.log" 2>&1
  echo "$4 exit=$?"
}
run 0   0   0   none
run 384 0   0   A
run 0   384 0   B
run 0   0   384 C
run 384 384 0   AB
run 384 0   384 AC
run 0   384 384 BC
run 384 384 384 ABC

#!/usr/bin/env bash
# Does the ld penalty follow the STRIDE or the total ALLOCATION FOOTPRINT?
#
# footprint = ((m+pad)*k + (k+pad)*n + (m+pad)*n) * batch * 4 B
#   pad=0   batch=512   ->  570 MB, all of it touched
#   pad=384 batch=512   -> 2280 MB allocated,  570 MB touched
#   pad=384 batch=128   ->  570 MB allocated,  143 MB touched
#   pad=0   batch=2048  -> 2280 MB, all of it touched
# Compare GFLOPS (not ms) since the batches differ.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_gemm_ld/gpu1"
GPU="${GPU:-1}"
run() { # pad batch tag
  BATCHLAS_GEMM_ROUTE=native BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD=$1 \
  GPU_GUARD_MAX_WAIT=1800 \
    ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
      --backend=CUDA --type=float --name=BM_GEMM --min_time=300 --min_iters=20 --max_iters=300 \
      --csv="$OUT/d-$3.csv" 128 1024 128 "$2" > "$OUT/d-$3.log" 2>&1
  echo "$3 exit=$?"
}
run 0   512  p0-b512
run 384 512  p384-b512
run 384 128  p384-b128
run 0   2048 p0-b2048
run 4   512  p4-b512
run 32  512  p32-b512
run 128 512  p128-b512
run 896 512  p896-b512

#!/usr/bin/env bash
# The two orders that were losing, with the trailing update now routed.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s16"
GPU="${GPU:-1}"
for route in vendor native; do
    BATCHLAS_TRSM_ROUTE="$route" GPU_GUARD_MAX_WAIT=5400 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/trsm_benchmark \
            --backend=CUDA --type=float --name=BM_TRSM_OrthoLeft \
            --min_time=200 --min_iters=10 --max_iters=200 \
            --csv="$OUT/quick-$route.csv" \
            256,512 1024,4096 128,512 > "$OUT/quick-$route.log" 2>&1
    echo "$route exit=$?"
done

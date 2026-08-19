#!/usr/bin/env bash
# The starvation profile -- PROFILE ONLY, NOT FOR RANKING.
#
# S10 asks for batch in {1,8,32} and q in {32,128} to be profiled but not
# ranked, because "profiling only at saturation is exactly what hid the
# batch-only-parallelism defect in this repo for months". Every number this
# produces is dominated by launch overhead; a ratio read off it is an overhead
# ratio and must not be quoted as an algorithm result.
#
# Runs on GPU 1 while the saturated sweep owns GPU 0. Both routes run on the
# SAME card, so the vendor/native comparison inside this file is consistent
# even though its absolute numbers are not comparable with the GPU-0 sweep's.
set -uo pipefail

cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s9"
BIN=./build/benchmarks/trsm_benchmark
GPU="${GPU:-1}"

for route in vendor native; do
    for side in Right Left; do
        echo "=== starved route=$route side=$side ==="
        BATCHLAS_TRSM_ROUTE="$route" \
            ./experiments/gpu_guard.sh "$GPU" "$BIN" \
                --backend=CUDA --min_time=100 --min_iters=10 --max_iters=100 \
                --name="BM_TRSM_Starved$side" \
                --csv="$OUT/starved-$(echo $side | tr A-Z a-z)-$route.csv" \
            2>&1 | tee "$OUT/starved-$(echo $side | tr A-Z a-z)-$route.log"
    done
done
echo "starved profile complete"

#!/usr/bin/env bash
# WP3 step 9 -- the ortho-shaped TRSM grid, vendor vs native.
#
# Two runs of the same binary over the same grid, differing ONLY in
# BATCHLAS_TRSM_ROUTE. That variable is the whole experiment, so the benchmark
# prints how it parsed the value; a run whose header does not say
# "trsm route forced: ..." measured the wrong thing and its CSV must be thrown
# away rather than compared.
#
# One GPU, guarded, for the whole sweep -- the second 4090 in this box is a
# co-tenant and a foreign process on the card silently inflates every row.
set -uo pipefail

cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s9"
BIN=./build/benchmarks/trsm_benchmark
GPU="${GPU:-0}"

# --min_time bounds the largest cells (n=256,q=4096,batch=128 is ~500 MB of
# traffic per iteration); the harness still honours its own warmup, which is
# what keeps a cold JIT out of the first timed iteration.
COMMON=(--backend=CUDA --min_time=200 --min_iters=10 --max_iters=200)

run_one() {
    local route="$1" name="$2" tag="$3"
    echo "=== route=$route name=$name ==="
    BATCHLAS_TRSM_ROUTE="$route" \
        ./experiments/gpu_guard.sh "$GPU" "$BIN" "${COMMON[@]}" \
            --name="$name" --csv="$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
}

for route in vendor native; do
    run_one "$route" BM_TRSM_OrthoRight     "right-$route"
    run_one "$route" BM_TRSM_OrthoLeft      "left-$route"
done

echo "sweep complete"

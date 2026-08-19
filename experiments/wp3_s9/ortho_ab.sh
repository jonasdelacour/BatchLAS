#!/usr/bin/env bash
# WP3 step 9 -- the END-TO-END half of the measurement.
#
# S10: "Validate every routing flip end-to-end through ortho, never at the
# kernel level alone. A prior 2.16x kernel win in this repo turned into an 11%
# gesvd loss." So this runs the real caller with the two routes and compares
# the CALLER's time, not the kernel's.
#
# THE ARGUMENT GRID IS NOT THE REGISTERED ONE, on purpose. ortho's trsm has
# triangular order = the COLUMN count n (the orthonormalisation block size) and
# q = the ROW count m. OrthoBenchSizes only emits n >= 64, so the registered
# grid would exercise the blocked driver exclusively and never once reach the
# CTA kernel -- the tier the coverage capture says the real callers actually hit
# (n = 5..12). n=16 and n=32 below are what put V1 on the path at all.
#
# A coverage file is written alongside each run. ortho_benchmark does not print
# the trsm route, so without it "native" and "vendor" would be indistinguishable
# from the outside and two identical numbers would read as a null result rather
# than as a failed A/B. compare the chosen_origin columns before believing any
# ratio out of this script.
set -uo pipefail

cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s9"
BIN=./build/benchmarks/ortho_benchmark
GPU="${GPU:-0}"

#      m           n                 batch     algo (0=Chol2 default, 2=ShiftChol3)
ARGS=( 1024,4096   16,32,64,128,256  128,512   0,2 )

for route in vendor native; do
    echo "=== ortho end-to-end, route=$route ==="
    BATCHLAS_TRSM_ROUTE="$route" \
    BATCHLAS_COVERAGE_OUT="$(pwd)/$OUT/ortho-$route.coverage.csv" \
        ./experiments/gpu_guard.sh "$GPU" "$BIN" \
            --backend=CUDA --min_time=200 --min_iters=5 --max_iters=50 \
            --csv="$OUT/ortho-$route.csv" "${ARGS[@]}" \
        2>&1 | tee "$OUT/ortho-$route.log"
done

echo "ortho A/B complete"
echo "--- trsm routes actually taken ---"
# NOTE THE GLOB. coverage.cc writes one shard PER PROCESS, named
# "$BATCHLAS_COVERAGE_OUT.<pid>" -- the bare path never exists. An earlier
# version of this loop tested `[ -f "$OUT/ortho-$route.coverage.csv" ]`, which
# is always false, so it printed nothing and, being the last command, exited 1
# on a run whose measurements had all succeeded.
for route in vendor native; do
    printf '%s: ' "$route"
    cat "$OUT/ortho-$route.coverage.csv."* 2>/dev/null \
        | grep ',trsm,' | cut -d, -f10,11 | sort -u | tr '\n' ' '
    printf '\n'
done

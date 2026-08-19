#!/usr/bin/env bash
# WP3 step 12 -- the END-TO-END check for the side the staging tile fixed.
#
# The step-9 ortho A/B validated Side::Right only, because ortho_benchmark
# hardcoded Transpose::NoTrans and ortho.cc:205,289 pick the side from exactly
# that flag. So the float Side::Left window -- the one this step widened from
# order 16 to order 128 -- had never been measured at caller level at all. arg4
# now selects it.
#
# TRANSPOSED ORIENTATION. With transA = Trans, ortho orthonormalises the ROW
# vectors: the triangular order is the ROW count m and the vector length is the
# COLUMN count n, the exact mirror of the NoTrans case. So m is swept over the
# orders that moved (16..256) and n is the long extent.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s12"
GPU="${GPU:-1}"

exec 9>"$OUT/.ortho.lock"
flock -n 9 || { echo "another ortho A/B holds the lock"; exit 3; }

#      m(order)          n(len)     batch     algo   trans=1 -> Side::Left
ARGS=( 16,32,64,128,256  1024,4096  128,512   0,2    1 )

for route in vendor native; do
    echo "=== ortho Side::Left, route=$route ==="
    BATCHLAS_TRSM_ROUTE="$route" GPU_GUARD_MAX_WAIT=5400 \
    BATCHLAS_COVERAGE_OUT="$(pwd)/$OUT/ortho-left-$route.coverage.csv" \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/ortho_benchmark \
            --backend=CUDA --min_time=200 --min_iters=5 --max_iters=50 \
            --csv="$OUT/ortho-left-$route.csv" "${ARGS[@]}" \
        > "$OUT/ortho-left-$route.log" 2>&1
    echo "  $route exit=$?"
    grep -q "WARNING -- foreign process" "$OUT/ortho-left-$route.log" && {
        echo "  *** CONTAMINATED -- discarding"; rm -f "$OUT/ortho-left-$route.csv"; }
done

echo "--- trsm routes actually taken (glob: coverage writes one shard per pid) ---"
for route in vendor native; do
    printf '%s: ' "$route"
    cat "$OUT/ortho-left-$route.coverage.csv."* 2>/dev/null \
        | grep ',trsm,' | cut -d, -f10,11 | sort -u | tr '\n' ' '
    printf '\n'
done
echo "ortho Side::Left A/B finished"

#!/usr/bin/env bash
# Crossover sweep for the SYR2K triangular-output route. Runs each route at each
# shape through the GPU guard and prints "route n k batch avg_ms stddev_ms".
#
#     experiments/syr2k_sweep.sh 0 shapes.txt
#
# where shapes.txt has one "n k batch" triple per line. The GFLOPS column of the
# benchmark is not comparable across routes here (it counts syr2k flops for all
# of them), so compare avg_ms. The name column is quoted and holds a comma of its
# own, which is why the timing fields are read one further right than the header.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?usage: syr2k_sweep.sh <gpu> <shapes-file>}"
SHAPES="${2:?usage: syr2k_sweep.sh <gpu> <shapes-file>}"
BIN="${SYR2K_BENCH:-./build/benchmarks/syr2k_benchmark}"
ROUTES="${SYR2K_ROUTES:-vendor gemm triangular}"
CSV=$(mktemp)

printf 'route\tn\tk\tbatch\tms\tstd\n'
for route in $ROUTES; do
    while read -r n k batch; do
        [ -z "${n:-}" ] && continue
        BATCHLAS_SYR2K_VARIANT="$route" experiments/gpu_guard.sh "$GPU" \
            "$BIN" --backend=CUDA --type=float --warmup=10 --warmup_internal=3 \
            --min_iters=30 --csv="$CSV" "$n" "$k" "$n" "$batch" >/dev/null 2>&1 \
            || { printf '%s\t%s\t%s\t%s\tGUARD-FAIL\t\n' "$route" "$n" "$k" "$batch"; continue; }
        awk -F, -v r="$route" -v n="$n" -v k="$k" -v b="$batch" \
            'NR==2 {printf "%s\t%s\t%s\t%s\t%s\t%s\n", r, n, k, b, $8, $9}' "$CSV"
    done < "$SHAPES"
done
rm -f "$CSV"

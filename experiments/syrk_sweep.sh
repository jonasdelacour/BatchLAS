#!/usr/bin/env bash
# Crossover sweep for the SYRK triangular-output route. Runs each route at each
# shape through the GPU guard and prints "route n batch avg_ms stddev_ms".
#
#     experiments/syrk_sweep.sh 0 shapes.txt
#
# where shapes.txt has one "n batch" pair per line. The GFLOPS column of the
# benchmark is not comparable across routes here (it counts syrk flops for all
# of them), so compare avg_ms. The name column is quoted and holds a comma of its
# own, which is why the timing fields are read one further right than the header.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?usage: syrk_sweep.sh <gpu> <shapes-file>}"
SHAPES="${2:?usage: syrk_sweep.sh <gpu> <shapes-file>}"
BIN="${SYRK_BENCH:-./build/benchmarks/syrk_benchmark}"
ROUTES="${SYRK_ROUTES:-vendor cublasdx gemm triangular}"
CSV=$(mktemp)

printf 'route\tn\tbatch\tms\tstd\n'
for route in $ROUTES; do
    while read -r n batch; do
        [ -z "${n:-}" ] && continue
        BATCHLAS_SYRK_VARIANT="$route" experiments/gpu_guard.sh "$GPU" \
            "$BIN" --backend=CUDA --type=float --warmup=10 --warmup_internal=3 \
            --min_iters=30 --csv="$CSV" "$n" "$n" "$n" "$batch" >/dev/null 2>&1 \
            || { printf '%s\t%s\t%s\tGUARD-FAIL\t\n' "$route" "$n" "$batch"; continue; }
        awk -F, -v r="$route" -v n="$n" -v b="$batch" \
            'NR==2 {printf "%s\t%s\t%s\t%s\t%s\n", r, n, b, $8, $9}' "$CSV"
    done < "$SHAPES"
done
rm -f "$CSV"

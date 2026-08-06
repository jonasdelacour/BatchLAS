#!/usr/bin/env bash
# Same as syrk_sweep.sh but with k free, to check that the crossover really is
# insensitive to the reduction depth. Shapes file holds "n k batch" per line.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?usage: syrk_kskew.sh <gpu> <shapes-file>}"
SHAPES="${2:?usage: syrk_kskew.sh <gpu> <shapes-file>}"
BIN="${SYRK_BENCH:-./build/benchmarks/syrk_benchmark}"
ROUTES="${SYRK_ROUTES:-cublasdx triangular}"
CSV=$(mktemp)

printf 'route\tn\tk\tbatch\tms\tstd\n'
for route in $ROUTES; do
    while read -r n k batch; do
        [ -z "${n:-}" ] && continue
        BATCHLAS_SYRK_VARIANT="$route" experiments/gpu_guard.sh "$GPU" \
            "$BIN" --backend=CUDA --type=float --warmup=10 --warmup_internal=3 \
            --min_iters=30 --csv="$CSV" "$n" "$k" "$n" "$batch" >/dev/null 2>&1 \
            || { printf '%s\t%s\t%s\t%s\tGUARD-FAIL\t\n' "$route" "$n" "$k" "$batch"; continue; }
        awk -F, -v r="$route" -v n="$n" -v k="$k" -v b="$batch" \
            'NR==2 {printf "%s\t%s\t%s\t%s\t%s\t%s\n", r, n, k, b, $8, $9}' "$CSV"
    done < "$SHAPES"
done
rm -f "$CSV"

#!/usr/bin/env bash
# The canonical saturation ladder for one (op, type), one process per cell,
# every cell through gpu_guard.sh on GPU 1.
#
#     benchmarks/run_factor_grid.sh potrf float benchmarks/results/factor_baseline_potrf_float.csv
#
# ONE PROCESS PER CELL is load-bearing, not tidiness: the SLM carve-out
# attribute is sticky per CUfunction, so an earlier, larger launch in the same
# process can make a later launch that should have failed succeed. It is also
# what makes the resolved-route readback below work at all -- dispatch coverage
# is emitted from an atexit handler, so one cell per process is exactly one
# coverage file per cell.
#
# THREE BATCHES PER ORDER (half, nominal, double). A single batch cannot tell a
# real ratio from an unsaturated one: below saturation both arms are measuring
# launch overhead and the ratio is meaningless (docs/perf/README.md). The
# nominal column is the SAT_LADDER value from docs/perf/lu.md.
#
# Output: the harness's own CSV columns plus `resolved_route`, which is read
# back per arm from BATCHLAS_COVERAGE_OUT and is the ONLY column that says what
# actually ran. The harness's `pin_parsed` says the pin was UNDERSTOOD; an
# unsupported pin still falls through to automatic() with pin_parsed=1.
#
# env:
#   FACTOR_BENCH  path to the factor_bench binary (required unless it is found
#                 in one of the usual build directories)
#   GPU           GPU index to pin (default 1)
#   REPS          timed reps per cell (default 7)
#   ORDERS        override the order ladder
set -uo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <op> <type> <out.csv>" >&2
    echo "  op   : potrf getrf getrs geqrf orgqr" >&2
    echo "  type : float double cfloat cdouble" >&2
    exit 2
fi

OP="$1"; TYPE="$2"; OUT="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
GPU="${GPU:-1}"
REPS="${REPS:-7}"
GUARD="$HERE/gpu_guard.sh"

BENCH="${FACTOR_BENCH:-}"
if [ -z "$BENCH" ]; then
    for cand in \
        "$ROOT/build/presets/benchmarks/benchmarks/factor_bench" \
        "$ROOT/build/presets/dev-tests/benchmarks/factor_bench" \
        "$ROOT/build/benchmarks/factor_bench"; do
        [ -x "$cand" ] && { BENCH="$cand"; break; }
    done
fi
if [ ! -x "${BENCH:-/nonexistent}" ]; then
    echo "run_factor_grid: no factor_bench binary; set FACTOR_BENCH=<path>" >&2
    exit 2
fi

# The order ladder. Powers of two AND their +1 neighbours, because a kernel that
# is tuned on 2^k and falls off a cliff at 2^k+1 is this repository's recurring
# small-n defect (docs/design/small-n-factorization-plan.md, section 1.3).
ORDERS="${ORDERS:-4 8 9 16 17 24 32 33 48 64 65 96 128 129 192 256 384 512}"

# SAT_LADDER from docs/perf/lu.md: the batch at which the order saturates the
# card. The script measures half and double as well, so an unsaturated reading
# is visible rather than assumed away.
nominal_batch() {
    local n="$1"
    if   [ "$n" -le 24 ];  then echo 16384
    elif [ "$n" -le 64 ];  then echo 8192
    elif [ "$n" -le 128 ]; then echo 4096
    elif [ "$n" -le 256 ]; then echo 2048
    else                        echo 512
    fi
}

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

printf 'op,type,m,n,nrhs,batch,arm,pin,pin_parsed,median_ms,mean_ms,rel_sd,reps,residual,extra_check,info_nonzero,bad,reason,resolved_route\n' > "$OUT"

# The resolved route for ONE arm of ONE cell: an extra, untimed run with dispatch
# coverage on. The coverage instrument writes $BATCHLAS_COVERAGE_OUT.<pid> from
# atexit, one file per process, and the `reached,` rows are what the run actually
# selected -- fields 10 and 11 are chosen_origin and chosen_algo. `linked` rows
# are NOT evidence that a route ran (docs/design/vendor-free-status.md).
readback_route() {
    local m="$1" n="$2" nrhs="$3" batch="$4" arm="$5"
    local base="$TMPD/cov_${arm}"
    rm -f "$base".* 2>/dev/null
    WARM_S=0 BATCHLAS_COVERAGE_OUT="$base" \
        "$GUARD" "$GPU" "$BENCH" "$OP" "$TYPE" "$m" "$n" "$nrhs" "$batch" 1 \
        --arms="$arm" >/dev/null 2>&1
    local row
    row="$(cat "$base".* 2>/dev/null | awk -F, -v op="$OP" '$1=="reached" && $2==op {print $10":"$11}' | sort -u | paste -sd'|' -)"
    [ -z "$row" ] && row="unknown"
    printf '%s' "$row"
}

run_cell() {
    local m="$1" n="$2" nrhs="$3" batch="$4"
    local cell="$TMPD/cell.csv"
    rm -f "$cell"
    "$GUARD" "$GPU" "$BENCH" "$OP" "$TYPE" "$m" "$n" "$nrhs" "$batch" "$REPS" \
        --csv="$cell" >/dev/null
    local rc=$?
    if [ "$rc" -eq 5 ]; then
        echo "run_factor_grid: GPU contaminated during ${OP}/${TYPE} ${m}x${n} nrhs=${nrhs} batch=${batch}; cell DISCARDED" >&2
        return 0
    fi
    if [ ! -s "$cell" ]; then
        echo "run_factor_grid: no output for ${OP}/${TYPE} ${m}x${n} nrhs=${nrhs} batch=${batch} (rc=$rc)" >&2
        return 0
    fi
    local rv rn
    rv="$(readback_route "$m" "$n" "$nrhs" "$batch" vendor)"
    rn="$(readback_route "$m" "$n" "$nrhs" "$batch" native)"
    awk -F, -v OFS=, -v rv="$rv" -v rn="$rn" \
        'NR>1 { print $0, ($7=="vendor" ? rv : rn) }' "$cell" >> "$OUT"
    tail -n +2 "$cell" >&2
}

case "$OP" in
    getrs) NRHS_LIST="1 4" ;;
    *)     NRHS_LIST="0" ;;
esac

for n in $ORDERS; do
    nb="$(nominal_batch "$n")"
    for batch in $((nb / 2)) "$nb" $((nb * 2)); do
        for nrhs in $NRHS_LIST; do
            run_cell "$n" "$n" "$nrhs" "$batch"
        done
    done
done

# The tall panels. They are not decoration: geqrf's real callers in this tree
# issue tall panels (band_reduction.cc, sytrd_sy2sb.cc), and every square cell
# above is blind to them.
if [ "$OP" = "geqrf" ] || [ "$OP" = "orgqr" ]; then
    for cell in "128 32" "512 32" "512 64" "1024 128"; do
        set -- $cell
        m="$1"; n="$2"
        nb="$(nominal_batch "$n")"
        for batch in $((nb / 2)) "$nb" $((nb * 2)); do
            run_cell "$m" "$n" 0 "$batch"
        done
    done
fi

echo "run_factor_grid: wrote $OUT" >&2

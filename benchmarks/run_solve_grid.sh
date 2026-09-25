#!/usr/bin/env bash
# P2's saturation grid for the FUSED solve ops (gesv, posv), one process per
# cell, every cell through gpu_guard.sh.
#
#     benchmarks/run_solve_grid.sh gesv float benchmarks/results/p2_gesv_float.csv
#
# Separate from run_factor_grid.sh for one reason: these ops have THREE arms,
# not two, and two of them are the same route. gesv's `Blocked` route is the
# composition `getrf; getrs`, so "vendor" and "native" here differ only in what
# the composed sub-ops are pinned to (factor_bench's composed_pins). The third
# arm, `tiny`, is the fused kernel. A two-arm script cannot express that, and
# collapsing it would measure whatever the sub-ops happened to route to.
#
# Everything else follows run_factor_grid.sh: one process per cell (the sticky
# SLM carve-out), a batch ladder around the nominal so an unsaturated reading is
# visible rather than assumed, and a resolved-route readback per arm, because
# `pin_parsed` only says the pin was UNDERSTOOD -- an unsupported pin still
# falls through to automatic() with pin_parsed=1.
#
# env: FACTOR_BENCH, GPU (default 1), REPS (default 7), ORDERS, NRHS_LIST,
#      BATCHES, ARMS, NO_ROUTE_READBACK=1 to skip the (slow) coverage runs.
set -uo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <gesv|posv> <type> <out.csv>" >&2
    exit 2
fi

OP="$1"; TYPE="$2"; OUT="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
GPU="${GPU:-1}"
REPS="${REPS:-7}"
GUARD="$HERE/gpu_guard.sh"
ARMS="${ARMS:-tiny,vendor,native}"

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
    echo "run_solve_grid: no factor_bench binary; set FACTOR_BENCH=<path>" >&2
    exit 2
fi

# The tier caps at 32 and the non-power-of-two orders are in the ladder for the
# reason section 1.3 of the plan gives: a kernel tuned on 2^k that falls off a
# cliff at 2^k+1 is this repository's recurring small-n defect.
ORDERS="${ORDERS:-4 8 9 16 17 24 32}"
NRHS_LIST="${NRHS_LIST:-1 4}"
BATCHES="${BATCHES:-8192 16384 32768}"

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

printf 'op,type,m,n,nrhs,batch,arm,pin,pin_parsed,median_ms,mean_ms,rel_sd,reps,residual,extra_check,info_nonzero,bad,reason,resolved_route\n' > "$OUT"

readback_route() {
    local n="$1" nrhs="$2" batch="$3" arm="$4"
    [ -n "${NO_ROUTE_READBACK:-}" ] && { printf 'skipped'; return; }
    local base="$TMPD/cov_${arm}"
    rm -f "$base".* 2>/dev/null
    WARM_S=0 BATCHLAS_COVERAGE_OUT="$base" \
        "$GUARD" "$GPU" "$BENCH" "$OP" "$TYPE" "$n" "$n" "$nrhs" "$batch" 1 \
        --arms="$arm" >/dev/null 2>&1
    local row
    row="$(cat "$base".* 2>/dev/null \
        | awk -F, '$1=="reached" {print $2":"$10":"$11}' | sort -u | paste -sd'|' -)"
    [ -z "$row" ] && row="unknown"
    printf '%s' "$row"
}

run_cell() {
    local n="$1" nrhs="$2" batch="$3"
    local cell="$TMPD/cell.csv"
    rm -f "$cell"
    "$GUARD" "$GPU" "$BENCH" "$OP" "$TYPE" "$n" "$n" "$nrhs" "$batch" "$REPS" \
        --arms="$ARMS" --csv="$cell" >/dev/null
    local rc=$?
    if [ "$rc" -eq 5 ]; then
        echo "run_solve_grid: GPU contaminated during ${OP}/${TYPE} n=${n} nrhs=${nrhs} batch=${batch}; cell DISCARDED" >&2
        return 0
    fi
    if [ ! -s "$cell" ]; then
        echo "run_solve_grid: no output for ${OP}/${TYPE} n=${n} nrhs=${nrhs} batch=${batch} (rc=$rc)" >&2
        return 0
    fi
    local names=() routes=()
    IFS=',' read -r -a names <<< "$ARMS"
    local i=0
    for a in "${names[@]}"; do
        routes[$i]="$(readback_route "$n" "$nrhs" "$batch" "$a")"
        i=$((i + 1))
    done
    local joined=""
    i=0
    for a in "${names[@]}"; do
        joined="${joined}${a}=${routes[$i]};"
        i=$((i + 1))
    done
    awk -F, -v OFS=, -v map="$joined" '
        NR>1 {
            r="unknown"
            nk=split(map, kv, ";")
            for (j = 1; j <= nk; j++) {
                p = index(kv[j], "=")
                if (p > 0 && substr(kv[j], 1, p-1) == $7) r = substr(kv[j], p+1)
            }
            print $0, r
        }' "$cell" >> "$OUT"
    tail -n +2 "$cell" >&2
}

for n in $ORDERS; do
    for batch in $BATCHES; do
        for nrhs in $NRHS_LIST; do
            run_cell "$n" "$nrhs" "$batch"
        done
    done
done

echo "run_solve_grid: wrote $OUT" >&2

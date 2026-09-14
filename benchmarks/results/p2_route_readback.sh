#!/usr/bin/env bash
# P2: what Auto actually resolves to at the window edges, after the flip.
# Reads the dispatch coverage instrument, not the timing: a route change is only
# proven by a `reached` row. evidence: docs/perf/lu.md#p2-the-measured-gesv-window
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="$ROOT/build/presets/dev-tests/benchmarks/factor_bench"
GUARD="$ROOT/benchmarks/gpu_guard.sh"

probe() {
    local op="$1" ty="$2" n="$3"
    rm -f /tmp/p2cov.*
    WARM_S=0 BATCHLAS_COVERAGE_OUT=/tmp/p2cov \
        "$GUARD" 1 "$BENCH" "$op" "$ty" "$n" "$n" 1 8192 1 --arms=auto >/dev/null 2>&1
    printf '%-6s %-8s n=%-3s -> ' "$op" "$ty" "$n"
    cat /tmp/p2cov.* 2>/dev/null \
        | awk -F, -v op="$op" '$1=="reached" && $2==op {print $10":"$11}' \
        | sort -u | paste -sd'|' -
}

probe gesv float 32
probe gesv float 33
probe gesv cfloat 16
probe gesv cfloat 17
probe gesv double 16
probe gesv cdouble 16
probe posv float 32
probe posv float 33
probe posv cfloat 32
probe posv double 32
probe posv cdouble 16
probe posv cdouble 17

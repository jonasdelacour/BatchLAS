#!/usr/bin/env bash
#
# Prove a change did not move any dispatch decision.
#
# WHY THIS EXISTS. WP1 retargets the terminal GEMM of the level-3 expand/tile
# routes at the public entry point. On a vendor-present box that MUST be a
# no-op: same route chosen for every shape any test reaches. Reading the diff
# cannot establish that -- the whole point of routing through a resolver is that
# the decision is not visible at the call site -- and timing cannot establish it
# either, because an unsaturated benchmark's ratios are overhead, not algorithm.
#
# So compare the DECISION. A coverage build records, per (op, scalar,
# shape_class), which Route the resolver returned. Two runs, one before and one
# after, must produce byte-identical `reached` rows. Anything else is a route
# change, and a route change has to be argued for rather than discovered later.
#
# USAGE
#   scripts/route_diff.sh capture <build-dir> <label>
#   scripts/route_diff.sh compare <label-a> <label-b>
#
# Any ordinary build works -- recording is gated at runtime on
# BATCHLAS_COVERAGE_OUT, which this script sets. No special configuration.
#
# What this script WILL NOT do is report success on an empty measurement. A
# capture with zero `reached` rows is treated as a hard error, not as "nothing
# changed". That is not hypothetical: the instrument has already produced a
# file with a correct header and no `reached` rows twice, for two unrelated
# reasons (a destroyed static, then a weak-symbol interposition), and both times
# it looked exactly like a clean result.

set -uo pipefail

STORE="${BATCHLAS_ROUTE_DIFF_STORE:-.route-diff}"

die() { printf 'route_diff: %s\n' "$*" >&2; exit 1; }

capture() {
    local build_dir="$1" label="$2"
    [[ -d "$build_dir" ]] || die "no such build dir: $build_dir"

    mkdir -p "$STORE"
    local raw="$STORE/$label.csv"

    # Tests are EXPECTED to fail in a vendor-free build; a non-zero ctest is not
    # a capture failure. What matters is that the file appears and has rows.
    BATCHLAS_COVERAGE_OUT="$(cd "$(dirname "$raw")" && pwd)/$(basename "$raw")" \
        ctest --test-dir "$build_dir" -LE slow >"$STORE/$label.ctest.log" 2>&1
    local ctest_status=$?

    [[ -s "$raw" ]] || die "no coverage file written to $raw (BATCHLAS_COVERAGE_OUT unread?)"

    local reached
    reached=$(grep -c '^reached,' "$raw" || true)
    [[ "$reached" -gt 0 ]] || die "coverage file has 0 'reached' rows.
        The dynamic half is not recording. Check that resolve_route still calls
        coverage::record_if_enabled, and that coverage.cc's g_dynamic_enabled
        initialiser saw BATCHLAS_COVERAGE_OUT."

    # Normalise for comparison: keep only the routing decision, drop the call
    # COUNT. Counts vary with test scheduling and iteration and are not part of
    # the decision; including them would make every comparison fail for reasons
    # that are not route changes.
    #   kind,op,scalar,backend,shape_class,m,n,k,batch,origin,algo,calls,...
    #   1    2  3      4       5           6 7 8 9     10     11   12
    grep '^reached,' "$raw" \
        | awk -F, '{print $1","$2","$3","$4","$5","$10","$11","$13","$14}' \
        | sort -u > "$STORE/$label.routes"

    printf 'captured %s: %s reached rows -> %s distinct decisions (ctest exit %s)\n' \
        "$label" "$reached" "$(wc -l < "$STORE/$label.routes")" "$ctest_status"
}

compare() {
    local a="$1" b="$2"
    for l in "$a" "$b"; do
        [[ -s "$STORE/$l.routes" ]] || die "no capture named '$l' (run capture first)"
    done

    if diff -u "$STORE/$a.routes" "$STORE/$b.routes" > "$STORE/$a-vs-$b.diff"; then
        printf 'IDENTICAL: every decision in %s matches %s (%s decisions)\n' \
            "$a" "$b" "$(wc -l < "$STORE/$a.routes")"
        return 0
    fi

    printf 'ROUTE CHANGE between %s and %s:\n\n' "$a" "$b"
    cat "$STORE/$a-vs-$b.diff"
    printf '\nA route change is not automatically wrong -- but it is a decision, and it
has to be argued for and measured, not discovered. Full diff: %s\n' "$STORE/$a-vs-$b.diff"
    return 1
}

case "${1:-}" in
    capture) [[ $# -eq 3 ]] || die "usage: $0 capture <build-dir> <label>"; capture "$2" "$3" ;;
    compare) [[ $# -eq 3 ]] || die "usage: $0 compare <label-a> <label-b>"; compare "$2" "$3" ;;
    *) die "usage: $0 {capture <build-dir> <label> | compare <label-a> <label-b>}" ;;
esac

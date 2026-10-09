#!/bin/sh
# Run every CI check locally. Same scripts the workflow runs, no build required.
#
#   .github/ci/run_local_checks.sh                 # what CI runs
#   .github/ci/run_local_checks.sh /tmp/inst       # ...plus the installed package
#   .github/ci/run_local_checks.sh /tmp/inst ctest.xml gtest-xml/
#                                                  # ...plus a real ctest run
#
# The first optional argument is an install prefix (cmake --install build
# --prefix /tmp/inst); passing it enables the check that CI cannot run, because
# CI has no SYCL compiler and so cannot generate the export at all.
#
# The second and third are a ctest --output-junit report and the directory a run
# under GTEST_OUTPUT=xml:<dir>/ wrote. Passing them gates that run against
# tests/known-failures.txt. Only these two arguments involve a build at all;
# everything above still runs on a bare checkout. The documentation site is
# built into build/docs when Doxygen 1.18+ is available, and skipped otherwise.
#
# BATCHLAS_BUILD_DIR=build .github/ci/run_local_checks.sh
#                                                  # ...plus the device-call scan
# of that build's NVPTX images (scripts/check_device_calls.py). CI has no SYCL
# build, so this too runs only locally; it skips when the build has no CUDA target.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
status=0

run() {
    printf '\n== %s\n' "$*"
    "$@" || status=1
}

run python3 "$here/check_cmake_syntax.py"
run python3 "$here/check_exported_package.py"
run python3 "$here/check_public_headers.py"
run python3 "$here/check_no_unprefixed_includes.py"
run python3 "$here/check_no_global_names.py"
# The self-test runs first and separately so that a broken counter is
# distinguishable from a genuine offender list. Without it, a scan that reports
# nothing is indistinguishable from a scan that cannot count.
run python3 "$here/check_comment_density.py" --self-test
run python3 "$here/check_comment_density.py"
# Same self-test-first reason: an empty finding list must be distinguishable
# from a walk that read nothing.
run python3 "$here/check_evidence_anchors.py" --self-test
run python3 "$here/check_evidence_anchors.py"
# Checks the INDEX, so it catches a raw benchmark file before it is committed, not after.
run python3 "$here/check_lfs_pointers.py" --self-test
run python3 "$here/check_lfs_pointers.py"
# Stale tuned tables are a warning (flat-kernel-selection.md §6.5): the check always exits 0;
# only its self-test can fail.
run python3 "$here/check_tuned_tables.py" --self-test
run python3 "$here/check_tuned_tables.py"
# Plans and write-ups outside docs/ are what the documentation site replaced.
run python3 "$here/check_markdown_locations.py" --self-test
run python3 "$here/check_markdown_locations.py"

# The docs build. The generator's self-test needs no Doxygen; the site does,
# and only 1.18+ (MARKDOWN_ID_STYLE=GITHUB keeps evidence anchors live links),
# so an older or missing doxygen is a skip, not a failure. CI's docs job pins
# 1.18.0. Set DOXYGEN=/path/to/doxygen to use one off PATH, and
# BATCHLAS_DOCS_STRICT=1 to fail on warnings as CI does.
root=$(CDPATH= cd -- "$here/../.." && pwd)
run python3 "$root/docs/tools/gen_db_pages.py" --self-test
_doxygen=${DOXYGEN:-doxygen}
_dver=$("$_doxygen" --version 2>/dev/null | sed 's/[^0-9.].*//') || _dver=""
_dmaj=$(echo "${_dver:-0}" | cut -d. -f1)
_dmin=$(echo "${_dver:-0.0}.0" | cut -d. -f2)
if [ -n "$_dver" ] && { [ "$_dmaj" -gt 1 ] || { [ "$_dmaj" -eq 1 ] && [ "$_dmin" -ge 18 ]; }; }; then
    run env DOXYGEN="$_doxygen" sh "$root/scripts/build_docs.sh" "$root/build/docs"
else
    printf '\n== docs build SKIPPED: %s\n' \
        "no Doxygen 1.18+ ('$_doxygen' reports '${_dver:-nothing}'); set DOXYGEN=/path/to/doxygen"
fi
if [ "$#" -gt 0 ]; then
    run python3 "$here/check_exported_package.py" --package "$1"
fi
if [ -n "${BATCHLAS_BUILD_DIR:-}" ]; then
    printf '\n== device-call scan of %s\n' "$BATCHLAS_BUILD_DIR"
    _rc=0
    python3 "$here/../../scripts/check_device_calls.py" "$BATCHLAS_BUILD_DIR" || _rc=$?
    # 77 = nothing to scan (CPU-only build or no cuobjdump), not a failure.
    if [ "$_rc" -ne 0 ] && [ "$_rc" -ne 77 ]; then
        status=1
    fi
fi
if [ "$#" -gt 1 ]; then
    # consumer_package_tests is 'slow'-labelled, so the normal local loop --
    # `ctest -LE slow`, which tests/README.md documents and the PR job uses --
    # never registers it. Requiring it unconditionally made this wrapper report
    # FAILED on the standard workflow, which is how a pre-push script stops
    # being run at all.
    #
    # So the caller declares what kind of run they made. Pass `full` as the 4th
    # argument after a complete `ctest` (no -LE slow); only then is a skipped
    # packaging test a failure rather than an absence.
    _require=""
    if [ "${4:-}" = "full" ]; then
        _require="--require consumer_package_tests"
    fi
    # An AdaptiveCpp tree is gated against tests/known-failures-acpp.txt.
    if [ -n "${BATCHLAS_BUILD_DIR:-}" ]; then
        _impl=$(sed -n 's/^BATCHLAS_SYCL_IMPL_RESOLVED:INTERNAL=//p' "$BATCHLAS_BUILD_DIR/CMakeCache.txt" 2>/dev/null)
        if [ -n "$_impl" ]; then
            _require="$_require --sycl-impl $_impl"
        fi
    fi
    if [ "$#" -gt 2 ] && [ -n "$3" ]; then
        # shellcheck disable=SC2086  # _require is a deliberate word split
        run python3 "$here/compare_failures.py" --junit "$2" --gtest-dir "$3" $_require
    else
        # shellcheck disable=SC2086
        run python3 "$here/compare_failures.py" --junit "$2" $_require
    fi
fi

printf '\n'
if [ "$status" -eq 0 ]; then
    echo "all checks passed"
else
    echo "checks FAILED"
fi
exit "$status"

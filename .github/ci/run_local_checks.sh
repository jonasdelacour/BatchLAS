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
# everything above still runs on a bare checkout.
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
if [ "$#" -gt 0 ]; then
    run python3 "$here/check_exported_package.py" --package "$1"
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

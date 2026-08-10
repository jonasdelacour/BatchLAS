#!/bin/sh
# Run every CI check locally. Same scripts the workflow runs, no build required.
#
#   .github/ci/run_local_checks.sh                 # what CI runs
#   .github/ci/run_local_checks.sh /tmp/inst       # ...plus the installed package
#
# The optional argument is an install prefix (cmake --install build --prefix
# /tmp/inst); passing it enables the check that CI cannot run, because CI has no
# SYCL compiler and so cannot generate the export at all.
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
if [ "$#" -gt 0 ]; then
    run python3 "$here/check_exported_package.py" --package "$1"
fi

printf '\n'
if [ "$status" -eq 0 ]; then
    echo "all checks passed"
else
    echo "checks FAILED"
fi
exit "$status"

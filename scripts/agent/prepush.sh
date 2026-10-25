#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Everything the hosted CI jobs check, run locally, plus the PR scope review. Run before every
# push; CI then has nothing left to find. (Every post-push CI failure in PRs #191-#195 was a
# docs warning or a tuned-table check this runs.) It does not build or run ctest: that is
# scripts/gate.sh, which needs the GPU and ~40 min.
#
#   scripts/agent/prepush.sh [--only docs|checks|scope|pins] [--base origin/main]
#
# Steps: fetch the tuning ledgers from LFS (sweep_to_table --check reads them; a fresh worktree
# holds pointers); kernel-hash pins current; .github/ci/run_local_checks.sh with CI's pinned
# Doxygen and BATCHLAS_DOCS_STRICT=1 (warnings are errors, as in the docs job); pr_scope.py.
# Exit 0 only when every step passed and the scope report flagged nothing.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
only="" base=origin/main
while [ "$#" -gt 0 ]; do
    case $1 in
        --only) only=$2; shift 2 ;;
        --base) base=$2; shift 2 ;;
        -h|--help) sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "prepush: unknown argument $1" >&2; exit 2 ;;
    esac
done
cd "$root"
want() { [ -z "$only" ] || [ "$only" = "$1" ]; }
status=0
summary=()
step() {
    local name=$1; shift
    echo; echo "== prepush: $name"
    if "$@"; then summary+=("ok    $name"); else summary+=("FAIL  $name"); status=1; fi
}

if want checks || want pins; then
    step "LFS ledgers present" git lfs pull --include "benchmarks/results/tuning/ledger/*"
fi
want pins && step "kernel-hash pins current" python3 "$here/repin_tune_hashes.py" --check
if want docs; then
    doxygen=$("$here/get_doxygen.sh") || { echo "prepush: cannot install Doxygen"; exit 1; }
    step "docs build, strict (CI's Doxygen)" env DOXYGEN="$doxygen" BATCHLAS_DOCS_STRICT=1 \
        sh scripts/build_docs.sh build/docs
fi
if want checks; then
    doxygen=$("$here/get_doxygen.sh") || doxygen=""
    # The docs build inside run_local_checks repeats --only docs; skip it there when docs ran.
    if [ -z "$only" ]; then
        step "run_local_checks.sh" env DOXYGEN=/nonexistent sh .github/ci/run_local_checks.sh
    else
        step "run_local_checks.sh" env DOXYGEN="${doxygen:-doxygen}" BATCHLAS_DOCS_STRICT=1 \
            sh .github/ci/run_local_checks.sh
    fi
fi
want scope && step "PR scope review (show REVIEW lines to the user)" python3 "$here/pr_scope.py" --base "$base"

echo
echo "== prepush summary"
printf '%s\n' "${summary[@]}"
[ "$status" = 0 ] && echo "prepush: PASS" || echo "prepush: FAIL"
exit "$status"

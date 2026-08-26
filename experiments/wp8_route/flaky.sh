#!/usr/bin/env bash
# Is sytrd_blocked_tests' GridMatchesLegacyTridiagonal failure caused by this
# pass's routing, or is it flaky on its own?
#
# THE ROUTE-DIFF ALREADY ANSWERS IT MECHANICALLY: zero non-AUTO gemv decisions
# moved between the before and after captures, and sytrd calls no LU op at all,
# so no clause this pass ships is reachable from this binary. This script is the
# empirical half -- run it N times with the shipped routing and N times with
# every op this pass touched PINNED BACK to the vendor, which restores the
# pre-pass decision without a rebuild.
set -u
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
export CUDA_VISIBLE_DEVICES=0
N="${N:-10}"
run () {
  local label="$1" fails=0
  for i in $(seq "$N"); do
    if ! ./build/tests/sytrd_blocked_tests >/dev/null 2>&1; then fails=$((fails+1)); fi
  done
  printf '%-28s %s/%s runs FAILED\n' "$label" "$fails" "$N"
}
unset BATCHLAS_GEMV_ROUTE BATCHLAS_GETRF_ROUTE BATCHLAS_GETRI_ROUTE BATCHLAS_GETRS_ROUTE
run "shipped routing"
export BATCHLAS_GEMV_ROUTE=vendor BATCHLAS_GETRF_ROUTE=vendor
export BATCHLAS_GETRI_ROUTE=vendor BATCHLAS_GETRS_ROUTE=vendor
run "all four pinned to vendor"
echo FLAKY_PROBE_DONE

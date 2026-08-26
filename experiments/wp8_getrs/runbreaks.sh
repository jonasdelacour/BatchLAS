#!/usr/bin/env bash
# GATE-D. Apply each break, REBUILD, run the getrs tests, record the failure
# count, revert, and rebuild clean.
#
# WHICH BUILD. build-novendor. The getrs tests reach the composed driver through
# the DIRECT entry point (sycl_getrs::getrs_blocked_dispatch), so they are
# non-vacuous in build/ too -- but preferred() is all-false at every width this
# pass touches, so in a vendor-present build nothing ROUTES there, and a break
# demonstrated only under a direct call is weaker evidence than one demonstrated
# in the build where the driver is what a caller actually gets. The route the
# tests resolve is printed by the decision-surface test's own spelling read-back.
#
# BATCHLAS_TEST_BACKEND=cuda is required: without it the LuTest fixture's queue
# is a CPU one and every getrs test SKIPS (route_getrf.hh gate 2). A break run
# against 24 skipped tests is the vacuous kind this campaign counts.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
export CUDA_VISIBLE_DEVICES=0 BATCHLAS_TEST_BACKEND=cuda
FILTER='*Getrs*'
LOG="$D/breaks_log.txt"
: > "$LOG"

run () {
  cmake --build "$W/build-novendor" -j --target getrf_tests > /dev/null 2>&1 || {
      echo "BUILD FAILED" >> "$LOG"; return; }
  "$W/build-novendor/tests/getrf_tests" --gtest_filter="$FILTER" > "$D/.brk.out" 2>&1
  local passed failed
  passed=$(grep -c '^\[       OK \]' "$D/.brk.out" || true)
  failed=$(grep -c '^\[  FAILED  \].*\.' "$D/.brk.out" || true)
  echo "  PASSED=$passed  FAILED_LINES=$failed" >> "$LOG"
  grep -E '^\[  FAILED  \] Lu' "$D/.brk.out" | sort -u | head -20 >> "$LOG"
}

echo "=== BASELINE (no break) ===" >> "$LOG"
run
for b in $(python3 "$D/breaks.py" list); do
  echo "=== BREAK $b ===" >> "$LOG"
  python3 "$D/breaks.py" apply "$b" >> "$LOG"
  run
  python3 "$D/breaks.py" revert "$b" >> "$LOG"
done
echo "=== RESTORED ===" >> "$LOG"
run
rm -f "$D/.brk.out"
cat "$LOG"

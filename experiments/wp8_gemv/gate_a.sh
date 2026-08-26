#!/usr/bin/env bash
# GATE-A, run by whoever changed anything last.
#
# TRAP 1: "N tests failed out of M" is a FAILURE count, not a pass count, and
# `ctest -L a -L b` ANDs labels and can select ZERO tests while exiting 0. The
# SELECTED count is printed explicitly below and must read 56.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
L=/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/gate_a

mkdir -p "$L"
echo "=== BUILDS ==="
cmake --build "$W/build" -j > "$L/build_v.log" 2>&1;            echo "build/          exit $?"
cmake --build "$W/build-novendor" -j > "$L/build_nv.log" 2>&1;  echo "build-novendor/ exit $?"

for b in build build-novendor; do
  echo "=== ctest -LE slow in $b ==="
  ( cd "$W/$b" && ctest -LE slow ) > "$L/ctest_$b.log" 2>&1
  grep -E 'Test project|tests passed|Total Test' "$L/ctest_$b.log" | tail -3
  echo -n "SELECTED: "; grep -cE '^\s+[0-9]+/[0-9]+ Test ' "$L/ctest_$b.log"
  grep -E '^\s+[0-9]+ - ' "$L/ctest_$b.log" | sed 's/.* - //; s/ (.*//' | sort > "$L/fail_$b.txt"
  echo "failing set ($(wc -l < "$L/fail_$b.txt")):"; tr '\n' ' ' < "$L/fail_$b.txt"; echo
done

echo "=== FAILING-SET DIFF, vendor-free, against the recorded 22 ==="
grep -E '^\s+[0-9]+ - ' "$W/.route-diff/wp8i1-after-nv.ctest.log" | sed 's/.* - //; s/ (.*//' | sort > "$L/fail_recorded.txt"
if diff "$L/fail_recorded.txt" "$L/fail_build-novendor.txt"; then
  echo "IDENTICAL ($(wc -l < "$L/fail_recorded.txt") names)"
fi

echo "=== TARGETED BINARIES ==="
for b in build build-novendor; do
  for t in gemv_tests getrf_tests route_vocabulary_tests; do
    r=$("$W/$b/tests/$t" 2>&1 | grep -E '^\[  (PASSED|FAILED)  \] [0-9]+ tests?' | tr '\n' ' ')
    echo "$b/$t : $r"
  done
done

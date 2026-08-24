#!/bin/bash
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
T=/home/jonaslacour/.claude/jobs/20812aa0/tmp
B="$1"
python3 "$T/break3.py" "$B" || exit 1
cmake --build "$W/build" -j 32 --target getrf_tests > "$T/build_$B.log" 2>&1
if [ $? -ne 0 ]; then echo "BUILD FAILED: $B"; tail -25 "$T/build_$B.log"; exit 1; fi
"$W/build/tests/getrf_tests" > "$T/break_$B.txt" 2>&1
echo "=== $B  exit=$?"
grep -E "^\[  PASSED  \]|^\[  FAILED  \] [0-9]" "$T/break_$B.txt" | head -3
grep -E "^\[  FAILED  \] Lu.*ms\)" "$T/break_$B.txt" | sed -E 's/^\[  FAILED  \] LuTest\/([0-9])\.([A-Za-z]+).*/type\1 \2/' | sort

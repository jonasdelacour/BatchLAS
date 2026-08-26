#!/usr/bin/env bash
# GATE-A, verbatim. Run by whoever changed anything last, after a final rebuild
# of BOTH builds at the shipped state.
#
# TRAP 1 IS CHECKED EXPLICITLY: "N tests failed out of M" is a FAILURE count, not
# a pass count, and `ctest -L a -L b` ANDs labels and can select ZERO tests while
# exiting 0. So the SELECTED count is grepped out of the log as well.
set -u
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
export CUDA_VISIBLE_DEVICES=0
D=/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7

for b in build build-novendor; do
  echo "=== ctest -LE slow in $b ==="
  ctest --test-dir "$b" -LE slow > "$D/gatea-$b.log" 2>&1
  printf '  selected: %s\n' "$(grep -cE 'Test +#[0-9]+:' "$D/gatea-$b.log")"
  grep -E 'tests passed|tests failed' "$D/gatea-$b.log" | tail -1 | sed 's/^/  /'
  grep -E '^\s+[0-9]+ - ' "$D/gatea-$b.log" | awk '{print $3}' | sort > "$D/fail-$b.txt"
  printf '  failing set (%s):\n' "$(wc -l < "$D/fail-$b.txt")"
  tr '\n' ' ' < "$D/fail-$b.txt" | sed 's/^/    /'; echo
done

echo "=== FAILING-SET DIFF, vendor-free, against the recorded 22 ==="
grep -E '^\s+[0-9]+ - ' .route-diff/wp8-before-nv.ctest.log | awk '{print $3}' | sort \
  > "$D/fail-before-nv.txt"
if diff "$D/fail-before-nv.txt" "$D/fail-build-novendor.txt"; then
  echo "  IDENTICAL ($(wc -l < "$D/fail-before-nv.txt") names)"
else
  echo "  ^^^ THE FAILING SET MOVED"
fi

echo "=== TARGETED BINARIES ==="
for b in build build-novendor; do
  for t in getrf_tests gemv_tests route_vocabulary_tests inverse_tests; do
    if [ -x "$b/tests/$t" ]; then
      out=$("$b/tests/$t" 2>&1 | tail -4 | tr '\n' ' ')
      printf '  %-16s %-24s %s\n' "$b" "$t" "$out"
    fi
  done
done
echo GATE_A_DONE

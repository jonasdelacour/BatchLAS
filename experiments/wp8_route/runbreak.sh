#!/usr/bin/env bash
# runbreak.sh <break-name> -- apply one GATE-D break, rebuild BOTH builds, run
# the three suites the clauses are guarded by, print the failing case names, then
# revert and (optionally) rebuild back.
#
# WHY BOTH BUILDS. A preferred() clause is consulted by the vendor-PRESENT walk
# (that is the whole point of a window), so build/ is where it is reachable. The
# vendor-free build reaches it too, through automatic()'s first pass. Running
# both is what makes "the break is not vacuous" a measurement rather than a
# claim.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"
export CUDA_VISIBLE_DEVICES=0
N="$1"
python3 experiments/wp8_route/breaks.py apply "$N" || exit 1
cmake --build build -j >/dev/null 2>&1 || { echo "BUILD FAILED"; }
cmake --build build-novendor -j >/dev/null 2>&1 || { echo "NV BUILD FAILED"; }
for b in build build-novendor; do
  for t in route_vocabulary_tests getrf_tests gemv_tests; do
    [ -x "$b/tests/$t" ] || continue
    o=$("$b/tests/$t" 2>&1)
    n=$(printf '%s' "$o" | grep -c '^\[  FAILED  \] .*\..*(' || true)
    printf '%-16s %-24s FAILED=%s\n' "$b" "$t" "$n"
    printf '%s' "$o" | grep '^\[  FAILED  \] ' | grep '(' | sed 's/^/      /' | sort -u | head -12
  done
done
python3 experiments/wp8_route/breaks.py revert "$N"
echo "BREAK_${N}_DONE"

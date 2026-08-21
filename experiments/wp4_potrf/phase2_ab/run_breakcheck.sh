#!/usr/bin/env bash
# Deliberate-break validation of the `blocked` mode's residual check.
# Each row must go RED (residual >> 1e-6) for the breaks that matter to it, and
# the `conj` break must be a NO-OP for real types by definition -- if it went red
# for float that would mean the check is reading something other than the answer.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
echo "break,type,csvrow"
for t in float double cfloat cdouble; do
  for b in none conj nofold stride; do
    v=""
    [ "$b" != none ] && v="$b"
    out=$(PHASE2_BREAK="$v" BENCH_WARM_S=0.05 ./phase2 blocked "$t" 256 64 128 4 2 2>&1 | tail -1)
    echo "$b,$t,$out"
  done
done

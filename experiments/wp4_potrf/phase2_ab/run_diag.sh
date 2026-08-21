#!/usr/bin/env bash
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
for t in float double; do
  for nb in 32 48 64; do
    for n in 512 1024; do
      echo "=== $t n=$n nb=$nb"
      PHASE2_DIAG=1 BENCH_WARM_S=0.02 ./phase2 blocked "$t" "$n" "$nb" 128 16 1 2>&1 | head -14
    done
  done
done

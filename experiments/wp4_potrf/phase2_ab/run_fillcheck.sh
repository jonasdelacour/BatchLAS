#!/usr/bin/env bash
# Was the n=1024 nb=48 float failure a DRIVER defect or a bad INPUT?
# Decisive test: run cuSOLVER (`vendorpotrf`) on the SAME matrices. If the
# vendor also reports info != 0, the input was not positive definite as stored.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
echo "--- per-item fill (the ORIGINAL, which failed) ---"
for t in float double; do
  for n in 512 1024; do
    echo -n "vendor  $t n=$n: "
    PHASE2_FILL=peritem BENCH_WARM_S=0.02 ./phase2 vendorpotrf "$t" "$n" 128 2 2>&1 | tail -1
    for nb in 32 48; do
      echo -n "blocked $t n=$n nb=$nb: "
      PHASE2_FILL=peritem BENCH_WARM_S=0.02 ./phase2 blocked "$t" "$n" "$nb" 128 128 2 2>&1 | tail -1
    done
  done
done
echo "--- shared-Gram fill (the replacement) ---"
for t in float double; do
  for n in 512 1024; do
    echo -n "vendor  $t n=$n: "
    BENCH_WARM_S=0.02 ./phase2 vendorpotrf "$t" "$n" 128 2 2>&1 | tail -1
    for nb in 32 48; do
      echo -n "blocked $t n=$n nb=$nb: "
      BENCH_WARM_S=0.02 ./phase2 blocked "$t" "$n" "$nb" 128 128 2 2>&1 | tail -1
    done
  done
done

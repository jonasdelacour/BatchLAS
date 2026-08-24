#!/usr/bin/env bash
# THE REGISTER-CAP A/B. Three passes, because the two arms are two BUILDS of the
# .so and therefore cannot be interleaved in one process -- the cross-pass median
# spread is the evidence that stands in for interleaving.
#
# usage: run.sh <before|after>
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
B="$W/experiments/wp6_perf/bench"
R="$W/experiments/wp6_perf/regcap"
TAG="${1:?before|after}"
export GPU=1 WARM_S=0.8 REPS=7 NPROBE=1 NTRANS=1
for p in 1 2 3; do
  CELLFILE="$R/cells.txt" bash "$B/run_cells.sh" "$R/${TAG}_p${p}.csv" lubench6_nv native:cta
done
echo "${TAG}-DONE"

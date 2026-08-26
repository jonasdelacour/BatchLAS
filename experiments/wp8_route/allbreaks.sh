#!/usr/bin/env bash
# Run the remaining GATE-D breaks one at a time, each applied / rebuilt in BOTH
# builds / run / shown red / reverted.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"
for n in getri_boundary getrf_boundary getrf_doubleleak getrs_boundary \
         getrs_nofloor gemv_axisswap gemv_nobatch gemv_alltypes; do
  echo "############ $n"
  bash experiments/wp8_route/runbreak.sh "$n"
done
echo ALL_BREAKS_DONE

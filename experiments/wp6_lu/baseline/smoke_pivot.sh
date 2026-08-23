#!/usr/bin/env bash
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export WARM_S=0.4
for v in nopiv swaponly pivman pivgrp; do
  "$D/pivotcost" "$v" "${1:-float}" "${2:-64}" "${3:-4096}" 5
done

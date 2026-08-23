#!/usr/bin/env bash
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=1
export WARM_S=${WARM_S:-0.5}
for m in getrs getrs_trsm getri getri_trsm; do
  "$D/lubench_v" "$m" "${1:-float}" "${2:-64}" "${3:-64}" "${4:-256}" 5
done

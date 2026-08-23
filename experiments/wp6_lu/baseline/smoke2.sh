#!/usr/bin/env bash
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=1
export WARM_S=0.4
for m in getrs_trsm getri_trsm; do
  printf 'list,';   env -u LASWP        "$D/lubench_v" "$m" "${1:-float}" "${2:-128}" "${3:-128}" "${4:-256}" 5
  printf 'gather,'; LASWP=gather        "$D/lubench_v" "$m" "${1:-float}" "${2:-128}" "${3:-128}" "${4:-256}" 5
done

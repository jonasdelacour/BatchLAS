#!/usr/bin/env bash
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=1
export WARM_S=0.4
"$D/lubench_v" getrf float 2048 1 32 3
"$D/lubench_v" getri float 2048 1 32 3
"$D/lubench_v" getri_trsm float 2048 1 32 3
"$D/lubench_v" getrf cdouble 1024 1 64 3
"$D/lubench_v" getri cdouble 1024 1 64 3

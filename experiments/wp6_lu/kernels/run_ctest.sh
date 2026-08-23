#!/usr/bin/env bash
# The burn-down check. ONE -L with an alternation, never repeated -L flags:
# repeated -L flags AND together and select ZERO tests while exiting 0.
#
# $1 = build dir (build | build-novendor)
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
B="${1:-build-novendor}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd "$W/$B" || exit 1
ctest -L "blas|ortho|util" -LE slow --output-on-failure 2>&1 | tail -60

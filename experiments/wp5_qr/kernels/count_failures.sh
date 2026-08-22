#!/usr/bin/env bash
# Per-suite [ FAILED ] counts, the burn-down's unit. Run against either build dir.
set -uo pipefail
B=${1:-/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/build-novendor}
cd "$B" || exit 1
for t in "${@:2}"; do
  n=$(CUDA_VISIBLE_DEVICES=1 timeout 900 ctest -R "^${t}$" --output-on-failure 2>&1 \
        | grep -c '^\[  FAILED  \]')
  echo "${t} failed_lines=${n}"
done

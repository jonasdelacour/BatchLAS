#!/usr/bin/env bash
# Does the pin take effect? An UNRECOGNISED BATCHLAS_POTRF_ROUTE silently means
# vendor, so this asks the question rather than assuming the answer.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_impl
export CUDA_VISIBLE_DEVICES=1
for route in blocked native:blocked cta vendor typo_not_a_route ""; do
  echo "=== BATCHLAS_POTRF_ROUTE='$route'"
  if [ -z "$route" ]; then unset BATCHLAS_POTRF_ROUTE; else export BATCHLAS_POTRF_ROUTE="$route"; fi
  "$D/verify" facade f 256 8 2>&1 | tail -3
done

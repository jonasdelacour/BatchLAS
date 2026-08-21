#!/usr/bin/env bash
# NOTE: POTRF2_BREAK DOES NOT EXIST IN THE SHIPPED DRIVER. It was patched into
# src/extensions/potrf_blocked.cc for one build and removed again; these scripts
# are kept for the record. README.md carries the results. Re-running requires
# re-applying that patch.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_impl
export CUDA_VISIBLE_DEVICES=1
export BATCHLAS_POTRF_ROUTE=blocked
echo "### control"
for t in f d c z; do "$D/verify" info "$t" 300 8 2>&1 | tail -2; done
for b in noquench nomerge; do
  echo "### POTRF2_BREAK=$b"
  export POTRF2_BREAK="$b"
  for t in f z; do "$D/verify" info "$t" 300 8 2>&1 | tail -2; done
  unset POTRF2_BREAK
done

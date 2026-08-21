#!/usr/bin/env bash
# NOTE: POTRF2_BREAK DOES NOT EXIST IN THE SHIPPED DRIVER. It was patched into
# src/extensions/potrf_blocked.cc for one build and removed again; these scripts
# are kept for the record. README.md carries the results. Re-running requires
# re-applying that patch.
# Every check in verify.cpp, against a driver with a deliberate defect injected.
# A check that stays GREEN under its own break is a blind guard.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_impl
export CUDA_VISIBLE_DEVICES=1
export BATCHLAS_POTRF_ROUTE=blocked

echo "### control (no break)"
"$D/verify" facade f 256 8 | tail -2
"$D/verify" facade z 256 8 | tail -2
"$D/verify" info  f 300 8 | tail -2

for b in nofold stride conj nozero nomerge noquench; do
  echo "### POTRF2_BREAK=$b"
  export POTRF2_BREAK="$b"
  "$D/verify" facade f 256 8 2>&1 | tail -2
  "$D/verify" facade z 256 8 2>&1 | tail -2
  "$D/verify" info  f 300 8 2>&1 | tail -2
  unset POTRF2_BREAK
done

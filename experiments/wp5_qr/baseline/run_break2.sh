#!/usr/bin/env bash
# Follow-up to run_break.sh: BREAK=1 (drop the LAST reflector) failed to turn
# the real types red. Print |tau| at the tail to test the hypothesis that
# LAPACK's larfg returns tau = 0 for a 1x1 trailing reflector on a REAL matrix,
# which would make the "short final panel" break vacuous at m == n -- and add
# BREAK=5, which drops a MIDDLE reflector instead.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export CUDA_VISIBLE_DEVICES=1 WARM_S=0.2
for t in float double cfloat cdouble; do
  echo "--- $t ---"
  SHOW_TAU=1 "$D/wp5qr_v" ormqrI "$t" 96 8 2 2>&1 >/dev/null | grep SHOW_TAU
  for b in 0 1 5; do
    printf 'BREAK=%s  geqrf_residual=%s\n' "$b" \
      "$(BREAK=$b "$D/wp5qr_v" ormqrI "$t" 96 8 2 | cut -d, -f9)"
  done
done

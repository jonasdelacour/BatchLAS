#!/usr/bin/env bash
# Cost probe: how long does one cell of the biggest shape take, so the sweep can
# be budgeted rather than discovered by timing out.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export CUDA_VISIBLE_DEVICES=1 WARM_S=0.2
for cell in "geqrf float 1024 128 3" "geqrf cdouble 512 512 3" "orgqr float 256 2048 3"; do
  set -- $cell
  echo "== $* =="
  /usr/bin/time -f "wall %e s" "$D/wp5qr_v" "$@"
done

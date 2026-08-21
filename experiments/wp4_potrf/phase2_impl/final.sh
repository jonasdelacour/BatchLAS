#!/usr/bin/env bash
# The full proof, re-run on the SHIPPED build (no break switch present).
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_impl
export CUDA_VISIBLE_DEVICES=1
rc=0
echo "=== facade, BATCHLAS_POTRF_ROUTE=blocked, ld=n+7, stride=ld*n+13, upper poisoned"
export BATCHLAS_POTRF_ROUTE=blocked
for t in f d c z; do
  for nb in "256 128" "512 32" "1000 16"; do
    "$D/verify" facade $t $nb 2>&1 | tail -2 || rc=1
  done
done
echo "=== direct entry point (cannot be served by a vendor)"
for t in f d c z; do "$D/verify" direct $t 256 128 2>&1 | tail -2 || rc=1; done
echo "=== vs cuSOLVER"
unset BATCHLAS_POTRF_ROUTE
for t in f d c z; do "$D/verify" vendorcmp $t 512 32 2>&1 | tail -2 || rc=1; done
echo "=== info: global index, first-wins, failed-item finiteness"
export BATCHLAS_POTRF_ROUTE=blocked
for t in f d c z; do "$D/verify" info $t 300 8 2>&1 | tail -2 || rc=1; done
echo "=== open question 9"
for t in c z; do "$D/verify" oq9 $t 512 8 2>&1 | tail -1; done
echo "=== the pin, asked rather than assumed"
for route in blocked native:blocked cta vendor typo_not_a_route NONE; do
  if [ "$route" = NONE ]; then unset BATCHLAS_POTRF_ROUTE; else export BATCHLAS_POTRF_ROUTE="$route"; fi
  printf "  BATCHLAS_POTRF_ROUTE=%-22s -> " "$route"
  out=$("$D/verify" facade f 256 8 2>&1 | tail -1)
  echo "$out"
done
exit $rc

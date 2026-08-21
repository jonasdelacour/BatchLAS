#!/usr/bin/env bash
# VERIFY THE PIN, do not assume it.  An UNRECOGNISED BATCHLAS_POTRF_ROUTE value
# silently means vendor (route_env.hh:214-230 -> legacy_unset_default), so a
# benchmark that thinks it pinned a native route can be timing cuSOLVER.  This
# asks the resolver itself, per type and per order, including a deliberate typo.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
BIN=${BIN:-./bench}
OUT=${OUT:-$D/route.txt}
: > "$OUT"
for t in float double cfloat cdouble; do
  for n in 64 128 256 512 1024 2048; do
    for e in UNSET blocked native:blocked cta vendor tyop; do
      if [ "$e" = UNSET ]; then unset BATCHLAS_POTRF_ROUTE; else export BATCHLAS_POTRF_ROUTE="$e"; fi
      CUDA_VISIBLE_DEVICES=$GPU $BIN route "$t" "$n" 128 >> "$OUT" 2>&1
    done
  done
done
unset BATCHLAS_POTRF_ROUTE
echo "wrote $OUT"

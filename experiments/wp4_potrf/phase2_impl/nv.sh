#!/usr/bin/env bash
# The vendor-free build, with NO route pinned at all. Before Phase 2 an order
# above potrf_cta_max_n<T>() had no native route here and threw NoRouteError.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_impl
export CUDA_VISIBLE_DEVICES=1
unset BATCHLAS_POTRF_ROUTE
for t in f d c z; do "$D/verify_nv" facade "$t" 256 64 2>&1 | tail -2; done
echo "--- and at an order BELOW the ceiling, which must still take the CTA leaf"
for t in f z; do "$D/verify_nv" facade "$t" 64 64 2>&1 | tail -2; done

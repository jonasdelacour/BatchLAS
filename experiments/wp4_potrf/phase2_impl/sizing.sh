#!/usr/bin/env bash
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_impl
export CUDA_VISIBLE_DEVICES=1
unset BATCHLAS_POTRF_ROUTE
echo "--- vendor-present (route is Vendor: native_need must stay 0 and nothing must move)"
for t in f z; do "$D/verify" sizing "$t" 5 16 2>&1 | tail -2; done
for t in f z; do "$D/verify" sizing "$t" 256 16 2>&1 | tail -2; done
echo "--- vendor-free (route is Native: the new max()-over-tiers query fires)"
for t in f d c z; do "$D/verify_nv" sizing "$t" 5 16 2>&1 | tail -2; done
for t in f d c z; do "$D/verify_nv" sizing "$t" 256 16 2>&1 | tail -2; done

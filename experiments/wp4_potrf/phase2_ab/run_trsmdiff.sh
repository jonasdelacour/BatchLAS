#!/usr/bin/env bash
# Native trsm vs vendor trsm on the potrf panel shape, same input, one process.
# Columns: maxrel = worst per-item max|Xnative - Xvendor| / max|Xvendor|;
#          items_diff = batch items where that exceeds 1e-3;
#          rn0/rnL = host residual ||X op(A) - B||inf / ||B||inf for the NATIVE
#          answer on items 0 and batch-1; rv0/rvL the same for the VENDOR.
# Neither side is assumed correct: the residual columns say which one is wrong.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/trsmdiff.csv"
echo "mode,type,n,ib,j,m2,batch,maxrel,items_diff,resid_native_0,resid_native_last,resid_vendor_0,resid_vendor_last" > "$OUT"
for t in float double cfloat cdouble; do
  for n in 512 1024; do
    for ib in 32 48 64 96 128; do
      for batch in 8 64 128; do
        ./phase2 trsmdiff "$t" "$n" "$ib" 0 "$batch" 2>&1 | tail -1 >> "$OUT"
      done
    done
  done
done
column -s, -t < "$OUT"

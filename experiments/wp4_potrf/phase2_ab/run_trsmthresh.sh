#!/usr/bin/env bash
# Where does the native-trsm wrong answer start? Two candidate axes:
#   (a) q*batch crossing 65535 -- the CUDA grid-dimension limit;
#   (b) the triangular order not being a multiple of trsm_cta_max_n = 32.
# Sweep both. flat operands throughout, so nothing here is about sub-views.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/trsmthresh.csv"
echo "mode,type,n,ib,j,m2,batch,store,slack,rep,maxrel,items_diff,resid_native_0,resid_native_last,resid_vendor_0,resid_vendor_last" > "$OUT"
# (a) fixed shape, batch sweep across q*batch = 65535 (q = m2 = 976)
for b in 16 32 48 64 66 67 68 70 96 128; do
  for rep in 0 1; do
    ./phase2 trsmdiff double 1024 48 0 "$b" flat 0 "$rep" 2>&1 | tail -1 >> "$OUT"
  done
done
# (b) triangular-order sweep at a fixed, large q*batch
for ib in 32 33 48 64 65 77 96 109 128 155 160; do
  for rep in 0 1; do
    ./phase2 trsmdiff double 1024 "$ib" 0 128 flat 0 "$rep" 2>&1 | tail -1 >> "$OUT"
  done
done
# (c) same order sweep at a SMALL q*batch, to separate the two axes
for ib in 32 48 64 77 109 155; do
  for rep in 0 1; do
    ./phase2 trsmdiff double 1024 "$ib" 0 32 flat 0 "$rep" 2>&1 | tail -1 >> "$OUT"
  done
done
column -s, -t < "$OUT"

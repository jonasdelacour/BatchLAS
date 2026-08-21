#!/usr/bin/env bash
# Characterise the native-trsm wrong answer along three axes:
#   store  sub | flat  -- is it specific to sub-views carrying a parent ld/stride?
#   slack  0 | n*n     -- is it the over-long MatrixView span (matrix.cc:1839)?
#   rep    0..4        -- is it deterministic?
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/trsmbug.csv"
echo "mode,type,n,ib,j,m2,batch,store,slack,rep,maxrel,items_diff,resid_native_0,resid_native_last,resid_vendor_0,resid_vendor_last" > "$OUT"
for t in float double; do
  for cfg in "1024 48" "1024 64" "1024 128" "512 96"; do
    set -- $cfg; n=$1; ib=$2
    for store in sub flat; do
      for slack in 0 1048576; do
        [ "$store" = flat ] && [ "$slack" != 0 ] && continue
        for rep in 0 1 2 3 4; do
          ./phase2 trsmdiff "$t" "$n" "$ib" 0 128 "$store" "$slack" "$rep" 2>&1 | tail -1 >> "$OUT"
        done
      done
    done
  done
done
column -s, -t < "$OUT"

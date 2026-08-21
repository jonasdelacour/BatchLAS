#!/usr/bin/env bash
# The blocked driver returns a WRONG ANSWER for some (n, nb, batch). Bisect it
# along the two axes that change ROUTE rather than shape: BATCHLAS_TRSM_ROUTE
# and BATCHLAS_GEMM_ROUTE. Residual and info count are the only columns that
# matter here; the timings are incidental.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/wrongans.csv"
echo "trsm,gemm,type,n,nb,batch,residual,info_nonzero" > "$OUT"
for t in double float; do
 for n in 512 1024; do
  for nb in 32 48 64 96; do
   for batch in 64 128; do
    for tr in vendor native; do
     for gm in vendor native; do
      r=$(BATCHLAS_TRSM_ROUTE=$tr BATCHLAS_GEMM_ROUTE=$gm BENCH_WARM_S=0.01 \
          ./phase2 blocked "$t" "$n" "$nb" 128 "$batch" 1 2>&1 | tail -1)
      res=$(echo "$r" | awk -F, '{print $(NF-1)}')
      bad=$(echo "$r" | awk -F, '{print $NF}')
      echo "$tr,$gm,$t,$n,$nb,$batch,$res,$bad" >> "$OUT"
     done
    done
   done
  done
 done
done
column -s, -t < "$OUT"

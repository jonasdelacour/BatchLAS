#!/usr/bin/env bash
# The control for §2: is the PANEL trsm sensitive to the sub-view leading
# dimension the way the 128x128 GEMM is? Same (m2, ib, batch), operands freshly
# allocated at ld == rows instead of sliced out of an n x n parent.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/panelflat.csv"
echo "route,mode,type,n,nb,j,m2,ib,batch,med_ms,min_ms,rel_sd,gflops" > "$OUT"
BATCH=${BATCH:-128}
REPS=${REPS:-7}
# (type, m2, ib) taken from the j = 0 row of panel.csv at n = 1024 and n = 2048.
run () {
  local t=$1 m2=$2 ib=$3
  for route in default vendor native; do
    unset BATCHLAS_TRSM_ROUTE
    [ "$route" != default ] && export BATCHLAS_TRSM_ROUTE="$route"
    ./phase2 panelflat "$t" "$m2" "$ib" "$BATCH" "$REPS" 2>&1 | sed "s/^/$route,/" >> "$OUT"
    unset BATCHLAS_TRSM_ROUTE
  done
}
run float   869 155
run float  1893 155
run double  915 109
run double 1939 109
run cfloat  915 109
run cdouble 947 77
column -s, -t < "$OUT"

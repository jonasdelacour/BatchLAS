#!/usr/bin/env bash
# OPEN QUESTION 5 -- the panel solve.
#
# trsm(Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit) on the
# REAL panel shapes, with the operands built as sub-views of the n x n parent so
# they carry the parent ld AND the parent batch stride.
#
# Three routes per cell, INTERLEAVED (default, vendor, native) rather than all
# of one then all of the other, so a drift in clock or in another process shows
# up as spread rather than as a ratio.
#
# batch = 128 throughout: 128 SMs on this box, and trsm's float/Side::Right
# preferred() clause is `s.batch >= 128 || order <= 32` (route_trsm.hh:304), so
# a smaller batch would measure a DIFFERENT route and call it the same thing.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/panel.csv"
echo "route,mode,type,n,nb,j,m2,ib,batch,med_ms,min_ms,rel_sd,gflops" > "$OUT"
BATCH=${BATCH:-128}
REPS=${REPS:-7}
for t in float double cfloat cdouble; do
  case "$t" in
    float)   nb=155 ;;
    double)  nb=109 ;;
    cfloat)  nb=109 ;;
    cdouble) nb=77  ;;
  esac
  for n in 512 1024 2048; do
    for route in default vendor native; do
      unset BATCHLAS_TRSM_ROUTE
      [ "$route" != default ] && export BATCHLAS_TRSM_ROUTE="$route"
      ./phase2 panel "$t" "$n" "$nb" "$BATCH" "$REPS" 2>&1 \
        | sed "s/^/$route,/" >> "$OUT"
      unset BATCHLAS_TRSM_ROUTE
    done
  done
done
echo "wrote $OUT"

#!/usr/bin/env bash
# OPEN QUESTION 6 -- the trailing update gemm.
#
# C := C - L21 * L21^H, i.e. transA = NoTrans, transB = ConjTrans, alpha = -1,
# beta = 1 (beta = 0 for the WxW diagonal block, priced separately in `blocked`).
#
# TWO CONTROLS, and the pair is the whole point:
#   sub  -- A, B and C are sub-views of ONE (max(m,n)+k)-square parent, so each
#           carries the parent ld and the parent batch stride. This is what a
#           blocked driver actually issues.
#   flat -- the same m,n,k freshly allocated at ld == rows. This is what every
#           square-matrix GEMM benchmark measures, and it is a DIFFERENT number.
#
# Shapes: (m, n, k) = (m2, W, ib) for the below-diagonal rectangle and
# (W, W, ib) for the diagonal block, with W = 128 and ib = potrf_cta_max_n<T>().
# The (m2, m2, ib) row prices the whole-A22 gemm the design rejects.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/trail.csv"
echo "route,mode,type,m,n,k,batch,store,tB,med_ms,min_ms,rel_sd,gflops" > "$OUT"
BATCH=${BATCH:-128}
REPS=${REPS:-7}

run () { # type m n k store tB
  local t=$1 m=$2 nn=$3 k=$4 st=$5 tb=$6
  for route in default vendor native; do
    unset BATCHLAS_GEMM_ROUTE
    [ "$route" != default ] && export BATCHLAS_GEMM_ROUTE="$route"
    ./phase2 trail "$t" "$m" "$nn" "$k" "$BATCH" "$st" "$tb" "$REPS" 2>&1 \
      | sed "s/^/$route,/" >> "$OUT"
    unset BATCHLAS_GEMM_ROUTE
  done
}

for t in float double cfloat cdouble; do
  case "$t" in
    float)   ib=155; m1=869;  m2=1893 ;;
    double)  ib=109; m1=915;  m2=1939 ;;
    cfloat)  ib=109; m1=915;  m2=1939 ;;
    cdouble) ib=77;  m1=947;  m2=1971 ;;
  esac
  for st in sub flat; do
    run "$t" "$m1" 128  "$ib" "$st" C     # rectangle, n = 1024 panel
    run "$t" 128   128  "$ib" "$st" C     # diagonal block
    run "$t" "$m2" 128  "$ib" "$st" C     # rectangle, n = 2048 panel
    run "$t" "$m1" "$m1" "$ib" "$st" C    # whole-A22 gemm (the rejected form)
  done
  # transB = Trans, real types only: mathematically identical there, but a
  # DIFFERENT enum value, and select_kernel_variant's float TN/NT/TT register
  # cases test for Trans exactly (gemm_kernels.cc:470-478).
  case "$t" in
    float|double)
      for st in sub flat; do
        run "$t" "$m1" 128 "$ib" "$st" T
        run "$t" 128   128 "$ib" "$st" T
      done ;;
  esac
done
echo "wrote $OUT"

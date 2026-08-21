#!/usr/bin/env bash
# The nb sweep, the panel-solve FRACTION, and the end-to-end scale.
#
# `blocked` runs the whole right-looking driver on real sub-views and reports
# BOTH an un-instrumented total (one wait at the end) and a per-stage split
# (a wait after each stage). Quote the total for ratios and the split only for
# proportions -- the split's waits inflate it by the difference between the two.
#
# EVERY ROW CARRIES residual AND info_nonzero. Some cells are NUMERICALLY WRONG
# (the native trsm defect characterised in trsmthresh.csv). A wrong cell still
# does the same arithmetic, so its timing is comparable -- but no cell may be
# quoted as a result without its residual.
#
# routemode:
#   default  -- what a vendor-PRESENT build gets today.
#   native   -- BATCHLAS_TRSM_ROUTE=native BATCHLAS_GEMM_ROUTE=native, i.e. what
#               a vendor-FREE build gets. This is the build WP4 exists for.
#   vtrsm    -- native gemm, VENDOR trsm: the only combination that is correct
#               at every nb today, and the one a vendor-present build should use
#               until the trsm defect is fixed.
#   nativeT  -- native, with the trailing gemm's transB = Trans instead of
#               ConjTrans. Mathematically identical for a REAL type; a different
#               enum value, and select_kernel_variant's float NT register case
#               tests for Trans exactly. Real types only.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/blocked.csv"
echo "routemode,mode,type,n,nb,W,batch,med_ms,min_ms,rel_sd,gflops,staged_ms,leaf_ms,panel_ms,trail_ms,residual,info_nonzero" > "$OUT"
BATCH=${BATCH:-128}
REPS=${REPS:-4}
export BENCH_WARM_S=${BENCH_WARM_S:-1.0}

one () { # routemode type n nb W
  local rm=$1 t=$2 n=$3 nb=$4 W=$5
  unset BATCHLAS_TRSM_ROUTE BATCHLAS_GEMM_ROUTE PHASE2_BREAK
  case "$rm" in
    native)  export BATCHLAS_TRSM_ROUTE=native BATCHLAS_GEMM_ROUTE=native ;;
    vtrsm)   export BATCHLAS_TRSM_ROUTE=vendor BATCHLAS_GEMM_ROUTE=native ;;
    nativeT) export BATCHLAS_TRSM_ROUTE=native BATCHLAS_GEMM_ROUTE=native PHASE2_BREAK=conj ;;
  esac
  ./phase2 blocked "$t" "$n" "$nb" "$W" "$BATCH" "$REPS" 2>&1 | sed "s/^/$rm,/" >> "$OUT"
  unset BATCHLAS_TRSM_ROUTE BATCHLAS_GEMM_ROUTE PHASE2_BREAK
}

for t in float double cfloat cdouble; do
  case "$t" in
    float)   NBS="32 48 64 96 128 155" ;;
    double)  NBS="32 48 64 80 96 109" ;;
    cfloat)  NBS="32 48 64 80 96 109" ;;
    cdouble) NBS="32 48 64 77" ;;
  esac
  for n in 512 1024; do
    for nb in $NBS; do
      for rm in default native vtrsm; do one "$rm" "$t" "$n" "$nb" 128; done
      case "$t" in float|double) one nativeT "$t" "$n" "$nb" 128 ;; esac
    done
  done
done

# W sweep at a fixed mid nb, native only -- W is a trailing-update knob and the
# trailing update is the only stage it touches.
for t in float double cfloat cdouble; do
  case "$t" in
    cdouble) nb=64 ;;
    *)       nb=96 ;;
  esac
  for W in 64 128 256 512; do
    one native  "$t" 1024 "$nb" "$W"
    one default "$t" 1024 "$nb" "$W"
  done
done

# End-to-end scale: the routed potrf (cuSOLVER here) on the same parents.
echo "vendorpotrf rows follow (own columns)" >> "$OUT"
for t in float double cfloat cdouble; do
  for n in 512 1024; do
    ./phase2 vendorpotrf "$t" "$n" "$BATCH" "$REPS" >> "$OUT" 2>&1
  done
done
echo "wrote $OUT"

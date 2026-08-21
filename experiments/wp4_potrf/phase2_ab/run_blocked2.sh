#!/usr/bin/env bash
# Supplement to run_blocked.sh.
#
# At n = 1024 the native trsm is WRONG for every nb tried (§4), so no row with
# BATCHLAS_TRSM_ROUTE=native there is a usable measurement of anything. The only
# correct configuration that still exercises the NATIVE trailing gemm is
# vendor-trsm + native-gemm. These rows price the Transpose::Trans trick and the
# nb trend on correct answers only.
#
#   vtrsm   native gemm, vendor trsm, transB = ConjTrans
#   vtrsmT  native gemm, vendor trsm, transB = Trans   (real types: same maths)
#   default everything routed, no env -- the vendor-present baseline
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/blocked2.csv"
echo "routemode,mode,type,n,nb,W,batch,med_ms,min_ms,rel_sd,gflops,staged_ms,leaf_ms,panel_ms,trail_ms,residual,info_nonzero" > "$OUT"
BATCH=${BATCH:-128}
REPS=${REPS:-4}
export BENCH_WARM_S=${BENCH_WARM_S:-1.0}

one () { # routemode type n nb W
  local rm=$1 t=$2 n=$3 nb=$4 W=$5
  unset BATCHLAS_TRSM_ROUTE BATCHLAS_GEMM_ROUTE PHASE2_BREAK
  case "$rm" in
    vtrsm)  export BATCHLAS_TRSM_ROUTE=vendor BATCHLAS_GEMM_ROUTE=native ;;
    vtrsmT) export BATCHLAS_TRSM_ROUTE=vendor BATCHLAS_GEMM_ROUTE=native PHASE2_BREAK=conj ;;
    vgemm)  export BATCHLAS_TRSM_ROUTE=vendor BATCHLAS_GEMM_ROUTE=vendor ;;
  esac
  ./phase2 blocked "$t" "$n" "$nb" "$W" "$BATCH" "$REPS" 2>&1 | sed "s/^/$rm,/" >> "$OUT"
  unset BATCHLAS_TRSM_ROUTE BATCHLAS_GEMM_ROUTE PHASE2_BREAK
}

for t in float double; do
  for n in 512 1024; do
    for nb in 64 96 128 155; do
      case "$t:$nb" in double:155) continue ;; esac
      for rm in vtrsm vtrsmT vgemm; do one "$rm" "$t" "$n" "$nb" 128; done
    done
  done
done
for t in cfloat cdouble; do
  case "$t" in cfloat) NBS="64 96 109" ;; cdouble) NBS="48 64 77" ;; esac
  for n in 512 1024; do
    for nb in $NBS; do
      for rm in vtrsm vgemm; do one "$rm" "$t" "$n" "$nb" 128; done
    done
  done
done

# W sweep down to 32. The main sweep only went 64..512 and W=64 won almost
# everywhere, so the bottom of the range was never bracketed.
for t in float double cfloat cdouble; do
  case "$t" in cdouble) nb=64 ;; *) nb=96 ;; esac
  for W in 16 32 64 96; do
    one vtrsm "$t" 512 "$nb" "$W"
    one vgemm "$t" 512 "$nb" "$W"
  done
done
echo "wrote $OUT"

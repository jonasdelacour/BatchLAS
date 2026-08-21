#!/usr/bin/env bash
# nb / W sweep on potrf ITSELF (BATCHLAS_POTRF_NB / BATCHLAS_POTRF_W are read by
# potrf_blocked_params, potrf_blocked.cc:177-178).  The shipped constants were
# measured on the STAGED driver of the measure phase, at n <= 1024 and batch 128
# only; this re-checks them on the real driver at the batch and order this
# campaign cares about.
#
# Swept in TWO configurations on purpose:
#   nn  the vendor-free one -- but its answers are WRONG at these batches (the
#       native trsm defect), so its optimum is only trustworthy if the wrongness
#       does not change the work, which for a right-looking Cholesky it does not
#       (no pivoting, no early exit) but which is stated rather than assumed.
#   nV  native gemm, vendor trsm -- CORRECT, and it isolates nb's effect on the
#       trailing update, which is where nb actually acts (k == nb).
# A value that wins in both is a value the driver can ship.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
BIN=${BIN:-./bench}
OUT=${OUT:-$D/nbsweep.csv}
REPS=${REPS:-4}
export BENCH_WARM_S=${BENCH_WARM_S:-1.0}
echo "cfg,nbreq,Wreq,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
: > "$D/nbsweep.err"

one () { # cfg type n batch nb W
  local cfg=$1 t=$2 n=$3 b=$4 nb=$5 W=$6
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
  case $cfg in
    nn) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native ;;
    nV) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=vendor ;;
    VV) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=vendor ;;
  esac
  BATCHLAS_POTRF_NB=$nb BATCHLAS_POTRF_W=$W CUDA_VISIBLE_DEVICES=$GPU \
    $BIN ab "$t" "$n" "$b" "$REPS" 2>>"$D/nbsweep.err" | sed "s/^/$cfg,$nb,$W,/" >> "$OUT"
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
}

# nb sweep at the shipped W.  Only multiples of trsm_cta_max_n<T>() == 32 are
# legal for the native trsm (potrf_blocked.cc rounds nb DOWN to one anyway, so a
# request of 48 or 80 silently becomes 32 or 64 -- sweeping them would print a
# fake data point).  Cell: n=1024, batch=256, which main.csv shows is saturated
# for every type.
for cfg in nn VV; do
  for t in float double cfloat cdouble; do
    case "$t" in cdouble) Wd=16 ;; *) Wd=32 ;; esac
    for nb in 32 64 96 128 160; do
      one "$cfg" "$t" 1024 256 "$nb" "$Wd"
    done
  done
done

# W sweep at the shipped nb.
for cfg in nn VV; do
  for t in float double cfloat cdouble; do
    case "$t" in float) nbd=128 ;; cdouble) nbd=64 ;; *) nbd=96 ;; esac
    for W in 16 32 64 128 256; do
      one "$cfg" "$t" 1024 256 "$nbd" "$W"
    done
  done
done
echo "wrote $OUT"

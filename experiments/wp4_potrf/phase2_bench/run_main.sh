#!/usr/bin/env bash
# THE MAIN GRID.  cuSOLVER against the blocked native driver, on potrf itself --
# never on a synthetic gemm/trsm, because only potrf issues the strided
# sub-views the native kernels are gated against.
#
# Three blocked configurations per cell, all with the SAME driver.  What differs
# is only where the two INJECTED calls land, which is what separates "the driver
# is slow" from "the native kernels under it are slow":
#
#   def  no BATCHLAS_{GEMM,TRSM}_ROUTE.  This is exactly what
#        BATCHLAS_POTRF_ROUTE=blocked delivers in a vendor-present build today,
#        and it is NOT all-vendor: the panel trsm already resolves Native:Blocked
#        for most types and batches.
#   nn   both forced native.  The VENDOR-FREE configuration -- what
#        build-novendor runs, and the number the campaign is about.
#   VV   both forced vendor.  A correctness control (the native trsm is wrong on
#        these shapes at large batch, see README) and the pure cost of the
#        driver's schedule against cuSOLVER's.
#
# Every arm's residual, upper-triangle-preservation, non-finite count and info
# are checked IN THE SAME PROCESS on the SAME buffer BEFORE any timing, so no
# row here can be a wrong answer without saying so.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
BIN=${BIN:-./bench}
OUT=${OUT:-$D/main.csv}
ERR=${ERR:-$D/main.err}
REPS=${REPS:-5}
export BENCH_WARM_S=${BENCH_WARM_S:-1.5}
echo "cfg,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
: > "$ERR"

cell () { # type n batch
  local t=$1 n=$2 b=$3
  for cfg in def nn VV; do
    unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
    case $cfg in
      nn) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native ;;
      VV) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=vendor ;;
    esac
    CUDA_VISIBLE_DEVICES=$GPU $BIN ab "$t" "$n" "$b" "$REPS" 2>>"$ERR" \
      | sed "s/^/$cfg,/" >> "$OUT"
  done
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
}

for t in float double cfloat cdouble; do
  cell "$t" 128   512
  cell "$t" 128  2048
  cell "$t" 256   256
  cell "$t" 256  1024
  cell "$t" 512   128
  cell "$t" 512   512
  cell "$t" 1024  128
  cell "$t" 1024  256
  cell "$t" 2048   32
  cell "$t" 2048   64
done
echo "wrote $OUT"

#!/usr/bin/env bash
# Is the info != 0 failure IN THE DRIVER, or in the kernels it injects?
# Forces BOTH injected calls onto the vendor explicitly (the default is not the
# same thing: the measure phase found the panel trsm already resolves
# Native:Blocked for float at batch >= 128 and for the other three types at
# batch >= 8 with no env set).  If the failures survive all-vendor, the driver
# is at fault; if they vanish, a native kernel is.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
BIN=${BIN:-./bench}
OUT=${OUT:-$D/allvendor.csv}
echo "cfg,rep,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
for t in float cdouble; do
for nb in "1024 256" "512 256" "1024 128"; do
  set -- $nb; n=$1; b=$2
  for cfg in VV Vn nV nn; do
    unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
    case $cfg in
      VV) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=vendor ;;
      Vn) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=native ;;
      nV) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=vendor ;;
      nn) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native ;;
    esac
    for r in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$GPU BENCH_WARM_S=0.2 $BIN ab "$t" "$n" "$b" 2 2>>"$D/allvendor.err" \
        | sed "s/^/$cfg,$r,/" >> "$OUT"
    done
  done
done
done
unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
echo "wrote $OUT"

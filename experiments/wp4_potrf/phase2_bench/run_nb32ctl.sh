#!/usr/bin/env bash
# The control for run_nb32.sh.  nb=32 fails for all four types with both calls
# native -- but that run forces the GEMM native too, and "the native gemm is
# innocent" was only established at the DEFAULT nb.  This repeats nb=32 with the
# gemm still native and the trsm on the vendor.  If it is clean, the V1 CTA trsm
# kernel is the whole defect at nb=32 and the localisation holds.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
OUT="$D/nb32ctl.csv"
echo "cfg,rep,variant,type,n,batch,nbused,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
: > "$D/nb32ctl.err"
export BENCH_WARM_S=0.3 BATCHLAS_POTRF_NB=32
for cfg in nV VV; do
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
  case $cfg in
    nV) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=vendor ;;
    VV) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=vendor ;;
  esac
  for t in float double cfloat cdouble; do
    for r in 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$GPU ./bench ab "$t" 1024 256 2 2>>"$D/nb32ctl.err" \
        | sed "s/^/$cfg,$r,/" >> "$OUT"
    done
  done
done
unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE BATCHLAS_POTRF_NB
echo "wrote $OUT"

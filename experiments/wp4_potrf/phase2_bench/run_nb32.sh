#!/usr/bin/env bash
# Does nb = 32 dodge the native-trsm defect?
#
# WHY THE QUESTION IS SHARP: at nb = 32 the panel solve's triangular order is
# exactly trsm_cta_max_n<T>(), so trsm_native_blocked does NOT block at all --
# it falls through to a single V1 CTA solve (trsm_native.cc:696 onward).  If the
# failures survive nb = 32 the defect is in the V1 KERNEL, not in the blocking
# above it, and that is a different bug and a different fix.  The nb sweep hinted
# both ways: float and cfloat came back clean at nb = 32 while double and
# cdouble did not, and the failure is non-deterministic, so one run of each
# settles nothing.  Six repeats per type.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
OUT="$D/nb32.csv"
echo "nb,rep,variant,type,n,batch,nbused,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
: > "$D/nb32.err"
export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native
export BENCH_WARM_S=0.3
for nb in 32 64; do
for t in float double cfloat cdouble; do
for r in 1 2 3 4 5 6; do
  BATCHLAS_POTRF_NB=$nb CUDA_VISIBLE_DEVICES=$GPU ./bench ab "$t" 1024 256 2 2>>"$D/nb32.err" \
    | sed "s/^/$nb,$r,/" >> "$OUT"
done
done
done
unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
echo "wrote $OUT"

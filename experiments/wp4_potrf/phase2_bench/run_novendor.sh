#!/usr/bin/env bash
# The SAME grid in build-novendor (-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF), with NO
# env set at all.  This is the cross-check that the vendor-present `nn`
# configuration really is what a vendor-free build runs: `nn` reaches native by
# FORCING, which bypasses preferred() but not supports(), while build-novendor
# reaches it through the vendor-free fallback at route_resolve.hh:60-63.  Those
# are different code paths and this repository has shipped four bugs from
# assuming forced == supported.
#
# There is no vendor arm here to interleave against -- the binary cannot link
# one -- so the reference is the vendor time from the SAME cell of main.csv.
# That is a cross-process comparison and is stated as such.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
OUT=${OUT:-$D/novendor.csv}
REPS=${REPS:-5}
export BENCH_WARM_S=${BENCH_WARM_S:-1.5}
echo "cfg,variant,type,n,batch,nb,W,med_ms,min_ms,rel_sd,gflops,residual,upper_changed,nonfinite,info_nonzero" > "$OUT"
: > "$D/novendor.err"
unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE BATCHLAS_POTRF_ROUTE
for t in float double cfloat cdouble; do
  for nb in "128 512" "256 256" "512 128" "1024 128" "1024 256" "2048 64"; do
    set -- $nb; n=$1; b=$2
    CUDA_VISIBLE_DEVICES=$GPU ./bench_nv ab "$t" "$n" "$b" "$REPS" 2>>"$D/novendor.err" \
      | sed "s/^/novendor,/" >> "$OUT"
  done
done
echo "wrote $OUT"

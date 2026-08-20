#!/usr/bin/env bash
# Same question at a REAL leading dimension.
#
# Every complex GEMM in the demand table that comes from herk/her2k/hemm/trmm or
# from a blocked factorisation is issued on SUB-VIEWS that carry the parent's ld,
# so ld == rows (pad 0) is the case those callers never present. A native GEMM in
# this tree has already been measured to lose ~2x when the ld is strided while
# being at parity at ld == rows, so a pad-0-only answer would overstate how close
# the vendor-free build is.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-1}"
OUT="$D/cost_strided_ld.csv"
REPS="${REPS:-7}"
PAD="${PAD:-384}"

SHAPES=(
  "panel_herk129_cn   129 129 48   2048 C N"
  "panel_her2k96_nc   96  96  64   2048 N C"
  "square512_nn       512 512 512  64   N N"
)

echo "route,type,m,n,k,batch,tA,tB,beta,padA,padB,padC,reps,median_ms,min_ms,rel_sd,gflops,tag" > "$OUT"
for spec in "${SHAPES[@]}"; do
  read -r tag m n k b tA tB <<< "$spec"
  for type in cfloat cdouble; do
    for beta in ${BETAS:-1}; do
      for route in vendor native; do
        line=$(BATCHLAS_BENCH_BETA="$beta" BATCHLAS_BENCH_LD_PAD="$PAD" \
               BATCHLAS_GEMM_ROUTE="$route" GPU_GUARD_MAX_WAIT=1800 \
               "$D/../../gpu_guard.sh" "$GPU" \
               "$D/cx_gemm_bench" "$type" "$m" "$n" "$k" "$b" "$tA" "$tB" "$REPS" 2>/dev/null)
        [[ -z "$line" ]] && { echo "FAILED: $route $type $tag" >&2; continue; }
        echo "$route,$line,$tag" >> "$OUT"
        echo "$route,$line,$tag"
      done
    done
  done
done
echo "PAD_DONE -> $OUT"

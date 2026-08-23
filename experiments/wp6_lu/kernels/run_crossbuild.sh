#!/usr/bin/env bash
# The SAME native routes in the two BUILDS.
#
# The A/B grid compares "vendor build, routes pinned vendor" against "vendor-free
# build, Auto". Any difference between those two therefore folds in TWO things:
# the LU arm, and whatever the routed trsm/gemm underneath it resolves to, which
# is not the same in the two builds. This run separates them by pinning the LU
# routes to native in the VENDOR-PRESENT build, where the trsm's own trailing
# work can still reach cuBLAS.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WARM_S="${WARM_S:-0.6}" NPROBE=1 NTRANS=1
export BATCHLAS_GETRF_ROUTE=native:blocked
export BATCHLAS_GETRS_ROUTE=native:blocked
export BATCHLAS_GETRI_ROUTE=native:blocked
for op in ${OPS:-getrs getri getrf}; do
  for t in ${TYPES:-float cdouble}; do
    for cell in ${CELLS:-512:512 2048:32}; do
      n="${cell%%:*}"; b="${cell##*:}"
      echo -n "V "; "$D/luverify_v" "$op" "$t" "$n" 1 "$b" "${REPS:-5}" 2>/dev/null
      echo -n "N "; "$D/luverify_nv" "$op" "$t" "$n" 1 "$b" "${REPS:-5}" 2>/dev/null
    done
  done
done

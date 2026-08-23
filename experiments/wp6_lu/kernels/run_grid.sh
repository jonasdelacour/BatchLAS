#!/usr/bin/env bash
# The WP6 A/B grid: the PUBLIC getrf / getrs / getri at the saturating batch
# schedule the baseline established.
#
# ONE ARM PER INVOCATION, and the arm is chosen by the BINARY, not by a pin:
#   luverify_v  with the routes pinned to `vendor`  -> cuBLAS
#   luverify_nv (linked against build-novendor)     -> native, with NO pin at all,
#                                                      so the tier is whatever
#                                                      resolve_route picks
# That is what makes "vendor-free" mean the BUILD rather than a forced route, and
# it sidesteps the pin trap: a bare BATCHLAS_GETRF_ROUTE=native is
# {Native, Algorithm::Auto}, which names NEITHER tier, so supports() refuses it and
# route_resolve.hh:175 falls through to automatic() -- i.e. silently to cuBLAS.
#
# MEASUREMENT HYGIENE ACTUALLY APPLIED:
#   * one GPU pinned (this box has two RTX 4090s; contention fabricates results);
#   * WARM_S seconds of untimed warm-up per run -- a cold SYCL JIT once fabricated
#     a 3.7x loss;
#   * medians of REPS, with mean and relative sd on every row so a noisy cell is
#     visible rather than averaged away;
#   * correctness checked IN PROCESS on every timed row against a HOST oracle, so
#     a fast wrong answer cannot be reported as a win;
#   * the RESOLVED ROUTE printed on every row;
#   * nothing run under BATCHLAS_KERNEL_TRACE (~60% inflation).
#
# usage: run_grid.sh <out.csv> <binary> <pin|none>
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?out.csv}"
BIN="$D/${2:?binary}"
PIN="${3:-none}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WARM_S="${WARM_S:-1.0}"
export NPROBE="${NPROBE:-1}"
REPS="${REPS:-5}"

unset BATCHLAS_GETRF_ROUTE BATCHLAS_GETRS_ROUTE BATCHLAS_GETRI_ROUTE
if [ "$PIN" != none ]; then
  export BATCHLAS_GETRF_ROUTE="$PIN" BATCHLAS_GETRS_ROUTE="$PIN" BATCHLAS_GETRI_ROUTE="$PIN"
fi

# n:batch -- the baseline's saturating schedule. Its two known pessimisms are
# stated in the write-up rather than silently corrected: cuBLAS getri float n=256
# is best at batch 256 and degrades to the grid's 2048, and cuBLAS getrf float
# n=1024 is better at batch 256 than at the grid's 128.
CELLS="${CELLS:-32:8192 64:8192 128:4096 256:2048 512:512 1024:128 2048:32}"
TYPES="${TYPES:-float double cfloat cdouble}"
OPS="${OPS:-getrf getri getrs}"

: > "$OUT"
echo "op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,extra,ntpiv,flag" >> "$OUT"
for op in $OPS; do
  for t in $TYPES; do
    for cell in $CELLS; do
      n="${cell%%:*}"; b="${cell##*:}"
      timeout "${TMO:-1200}" "$BIN" "$op" "$t" "$n" 1 "$b" "$REPS" >> "$OUT" 2>/dev/null \
        || echo "$op,$t,$n,1,$b,TIMEOUT_OR_THROW,-,-,-,-,-,-,-,-,BAD" >> "$OUT"
    done
  done
done
echo "wrote $OUT"

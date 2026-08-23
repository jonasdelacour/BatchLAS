#!/usr/bin/env bash
# CTA against BLOCKED at the orders where BOTH are supported, in the vendor-free
# build. This is the question native_tier_preferred() exists to answer, and it is
# NOT the question preferred() can answer -- preferred() runs above the
# vendor-free walk regardless of vendor_available, so a window written to fix the
# tier choice would also drag vendor-present traffic onto shapes where cuBLAS
# beats both natives (route_resolve.hh's header block).
#
# The CTA arm is refused above the per-type ceiling (float 155, double 109,
# cfloat 109, cdouble 77), and a refused pin falls through to automatic() -- so
# the ROUTE COLUMN must be read on every row, not assumed.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export WARM_S="${WARM_S:-0.6}" NPROBE=1
for t in float double cfloat cdouble; do
  for cell in ${CELLS:-32:8192 64:8192 76:8192 100:4096 128:4096}; do
    n="${cell%%:*}"; b="${cell##*:}"
    for pin in native:cta native:blocked; do
      BATCHLAS_GETRF_ROUTE="$pin" "$D/${BIN:-luverify_nv}" getrf "$t" "$n" 1 "$b" "${REPS:-5}" 2>/dev/null
    done
  done
done

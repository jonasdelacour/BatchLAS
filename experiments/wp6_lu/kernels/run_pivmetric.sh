#!/usr/bin/env bash
# The pivot METRIC, on a matrix built so cabs1 and the modulus disagree.
# See run_pivmetric() in luverify.cpp for why the ordinary sweep is blind to this.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
for pin in vendor native:cta native:blocked; do
  for t in cfloat cdouble; do
    BATCHLAS_GETRF_ROUTE="$pin" "$D/${BIN:-luverify_v}" pivmetric "$t" "${N:-4}" 1 2 1
  done
done

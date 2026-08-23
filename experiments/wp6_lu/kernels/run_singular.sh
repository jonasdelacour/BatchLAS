#!/usr/bin/env bash
# The exact-zero info contract, native arms against the VENDOR arm on the SAME
# matrix, in the SAME build -- which is the comparison that matters, because the
# contract getrf must not break is cuBLAS's, and cuBLAS itself diverges from host
# LAPACK on the predicate "U(i,i) == 0" (measured in the interface-contract
# brief).
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
for pin in vendor native:cta native:blocked; do
  echo "--- BATCHLAS_GETRF_ROUTE=$pin"
  for t in float double cfloat cdouble; do
    BATCHLAS_GETRF_ROUTE="$pin" "$D/${BIN:-luverify_v}" singular "$t" "${N:-6}" 1 3 1
  done
done

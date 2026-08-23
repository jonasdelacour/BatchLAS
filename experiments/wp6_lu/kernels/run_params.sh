#!/usr/bin/env bash
# The launch geometry, asked through the same pure functions the driver calls.
# Evidence for two things a residual cannot show: which orders the CTA tier
# actually admits (its ceiling is a function of the RUNTIME local-memory budget,
# never a baked constant), and which orders take the GLOBAL panel leaf.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
for t in float double cfloat cdouble; do
  for n in ${NS:-32 64 76 100 110 128 155 156 190 256 760 800 2048}; do
    "$D/${BIN:-luverify_v}" params "$t" "$n" 1 1 1
  done
done

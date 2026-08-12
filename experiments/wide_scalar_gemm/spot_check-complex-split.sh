#!/usr/bin/env bash
# Quick post-edit re-check on the SYCL CPU device (no GPU touched).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
for dt in cfloat cdouble double float; do
  ./verify-cpu-complex-split --dtype "$dt" --m 128 --n 128 --k 32 --batch 2 --iters 1 --warmup 0 \
      --beta 1 --device cpu 2>&1 | grep -E '^RESULT|check |CORRECTNESS'
done

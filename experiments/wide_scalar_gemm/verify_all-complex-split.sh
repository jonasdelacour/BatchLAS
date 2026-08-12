#!/usr/bin/env bash
# Correctness matrix on a SYCL CPU device -- no GPU is touched.  The kernel
# source is identical on either device, so this validates the index arithmetic,
# the planar split/join, the interleaved path, the predicated path and the
# beta != 0 epilogue.  Each row itself runs four checks (fast + predicated on
# an aligned shape, a ragged shape, and the timed shape).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# Exactly the DPC++ lib dir: an inherited LD_LIBRARY_PATH (the HPC SDK is on it
# here) shadows the OpenCL ICD loader and the CPU device disappears.
export LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
fail=0
for dt in cfloat cdouble double float; do
  layouts="planar interleaved"
  case "$dt" in double|float) layouts="planar" ;; esac
  for lay in $layouts; do
    for tile in "128x128x8/8x8" "128x128x8/8x4" "64x64x8/4x4" "64x64x16/4x4"; do
      for beta in 0 1; do
        out=$(./verify-cpu-complex-split --dtype "$dt" --tile "$tile" --layout "$lay" --m 128 --n 128 \
                --k 32 --batch 2 --iters 1 --warmup 0 --beta "$beta" --device cpu 2>&1)
        rc=$?
        line=$(printf '%s\n' "$out" | grep '^RESULT' || true)
        err=$(printf '%s\n' "$out" | grep -o 'maxrelerr=[0-9.e+-]*' | cut -d= -f2)
        if [ $rc -ne 0 ] || [ -z "$line" ]; then
          printf '%-8s %-14s %-11s beta=%s  FAILED rc=%s\n' "$dt" "$tile" "$lay" "$beta" "$rc"
          printf '%s\n' "$out" | tail -3
          fail=1
        else
          printf '%-8s %-14s %-11s beta=%s  maxrelerr=%s\n' "$dt" "$tile" "$lay" "$beta" "$err"
        fi
      done
    done
  done
done
if [ $fail -eq 0 ]; then echo "ALL PASS"; else echo "FAILURES PRESENT"; fi
exit $fail

#!/usr/bin/env bash
# Run-to-run variance on the one cell where a "beats cuBLAS" claim would be made.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="$HERE/../../gpu_guard.sh"
cd "$HERE"
"$G" 0 ./cublas_baseline --dtype double --m 512 --n 512 --k 512 --batch 32 --iters 60 --warmup 10 >/dev/null 2>&1
for i in 1 2 3 4 5; do
  "$G" 0 ./cublas_baseline --dtype cfloat --m 512 --n 512 --k 512 --batch 128 --beta 1 --iters 50 --warmup 20 2>/dev/null
done
for i in 1 2 3 4 5; do
  "$G" 0 ./tile-complex-split --dtype cfloat --tile 128x128x8/8x8 --layout planar --m 512 --n 512 --k 512 --batch 128 --beta 1 --iters 50 --warmup 20 2>/dev/null | grep RESULT
done
for i in 1 2 3; do
  "$G" 0 ./cublas_baseline --dtype cdouble --m 512 --n 512 --k 512 --batch 128 --beta 1 --iters 20 --warmup 5 2>/dev/null
done
for i in 1 2 3; do
  "$G" 0 ./tile-complex-split --dtype cdouble --tile 64x64x8/4x4 --layout interleaved --m 512 --n 512 --k 512 --batch 128 --beta 1 --iters 20 --warmup 5 2>/dev/null | grep RESULT
done
echo "REPEAT DONE"

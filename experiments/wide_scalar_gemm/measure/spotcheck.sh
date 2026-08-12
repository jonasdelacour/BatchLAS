#!/usr/bin/env bash
# Independent reproduction of three cells from bench.log, same iters/warmup.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="$HERE/../../gpu_guard.sh"
cd "$HERE"

echo "== clocks before =="
nvidia-smi --id=0 --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader

echo "== warm =="
"$G" 0 ./cublas_baseline --dtype double --m 512 --n 512 --k 512 --batch 32 --iters 60 --warmup 10 >/dev/null 2>&1
nvidia-smi --id=0 --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader

echo "== cell 1: cublas double 512^3 b128 beta=1  (log: ms=27.5846 tflops=1.246) =="
"$G" 0 ./cublas_baseline --dtype double --m 512 --n 512 --k 512 --batch 128 --beta 1 --iters 30 --warmup 10 2>/dev/null

echo "== cell 2: 128x128-t8x4 double 512^3 b128 beta=1  (log: ms=25.1153 tflops=1.368) =="
"$G" 0 ./tile-128x128-k8-t8x4 --dtype double --m 512 --n 512 --k 512 --batch 128 --beta 1 --iters 30 --warmup 10 2>/dev/null | grep RESULT

echo "== cell 3: 64x64-k16-t4x4 cfloat 1024^3 b32 beta=1 (log: ms=5.6659 tflops=48.515) =="
"$G" 0 ./tile-64x64-k16-t4x4 --dtype cfloat --m 1024 --n 1024 --k 1024 --batch 32 --beta 1 --iters 30 --warmup 10 --skip-check 2>/dev/null | grep RESULT

echo "== cell 4: cublas cfloat 512^3 b128 beta=1 (log: ms=2.8012 tflops=49.064) =="
"$G" 0 ./cublas_baseline --dtype cfloat --m 512 --n 512 --k 512 --batch 128 --beta 1 --iters 30 --warmup 10 2>/dev/null

echo "== cell 5: 128x64-t8x4 cfloat 256^3 b512 beta=1 (log: ms=1.6491 tflops=41.670) =="
"$G" 0 ./tile-128x64-k8-t8x4 --dtype cfloat --m 256 --n 256 --k 256 --batch 512 --beta 1 --iters 30 --warmup 10 2>/dev/null | grep RESULT

echo "== clocks after =="
nvidia-smi --id=0 --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader

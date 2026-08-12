#!/usr/bin/env bash
# GAP FILL A: tile-complex-split was never run at all (absent from bench.log).
# Tile/layout scan at the middle shape, beta=1, with its correctness checks ON.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="$HERE/../../gpu_guard.sh"
cd "$HERE"
OUT="$HERE/bench_complex_split.log"
ERRF="$HERE/bench_complex_split.stderr"
: > "$OUT"
: > "$ERRF"

run() { # dtype tile layout m n k batch beta
    local line err
    echo "### $1 $2 $3 $4x$5x$6 b$7 beta=$8" >> "$ERRF"
    line=$("$G" 0 ./tile-complex-split --dtype "$1" --tile "$2" --layout "$3" \
            --m "$4" --n "$5" --k "$6" --batch "$7" --beta "$8" \
            --iters 30 --warmup 10 2>>"$ERRF" | grep '^RESULT')
    [ -z "$line" ] && line="RESULT dtype=$1 tile=$2:$3 m=$4 n=$5 k=$6 batch=$7 beta=$8 ms=FAILED tflops=FAILED"
    err=$(tail -12 "$ERRF" | grep -oE "maxrelerr [0-9.e+-]+|relerr=[0-9.e+-]+" | tr '\n' ' ')
    echo "CAND=complex-split $line || $err" | tee -a "$OUT"
}

# warm
"$G" 0 ./cublas_baseline --dtype double --m 512 --n 512 --k 512 --batch 32 --iters 60 --warmup 10 >/dev/null 2>&1

for TILE in 128x128x8/8x8 128x128x8/8x4 64x64x8/4x4 64x64x16/4x4; do
  for LAY in planar interleaved; do
    run cfloat  "$TILE" "$LAY" 512 512 512 128 1
  done
done
for TILE in 128x128x8/8x8 128x128x8/8x4 64x64x8/4x4 64x64x16/4x4; do
  for LAY in planar interleaved; do
    run cdouble "$TILE" "$LAY" 512 512 512 128 1
  done
done
for TILE in 128x128x8/8x8 128x128x8/8x4 64x64x8/4x4 64x64x16/4x4; do
    run double  "$TILE" planar 512 512 512 128 1
done
run float 128x128x8/8x8 planar 512 512 512 128 1
run float 128x128x8/8x8 planar 512 512 512 128 0
echo "done -> $OUT"

#!/usr/bin/env bash
# GAP FILL B:
#   (1) the INCUMBENT (in-tree Tiled16, which is what BatchLAS actually runs
#       today for every non-float type) across the full grid -- without this the
#       comparison has no floor;
#   (2) tile-complex-split's best tile per dtype across the full grid.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="$HERE/../../gpu_guard.sh"
cd "$HERE"
OUT="$HERE/bench_gapfill_b.log"
: > "$OUT"
SHAPES=("256 256 256 512" "512 512 512 128" "1024 1024 1024 32")

warm() { "$G" 0 ./cublas_baseline --dtype double --m 512 --n 512 --k 512 --batch 32 \
            --iters 60 --warmup 10 >/dev/null 2>&1; }

warm
# ---- incumbent -------------------------------------------------------------
for s in "${SHAPES[@]}"; do
  read -r M N K B <<< "$s"
  for dt in double cfloat cdouble float; do
    for beta in 0 1; do
      line=$("$G" 0 ./tiled16_incumbent --dtype "$dt" --m "$M" --n "$N" --k "$K" \
              --batch "$B" --beta "$beta" --iters 15 --warmup 5 2>/dev/null | grep '^RESULT')
      [ -z "$line" ] && line="RESULT lib=tiled16 dtype=$dt m=$M n=$N k=$K batch=$B beta=$beta ms=FAILED tflops=FAILED"
      echo "CAND=incumbent-tiled16 $line" | tee -a "$OUT"
    done
  done
done

# ---- complex-split, best tile per dtype ------------------------------------
cs() { # dtype tile layout
  for s in "${SHAPES[@]}"; do
    read -r M N K B <<< "$s"
    for beta in 0 1; do
      line=$("$G" 0 ./tile-complex-split --dtype "$1" --tile "$2" --layout "$3" \
              --m "$M" --n "$N" --k "$K" --batch "$B" --beta "$beta" \
              --iters 30 --warmup 10 2>/dev/null | grep '^RESULT')
      [ -z "$line" ] && line="RESULT dtype=$1 tile=$2:$3 m=$M n=$N k=$K batch=$B beta=$beta ms=FAILED tflops=FAILED"
      echo "CAND=complex-split $line" | tee -a "$OUT"
    done
  done
}
warm
cs cfloat  128x128x8/8x8 planar
cs cdouble 64x64x8/4x4   interleaved
cs double  64x64x16/4x4  planar
cs float   128x128x8/8x8 planar
echo "done -> $OUT"

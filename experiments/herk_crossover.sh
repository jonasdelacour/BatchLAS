#!/usr/bin/env bash
# Sweep the GEMM route against the per-batch vendor loop for herk/her2k, so the
# routing threshold in cublas.cc is a measurement rather than a guess.
set -euo pipefail

BIN=${1:?usage: herk_crossover.sh <benchmark binary> <outdir> [dims] [batches]}
OUT=${2:?usage: herk_crossover.sh <benchmark binary> <outdir> [dims] [batches]}
DIMS=${3:-"32 64 128 256 512 1024"}
BATCHES=${4:-"1 2 4 8 16 64 256"}
mkdir -p "$OUT"

for route in expand loop; do
  for n in $DIMS; do
    for batch in $BATCHES; do
      BATCHLAS_EXPAND_ROUTE=$route "$BIN" --backend=CUDA --type=cfloat \
        --min_iters=20 --warmup=3 \
        --csv="$OUT/${route}_${n}_${batch}.csv" "$n" "$n" "$n" "$batch" >/dev/null
    done
  done
done

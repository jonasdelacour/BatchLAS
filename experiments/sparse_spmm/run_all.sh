#!/usr/bin/env bash
# WP8 spmm bake-off: the whole grid, one pass.
#
#   usage: run_all.sh <pass-name>        e.g. run_all.sh pass1
#
# Writes to experiments/sparse_spmm/<pass-name>/. Runs strictly SEQUENTIALLY --
# one measuring process on the box at a time, on device 1 only. See README.md.
#
# Every sweep is run twice, once per route (vendor, native:direct), in separate
# processes, and a coverage table is written beside each timing CSV so the arm
# that ran is read off the instrument rather than off the environment.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
PASS="${1:?pass name}"
OUT="$D/$PASS"
mkdir -p "$OUT"

R="$D/run_spmm.sh"
TYPES="${TYPES:-float double cfloat cdouble}"
ROUTES="${ROUTES:-vendor native:direct}"

# ---------------------------------------------------------------- the grids
# args are: m nnzrow nrhs batch transB beta pattern transA
#
# LANCZOS: m 1024, 3 nnz/row, nrhs 1 vs 2 (lanczos pads 1 -> 2 to defeat a vendor
# SpMV fallback, src/extensions/lanczos.cc:52-54), and a full batch LADDER, because
# the recorded campaign failure mode is reading a ratio off a grid with one batch
# per size. Both patterns: banded is lanczos's real locality, random is the harder
# gather.
LANCZOS_ARGS="1024 3 1,2 8,32,64,128,256,512,1024 0 0 0,1"

# LOBPCG / syevx_filtered: m 1024-4096, 16 nnz/row, nrhs 12-50, batch ladder.
# Scattered pattern only -- that is the pattern a filtered eigensolve has.
LOBPCG_ARGS="1024,2048,4096 16 12,25,50 8,32,128,512 0 0 1"

# The three named cells crossed with the two layout levers and both beta values,
# so a ratio quoted at a cell has its transB / beta neighbours measured beside it.
run_cells() {
  local route=$1 type=$2 ta=$3
  "$R" "$route" "cellsA${ta}" BM_SPMM_Grid "$type" "$OUT" 1024 3  2  512 0,1 0,1 0,1 "$ta"
  "$R" "$route" "cellsB${ta}" BM_SPMM_Grid "$type" "$OUT" 1024 16 12 512 0,1 0,1 0,1 "$ta"
  "$R" "$route" "cellsC${ta}" BM_SPMM_Grid "$type" "$OUT" 2048 16 25 128 0,1 0,1 0,1 "$ta"
}

for type in $TYPES; do
  for route in $ROUTES; do
    for ta in 0 1; do
      "$R" "$route" "lanczos_ta${ta}" BM_SPMM_Grid "$type" "$OUT" $LANCZOS_ARGS "$ta"
      "$R" "$route" "lobpcg_ta${ta}"  BM_SPMM_Grid "$type" "$OUT" $LOBPCG_ARGS  "$ta"
      run_cells "$route" "$type" "$ta"
    done
  done
done
echo "PASS $PASS complete -> $OUT"

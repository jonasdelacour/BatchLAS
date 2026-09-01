#!/usr/bin/env bash
# THE SCATTER ARM'S nrhs BOUNDARY, WITH A BATCH LADDER UNDER EVERY CELL.
#
# WHY THIS SWEEP EXISTS AND THE bnd_* ONES DO NOT REPLACE IT. bnd_scatter_a/b
# locate the boundary beautifully -- ratio rises monotonically with nrhs and the
# 1.10 line falls between nrhs 4 and 12 for every type -- but each of their cells
# is measured at ONE batch. A single batch is not a saturation measurement, and
# this campaign's rule is that a ratio may be quoted only where saturation was
# MEASURED, on the ladder, not interpolated from a neighbouring nnz/row.
#
# So: the same nrhs walk, at two m and two nnz/row, with a real batch ladder under
# every cell, so that the recommended clause's boundary is bracketed by a
# measured non-winner AT SATURATION rather than next to one.
#
#   usage: run_scatter_ladder.sh <pass-name>
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
PASS="${1:?pass name}"
OUT="$D/$PASS"
mkdir -p "$OUT"
R="$D/run_spmm.sh"

for type in ${TYPES:-float double cfloat cdouble}; do
  for route in ${ROUTES:-vendor native:direct}; do
    "$R" "$route" "scl" BM_SPMM_Grid "$type" "$OUT" \
        1024,2048 3,16 1,2,4,8,12 128,256,512,1024 0 0 1 1
  done
done
echo "SCATTER LADDER $PASS complete -> $OUT"

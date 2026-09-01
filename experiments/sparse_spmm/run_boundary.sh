#!/usr/bin/env bash
# THE BOUNDARY SWEEPS. pass1/pass2 establish two clean regimes and two clean
# verdicts; these locate the LINES between them, because a recommendation whose
# boundary sits at the edge of the sampled grid is exactly the objection this
# campaign's own WP7 audit raised and refused.
#
# BOUNDARY 1 -- the transposed (scatter) arm.
#   pass1: transA=Trans WINS the lanczos regime (nnz/row=3, nrhs<=2: ratio
#   0.054-0.918 over all four types, 0 rows above the 1.10 gate) and LOSES the
#   LOBPCG regime (nnz/row=16, nrhs 12-50: median 1.08-1.24, 119 of 179
#   saturated rows above the gate). Two axes moved at once. This sweep separates
#   them: nnz/row 3..16 crossed with nrhs 1..25, at fixed m and batch.
#
# BOUNDARY 2 -- the ONE non-winning family on the NoTrans (gather) arm.
#   pass1 has exactly two saturated NoTrans rows above the gate, both cfloat, both
#   m=2048 nnz/row=16 nrhs=25 batch=128 transB=Trans pattern=BANDED (ratio 1.934
#   and 1.981), while the same cell at float is 0.460 and at the scattered pattern
#   is 0.760-1.019. So the suspect combination is (complex<float>, banded,
#   transB=Trans). This sweep walks it over m, nrhs and batch, on all four types,
#   so the family is bounded rather than anecdotal.
#
#   usage: run_boundary.sh <pass-name>
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
PASS="${1:?pass name}"
OUT="$D/$PASS"
mkdir -p "$OUT"
R="$D/run_spmm.sh"

for type in ${TYPES:-float double cfloat cdouble}; do
  for route in ${ROUTES:-vendor native:direct}; do
    # Boundary 1: the (nnz/row, nrhs) plane on the transposed arm, at two
    # (m, batch) points so a verdict is not read off one geometry.
    "$R" "$route" "bnd_scatter_a" BM_SPMM_Grid "$type" "$OUT" \
        1024 3,6,8,12,16 1,2,4,8,12,25 512 0 0 1 1
    "$R" "$route" "bnd_scatter_b" BM_SPMM_Grid "$type" "$OUT" \
        2048 3,6,8,12,16 1,2,4,8,12,25 128 0 0 1 1
    # ... and the same plane on the gather arm, so the two arms are compared on
    # ONE grid rather than across two different ones.
    "$R" "$route" "bnd_gather_a" BM_SPMM_Grid "$type" "$OUT" \
        1024 3,6,8,12,16 1,2,4,8,12,25 512 0 0 1 0

    # Boundary 2: the banded + transB=Trans family on the gather arm.
    "$R" "$route" "bnd_banded" BM_SPMM_Grid "$type" "$OUT" \
        1024,2048,4096 16 12,25,50 128,512 0,1 0 0 0
  done
done
echo "BOUNDARY $PASS complete -> $OUT"

#!/usr/bin/env bash
# WHERE EXACTLY DOES THE ONE NoTrans NON-WINNER START?
#
# bnd1/bnd2 isolate it to (complex<float>, transB=Trans, banded pattern, nnz/row
# 16) on the GATHER arm: ratio 0.74-1.00 at nrhs=12 and 1.69-2.03 at nrhs=25 and
# 50, at every m in {1024, 2048, 4096} and both batches. The boundary is
# therefore somewhere in 12 < nrhs <= 25 and this walks it, with the scattered
# pattern measured alongside so the pattern's contribution is visible rather than
# assumed.
#
# The suspect mechanism is the register block: kNCmax<Cx<float>> is 8
# (spmm_native.cc:88), so nrhs=25 needs ceil(25/8)=4 passes over A's values and
# indices while float's kNCmax=16 needs 2. If that were the whole story the same
# cliff would appear at nrhs=9, so nrhs=9 and 17 are measured explicitly.
#
#   usage: run_cfloat_edge.sh <pass-name>
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
PASS="${1:?pass name}"
OUT="$D/$PASS"
mkdir -p "$OUT"
R="$D/run_spmm.sh"

for type in ${TYPES:-cfloat float}; do
  for route in ${ROUTES:-vendor native:direct}; do
    "$R" "$route" "cfedge" BM_SPMM_Grid "$type" "$OUT" \
        2048 16 8,9,12,16,17,20,25,32,50 512 1 0 0,1 0
  done
done
echo "CFEDGE $PASS complete -> $OUT"

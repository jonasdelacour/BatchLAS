#!/usr/bin/env bash
# THE SATURATION EXTENSION.
#
# pass1's lanczos ladder stops at batch 1024, and at that batch the NATIVE gather
# is still falling: per-item time 0.023 -> 0.016 -> 0.012 -> 0.010 us over batch
# 128/256/512/1024 (float, m=1024, 3 nnz/row, nrhs=1). The vendor arm is flat from
# batch 256. So on that ladder only one arm is saturated, and the campaign rule is
# that a ratio may be quoted only at saturation.
#
# The bias has a KNOWN SIGN -- an unsaturated native arm makes the native number
# look WORSE than it is -- so the pass-1 lanczos ratios are a conservative bound
# rather than a flattering one. This sweep removes the caveat instead of arguing
# about it: it walks the batch ladder up to 8192 until the native per-item time
# goes flat too.
#
#   usage: run_satext.sh <pass-name>
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
PASS="${1:?pass name}"
OUT="$D/$PASS"
mkdir -p "$OUT"
R="$D/run_spmm.sh"

# batch 8192 x m=1024 x 3 nnz/row x nrhs=2, cdouble: A 402 MB + indices 100 MB +
# B and C 268 MB each -- about 1.0 GB, comfortably inside the card.
LADDER="1024 3 1,2 1024,2048,4096,8192 0 0 0,1"

for type in ${TYPES:-float double cfloat cdouble}; do
  for route in ${ROUTES:-vendor native:direct}; do
    "$R" "$route" "satext_ta0" BM_SPMM_Grid "$type" "$OUT" $LADDER 0
  done
done
echo "SATEXT $PASS complete -> $OUT"

#!/usr/bin/env bash
# WP8 -- THE SMALL-BATCH CORNER OF THE GATHER ARM (transA == NoTrans only).
#
# WHY THIS SWEEP EXISTS
# ---------------------
# Every ratio in pass1/pass2/bnd*/cfedge*/sat*/scl* is batch >= 128 at
# saturation, because the campaign rule forbids quoting an unsaturated ratio as
# an algorithmic result. But preferred() is consulted on EVERY call, batch = 1
# included. The recommended gather clause therefore either needs a batch floor
# or needs this corner measured -- and the campaign has a direct precedent for
# getting that wrong (getri at batch <= 32 beat cuBLAS 1.7-28x and was left
# UNROUTED because its low-batch region was never bracketed at every admitted
# point).
#
# WHAT THIS SWEEP IS AND IS NOT
# -----------------------------
# It is NOT an algorithm comparison. At batch 1-16 the timed region is dominated
# by launch latency, route dispatch and the vendor's per-call bufferSize re-query;
# no number below is a kernel-efficiency ratio and none is quoted as one.
# It IS a HARM CHECK. The question the clause needs answered is not "is native
# faster at batch 4" but "is native ever materially SLOWER below batch 128, and
# where". Both a win and a wash license the clause; only a reproducible loss
# forces a floor.
#
# THE GRID -- drawn from the cells the recommended clause would admit
#   sbL  m=1024 nnz/row=3  nrhs=2  transB=0 both patterns, beta 0 AND 1  (lanczos)
#   sbM  m=1024 nnz/row=16 nrhs=12 transB=0 scattered                    (LOBPCG M)
#   sbS  m=2048 nnz/row=16 nrhs=25 transB=0 scattered                    (LOBPCG S)
#   sbB  m=4096 nnz/row=16 nrhs=50 transB=0 scattered                    (LOBPCG max)
#   sbT  m=2048 nnz/row=16 nrhs=25 transB=1 BANDED                       (the known
#        large-batch loser family: cfloat measures 1.71-1.73 here at batch 512.
#        Measured for ALL FOUR types so that if the loss also exists small, the
#        exclusion can be widened on evidence rather than on suspicion.)
# Batch ladder on every family: 1 2 4 8 16 32 64 128. The 128 rung overlaps the
# saturated grid on purpose -- it is the cross-sweep anchor that says this sweep
# is measuring the same thing pass1/pass2 measured.
#
#   usage: run_smallbatch.sh <pass-name>       e.g. run_smallbatch.sh sb1
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
PASS="${1:?pass name}"
OUT="$D/$PASS"
mkdir -p "$OUT"
R="$D/run_spmm.sh"

B="1,2,4,8,16,32,64,128"

for type in ${TYPES:-float double cfloat cdouble}; do
  for route in ${ROUTES:-vendor native:direct}; do
    # args: m nnzrow nrhs batch transB beta pattern transA
    "$R" "$route" sbL BM_SPMM_Grid "$type" "$OUT" 1024 3  2  "$B" 0 0,1 0,1 0
    "$R" "$route" sbM BM_SPMM_Grid "$type" "$OUT" 1024 16 12 "$B" 0 0   1   0
    "$R" "$route" sbS BM_SPMM_Grid "$type" "$OUT" 2048 16 25 "$B" 0 0   1   0
    "$R" "$route" sbB BM_SPMM_Grid "$type" "$OUT" 4096 16 50 "$B" 0 0   1   0
    "$R" "$route" sbT BM_SPMM_Grid "$type" "$OUT" 2048 16 25 "$B" 1 0   0   0
  done
done
echo "SMALLBATCH $PASS complete -> $OUT"

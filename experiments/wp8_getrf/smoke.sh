#!/usr/bin/env bash
# Correctness smoke over shapes the timing grid never reaches: orders that
# straddle the block width, the SHORT FINAL PANEL (n = 129, 100, 33, 5) and tiny
# batches. The A/B harness's bit-identity check is the assertion here, not the
# time -- WARM_S is deliberately short.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export WARM_S="${WARM_S:-0.2}"
A="${A:-inloop}"
B="${B:-defer_gather}"
rc=0
while read -r t n b; do
  [ -z "$t" ] && continue
  "$D/getrfab_nv" "$t" "$n" "$b" 3 "$A" "$B" || rc=1
done <<'CELLS'
float 5 3
double 5 3
cfloat 31 3
cdouble 32 3
float 33 3
double 33 5
cfloat 63 3
cdouble 64 3
float 96 3
double 100 4
cfloat 128 3
cdouble 129 3
float 129 3
double 129 3
cfloat 129 3
float 257 2
double 512 2
cfloat 1024 2
cdouble 1024 8
CELLS
echo "smoke rc=$rc"
exit $rc

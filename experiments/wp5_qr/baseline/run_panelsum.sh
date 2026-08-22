#!/usr/bin/env bash
# THE BLAS-3 LOWER BOUND for a blocked vendor-free geqrf.
#
# Walk EVERY panel position a right-looking blocked geqrf would visit on an
# N x N parent at block width nb, time both trailing GEMMs at each, and sum. The
# sum is what the trailing update alone will cost -- a lower bound on the driver,
# ignoring the panel factorisation and any launch gaps. Comparing that sum to
# the measured cuSOLVER geqrf of the same (type, N, batch) turns "the target"
# into a number with headroom attached, instead of an aspiration.
#
# Both builds, interleaved per cell.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export WARM_S=${WARM_S:-0.2}
echo "build,which,type,N,nb,j0,batch,m,n,k,med_ms,rel_sd,GFLOPs"
body() {
  N=1024; nb=56; b=128
  for t in float double cfloat cdouble; do
    j0=0
    while [ "$j0" -lt "$N" ]; do
      for w in G1 G3; do
        echo -n "vendor,";     "$D/gemmtrail_v"  "$t" "$N" "$nb" "$j0" "$b" "$w" 3
        echo -n "vendorfree,"; "$D/gemmtrail_nv" "$t" "$N" "$nb" "$j0" "$b" "$w" 3
      done
      j0=$((j0 + nb))
    done
  done
}
bash /home/jonaslacour/BatchLAS/experiments/gpu_guard.sh 1 bash -c "$(declare -f body); D=$D WARM_S=$WARM_S body"

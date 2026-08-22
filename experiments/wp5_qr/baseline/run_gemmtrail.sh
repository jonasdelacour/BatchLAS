#!/usr/bin/env bash
# geqrf's WY trailing-update GEMM pair on real sub-views, vendor build vs
# vendor-free build. INTERLEAVED: each cell runs _v then _nv back to back, so a
# clock or contention drift moves both halves of a ratio together.
#
# nb per N is the value the shipped ormqr tuning table returns for that N
# (16/16/24/48/56); j0 = 0 is the first (widest) panel, j0 = N/2 the middle.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export WARM_S=${WARM_S:-1.0}
echo "build,which,type,N,nb,j0,batch,m,n,k,med_ms,rel_sd,GFLOPs"
body() {
  for t in float double cfloat cdouble; do
    for cell in "256 24 2048" "512 48 512" "1024 56 128" "2048 56 32"; do
      set -- $cell; N=$1; nb=$2; b=$3
      for j0 in 0 $((N/2)); do
        for w in G1 G3; do
          echo -n "vendor,";     "$D/gemmtrail_v"  "$t" "$N" "$nb" "$j0" "$b" "$w" 5
          echo -n "vendorfree,"; "$D/gemmtrail_nv" "$t" "$N" "$nb" "$j0" "$b" "$w" 5
        done
      done
    done
  done
}
bash /home/jonaslacour/BatchLAS/experiments/gpu_guard.sh 1 bash -c "$(declare -f body); D=$D WARM_S=$WARM_S body"

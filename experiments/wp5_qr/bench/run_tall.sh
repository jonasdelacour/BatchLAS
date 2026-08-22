#!/usr/bin/env bash
# THE TALL-PANEL SWEEP -- the shape the library ITSELF asks geqrf for.
#
# Every cell in the baseline table and in the order sweep is square, and the two
# in-tree callers of geqrf are not: src/extensions/band_reduction.cc:595 and
# src/extensions/sytrd_sy2sb.cc:504 both hand it an m x r PANEL with r << m. A
# routing decision taken on square evidence alone would be taken on shapes the
# library never issues.
#
# It also straddles the blocked driver's leaf boundary: a 1024 x 32 float panel
# is 128 KB against this box's ~97 KB budget, so the leading panel takes the
# GLOBAL leaf, while a 128 x 32 one is resident.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=${WARM_S:-1.5}

echo "bin,op,type,m,n,batch,med_ms,mean_ms,relsd,GFLOPs,geqrf_res,ortho,recon,ws_bytes,route,cta_max_elems,flag"
for cell in "128 32 4096" "512 32 2048" "1024 32 1024" "2048 32 512" \
            "512 64 1024" "1024 64 512" "2048 64 256" "1024 128 256"; do
  set -- $cell
  m=$1; n=$2; b=$3
  for t in float double cfloat cdouble; do
    for bin in qrbench_v qrbench_nv; do
      printf '%s,' "$bin"
      timeout 1800 "$D/$bin" geqrf "$t" "$m" "$n" "$b" 7 || echo "geqrf,$t,$m,$n,$b,TIMEOUT_OR_CRASH"
    done
  done
done

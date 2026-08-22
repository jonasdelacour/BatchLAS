#!/usr/bin/env bash
# THE NATIVE-INTERNAL TIER SWEEP: CTA against Blocked, vendor-free build only.
#
# It answers a question the vendor A/B cannot: WHERE the two native tiers cross,
# which is the number a future preferred() has to encode. Vendor-free build only,
# because in a vendor-present build a forced route that turns out to be
# unsupported falls through to automatic() (route_resolve.hh:101) and automatic()
# returns {Vendor, Auto} -- so a "cta" pin at an order the CTA tile cannot hold
# would silently MEASURE cuSOLVER and be tabulated as a CTA number. That is
# WP4's recorded trap, and the route column below is what makes it visible:
# ONLY rows whose route reads native:cta belong to the cta series.
#
# The CTA capacity is a per-type AREA (m*n <= cta_max_elems, printed in the last
# column but one), so the largest square order each type can hold is:
#   float 24320 -> 155   double/cfloat 12160 -> 110   cdouble 6080 -> 77
# The n list straddles all three boundaries.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=${WARM_S:-1.5}

batch_for() { case "$1" in 16|32|48|64) echo 8192;; 80|96|112|128) echo 4096;; *) echo 2048;; esac; }

echo "bin,op,type,m,n,batch,med_ms,mean_ms,relsd,GFLOPs,geqrf_res,ortho,recon,ws_bytes,route,cta_max_elems,flag,pin"
for n in 16 32 48 64 80 96 112 128 160 192 256; do
  b="$(batch_for "$n")"
  for t in float double cfloat cdouble; do
    for pin in cta blocked; do
      printf '%s,' "qrbench_nv"
      out="$(BATCHLAS_GEQRF_ROUTE="$pin" timeout 1800 "$D/qrbench_nv" geqrf "$t" "$n" "$n" "$b" 7)" \
        || out="geqrf,$t,$n,$n,$b,TIMEOUT_OR_CRASH"
      echo "$out,$pin"
    done
  done
done

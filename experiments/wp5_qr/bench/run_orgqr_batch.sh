#!/usr/bin/env bash
# ORGQR'S BATCH AXIS AT THE ORDERS WHERE NATIVE LOSES.
#
# The order sweep found native orgqr losing at n >= 1024 (float 0.41x at
# n=2048/b=32, cfloat 0.31x, cdouble 0.46x), and the WP5 baseline recorded that
# the ANALOGOUS loss in the VENDOR build was partly a batch artefact -- 0.67x at
# b=32 became 1.12x at b=128 for float n=2048. The vendor arm is a per-batch-item
# cuSOLVER loop (cublas.cc:1414-1419), so its time is LINEAR in batch by
# construction while a batched kernel's is not until it saturates: at b = 32 on a
# 128-SM card the vendor loop is not yet paying for serialisation.
#
# So the losing cells have to be re-measured against batch before they can be
# called losses. Memory is the binding constraint: orgqr mode holds THREE
# n*n*batch arrays (A0, A, F) plus both workspaces, and at cdouble n=2048 even
# b=32 is 6.4 GB of matrix data.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=${WARM_S:-1.5}

cell() {   # type n batch reps
  for bin in qrbench_v qrbench_nv; do
    printf '%s,' "$bin"
    timeout 3600 "$D/$bin" orgqr "$1" "$2" "$2" "$3" "$4" || echo "orgqr,$1,$2,$2,$3,TIMEOUT_OR_CRASH"
  done
}

echo "bin,op,type,m,n,batch,med_ms,mean_ms,relsd,GFLOPs,geqrf_res,ortho,recon,ws_bytes,route,cta_max_elems,flag"
for b in 32 64 128 256; do cell float   1024 "$b" 5; done
for b in 32 64 128 256; do cell double  1024 "$b" 5; done
for b in 32 64 128 256; do cell cfloat  1024 "$b" 5; done
for b in 32 64 128;     do cell cdouble 1024 "$b" 5; done
for b in 16 32 64 128;  do cell float   2048 "$b" 3; done
for b in 16 32 64;      do cell cfloat  2048 "$b" 3; done
for b in 16 32;         do cell cdouble 2048 "$b" 3; done

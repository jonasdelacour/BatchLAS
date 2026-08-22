#!/usr/bin/env bash
# THE BATCH SWEEP -- the other axis. WP4 found its potrf crossover was in ORDER
# and not in BATCH, which is not the intuition, so the two axes are swept
# separately here rather than inferred from one another.
#
# geqrf only: orgqr's vendor arm is a per-batch-item loop whose time is linear in
# batch BY CONSTRUCTION (cublas.cc:1414-1419), so its batch axis carries no
# information the order sweep does not already have.
#
# Batch lists are capped per (n, type) by DEVICE MEMORY, not by taste: the
# harness holds two n*n*batch arrays plus the vendor-free workspace, and a cell
# that OOMs prints THREW rather than a number.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=${WARM_S:-1.5}

run_cell() {   # n type batch reps
  for bin in qrbench_v qrbench_nv; do
    printf '%s,' "$bin"
    timeout 3600 "$D/$bin" geqrf "$2" "$1" "$1" "$3" "$4" || echo "geqrf,$2,$1,$1,$3,TIMEOUT_OR_CRASH"
  done
}

echo "bin,op,type,m,n,batch,med_ms,mean_ms,relsd,GFLOPs,geqrf_res,ortho,recon,ws_bytes,route,cta_max_elems,flag"
# n = 64: the CTA tier
for t in float double cfloat cdouble; do
  for b in 32 128 512 2048 8192 16384; do run_cell 64 "$t" "$b" 7; done
done
# n = 256: blocked for every type (256*256 = 65536 > every cta_max_elems)
for t in float double cfloat cdouble; do
  for b in 32 128 512 2048; do run_cell 256 "$t" "$b" 7; done
done
# n = 1024: blocked, several panels
for b in 8 32 128 256; do run_cell 1024 float "$b" 5; done
for b in 8 32 128 256; do run_cell 1024 double "$b" 5; done
for b in 8 32 128 256; do run_cell 1024 cfloat "$b" 5; done
for b in 8 32 64 128; do run_cell 1024 cdouble "$b" 5; done

#!/usr/bin/env bash
# BLOCK-WIDTH PROBE, two independent instruments.
#
# The shipped ORMQR_BLOCK_SIZE_* buckets (16/16/24/48/56) were tuned on
# CUDA/float ONLY: evaluation/tuning/tune.py takes a single --type for the whole
# run, and every example in evaluation/tuning/README.md is --type float. geqrf
# will inherit that table unless WP5 measures its own width, so:
#
#  PART 1  the geqrf trailing GEMM pair alone, at fixed N and sweeping nb. Gives
#          effective throughput per call; the peak is the width the BLAS-3 core
#          wants. Vendor-FREE build -- the configuration WP5 must be fast in.
#  PART 2  end-to-end WY apply (ormqr on an identity) with
#          BATCHLAS_TUNE_ORMQR_BLOCK_SIZE forced. This one DOES pay the launch
#          count and the per-panel larft, so it is the honest end-to-end answer;
#          PART 1 explains it.
#
# The env var must not change between a buffer-size query and its call
# (tuning_params.hh's HAZARD note). It is fixed for the whole process here.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export WARM_S=${WARM_S:-0.3}
body() {
  echo "== PART 1: trailing GEMM pair vs nb, vendor-free build, N=1024 b=128 =="
  echo "build,which,type,N,nb,j0,batch,m,n,k,med_ms,rel_sd,GFLOPs"
  for t in float double cfloat cdouble; do
    for nb in 8 16 24 32 48 56 64 96 128; do
      for j0 in 0 512; do
        for w in G1 G3; do
          echo -n "vendorfree,"; "$D/gemmtrail_nv" "$t" 1024 "$nb" "$j0" 128 "$w" 3
        done
      done
    done
  done
  echo "== PART 2: end-to-end WY apply vs forced nb, vendor-free build =="
  echo "nb_forced,op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,res,ortho,recon,ws,route,nbused,dQ"
  for cell in "256 512" "1024 64"; do
    set -- $cell; n=$1; b=$2
    for t in float double cfloat cdouble; do
      for nb in 8 16 24 32 48 56 64 96; do
        echo -n "$nb,"
        BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=$nb "$D/wp5qr_nv" synthI "$t" "$n" "$b" 3
      done
    done
  done
}
bash /home/jonaslacour/BatchLAS/experiments/gpu_guard.sh 1 bash -c "$(declare -f body); D=$D WARM_S=$WARM_S body"

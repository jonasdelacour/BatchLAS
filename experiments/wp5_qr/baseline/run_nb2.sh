#!/usr/bin/env bash
# Two follow-ups run_nb.sh left open.
#
# (a) PART 1 showed a CLIFF for float at nb=128: effective throughput of the
#     trailing pair jumps 6896 -> 18906 GFLOP/s. That is the float TN register
#     kernel (Tiled128x32RegisterK32TN, gemm_kernels.cc:474) becoming reachable,
#     because its gate is m >= 128 and m of the G1 gemm IS the block width.
#     PART 2 never tested nb >= 112, so it could not see whether that cliff
#     survives the extra panel cost end to end.
#
# (b) The shipped ORMQR_BLOCK_SIZE_* table was tuned in a VENDOR-PRESENT build.
#     PART 2 ran vendor-FREE. So "the shipped width is 1.1-1.55x off" could be a
#     BUILD difference rather than a TYPE difference. This runs the same ladder
#     in the vendor build as the control that separates the two.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export WARM_S=${WARM_S:-0.3}
body() {
  echo "== (a) wide nb, vendor-free =="
  echo "nb_forced,op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,res,ortho,recon,ws,route,nbused,dQ"
  for cell in "256 512" "1024 64"; do
    set -- $cell; n=$1; b=$2
    for t in float double cfloat cdouble; do
      for nb in 112 128 160 192; do
        echo -n "$nb,"; BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=$nb "$D/wp5qr_nv" synthI "$t" "$n" "$b" 3
      done
    done
  done
  echo "== (b) same ladder, VENDOR build =="
  echo "nb_forced,op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,res,ortho,recon,ws,route,nbused,dQ"
  for cell in "256 512" "1024 64"; do
    set -- $cell; n=$1; b=$2
    for t in float double cfloat cdouble; do
      for nb in 8 16 24 32 48 56 64 96 128; do
        echo -n "$nb,"; BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=$nb "$D/wp5qr_v" synthI "$t" "$n" "$b" 3
      done
    done
  done
}
bash /home/jonaslacour/BatchLAS/experiments/gpu_guard.sh 1 bash -c "$(declare -f body); D=$D WARM_S=$WARM_S body"

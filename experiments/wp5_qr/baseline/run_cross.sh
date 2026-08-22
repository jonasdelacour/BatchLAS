#!/usr/bin/env bash
# THE ONE CELL WHERE ormqr-on-identity LOST. In the main sweep, n=2048 batch=32
# is the only n where cuSOLVER orgqr beat routed ormqr-on-identity (0.43-0.97x).
# That cell has TWO confounds:
#   * batch 32 -- run_sat.sh proved the vendor geqrf is nowhere near saturated
#     there, and orgqr is a per-item LOOP whose cost is linear in batch either way;
#   * block width 56 -- the nb ladder measured 56 as 1.24-1.39x off the best.
# Re-measure with the batch raised as far as memory allows and the width set to
# the measured best, so the crossover claim is not an artefact of either.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export WARM_S=${WARM_S:-0.5}
echo "nb_forced,op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,res,ortho,recon,ws[,route,nb,dQ]"
body() {
  for b in 32 64 128; do
    for nb in 56 32; do
      echo -n "$nb,"; BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=$nb "$D/wp5qr_v" qcheck float 2048 "$b" 3
    done
  done
  for nb in 56 32; do
    echo -n "$nb,"; BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=$nb "$D/wp5qr_v" qcheck cdouble 2048 32 3
    echo -n "$nb,"; BATCHLAS_TUNE_ORMQR_BLOCK_SIZE=$nb "$D/wp5qr_v" qcheck cfloat  2048 64 3
  done
}
bash /home/jonaslacour/BatchLAS/experiments/gpu_guard.sh 1 bash -c "$(declare -f body); D=$D WARM_S=$WARM_S body"

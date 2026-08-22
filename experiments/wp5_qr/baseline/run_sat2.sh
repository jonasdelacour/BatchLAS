#!/usr/bin/env bash
# EXTENDED SATURATION for the vendor geqrf, at the two n where the first ladder
# left the question open. run_sat.sh showed cuBLAS geqrfBatched is NOT saturated
# at n=1024 even at batch 256 -- its wall time is nearly flat in batch, so
# GFLOP/s is still climbing linearly. That means the n>=512 rows of the vendor
# baseline table are absolute wall-clock targets at those cells, NOT statements
# about cuBLAS's ceiling. This pins how far off saturation they are.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export WARM_S=${WARM_S:-0.3}
echo "op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,res,ws"
body() {
  for b in 128 256 512 1024 2048; do "$D/wp5qr_v" geqrf float   512 "$b" 3; done
  for b in 128 256 512 1024;      do "$D/wp5qr_v" geqrf cdouble 512 "$b" 3; done
  for b in 32 64 128 256;         do "$D/wp5qr_v" geqrf float  2048 "$b" 3; done
}
bash /home/jonaslacour/BatchLAS/experiments/gpu_guard.sh 1 bash -c "$(declare -f body); D=$D WARM_S=$WARM_S body"

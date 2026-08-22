#!/usr/bin/env bash
# SATURATION LADDER. Every ratio in this directory is quoted at one (n, batch)
# cell, and a ratio taken below saturation is a ratio of overheads. This walks
# batch at fixed n so the READER can see where each op's throughput plateaus --
# and, for geqrf, whether the collapse at large n is the algorithm or merely a
# batch that stopped filling the card.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export WARM_S=${WARM_S:-1.0}
echo "op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,..."
body() {
  for t in float cdouble; do
    for b in 32 64 128 256 512 1024 2048 4096; do "$D/wp5qr_v" geqrf "$t" 256 "$b" 3; done
    for b in 8 16 32 64 128 256;              do "$D/wp5qr_v" geqrf "$t" 1024 "$b" 3; done
    for b in 32 64 128 256 512 1024 2048 4096; do "$D/wp5qr_v" synthI "$t" 256 "$b" 3; done
    for b in 8 16 32 64 128 256;              do "$D/wp5qr_v" synthI "$t" 1024 "$b" 3; done
  done
}
bash /home/jonaslacour/BatchLAS/experiments/gpu_guard.sh 1 bash -c "$(declare -f body); D=$D WARM_S=$WARM_S body"

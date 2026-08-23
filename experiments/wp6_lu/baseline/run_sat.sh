#!/usr/bin/env bash
# SATURATION. Sweep batch at fixed n and report where the ms/batch curve goes
# flat, for cuBLAS getrf and getri -- BEFORE any comparison exists.
#
# WP5 found geqrfBatched nearly independent of batch at large n, which made every
# wall-clock ratio flatter the native side enormously and had to be caveated in
# every table. This establishes whether getrfBatched / getriBatched do the same.
#
#   bash run_sat.sh > sat.csv 2> sat_err.txt
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=1
export WARM_S=${WARM_S:-1.0}

echo "op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws_bytes,route,extra,ntpiv,flag"
for t in float cdouble; do
  for b in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
    "$D/lubench_v" getrf "$t" 64 1 "$b" 5
    "$D/lubench_v" getri "$t" 64 1 "$b" 5
  done
  for b in 1 2 4 8 16 32 64 128 256 512 1024 2048; do
    "$D/lubench_v" getrf "$t" 256 1 "$b" 5
    "$D/lubench_v" getri "$t" 256 1 "$b" 5
  done
  for b in 1 2 4 8 16 32 64 128 256; do
    "$D/lubench_v" getrf "$t" 1024 1 "$b" 3
  done
  for b in 1 2 4 8 16 32 64; do
    "$D/lubench_v" getrf "$t" 2048 1 "$b" 3
  done
done

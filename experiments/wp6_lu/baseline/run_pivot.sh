#!/usr/bin/env bash
# WP6 question 3: what does partial pivoting cost, as a function of n and batch,
# and does the pivot reduction reopen the 48 KB launch hole.
#
#   bash run_pivot.sh > pivot.csv 2> pivot_err.txt
#
# EVERY ROW IS ITS OWN PROCESS. The SLM attribute is sticky per CUfunction, so a
# ladder run inside one process would have its first (largest) launch raise the
# cap for the rest and the hole would vanish by execution order.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=${WARM_S:-1.0}
export WG=${WG:-256}

echo "section,variant,type,n,batch,wg,ld,slm_bytes,med_ms,mean_ms,relsd,GFLOPs,resid,ntpiv,flag"

# ---- (a) THE HOLE LADDER, float, ascending, one process per point. slm crosses
# 49152 B between n=110 (48852 B) and n=111 (49296 B) for the non-reducing arms.
for n in 104 106 108 109 110 111 112 113 116; do
  for v in nopiv pivman pivgrp; do
    printf 'hole,'; "$D/pivotcost" "$v" float "$n" 1024 3
  done
done

# ---- (b) the n ladder, per type, at the largest n that still fits SLM.
for n in 16 24 32 48 64 96 128 152; do
  for v in nopiv swaponly pivman pivgrp; do printf 'n,'; "$D/pivotcost" "$v" float "$n" 4096 7; done
done
for n in 16 24 32 48 64 96 110; do
  for v in nopiv swaponly pivman pivgrp; do printf 'n,'; "$D/pivotcost" "$v" double "$n" 4096 7; done
  for v in nopiv swaponly pivman pivgrp; do printf 'n,'; "$D/pivotcost" "$v" cfloat "$n" 4096 7; done
done
for n in 16 24 32 48 64 78; do
  for v in nopiv swaponly pivman pivgrp; do printf 'n,'; "$D/pivotcost" "$v" cdouble "$n" 4096 7; done
done

# ---- (c) the batch sweep, so the delta is not read off one batch. n=64 float.
for b in 128 256 512 1024 2048 4096 8192 16384; do
  for v in nopiv swaponly pivman pivgrp; do printf 'batch,'; "$D/pivotcost" "$v" float 64 "$b" 7; done
done

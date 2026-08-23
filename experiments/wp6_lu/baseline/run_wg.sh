#!/usr/bin/env bash
# Work-group width sensitivity, so the pivot delta is not reported at one
# arbitrary wg. n=64 and n=128, float, at a saturating batch.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export WARM_S=1.0
echo "variant,type,n,batch,wg,ld,slm_bytes,med_ms,mean_ms,relsd,GFLOPs,resid,ntpiv,flag"
for n in 64 128; do
  for w in 32 64 128 256 512; do
    for v in nopiv pivman pivgrp; do
      WG=$w "$D/pivotcost" "$v" float "$n" 4096 7
    done
  done
done

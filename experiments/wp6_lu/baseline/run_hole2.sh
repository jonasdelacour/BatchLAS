#!/usr/bin/env bash
# Bracket the hole found by run_hole.sh: at EXACTLY 49152 declared bytes the
# reduce_over_group arm fails to launch while the identical shape with an
# explicit SLM tree does not. Repeat it (is it deterministic?), walk the
# neighbouring byte counts at 128 B granularity, and try the other scalar types.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export WARM_S=0.2
export WG=256
echo "pad_request,variant,type,n,batch,wg,ld,slm_bytes,med_ms,mean_ms,relsd,GFLOPs,resid,ntpiv,flag"
# determinism: the same point five times, five processes
for i in 1 2 3 4 5; do
  printf '49152,'; PAD=49152 "$D/pivotcost" pivgrp float 64 1024 2
done
# fine walk either side, 128 B steps
for pad in 48768 48896 49024 49152 49280 49408 49536; do
  printf '%s,' "$pad"; PAD=$pad "$D/pivotcost" pivgrp float 64 1024 2
done
# does it depend on the scalar type, or on the byte count alone?
for t in double cfloat cdouble; do
  for pad in 48896 49152 49664; do
    printf '%s,' "$pad"; PAD=$pad "$D/pivotcost" pivgrp "$t" 32 1024 2
  done
done
# and on the work-group width?
for w in 32 64 128 512; do
  printf '49152,'; WG=$w PAD=49152 "$D/pivotcost" pivgrp float 64 1024 2
done
# control: the same byte count on the arm WITHOUT a group collective
for w in 32 64 128 256 512; do
  printf '49152,'; WG=$w PAD=49152 "$D/pivotcost" pivman float 64 1024 2
done

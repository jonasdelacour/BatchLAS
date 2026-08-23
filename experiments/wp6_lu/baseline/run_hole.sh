#!/usr/bin/env bash
# THE 48 KB LAUNCH HOLE, at exact byte counts, one process per point.
#
# WP4's record is not a RANGE, it is specific byte counts: 48896 passes, 49152
# FAILS, 49664 passes. An n ladder steps over 49152 rather than landing on it,
# which is why run_pivot.sh's hole section found nothing. Here the SLM request is
# padded to a named byte count so the kernel, the shape and the work-group are
# held fixed and only the byte count moves.
#
# ONE PROCESS PER POINT, and the points are emitted in ASCENDING order. The SLM
# attribute is sticky per CUfunction: a larger launch earlier in the same process
# raises the cap and hides the whole class by execution order.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export WARM_S=0.3
export WG=256
echo "pad_request,variant,type,n,batch,wg,ld,slm_bytes,med_ms,mean_ms,relsd,GFLOPs,resid,ntpiv,flag"
for pad in 46080 47104 48640 48896 49152 49408 49664 50176 50688 50944 51200; do
  for v in nopiv pivman pivgrp; do
    printf '%s,' "$pad"; PAD=$pad "$D/pivotcost" "$v" float 64 1024 3
  done
done

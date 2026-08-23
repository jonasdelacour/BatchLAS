#!/usr/bin/env bash
# ANTI-VACUITY. Corrupt the exact thing each probe claims to check and confirm the
# probe goes RED. A break that leaves the residual small is a finding.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/baseline
export CUDA_VISIBLE_DEVICES=1
export WARM_S=0.3
echo "break,op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,extra,flag"
for t in float cdouble; do
  printf 'none,';   env -u BREAK "$D/lubench_v" getrf      "$t" 128 128 256 3
  printf 'piv,';    BREAK=piv    "$D/lubench_v" getrf      "$t" 128 128 256 3
  printf 'factor,'; BREAK=factor "$D/lubench_v" getrf      "$t" 128 128 256 3
  printf 'none,';   env -u BREAK "$D/lubench_v" getrs_trsm "$t" 128 128 256 3
  printf 'laswp,';  BREAK=laswp  "$D/lubench_v" getrs_trsm "$t" 128 128 256 3
  printf 'sol,';    BREAK=sol    "$D/lubench_v" getrs_trsm "$t" 128 128 256 3
  printf 'none,';   env -u BREAK "$D/lubench_v" getri_trsm "$t" 128 128 256 3
  printf 'laswp,';  BREAK=laswp  "$D/lubench_v" getri_trsm "$t" 128 128 256 3
  printf 'sol,';    BREAK=sol    "$D/lubench_v" getri_trsm "$t" 128 128 256 3
  printf 'none,';   env -u BREAK "$D/lubench_v" getri      "$t" 128 128 256 3
  printf 'sol,';    BREAK=sol    "$D/lubench_v" getri      "$t" 128 128 256 3
done

#!/usr/bin/env bash
# THE VENDOR BASELINE GRID + the routed-trsm composition, on the same shapes, in
# the same process, one cell after another so the A/B is interleaved at cell
# granularity and nothing drifts between the two arms of a comparison.
#
#   bash run_grid.sh > grid.csv 2> grid_err.txt
#
# GPU 1 only, claimed for the whole run by gpu_guard.sh. Never run under
# BATCHLAS_KERNEL_TRACE.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp6_lu/baseline"
export CUDA_VISIBLE_DEVICES=1
export WARM_S=${WARM_S:-1.5}

echo "laswp,op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws_bytes,route,extra,ntpiv,flag"

# Saturating batch per n: pair small n with large batch (plan section 6 rule 2).
batch_for() {
  case "$1" in
    32)   echo 8192 ;; 64)   echo 8192 ;; 128)  echo 4096 ;; 256) echo 2048 ;;
    512)  echo 512  ;; 1024) echo 128  ;; 2048) echo 32   ;; *) echo 128 ;;
  esac
}
reps_for() { if [ "$1" -le 512 ]; then echo 5; else echo 3; fi; }

for t in float double cfloat cdouble; do
  for n in 32 64 128 256 512 1024 2048; do
    b=$(batch_for "$n"); r=$(reps_for "$n")
    printf '%s,' '-';      env -u LASWP "$D/lubench_v" getrf      "$t" "$n" 1  "$b" "$r"
    printf '%s,' '-';      env -u LASWP "$D/lubench_v" getri      "$t" "$n" 1  "$b" "$r"
    printf 'list,';   env -u LASWP "$D/lubench_v" getri_trsm "$t" "$n" 1  "$b" "$r"
    printf 'gather,'; LASWP=gather "$D/lubench_v" getri_trsm "$t" "$n" 1  "$b" "$r"
    for nr in 1 64; do
      printf '%s,' '-';      env -u LASWP "$D/lubench_v" getrs      "$t" "$n" "$nr" "$b" "$r"
      printf 'list,';   env -u LASWP "$D/lubench_v" getrs_trsm "$t" "$n" "$nr" "$b" "$r"
      printf 'gather,'; LASWP=gather "$D/lubench_v" getrs_trsm "$t" "$n" "$nr" "$b" "$r"
    done
  done
done

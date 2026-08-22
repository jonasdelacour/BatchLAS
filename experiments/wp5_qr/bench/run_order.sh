#!/usr/bin/env bash
# THE ORDER SWEEP. n varies, batch follows a fixed memory-bounded schedule that
# is the baseline table's (experiments/wp5_qr/baseline/README.md section 3)
# extended with the four orders the baseline had no reason to visit but that the
# CTA/blocked tier boundary falls between (32, 96, 160) and 2048.
#
# A/B IS INTERLEAVED PER CELL -- qrbench_v then qrbench_nv, same n, same batch,
# back to back -- rather than all-of-A-then-all-of-B, so a clock or thermal
# drift lands on both arms of every ratio instead of on one of them.
#
# WARM_S=1.5 inside the harness, so the JIT and the clocks are warm before the
# first timed rep. NEVER run this under BATCHLAS_KERNEL_TRACE (~60% inflation).
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export WARM_S=${WARM_S:-1.5}

batch_for() { case "$1" in
  32) echo 8192;; 64) echo 8192;; 96) echo 4096;; 128) echo 4096;;
  160) echo 2048;; 256) echo 2048;; 512) echo 512;; 1024) echo 128;; 2048) echo 32;;
  *) echo 128;; esac; }
reps_for()  { case "$1" in 32|64|96|128|160|256|512) echo 7;; 1024) echo 5;; *) echo 3;; esac; }

echo "bin,op,type,m,n,batch,med_ms,mean_ms,relsd,GFLOPs,geqrf_res,ortho,recon,ws_bytes,route,cta_max_elems,flag"
for op in geqrf orgqr; do
  for n in 32 64 96 128 160 256 512 1024 2048; do
    b="$(batch_for "$n")"; r="$(reps_for "$n")"
    for t in float double cfloat cdouble; do
      for bin in qrbench_v qrbench_nv; do
        printf '%s,' "$bin"
        timeout 3600 "$D/$bin" "$op" "$t" "$n" "$n" "$b" "$r" || echo "$op,$t,$n,$n,$b,TIMEOUT_OR_CRASH"
      done
    done
  done
done

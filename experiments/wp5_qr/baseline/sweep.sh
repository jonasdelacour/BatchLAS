#!/usr/bin/env bash
# THE WP5 BASELINE SWEEP.
#
#   A  geqrf   vendor build   -- the number WP5's geqrf is judged against
#   B  qcheck  vendor build   -- cuSOLVER orgqr AND routed ormqr-on-identity in
#                                ONE process, back to back (interleaved A/B),
#                                plus the elementwise agreement of the two Qs
#   C  synthI  vendor build   -- ormqr-on-identity from synthetic reflectors
#   D  synthI  vendor-FREE    -- the same, in the build with no cuBLAS at all.
#                                C vs D isolates what the GEMM layer costs, with
#                                the ormqr route held fixed at Native:Blocked.
#
# Batch schedule: saturating at small n, memory-bounded at large n. qcheck holds
# SIX arrays of n*n*batch (A0, A, F, Qref, C, C0), which at cdouble is the
# binding constraint on this 24 GB card.
#
# Discard rule: any cell with rel_sd > 10% is re-run once and dropped if it does
# not settle. Reported cells all have rel_sd printed in column 7.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
G=/home/jonaslacour/BatchLAS/experiments/gpu_guard.sh
export WARM_S=${WARM_S:-1.5}

batch_for() { case "$1" in 64) echo 8192;; 128) echo 4096;; 256) echo 2048;;
                           512) echo 512;; 1024) echo 128;; 2048) echo 32;; esac; }
reps_for()  { case "$1" in 64|128|256|512) echo 5;; *) echo 3;; esac; }

run_grid() {   # $1 = exe, $2 = mode
  for t in float double cfloat cdouble; do
    for n in 64 128 256 512 1024 2048; do
      "$1" "$2" "$t" "$n" "$(batch_for "$n")" "$(reps_for "$n")"
    done
  done
}

echo "# columns: op,type,n,batch,med_ms,mean_ms,rel_sd,GFLOPs,geqrf_res,[ortho,recon,ws,route,nb,dQ]"
echo "## A geqrf, vendor build"
bash "$G" 1 bash -c "$(declare -f batch_for reps_for run_grid); run_grid $D/wp5qr_v geqrf"
echo "## B qcheck (orgqr + ormqrI), vendor build"
bash "$G" 1 bash -c "$(declare -f batch_for reps_for run_grid); run_grid $D/wp5qr_v qcheck"
echo "## C synthI, vendor build"
bash "$G" 1 bash -c "$(declare -f batch_for reps_for run_grid); run_grid $D/wp5qr_v synthI"
echo "## D synthI, vendor-FREE build"
bash "$G" 1 bash -c "$(declare -f batch_for reps_for run_grid); run_grid $D/wp5qr_nv synthI"

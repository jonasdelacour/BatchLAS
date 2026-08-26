#!/usr/bin/env bash
# WP8/I3 -- REACHABILITY. "A kernel being LINKED is not evidence it RUNS"
# (campaign trap 4), and {Native, CTA} now names TWO kernels so the resolved
# route column cannot tell them apart. This asks the profiler which kernel
# actually launched.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENBLAS_CORETYPE=SKYLAKEX
export BATCHLAS_GEMV_ROUTE=native:cta
export WARM_S=0
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN="$W/experiments/wp7_gemv/ab/gemvab_v"
NCU=/usr/local/cuda-13.2/bin/ncu
# m n batch : Trans, so m = red_len.
for ty in cdouble double cfloat float; do
  for m in 32 64 65 128; do
    for mode in auto off; do
      k=$(BATCHLAS_GEMV_SEGT=$mode "$NCU" --target-processes all -k regex:Gemv \
            --launch-count 1 --metrics launch__registers_per_thread,launch__shared_mem_per_block_static,launch__block_size,launch__grid_size \
            --csv "$BIN" "$ty" "$m" 256 64 T 1 2>/dev/null | tr -d '"' \
          | awk -F, 'NR>1 && NF>5 {print $5"|"$(NF-2)"="$NF}' | paste -sd' ')
      echo "$ty red_len=$m SEGT=$mode  $k"
    done
  done
done

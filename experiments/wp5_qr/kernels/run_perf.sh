#!/usr/bin/env bash
# A SANITY-SCALE performance check, not a routing gate.
#
# It reuses ../baseline/wp5qr.cpp UNCHANGED -- the same program linked once
# against build/ and once against build-novendor/, so "vendor-free" is the BUILD
# (experiments/wp4_potrf/phase2_ab/realpotrf.cpp's pattern). It times the PUBLIC
# geqrf, warms the JIT and the clocks inside the harness (WARM_S), and checks the
# residual in the same process, so a fast wrong answer cannot be reported as a
# win.
#
# The cells are the baseline table's, so the ms column is directly comparable.
# preferred() is FALSE, so this decides nothing -- it exists so that a
# catastrophic number (the batch-only-parallelism defect) cannot go unnoticed.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
B="$W/experiments/wp5_qr/baseline"
echo "mode,type,n,batch,med_ms,mean_ms,relsd,GFLOPs,residual,ws_bytes"
for cell in "64 8192" "128 4096" "256 2048" "512 512" "1024 128"; do
  set -- $cell
  for t in float double cfloat cdouble; do
    for bin in wp5qr_v wp5qr_nv; do
      printf '%s ' "$bin"
      CUDA_VISIBLE_DEVICES=1 WARM_S=${WARM_S:-1.5} timeout 1800 "$B/$bin" geqrf "$t" "$1" "$2" "${REPS:-7}"
    done
  done
done

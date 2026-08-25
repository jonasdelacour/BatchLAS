#!/usr/bin/env bash
# WP7 R4 follow-up. The main sweep found two cells where cuBLAS reproduces at
# ~1/3 of the roof across two independent passes, both complex<double> + Trans.
# This maps the (m, n) extent of that slow path on a fixed ~1 GB A footprint, so
# that the only thing varying across the grid is the SHAPE, not the traffic.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/baseline"
BIN="$D/gemvbase_v"
OUT="${OUT:-$D/slowpath_probe.csv}"
REPS="${REPS:-11}"
TARGET=$((1024*1024*1024))
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX
export WARM_S=0.6

echo "type,m,n,batch,transA,median_ms,mean_ms,rel_sd,GBs,frac_of_900,relerr" > "$OUT"
for ty in cdouble cfloat float; do
  case $ty in cdouble) el=16;; cfloat) el=8;; float) el=4;; esac
  for m in 32 64 128 256 512 1024 2048; do
    for n in 32 64 128 256 512 1024 2048; do
      b=$(( TARGET / (m*n*el) ))
      [ "$b" -lt 32 ] && b=32
      [ "$b" -gt 32768 ] && b=32768
      for tr in T N; do
        "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" >> "$OUT" 2>>"$D/probe_err.txt" \
          || echo "FAILED,$ty,$m,$n,$b,$tr" >> "$OUT"
      done
    done
  done
done
echo "wrote $OUT"

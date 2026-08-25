#!/usr/bin/env bash
# WP7 -- does the complex<double> transposed win survive DOWN THE BATCH LADDER?
#
# WHY THIS EXISTS. preferred() has no footprint field and B4 forbids an
# L2-residency gate (the recon phase's own numbers contradict the "it switches
# on when A leaves the 72 MB L2" story, and such a gate admits cells where
# cuBLAS runs at 92-96% of roof). But a BATCH FLOOR is a different object: it is
# a measured threshold on a field the shape actually carries, exactly like
# TrsmShape's `s.batch < 8`. This sweep is what decides whether one is needed --
# at small batch A fits in L2, and the whole question is whether cuBLAS's dip
# goes away there and takes the win with it.
#
# Two in-band shapes, both from the measured rectangle: the best cell
# (m=256, n=256) and the weakest one (m=64, n=2048).
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/ab"
BIN="${BIN:-$D/gemvab_v}"
OUT="${OUT:-$D/batchdep_p1.csv}"
REPS="${REPS:-9}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld" > "$OUT"
for shape in "256 256" "64 2048"; do
  set -- $shape
  m=$1; n=$2
  for b in 1 8 32 64 128 256 512 1024 2048; do
    for arm in vendor native:cta; do
      row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" cdouble "$m" "$n" "$b" C "$REPS" 2>>"$D/run_err.txt")
      if [ -z "$row" ]; then
        echo "$arm,FAILED,cdouble,$m,$n,$b,C" >> "$OUT"
      else
        echo "$arm,$row" >> "$OUT"
      fi
    done
  done
done
echo "wrote $OUT"

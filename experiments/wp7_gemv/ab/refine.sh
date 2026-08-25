#!/usr/bin/env bash
# WP7 -- map the ONE region where a preferred() clause could be justified:
# complex<double> with a TRANSPOSED transA, where the recon phase measured
# cuBLAS at 310-380 GB/s (33-40% of the ~950 GB/s roof) while float, double and
# complex<float> run 936-967 GB/s at IDENTICAL bytes and IDENTICAL (m, n).
#
# THE AXIS IS m, NOT out_len(). Under transA = Trans, out_len() == n and
# red_len() == m; the measured band is on m. A predicate written on out_len()
# tests n, never touches m, and inverts the window -- which is why this grid
# sweeps m and n independently rather than sweeping "size".
#
# The grid deliberately straddles BOTH edges recon reported (m = 48 and m = 384
# are outside the band; n = 64 is below the n >= 128 floor) so the clause can be
# bounded from measurement rather than from the recon prose.
#
# FOOTPRINT IS HELD AT ~1 GB so every cell is DRAM-resident: the RTX 4090's L2
# is 72 MB and a cell that fits in it measures L2 bandwidth, reports over 100%
# of roof, and means nothing.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/ab"
BIN="${BIN:-$D/gemvab_v}"
OUT="${OUT:-$D/refine_p1.csv}"
REPS="${REPS:-9}"
TR="${TR:-C}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld" > "$OUT"
for m in 48 64 96 128 192 256 320 384; do
  for n in 64 128 256 512 1024 2048; do
    b=$(( 1000000000 / (m * n * 16) ))
    [ "$b" -lt 128 ] && b=128
    for arm in vendor native:cta; do
      row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" cdouble "$m" "$n" "$b" "$TR" "$REPS" 2>>"$D/run_err.txt")
      if [ -z "$row" ]; then
        echo "$arm,FAILED,cdouble,$m,$n,$b,$TR" >> "$OUT"
      else
        echo "$arm,$row" >> "$OUT"
      fi
    done
  done
done
echo "wrote $OUT"

#!/usr/bin/env bash
# WP7 -- the native-vs-vendor gemv A/B sweep.
#
# METHOD, per the campaign's measurement rules:
#   * CUDA_VISIBLE_DEVICES=0, one dedicated RTX 4090 (this box has two).
#   * SATURATION ONLY: every cell has batch >= 128 and a DRAM-resident A. An
#     unsaturated ratio measures overhead, not the kernel.
#   * The arms are INTERLEAVED WITHIN ONE SESSION -- vendor, native:cta and
#     native:direct for the same cell run back to back -- so a clock or
#     contention drift has to hit all three identically to survive.
#   * One process per cell so an OOM cannot take the sweep with it; each process
#     re-warms for 1 s (JIT, clocks, first-touch migration of a multi-GB shared
#     allocation -- a cold first run has fabricated a 3.7x result here).
#   * The ROUTE COLUMN is printed by the binary, resolved through the real
#     table. A kernel being LINKED is not evidence it RAN.
#
# WHY native:cta IS NOT RUN FOR transA = N: supports() refuses it (there is no
# NoTrans CTA body), so the request falls back to the automatic choice and the
# row would be a duplicate of the vendor one wearing a native label. The route
# column would say so, but not running it is cheaper and clearer.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/ab"
BIN="${BIN:-$D/gemvab_v}"
OUT="${OUT:-$D/ab_p1.csv}"
REPS="${REPS:-9}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX

# m n batch. Square ladder at batch = max(128, 262144/n), plus the two extreme
# aspect ratios, plus the two cells the recon phase found cuBLAS slow on
# (cdouble + Trans: 256x256x1024 and 64x2048x1024).
CELLS="
64 64 4096
128 128 2048
256 256 1024
512 512 512
1024 1024 256
2048 64 1024
64 2048 1024
"

echo "arm,type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld" > "$OUT"
while read -r m n b; do
  [ -z "$m" ] && continue
  for tr in N T C; do
    for ty in float double cfloat cdouble; do
      for arm in vendor native:direct native:cta; do
        if [ "$tr" = "N" ] && [ "$arm" = "native:cta" ]; then continue; fi
        row=$(BATCHLAS_GEMV_ROUTE="$arm" "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" 2>>"$D/run_err.txt")
        if [ -z "$row" ]; then
          echo "$arm,FAILED,$ty,$m,$n,$b,$tr" >> "$OUT"
        else
          echo "$arm,$row" >> "$OUT"
        fi
      done
    done
  done
done <<< "$CELLS"
echo "wrote $OUT"

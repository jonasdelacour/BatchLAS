#!/usr/bin/env bash
# WP7 R4 -- sweep the vendor gemv baseline. One process per cell so that a cell
# that OOMs or aborts cannot take the rest of the sweep with it.
#
# Cells are interleaved by TYPE inside one session; each process re-warms, so a
# cold-clock artefact would have to hit every type identically to survive.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/baseline"
BIN="${BIN:-$D/gemvbase_v}"
OUT="${OUT:-$D/vendor_baseline.csv}"
REPS="${REPS:-15}"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_CORETYPE=SKYLAKEX

# m n batch  -- the ladder the lead specified, plus two L2-escape cells.
# batch = max(128, 262144/n) for the square ladder.
CELLS="
32 32 8192
64 64 4096
128 128 2048
256 256 1024
512 512 512
1024 1024 256
2048 2048 128
64 64 8192
2048 2048 64
2048 64 1024
64 2048 1024
32 32 65536
64 64 16384
"

echo "type,m,n,batch,transA,median_ms,mean_ms,rel_sd,GBs,frac_of_900,relerr" > "$OUT"
while read -r m n b; do
  [ -z "$m" ] && continue
  for tr in N T; do
    for ty in float double cfloat cdouble; do
      if ! "$BIN" "$ty" "$m" "$n" "$b" "$tr" "$REPS" >> "$OUT" 2>>"$D/run_err.txt"; then
        echo "FAILED,$ty,$m,$n,$b,$tr" >> "$OUT"
      fi
    done
  done
done <<< "$CELLS"
echo "wrote $OUT"

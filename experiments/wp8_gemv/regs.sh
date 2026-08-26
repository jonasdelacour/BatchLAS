#!/usr/bin/env bash
# WP8/I3 -- REGISTERS, WORK-GROUP CAP AND LOCAL MEMORY for every new entry
# function, plus body 3's for comparison.
#
# The build emits PTX rather than a cubin into libbatchlas_sycl.so (cuobjdump
# reports "does not contain device code"), so the register count is a RUNTIME
# fact here and ncu is the only instrument that has it. BATCHLAS_GEMV_SEGT
# forces each W, bypassing all three gates, so every instantiation can be
# reached from one shape; SEGT=off gives body 3's figures for comparison.
#
# THE CAP: a work-group of `wg` threads needs regs*wg registers of the 65536 an
# Ada SM has, so the largest admissible wg is 65536/regs rounded down to a
# multiple of 32. gemv_wg_ladder never proposes more than 256.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENBLAS_CORETYPE=SKYLAKEX
export BATCHLAS_GEMV_ROUTE=native:cta
export WARM_S=0
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
BIN="$D/gemvsegab_v"
NCU=/usr/local/cuda-13.2/bin/ncu
OUT="${OUT:-$D/regs.csv}"
M="launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__block_size"
TMP="$D/regs.raw"

: > "$TMP"
for ty in float double cfloat cdouble; do
  for w in off 2 4 8; do
    BATCHLAS_GEMV_SEGT=$w "$NCU" --target-processes all -k regex:Gemv --launch-count 8 \
      --metrics "$M" --csv "$BIN" "$ty" 8 2048 512 T 1 "$w" 2>/dev/null \
      | tr -d '"' \
      | awk -F, -v ty="$ty" -v w="$w" 'NR>1 && NF>5 {
            k=$5; sub(/Typeinfo name for unnamed>::/, "", k);
            printf "%s,%s,%s,%s,%s\n", ty, w, k, $(NF-2), $NF }' >> "$TMP"
  done
done
echo "type,segt,kernel,metric,value" > "$OUT"
sort -u "$TMP" >> "$OUT"
rm -f "$TMP"
echo "wrote $OUT"

#!/usr/bin/env bash
# WP8/I3 -- D3's PART D pre-check, run BEFORE body 5 is written.
#
# PREDICTION UNDER TEST (D3): the short-reduction shortfall in body 3 is FP64
# work in the shuffle fold on a 1/64-rate GeForce part, not the shuffle COUNT.
# If it were the count, `double` (5 shift_group_left on a 64-bit value) and
# `complex<float>` (10 shift_group_left on a 32-bit value) would be hurt
# equally; they are not.
#
# EXPECT: fp64 pipe near saturation for cdouble at red_len 32-64 and low at 128;
# near zero for cfloat and float at all three; double intermediate.
#
# NOTE ON THE SHUFFLE COUNTER. This CUDA 13.2 ncu exposes no
# smsp__inst_executed_op_shfl on sm_89, so smsp__inst_executed_pipe_lsu.sum
# stands in: SHFL issues on the LSU pipe, and the two types under comparison
# execute the same number of global loads per output, so a DIFFERENCE in that
# counter between double and cfloat would be a difference in shuffles. It is a
# proxy and is labelled as one; the decisive counter is the FP64 pipe.
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENBLAS_CORETYPE=SKYLAKEX
export BATCHLAS_GEMV_ROUTE=native:cta
export WARM_S=0
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
BIN="${BIN:-$W/experiments/wp7_gemv/ab/gemvab_v}"
NCU=/usr/local/cuda-13.2/bin/ncu
OUT="${OUT:-$D/ncu_precheck.csv}"
KRX="${KRX:-GemvCtaT}"
RLS="${RLS:-32 64 128}"
OL="${OL:-2048}"
M="sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_lsu.sum,\
smsp__inst_executed_pipe_fp64.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
launch__grid_size,launch__registers_per_thread,launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic"

echo "type,out_len,red_len,batch,kernel,metric,value" > "$OUT"
for ty in cdouble double cfloat float; do
  for rl in $RLS; do
    # Trans: m = red_len, n = out_len
    "$NCU" --target-processes all -k "regex:$KRX" --launch-count 1 \
      --metrics "$M" --csv "$BIN" "$ty" "$rl" "$OL" 512 T 1 2>/dev/null \
      | tr -d '"' \
      | awk -F, -v ty="$ty" -v rl="$rl" -v ol="$OL" 'NR>1 && NF>5 {
            k=$5; g=$(NF-2); v=$NF;
            printf "%s,%s,%s,512,%s,%s,%s\n", ty, ol, rl, k, g, v }' >> "$OUT"
  done
done
echo "wrote $OUT"

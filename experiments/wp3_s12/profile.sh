#!/usr/bin/env bash
# WP3 step 12 -- establish the MECHANISM behind the float/Side::Left cliff
# before building anything to fix it.
#
# The spec (S3.4) predicts 8x over-fetch "on both the read and the
# write-allocate" for Side::Left, on the grounds that 32 lanes each use 4 B of a
# 32 B sector. But over-fetched bytes here are not wasted in the usual way: the
# bytes lane u does not use at step s are exactly the bytes lane u DOES use at
# steps s+1..s+7. If they survive in L1 that long there is no DRAM over-fetch at
# all, and the cost is L1 transaction count instead -- a different defect with a
# different fix. This repo has already had one panel kernel misdiagnosed that
# way round (see the latrd panel note).
#
# So measure both levels, on three cells whose ranking is already known:
#   float Left  n=32  -- LOSES 0.71x
#   float Left  n=8   -- WINS  3.49x  (same side, same kernel, below the cliff)
#   float Right n=32  -- WINS  3.50x  (same order, coalesced side)
#
# NOTE THE FILTER. --kernel-name matches the MANGLED name, which for a SYCL
# kernel is _ZTSN8batchlas...; "regex:TrsmCtaKernel" matches nothing and ncu
# then profiles the run without emitting a single metric row, which looks
# exactly like a kernel that does no memory traffic. --kernel-name-base
# demangled is what makes the readable name the thing being matched.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s12"
mkdir -p "$OUT"
BIN=./build/benchmarks/trsm_benchmark

M="dram__bytes_read.sum,dram__bytes_write.sum"
M="$M,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
M="$M,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
M="$M,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
M="$M,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
M="$M,gpu__time_duration.sum"

run() {
    local name="$1" n="$2" q="$3" bs="$4" tag="$5"
    echo "=== $tag : $name n=$n q=$q batch=$bs ==="
    CUDA_VISIBLE_DEVICES=${PROF_GPU:-0} BATCHLAS_TRSM_ROUTE=native \
        ncu --kernel-name-base demangled --kernel-name regex:TrsmCtaKernel \
            --launch-count 1 --metrics "$M" --csv \
            "$BIN" --backend=CUDA --type=float --name="$name" \
            --min_time=1 --min_iters=1 --max_iters=1 "$n" "$q" "$bs" \
        > "$OUT/$tag.csv" 2>"$OUT/$tag.err"
    python3 experiments/wp3_s12/summarise.py "$OUT/$tag.csv" "$n" "$q" "$bs"
}

run BM_TRSM_OrthoLeft  32 1024 512 left-n32
run BM_TRSM_OrthoLeft   8 1024 512 left-n8
run BM_TRSM_OrthoRight 32 1024 512 right-n32

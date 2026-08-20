#!/usr/bin/env bash
# The outer trailing shapes again, but with a REALISTIC leading dimension.
#
# The first run used ld == rows, and that is not what trsm hands gemm. V2's
# operands are SUB-VIEWS of the full matrices and carry the parent's ld: at
# n=512 the C block is 128 x q with ld = 512, i.e. four times its own row count.
# A kernel that reads C well at ld=128 may not at ld=512, and the whole question
# here is whether cuBLAS's 2x advantage at k=256/384 survives the real layout.
#
# BATCHLAS_BENCH_LD_PAD adds a constant pad to every operand's ld, so pad=384
# gives the m=128 operands (A and C) exactly the ld=512 they have in situ. B's
# ld comes out at k+384 rather than 512; that is the one inexactness and it errs
# toward MORE stride, not less.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s16"
GPU="${GPU:-1}"

for route in vendor native; do
    BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_BENCH_BETA=1 BATCHLAS_BENCH_LD_PAD=384 \
    GPU_GUARD_MAX_WAIT=5400 \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
            --backend=CUDA --type=float --name=BM_GEMM \
            --min_time=200 --min_iters=10 --max_iters=200 \
            --csv="$OUT/outerpad-$route.csv" \
            128 1024 128,256,384 512 > "$OUT/outerpad-$route.log" 2>&1
    echo "$route exit=$?"
done

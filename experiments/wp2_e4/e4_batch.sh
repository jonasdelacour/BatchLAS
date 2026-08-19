#!/usr/bin/env bash
# Batch sensitivity across the accepted float window.
#
# register_128x128.hh's own header records 43.6 TFLOP/s against cuBLAS 43.9 at
# 512^3 BATCH 512 -- essentially parity. The main E4 sweep ran n>=384 at batch
# 96-128 for working-set reasons and saw 0.77-0.91x. If the gap closes at batch
# 512 then the window is fine in the regime that matters and the sweep's batch
# scaling was the confound; if it does not, the window is wrong.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,beta,arm,rep,gflops" > "$OUT"
for n in 128 192 256 384 512; do
  for batch in 128 512 1024; do
    for beta in 0 1; do
      for arm in vendor sycl; do
        for rep in 1 2 3; do
          v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm timeout 900 "$BIN" \
              --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
              "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
          echo "$n,$batch,$beta,$arm,$rep,${v:-NA}" >> "$OUT"
        done
      done
    done
  done
done
echo "e4 batch sweep done"

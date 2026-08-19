#!/usr/bin/env bash
# preferred() for double is `max_dim <= 512` with no floor, so it accepts n=4
# upward. The main sweep started at 32; this closes the bottom of the window,
# which is where real demand actually sits (double 4/5/8/10/15/16 all appear).
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,beta,arm,rep,gflops" > "$OUT"
for n in 4 8 16 24 32; do
  for beta in 0 1; do
    for arm in vendor sycl; do
      for rep in 1 2 3; do
        v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm \
            timeout 900 "$BIN" --backend=CUDA --type=double --name=BM_GEMM_FIXED128 \
            --warmup=5 "$n" "$n" "$n" 4096 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        echo "$n,4096,$beta,$arm,$rep,${v:-NA}" >> "$OUT"
      done
    done
  done
done
echo "small-n sweep done"

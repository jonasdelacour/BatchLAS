#!/usr/bin/env bash
# Where should a widened double window STOP?
#
# Panel shapes (224..992 x k=8..32) all win 1.10-1.41x, and square 4..512 wins
# 1.01-4.51x. Before widening preferred() for double I need the edges: very
# large square, very skewed, and a tiny-k extreme -- because a predicate must be
# set where the evidence stops, not where the trend is assumed to continue.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_transpose_benchmark
OUT="$1"
echo "m,n,k,batch,beta,tA,tB,arm,rep,gflops" > "$OUT"
run() {
  local m=$1 n=$2 k=$3 batch=$4 tA=$5 tB=$6
  for beta in 0 1; do
    for arm in vendor sycl; do
      for rep in 1 2 3; do
        v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm timeout 900 "$BIN" \
            --backend=CUDA --type=double --name=BM_GEMM_TRANSPOSE --warmup=5 \
            "$m" "$n" "$k" "$batch" "$tA" "$tB" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        echo "$m,$n,$k,$batch,$beta,$tA,$tB,$arm,$rep,${v:-NA}" >> "$OUT"
      done
    done
  done
}
# very large square
run 1024 1024 1024 64 0 0
run 2048 2048 2048 16 0 0
# very skewed: tall-thin and wide-flat
run 4096 64 64 64 0 0
run 64 4096 64 64 0 0
# k = 1, the degenerate rank-1 update that appears in real demand
run 512 512 1 256 0 0
run 992 992 8 128 0 0
echo "e5 edges done"

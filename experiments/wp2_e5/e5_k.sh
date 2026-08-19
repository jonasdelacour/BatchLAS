#!/usr/bin/env bash
# Where exactly does small k turn against the native double route?
# k=1 loses (0.49x), k=8 wins (1.34-1.46x). Place the boundary on evidence.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_transpose_benchmark
OUT="$1"
echo "m,n,k,batch,beta,tA,tB,arm,rep,gflops" > "$OUT"
for k in 1 2 3 4 6 8 12 16; do
  for beta in 0 1; do
    for arm in vendor sycl; do
      for rep in 1 2 3; do
        v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm timeout 900 "$BIN" \
            --backend=CUDA --type=double --name=BM_GEMM_TRANSPOSE --warmup=5 \
            512 512 "$k" 256 0 0 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        echo "512,512,$k,256,$beta,0,0,$arm,$rep,${v:-NA}" >> "$OUT"
      done
    done
  done
done
echo "k boundary done"

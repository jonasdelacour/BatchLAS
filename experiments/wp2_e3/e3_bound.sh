#!/usr/bin/env bash
# Confirm the Direct/Tiled16 boundary finding with repeats before changing code.
# The single-rep sweep said Tiled16 beats Direct at n=32 at every batch, and
# n=32 is the only cell in the whole double window where native lost to cuBLAS.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,beta,kernel,rep,gflops" > "$OUT"
for n in 25 28 32; do
  for batch in 512 4096; do
    for beta in 0 1; do
      for kern in direct tiled16; do
        for rep in 1 2 3; do
          s=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=sycl BATCHLAS_GEMM_SYCL_KERNEL=$kern \
              timeout 900 "$BIN" --backend=CUDA --type=double --name=BM_GEMM_FIXED128 \
              --warmup=5 "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
          echo "$n,$batch,$beta,$kern,$rep,${s:-NA}" >> "$OUT"
        done
      done
      for rep in 1 2 3; do
        v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=vendor timeout 900 "$BIN" \
            --backend=CUDA --type=double --name=BM_GEMM_FIXED128 --warmup=5 \
            "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        echo "$n,$batch,$beta,cublas,$rep,${v:-NA}" >> "$OUT"
      done
    done
  done
done
echo "boundary confirm done"

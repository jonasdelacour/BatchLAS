#!/usr/bin/env bash
# n=192 is the worst cell in the whole float sweep: 0.39-0.49x of cuBLAS, at
# every batch. 192 is squareish but not a multiple of 128, so
# can_use_aligned_nn_fast_path fails and the selector falls to the GENERIC
# 128x32x32 variant. Is any other native kernel better there?
#
# This is a native-vs-native question and is worth answering regardless of what
# happens to preferred()'s window: a vendor-free build has no choice but native,
# so the selector's pick at n=192 is what a vendor-free user actually gets.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,beta,kernel,gflops" > "$OUT"
for n in 160 192 224 320; do
  for batch in 512; do
    for beta in 0 1; do
      v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=vendor timeout 900 "$BIN" \
          --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
          "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
      echo "$n,$batch,$beta,cublas,${v:-NA}" >> "$OUT"
      for kern in auto 128x32x32_s2_u1_generic 128x128x8 tiled16 32x32 64x64x16 128x32x16 128x32x32_s2_u2; do
        if [ "$kern" = auto ]; then
          s=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=sycl timeout 900 "$BIN" \
              --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
              "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        else
          s=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=sycl BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
              timeout 900 "$BIN" --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
              "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        fi
        echo "$n,$batch,$beta,$kern,${s:-NA}" >> "$OUT"
      done
    done
  done
done
echo "n192 alternatives done"

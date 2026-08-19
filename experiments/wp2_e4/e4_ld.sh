#!/usr/bin/env bash
# The UNALIGNED-LEADING-DIMENSION case, which the dimension sweep did not cover.
#
# tests/gemm_tests.cc pins 256x256x256 at ld=258 to the generic 128x32x32 route.
# That case is different from the ones measured: the DIMENSIONS tile perfectly
# (256 = 2 x 128) and only the addressing is unaligned, so the predicated path
# does bounds checks that never fire. Whether it still wins is a separate
# question and this answers it rather than extrapolating.
#
# It is not a corner case: a panel is a sub-view carrying its parent's ld, so
# unaligned ld is what the factorisations actually hand to gemm.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,pad,beta,kernel,rep,gflops" > "$OUT"
for n in 256 512; do
  for pad in 2 3; do
    for beta in 0 1; do
      for kern in vendor 128x32x32_s2_u1_generic 128x128x8; do
        for rep in 1 2 3; do
          if [ "$kern" = vendor ]; then
            v=$(BATCHLAS_BENCH_LD_PAD=$pad BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=vendor \
                timeout 900 "$BIN" --backend=CUDA --type=float --name=BM_GEMM_FIXED128 \
                --warmup=5 "$n" "$n" "$n" 256 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
          else
            v=$(BATCHLAS_BENCH_LD_PAD=$pad BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=sycl \
                BATCHLAS_GEMM_SYCL_KERNEL="$kern" timeout 900 "$BIN" --backend=CUDA --type=float \
                --name=BM_GEMM_FIXED128 --warmup=5 "$n" "$n" "$n" 256 2>/dev/null \
                | tail -1 | awk '{print $(NF-1)}')
          fi
          echo "$n,256,$pad,$beta,$kern,$rep,${v:-NA}" >> "$OUT"
        done
      done
    done
  done
done
echo "ld sweep done"

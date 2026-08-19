#!/usr/bin/env bash
# Does the predicated-128x128 win hold at LARGER unaligned squareish shapes, and
# at a non-square one? The n=160..320 result must not be extrapolated blindly:
# the generic 128x32x32 route it replaces was tuned somewhere, and this is the
# region where that tuning is most likely to be real.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,beta,kernel,gflops" > "$OUT"
for n in 544 672 800 1056; do
  for beta in 0 1; do
    v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=vendor timeout 900 "$BIN" \
        --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
        "$n" "$n" "$n" 96 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
    echo "$n,96,$beta,cublas,${v:-NA}" >> "$OUT"
    for kern in auto 128x128x8; do
      if [ "$kern" = auto ]; then
        s=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=sycl timeout 900 "$BIN" \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
            "$n" "$n" "$n" 96 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
      else
        s=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=sycl BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
            timeout 900 "$BIN" --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
            "$n" "$n" "$n" 96 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
      fi
      echo "$n,96,$beta,$kern,${s:-NA}" >> "$OUT"
    done
  done
done
echo "large unaligned done"

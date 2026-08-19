#!/usr/bin/env bash
# E5, non-square -- measured on the shapes the library ACTUALLY ISSUES rather
# than on a shape cross-product.
#
# preferred() requires m == n == k, so no non-square GEMM has ever routed
# native. The demand analysis (WP2 E2-prep) says that is where nearly all the
# real work is: the dominant internal GEMM is a PANEL UPDATE -- large m, large
# n, small k -- and k is the blocking factor, clustered at 8/32/48/96/136.
#
# The m/n/k values below are taken from the real-demand table:
#   992x992x32, 480x480x32, 288x288x32, 224x224x32, 248x248x8, 312x312x8
# and the transpose forms are the ones those call sites use (NC/NT dominate).
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_transpose_benchmark
OUT="$1"
TYPE="${2:-double}"
echo "m,n,k,batch,beta,tA,tB,arm,rep,gflops" > "$OUT"
run() { # m n k batch tA tB
  local m=$1 n=$2 k=$3 batch=$4 tA=$5 tB=$6
  for beta in 0 1; do
    for arm in vendor sycl; do
      for rep in 1 2 3; do
        v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm timeout 900 "$BIN" \
            --backend=CUDA --type="$TYPE" --name=BM_GEMM_TRANSPOSE --warmup=5 \
            "$m" "$n" "$k" "$batch" "$tA" "$tB" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        echo "$m,$n,$k,$batch,$beta,$tA,$tB,$arm,$rep,${v:-NA}" >> "$OUT"
      done
    done
  done
}
for shape in "992 992 32" "480 480 32" "288 288 32" "224 224 32" "248 248 8" "312 312 8"; do
  set -- $shape
  for tt in "0 0" "0 1" "1 0"; do
    set -- $shape; m=$1; n=$2; k=$3
    ttv=$tt; set -- $ttv; tA=$1; tB=$2
    run "$m" "$n" "$k" 128 "$tA" "$tB"
  done
done
echo "e5 panel sweep done"

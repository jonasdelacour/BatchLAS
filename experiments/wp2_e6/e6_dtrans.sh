#!/usr/bin/env bash
# The pre-flip prediction turned up an unmeasured region: preferred()'s DOUBLE
# branch is a bare `max_dim <= 512` with NO transpose test, so the flip would
# route double TN/NT/TT natively. E3 measured NN only, and float's transposed
# window turned out to be 0.34-0.55x of cuBLAS -- so this cannot be assumed.
#
# ConjTrans is included: float's branch explicitly rejects it as meaningless for
# a real type, double's does not, so the flip would route it too.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_transpose_benchmark
OUT="$1"
echo "n,batch,beta,tA,tB,arm,rep,gflops" > "$OUT"
for n in 32 64 128 256 512; do
  for tt in "1 0" "0 1" "1 1" "2 0"; do
    set -- $tt; tA=$1; tB=$2
    for beta in 0 1; do
      for arm in vendor sycl; do
        for rep in 1 2 3; do
          v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm timeout 900 "$BIN" \
              --backend=CUDA --type=double --name=BM_GEMM_TRANSPOSE --warmup=5 \
              "$n" "$n" "$n" 512 "$tA" "$tB" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
          echo "$n,512,$beta,$tA,$tB,$arm,$rep,${v:-NA}" >> "$OUT"
        done
      done
    done
  done
done
echo "double transposed done"

#!/usr/bin/env bash
# E4, float TRANSPOSED. preferred() accepts transposed float only at
# batch >= 128 and 128 <= max_dim <= 512 (and rejects ConjTrans outright, which
# is meaningless for a real type). Is that window in the right place?
#
# All three transposed forms, because the kernels differ per form: the ladder
# has separate TN / NT / TT entries.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_transpose_benchmark
OUT="$1"
echo "n,batch,beta,tA,tB,arm,rep,gflops" > "$OUT"
for n in 64 96 128 192 256 384 512 768; do
  batch=256
  for tt in "1 0" "0 1" "1 1"; do
    set -- $tt; tA=$1; tB=$2
    for beta in 0 1; do
      for arm in vendor sycl; do
        for rep in 1 2 3; do
          v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm timeout 900 "$BIN" \
              --backend=CUDA --type=float --name=BM_GEMM_TRANSPOSE --warmup=5 \
              "$n" "$n" "$n" "$batch" "$tA" "$tB" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
          echo "$n,$batch,$beta,$tA,$tB,$arm,$rep,${v:-NA}" >> "$OUT"
        done
      done
    done
  done
done
echo "e4 transposed done"

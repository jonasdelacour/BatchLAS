#!/usr/bin/env bash
# E4, float NN. preferred() for float accepts square + batch>=64 + either
# max_dim <= 32 or 128 <= max_dim <= 512. So there are FOUR regions and the
# question is different in each:
#
#   n <= 32      ACCEPTED  -- is it right to?
#   33..127      REJECTED  -- should the window widen down?
#   128..512     ACCEPTED  -- is it right to?
#   > 512        REJECTED  -- should the ceiling lift?
#
# Batch is scaled with n to keep the working set sane; native and vendor are
# always compared at the SAME (n, batch), so that is not a confound.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
echo "n,batch,beta,arm,rep,gflops" > "$OUT"
run() { # n batch
  local n="$1" batch="$2"
  for beta in 0 1; do
    for arm in vendor sycl; do
      for rep in 1 2 3; do
        v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm timeout 900 "$BIN" \
            --backend=CUDA --type=float --name=BM_GEMM_FIXED128 --warmup=5 \
            "$n" "$n" "$n" "$batch" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
        echo "$n,$batch,$beta,$arm,$rep,${v:-NA}" >> "$OUT"
      done
    done
  done
}
for n in 8 16 32 33 48 64 96 127 128; do run "$n" 512; done
for n in 192 256;  do run "$n" 256; done
for n in 384 512;  do run "$n" 128; done
for n in 640 768;  do run "$n" 96;  done
for n in 1024;     do run "$n" 64;  done
echo "e4 nn sweep done"

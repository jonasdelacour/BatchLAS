#!/usr/bin/env bash
# E3: does the native route beat cuBLAS DGEMM in the window preferred() accepts?
#
# preferred() for double is a transcribed `max_dim <= 512` plus square plus
# batch >= 64, so it accepts everything from n=4 to n=512. The flip (E6) would
# move 666 real double calls into that window -- and 585 of them land on
# Tiled16, not on the wide-scalar tile WP2 measured. Every Tiled16-vs-cuBLAS
# number in the tree is at 256^3 or above. This fills that gap.
#
# Arms are the ROUTES, not hand-picked kernels: `sycl` lets
# select_kernel_variant choose exactly as it would after the flip.
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=./build/benchmarks/gemm_benchmark
OUT="$1"
TYPE="${2:-double}"
REPS="${3:-3}"

echo "n,batch,beta,arm,rep,gflops" > "$OUT"
for n in 32 48 64 96 136 200 256 320 512; do
  for batch in 512; do
    for beta in 0 1; do
      for arm in vendor sycl; do
        for rep in $(seq 1 "$REPS"); do
          v=$(BATCHLAS_BENCH_BETA=$beta BATCHLAS_GEMM_VARIANT=$arm \
              timeout 600 "$BIN" --backend=CUDA --type="$TYPE" \
              --name=BM_GEMM_FIXED128 --warmup=5 "$n" "$n" "$n" "$batch" 2>/dev/null \
              | tail -1 | awk '{print $(NF-1)}')
          echo "$n,$batch,$beta,$arm,$rep,${v:-NA}" >> "$OUT"
        done
      done
    done
  done
done
echo "done -> $OUT"

#!/usr/bin/env bash
# Where is the Direct/Tiled16 crossover for COMPLEX, NN?
#
# gemm_kernels.cc:695 ends the complex NN ladder with
#   `return max_dim <= 64 ? KernelVariant::Direct : KernelVariant::Tiled16;`
# That 64 has never been measured for complex -- the same constant was measured
# for DOUBLE and moved to 24 (gemm_kernels.cc:637-660, WP2_GEMM_SPEC E3), and
# the trace shows the complex demand's hottest internal shape (16x64x16 NN,
# syev_two_stage, 255 calls) landing on Direct because max_dim == 64.
#
# Forced kernel, not forced route: BATCHLAS_GEMM_SYCL_KERNEL names the kernel,
# BATCHLAS_GEMM_VARIANT does not (an unrecognised value there silently means
# vendor).
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-1}"
OUT="$D/small_nn_crossover.csv"
REPS="${REPS:-7}"

SHAPES=(
  "n16     16 16 16   8192"
  "n32     32 32 32   8192"
  "n48     48 48 48   4096"
  "n64     64 64 64   4096"
  "s16x64  16 64 16   8192"
)
echo "kernel,type,m,n,k,batch,tA,tB,beta,padA,padB,padC,reps,median_ms,min_ms,rel_sd,gflops,tag" > "$OUT"
for spec in "${SHAPES[@]}"; do
  read -r tag m n k b <<< "$spec"
  for type in cfloat cdouble; do
    for beta in ${BETAS:-1}; do
      for kern in direct tiled16; do
        line=$(BATCHLAS_BENCH_BETA="$beta" BATCHLAS_GEMM_ROUTE=native \
               BATCHLAS_GEMM_SYCL_KERNEL="$kern" GPU_GUARD_MAX_WAIT=1800 \
               "$D/../../gpu_guard.sh" "$GPU" \
               "$D/cx_gemm_bench" "$type" "$m" "$n" "$k" "$b" N N "$REPS" 2>/dev/null)
        [[ -z "$line" ]] && { echo "FAILED: $kern $type $tag" >&2; continue; }
        echo "$kern,$line,$tag" >> "$OUT"
        echo "$kern,$line,$tag"
      done
    done
  done
done
echo "SMALL_DONE -> $OUT"

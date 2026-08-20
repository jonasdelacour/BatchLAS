#!/usr/bin/env bash
# The transposed complex path has its OWN Direct/Tiled16 boundary
# (gemm_kernels.cc:472, `max_dim <= 32 ? Direct : Tiled16`), reached for every
# transposed complex GEMM. Same question, same method as small_nn_crossover.sh.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-1}"
OUT="$D/small_trans_crossover.csv"
REPS="${REPS:-7}"
echo "kernel,type,m,n,k,batch,tA,tB,beta,padA,padB,padC,reps,median_ms,min_ms,rel_sd,gflops,tag" > "$OUT"
for spec in "n8 8 8 8 8192" "n16 16 16 16 8192" "n24 24 24 24 8192" "n32 32 32 32 8192"; do
  read -r tag m n k b <<< "$spec"
  for type in cfloat cdouble; do
    for kern in direct tiled16; do
      line=$(BATCHLAS_BENCH_BETA=1 BATCHLAS_GEMM_ROUTE=native \
             BATCHLAS_GEMM_SYCL_KERNEL="$kern" GPU_GUARD_MAX_WAIT=1800 \
             "$D/../../gpu_guard.sh" "$GPU" \
             "$D/cx_gemm_bench" "$type" "$m" "$n" "$k" "$b" C N "$REPS" 2>/dev/null)
      [[ -z "$line" ]] && { echo "FAILED: $kern $type $tag" >&2; continue; }
      echo "$kern,$line,$tag" >> "$OUT"; echo "$kern,$line,$tag"
    done
  done
done
echo "TRANS_DONE -> $OUT"

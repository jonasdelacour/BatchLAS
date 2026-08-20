#!/usr/bin/env bash
# What does vendor-freedom cost COMPLEX GEMM today?
#
# Every shape below comes from the measured demand table
# (experiments/wp4_complex/gpu1/complex_gemm_by_suite.csv), scaled to a batch
# that saturates an RTX 4090 -- the test-suite batches are 1-5 and a ratio taken
# there is overhead, not algorithm.
#
# vendor  = BATCHLAS_GEMM_ROUTE=vendor -> cublasCgemmStridedBatched/ZgemmStridedBatched
#           via cublasGemmStridedBatchedEx (src/backends/cublas.cc:117). There is
#           NO complex diversion on the GEMM path, unlike TRSM (cublas.cc:1111).
# native  = BATCHLAS_GEMM_ROUTE=native -> exactly what a vendor-free build runs,
#           since preferred() is bypassed the same way in both cases and
#           select_kernel_variant is unchanged.
#
# The two routes for one cell run BACK TO BACK, because this campaign has
# already recorded a 16% process-to-process outlier on a control whose trace
# proved the kernels identical.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-1}"
OUT="$D/cost_of_vendor_freedom.csv"
REPS="${REPS:-9}"

# tag type m n k batch tA tB
SHAPES=(
  "panel_herk129_cn   129 129 48   2048 C N"
  "panel_herk128_cn   128 128 48   2048 C N"
  "panel_her2k96_nc   96  96  64   2048 N C"
  "panel_syevx184_nc  184 184 16   1024 N C"
  "panel_2stage97_nc  97  97  16   2048 N C"
  "expand_trmm129_nn  129 96  129  1024 N N"
  "skinny_2stage_nn   16  64  16   8192 N N"
  "skinny_2stage_cn   16  64  16   8192 C N"
  "square256_nn       256 256 256  256  N N"
  "square512_nn       512 512 512  64   N N"
  "square1024_nn      1024 1024 1024 8   N N"
)

echo "route,type,m,n,k,batch,tA,tB,beta,padA,padB,padC,reps,median_ms,min_ms,rel_sd,gflops,tag" > "$OUT"

for spec in "${SHAPES[@]}"; do
  read -r tag m n k b tA tB <<< "$spec"
  for type in cfloat cdouble; do
    for beta in 1 0; do
      for route in vendor native; do
        line=$(BATCHLAS_BENCH_BETA="$beta" BATCHLAS_GEMM_ROUTE="$route" \
               GPU_GUARD_MAX_WAIT=1800 \
               "$D/../../gpu_guard.sh" "$GPU" \
               "$D/cx_gemm_bench" "$type" "$m" "$n" "$k" "$b" "$tA" "$tB" "$REPS" 2>/dev/null)
        if [[ -z "$line" ]]; then
          echo "FAILED: $route $type $tag" >&2
          continue
        fi
        echo "$route,$line,$tag" >> "$OUT"
        echo "$route,$line,$tag"
      done
    done
  done
done
echo "COST_DONE -> $OUT"

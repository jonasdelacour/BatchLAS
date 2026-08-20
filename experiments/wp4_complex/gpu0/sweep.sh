#!/usr/bin/env bash
# WP4 complex GEMM: is the wide-scalar register kernel's PREDICATED leg being
# routed away from, exactly as the float 128x128 kernel's was?
#
# gemm_kernels.cc:631-635 gates Tiled64x64RegisterK16Wide on
#   min_dim >= 256 && can_use_64x64_k16_wide_fast_path<T>(A,B,C)
# but the dispatcher at gemm_kernels.cc:794-810 re-evaluates that same
# predicate and picks <true>/<false> itself. It is a LEG predicate. Failing it
# does not demote the call to the predicated leg -- it hands the call to
# Tiled16 (or Direct).
#
# Arms, all on the NATIVE route (BATCHLAS_GEMM_ROUTE=native = the vendor-free
# build's behaviour, since route_gemm.hh's preferred() refuses complex):
#   auto   -- what select_kernel_variant picks today
#   wide   -- the wide register kernel forced onto whichever leg it can use
#   vendor -- cuBLAS, for scale only
#
# Interleaved: every (shape, beta, pad) triple runs its three arms back to back,
# and the whole grid is repeated REPS times. rep 0 is a discarded warm-up.
#
# Any cell whose gpu_guard reports a foreign process (exit 5) or a timeout
# (exit 4) has its CSV DELETED and is retried, so no tainted number can reach
# the aggregator.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_complex/gpu0/raw"
mkdir -p "$OUT"
GPU="${GPU:-0}"
REPS="${REPS:-2}"
TYPES="${TYPES:-cfloat,cdouble}"

SHAPES=(
  "P_k8      1024 1024 8   128"
  "P_k32     1024 1024 32  128"
  "P_k64     1024 1024 64  128"
  "P_k96     1024 1024 96  128"
  "P_k136    1024 1024 136 128"
  "P_992k32  992  992  32  128"
  "P_480k32  480  480  32  256"
  "S_128     128  128  128 512"
  "S_256     256  256  256 256"
  "S_512     512  512  512 128"
)

run() { # tag cfg m n k batch beta pad rep
    local tag=$1 cfg=$2 m=$3 n=$4 k=$5 b=$6 beta=$7 pad=$8 rep=$9
    local f="$OUT/${tag}-${cfg}-b${beta}-pad${pad}-r${rep}"
    local route=native kern=""
    case "$cfg" in
        auto)   route=native; kern="" ;;
        wide)   route=native; kern="reg64x64k16wide" ;;
        vendor) route=vendor; kern="" ;;
    esac
    local attempt rc
    for attempt in 1 2 3 4 5; do
        rm -f "$f.csv"
        BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
        BATCHLAS_BENCH_BETA="$beta" BATCHLAS_BENCH_LD_PAD="$pad" \
        GPU_GUARD_MAX_WAIT=3600 \
            ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/gemm_benchmark \
                --backend=CUDA --type="$TYPES" --name=BM_GEMM_FIXED128 \
                --min_time=300 --min_iters=10 --max_iters=200 \
                --csv="$f.csv" "$m" "$n" "$k" "$b" > "$f.log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "  $tag $cfg beta=$beta pad=$pad rep=$rep ok"
            return 0
        fi
        echo "  $tag $cfg beta=$beta pad=$pad rep=$rep RETRY (guard rc=$rc, attempt $attempt)"
        rm -f "$f.csv"
        sleep 20
    done
    echo "  $tag $cfg beta=$beta pad=$pad rep=$rep GAVE UP (rc=$rc)"
    return 1
}

for rep in $(seq 0 "$REPS"); do
  for pad in 0 384; do
    for beta in 1 0; do
      for s in "${SHAPES[@]}"; do
        read -r tag m n k b <<< "$s"
        for cfg in auto wide vendor; do
          run "$tag" "$cfg" "$m" "$n" "$k" "$b" "$beta" "$pad" "$rep"
        done
      done
    done
  done
done
echo "sweep complete"

#!/usr/bin/env bash
# WP4 complex GEMM, part 3: WHERE DOES THE WIN STOP?
#
# sweep.sh and sweep2.sh establish that forcing the wide kernel wins on the
# shapes the gate refuses. A routing relaxation needs a LOWER BOUND, and a
# bound is only worth stating if there is a measured counterexample on the
# other side of it. Two families:
#
#   L_*  a square ladder down through the Direct/Tiled16 crossover. For complex
#        select_kernel_variant ends at `max_dim <= 64 ? Direct : Tiled16`
#        (gemm_kernels.cc:638), so L_16..L_64 are the Direct arm and L_96+ the
#        Tiled16 arm.
#   C_*  the actual NN complex shapes in build/coverage.csv that a relaxation
#        would newly capture, cheapest-first. 1x129x1 and 7x200x7 are there to
#        FAIL: a 64x64 macro tile on a 1-row output wastes 98% of every CTA.
#
# Arms auto vs wide only -- the vendor is not the question here.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_complex/gpu0/raw3"
mkdir -p "$OUT"
GPU="${GPU:-0}"
# Which guard. experiments/gpu_guard.sh is the default and the strict one;
# GUARD=ctx swaps in gpu_guard_ctx.sh, which tolerates a foreign process that
# holds only an idle context on this card. See gpu_guard_ctx.sh.
GUARD_SH="${GUARD_SH:-./experiments/gpu_guard.sh}"
REPS="${REPS:-1}"
TYPES="${TYPES:-cfloat,cdouble}"

CASES=(
  "L_16          16   16  16 4096"
  "L_32          32   32  32 4096"
  "L_48          48   48  48 2048"
  "L_64          64   64  64 2048"
  "L_96          96   96  96 1024"
  "L_128        128  128 128  512"
  "C_1x129x1      1  129   1 4096"
  "C_7x200x7      7  200   7 4096"
  "C_31x320x31   31  320  31 2048"
  "C_64x128x32   64  128  32 2048"
  "C_128x256x64 128  256  64 1024"
  "C_256x512x128 256 512 128  512"
  "C_129x96x129 129   96 129 1024"
  "C_300x32x300 300   32 300 1024"
)

run() { # tag cfg m n k batch beta rep
    local tag=$1 cfg=$2 m=$3 n=$4 k=$5 b=$6 beta=$7 rep=$8
    local f="$OUT/${tag}-${cfg}-b${beta}-r${rep}"
    local route=native kern=""
    case "$cfg" in
        auto)   route=native; kern="" ;;
        wide)   route=native; kern="reg64x64k16wide" ;;
        vendor) route=vendor; kern="" ;;
    esac
    # Resumable: a cell that already has a non-empty CSV is left alone, so the
    # sweep can be re-invoked after an interruption without re-measuring or
    # discarding what is already on disk.
    if [ -s "$f.csv" ]; then return 0; fi
    local attempt rc
    for attempt in 1 2 3 4 5; do
        rm -f "$f.csv"
        BATCHLAS_GEMM_ROUTE="$route" BATCHLAS_GEMM_SYCL_KERNEL="$kern" \
        BATCHLAS_BENCH_BETA="$beta" BATCHLAS_BENCH_LD_PAD=0 \
        GPU_GUARD_MAX_WAIT=3600 \
            "$GUARD_SH" "$GPU" ./build/benchmarks/gemm_benchmark \
                --backend=CUDA --type="$TYPES" --name=BM_GEMM_FIXED128 \
                --min_time=300 --min_iters=10 --max_iters=200 \
                --csv="$f.csv" "$m" "$n" "$k" "$b" > "$f.log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "  $tag $cfg beta=$beta rep=$rep ok"
            return 0
        fi
        echo "  $tag $cfg beta=$beta rep=$rep RETRY (guard rc=$rc, attempt $attempt)"
        rm -f "$f.csv"
        sleep 20
    done
    echo "  $tag $cfg beta=$beta rep=$rep GAVE UP (rc=$rc)"
    return 1
}

for rep in $(seq 0 "$REPS"); do
  for beta in 1 0; do
    for c in "${CASES[@]}"; do
      read -r tag m n k b <<< "$c"
      for cfg in auto wide; do
        run "$tag" "$cfg" "$m" "$n" "$k" "$b" "$beta" "$rep"
      done
    done
  done
done
echo "sweep3 complete"

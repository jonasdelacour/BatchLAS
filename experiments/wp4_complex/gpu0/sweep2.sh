#!/usr/bin/env bash
# WP4 complex GEMM, part 2: the PREDICATED LEG'S OWN COST, isolated -- plus an
# explicit Tiled16 arm.
#
# sweep.sh answers "auto vs wide". Two things it cannot answer:
#
#  1. What does the wide kernel lose by running its PREDICATED leg instead of
#     its aligned one, on work of (almost) the same size? Three ways to knock
#     the same kernel off the aligned leg, which are NOT equivalent for complex
#     -- can_use_64x64_k16_wide_fast_path's VecLen is 16/sizeof(T), i.e. 2 for
#     complex<float> and 1 for complex<double>, so its ld/stride tests are %2
#     and %1 and an ld can barely fail them:
#       pad 0 vs pad 1   -- odd ld. Blocks cfloat; CANNOT block cdouble (%1).
#       k 256 vs 260     -- k%16 != 0. Blocks both. +1.6% arithmetic.
#       320^3 vs 300^3   -- m,n,k %64 != 0. Blocks both. 0.82x the arithmetic.
#     The 300^3 / 320^3 pair is the routing defect in its purest form: both
#     clear the min_dim >= 256 floor, both are NN, and the only thing between
#     them is an extent predicate the DISPATCHER re-evaluates anyway.
#
#  2. What in-tree Tiled16 costs on a shape where `auto` already IS the wide
#     kernel. register_64x64_k16_wide.hh's header claims 7.0-7.7x over Tiled16
#     for complex<float>, but that was measured against a standalone REPLICA of
#     Tiled16 in experiments/wide_scalar_gemm/measure, not the in-tree one. The
#     `t16` arm forces the real one.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT="experiments/wp4_complex/gpu0/raw2"
mkdir -p "$OUT"
GPU="${GPU:-0}"
# Which guard. experiments/gpu_guard.sh is the default and the strict one;
# GUARD=ctx swaps in gpu_guard_ctx.sh, which tolerates a foreign process that
# holds only an idle context on this card. See gpu_guard_ctx.sh.
GUARD_SH="${GUARD_SH:-./experiments/gpu_guard.sh}"
REPS="${REPS:-2}"
TYPES="${TYPES:-cfloat,cdouble}"

# tag m n k batch pad
CASES=(
  "A_128      128 128 128 512 0"
  "A_256      256 256 256 256 0"
  "A_256pad1  256 256 256 256 1"
  "A_256k260  256 256 260 256 0"
  "A_512      512 512 512 128 0"
  "A_512pad1  512 512 512 128 1"
  "A_320      320 320 320 256 0"
  "A_300      300 300 300 256 0"
)

run() { # tag cfg m n k batch pad beta rep
    local tag=$1 cfg=$2 m=$3 n=$4 k=$5 b=$6 pad=$7 beta=$8 rep=$9
    local f="$OUT/${tag}-${cfg}-b${beta}-r${rep}"
    local route=native kern=""
    case "$cfg" in
        auto)   route=native; kern="" ;;
        wide)   route=native; kern="reg64x64k16wide" ;;
        t16)    route=native; kern="tiled16" ;;
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
        BATCHLAS_BENCH_BETA="$beta" BATCHLAS_BENCH_LD_PAD="$pad" \
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
      read -r tag m n k b pad <<< "$c"
      for cfg in auto wide t16 vendor; do
        run "$tag" "$cfg" "$m" "$n" "$k" "$b" "$pad" "$beta" "$rep"
      done
    done
  done
done
echo "sweep2 complete"

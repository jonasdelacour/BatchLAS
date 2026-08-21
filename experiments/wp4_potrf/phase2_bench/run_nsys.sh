#!/usr/bin/env bash
# WHERE DOES THE TIME GO.  nsys, not BATCHLAS_KERNEL_TRACE: the implementer
# already recorded that the trace emits only sycl_submit / sycl_parallel_for with
# no kernel names in this build, so it cannot attribute time to a stage, and it
# inflates wall time ~60% besides.  nsys names the CUDA kernels, which is exactly
# the leaf / panel-trsm / trailing-gemm / fold split we need.
#
# NOT a timing run.  The medians in main.csv are the timings; this only
# apportions them.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
GPU=${GPU:-1}
NSYS=${NSYS:-/usr/local/cuda-13.2/bin/nsys}
mkdir -p "$D/nsys"
run () { # tag cfg type n batch
  local tag=$1 cfg=$2 t=$3 n=$4 b=$5
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
  case $cfg in
    nn) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native ;;
    nV) export BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=vendor ;;
    VV) export BATCHLAS_GEMM_ROUTE=vendor BATCHLAS_TRSM_ROUTE=vendor ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU BENCH_WARM_S=0.2 BENCH_CHECK_COLS=1 \
    $NSYS profile -t cuda -f true -o "$D/nsys/$tag" --cuda-memory-usage=false \
    ./bench ab "$t" "$n" "$b" 2 > "$D/nsys/$tag.stdout" 2> "$D/nsys/$tag.stderr"
  $NSYS stats --report cuda_gpu_kern_sum --format csv \
    -o "$D/nsys/$tag" "$D/nsys/$tag.nsys-rep" >/dev/null 2>&1
  unset BATCHLAS_GEMM_ROUTE BATCHLAS_TRSM_ROUTE
}
for t in float cdouble; do
  run "${t}_1024_256_nn" nn "$t" 1024 256
  run "${t}_1024_256_nV" nV "$t" 1024 256
  run "${t}_1024_256_VV" VV "$t" 1024 256
done
echo "wrote $D/nsys"

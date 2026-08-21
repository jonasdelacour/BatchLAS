#!/usr/bin/env bash
# Build the Phase 2 benchmark against an ALREADY-BUILT library tree.
#   build.sh            -> vendor-present  build/      -> ./bench
#   build.sh novendor   -> vendor-free     build-novendor/ -> ./bench_nv
# Flags copied from build/benchmarks/CMakeFiles/gemm_benchmark.dir/{flags.make,link.txt}
# so the harness sees exactly the library ctest does.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp4_potrf/phase2_bench"
KIND=${1:-vendor}
if [ "$KIND" = novendor ]; then B="$W/build-novendor"; OUT="$D/bench_nv"; else B="$W/build"; OUT="$D/bench"; fi
LIBS=""
for f in batchlas_core batchlas_backends batchlas_extensions_eigen \
         batchlas_extensions_factorization batchlas_extensions_symmetric \
         batchlas_extensions_tridiag batchlas_extensions_sytrd batchlas_extensions_latrd \
         batchlas_extensions_stedc batchlas_extensions_cta batchlas_util batchlas_extra \
         batchlas_sycl batchlas_backends_cuda; do
  [ -f "$B/src/lib$f.so" ] && LIBS="$LIBS $B/src/lib$f.so"
done
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$B/include" -I"$W" \
  -Wl,--no-as-needed \
  "$D/bench.cpp" -o "$OUT" \
  -Wl,-rpath,"$B/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  $LIBS
echo "built $OUT"

#!/usr/bin/env bash
# Build the Phase 2 A/B harness against the ALREADY-BUILT libraries in build/.
# Flags and link line copied from experiments/wp4_complex/gpu1/build_bench.sh,
# which copied them from build/benchmarks/CMakeFiles/gemm_benchmark.dir/
# {flags.make,link.txt}, so the harness sees exactly the library ctest does.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp4_potrf/phase2_ab"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$W/build/include" -I"$W" \
  -Wl,--no-as-needed \
  "$D/realpotrf.cpp" -o "$D/realpotrf_nv" \
  -Wl,-rpath,"$W/build-novendor/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$W"/build-novendor/src/libbatchlas_core.so "$W"/build-novendor/src/libbatchlas_backends.so \
  "$W"/build-novendor/src/libbatchlas_extensions_eigen.so "$W"/build-novendor/src/libbatchlas_extensions_factorization.so \
  "$W"/build-novendor/src/libbatchlas_extensions_symmetric.so "$W"/build-novendor/src/libbatchlas_extensions_tridiag.so \
  "$W"/build-novendor/src/libbatchlas_extensions_sytrd.so "$W"/build-novendor/src/libbatchlas_extensions_latrd.so \
  "$W"/build-novendor/src/libbatchlas_extensions_stedc.so "$W"/build-novendor/src/libbatchlas_extensions_cta.so \
  "$W"/build-novendor/src/libbatchlas_util.so "$W"/build-novendor/src/libbatchlas_extra.so \
  "$W"/build-novendor/src/libbatchlas_sycl.so 
echo "built $D/phase2"

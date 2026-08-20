#!/usr/bin/env bash
# Build the standalone complex-GEMM harness against the ALREADY-BUILT libraries.
# Flags and link line copied from build/benchmarks/CMakeFiles/gemm_benchmark.dir/
# {flags.make,link.txt} so the harness sees exactly the library the test suite does.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp4_complex/gpu1"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89,native_cpu \
  -I"$W/include" -I"$W/build/include" \
  -Wl,--no-as-needed \
  "$D/cx_gemm_bench.cpp" -o "$D/cx_gemm_bench" \
  -Wl,-rpath,"$W/build/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$W"/build/src/libbatchlas_core.so "$W"/build/src/libbatchlas_backends.so \
  "$W"/build/src/libbatchlas_extensions_eigen.so "$W"/build/src/libbatchlas_extensions_factorization.so \
  "$W"/build/src/libbatchlas_extensions_symmetric.so "$W"/build/src/libbatchlas_extensions_tridiag.so \
  "$W"/build/src/libbatchlas_extensions_sytrd.so "$W"/build/src/libbatchlas_extensions_latrd.so \
  "$W"/build/src/libbatchlas_extensions_stedc.so "$W"/build/src/libbatchlas_extensions_cta.so \
  "$W"/build/src/libbatchlas_util.so "$W"/build/src/libbatchlas_extra.so \
  "$W"/build/src/libbatchlas_sycl.so "$W"/build/src/libbatchlas_backends_cuda.so
echo "built $D/cx_gemm_bench"

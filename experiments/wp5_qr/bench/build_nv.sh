#!/usr/bin/env bash
# Build the WP5 bench harness against the ALREADY-BUILT VENDOR-FREE libraries in
# build-novendor/ (-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF). Same source, same flags,
# different libraries -- so the A/B is the BUILD.
#
# -I"$W/build-novendor/include" supplies backend_config.h with
# BATCHLAS_HAS_CUBLAS 0, so factorization_vendor_available<CUDA> is FALSE here
# and the route column reports the vendor-free resolution.
# libbatchlas_backends_cuda.so is deliberately absent from the link line: it does
# not exist in this build.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/bench"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$W/build-novendor/include" -I"$W" \
  -Wl,--no-as-needed \
  "$D/qrbench.cpp" -o "$D/qrbench_nv" \
  -Wl,-rpath,"$W/build-novendor/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$W"/build-novendor/src/libbatchlas_core.so "$W"/build-novendor/src/libbatchlas_backends.so \
  "$W"/build-novendor/src/libbatchlas_extensions_eigen.so "$W"/build-novendor/src/libbatchlas_extensions_factorization.so \
  "$W"/build-novendor/src/libbatchlas_extensions_symmetric.so "$W"/build-novendor/src/libbatchlas_extensions_tridiag.so \
  "$W"/build-novendor/src/libbatchlas_extensions_sytrd.so "$W"/build-novendor/src/libbatchlas_extensions_latrd.so \
  "$W"/build-novendor/src/libbatchlas_extensions_stedc.so "$W"/build-novendor/src/libbatchlas_extensions_cta.so \
  "$W"/build-novendor/src/libbatchlas_util.so "$W"/build-novendor/src/libbatchlas_extra.so \
  "$W"/build-novendor/src/libbatchlas_sycl.so
echo "built $D/qrbench_nv"

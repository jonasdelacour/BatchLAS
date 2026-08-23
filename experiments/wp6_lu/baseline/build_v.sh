#!/usr/bin/env bash
# Build the WP6 baseline harness against the ALREADY-BUILT VENDOR-PRESENT
# libraries in build/. Link line copied verbatim from
# experiments/wp5_qr/bench/build_v.sh.
#
# -I"$W/build/include" supplies backend_config.h with BATCHLAS_HAS_CUBLAS 1, so
# dispatch::factorization_vendor_available<CUDA> and level3_vendor_available are
# TRUE in this binary and the route column it prints is this build's answer.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp6_lu/baseline"
SRC="${1:-$D/lubench.cpp}"
OUT="${2:-$D/lubench_v}"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$W/build/include" -I"$W" \
  -Wl,--no-as-needed \
  "$SRC" -o "$OUT" \
  -Wl,-rpath,"$W/build/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$W"/build/src/libbatchlas_core.so "$W"/build/src/libbatchlas_backends.so \
  "$W"/build/src/libbatchlas_extensions_eigen.so "$W"/build/src/libbatchlas_extensions_factorization.so \
  "$W"/build/src/libbatchlas_extensions_symmetric.so "$W"/build/src/libbatchlas_extensions_tridiag.so \
  "$W"/build/src/libbatchlas_extensions_sytrd.so "$W"/build/src/libbatchlas_extensions_latrd.so \
  "$W"/build/src/libbatchlas_extensions_stedc.so "$W"/build/src/libbatchlas_extensions_cta.so \
  "$W"/build/src/libbatchlas_util.so "$W"/build/src/libbatchlas_extra.so \
  "$W"/build/src/libbatchlas_sycl.so "$W"/build/src/libbatchlas_backends_cuda.so
echo "built $OUT"

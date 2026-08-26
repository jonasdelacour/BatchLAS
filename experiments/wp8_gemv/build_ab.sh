#!/usr/bin/env bash
# Build the WP8/I3 body5-vs-body3 A/B harness against the ALREADY-BUILT
# libraries in build/ (vendor-present, so the harness can also be pointed at the
# vendor for a reference row without a second build).
#
# CAMPAIGN TRAP 2: this binary resolves the gate in its OWN TU via
# gemv_seg_trans_width_debug, so it must be rebuilt after ANY change to
# src/sycl/gemv_native.{cc,hh} or the printed wA/wB columns lie.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$W/build/include" -I"$W" \
  -Wl,--no-as-needed \
  "$D/gemvsegab.cpp" -o "$D/gemvsegab_v" \
  -Wl,-rpath,"$W/build/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$W"/build/src/libbatchlas_core.so "$W"/build/src/libbatchlas_backends.so \
  "$W"/build/src/libbatchlas_extensions_eigen.so "$W"/build/src/libbatchlas_extensions_factorization.so \
  "$W"/build/src/libbatchlas_extensions_symmetric.so "$W"/build/src/libbatchlas_extensions_tridiag.so \
  "$W"/build/src/libbatchlas_extensions_sytrd.so "$W"/build/src/libbatchlas_extensions_latrd.so \
  "$W"/build/src/libbatchlas_extensions_stedc.so "$W"/build/src/libbatchlas_extensions_cta.so \
  "$W"/build/src/libbatchlas_util.so "$W"/build/src/libbatchlas_extra.so \
  "$W"/build/src/libbatchlas_sycl.so "$W"/build/src/libbatchlas_backends_cuda.so
echo "built $D/gemvsegab_v"

#!/usr/bin/env bash
# Build the WP7 native-vs-vendor gemv A/B harness against the ALREADY-BUILT
# libraries in build/ (the VENDOR-PRESENT build -- both arms must be reachable
# from one process so they can be interleaved inside one session).
#
# CAMPAIGN TRAP 2: this binary resolves the printed route in its OWN TU, so it
# must be rebuilt after any preferred() change or the route column lies.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp7_gemv/ab"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$W/build/include" -I"$W" \
  -Wl,--no-as-needed \
  "$D/gemvab.cpp" -o "$D/gemvab_v" \
  -Wl,-rpath,"$W/build/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$W"/build/src/libbatchlas_core.so "$W"/build/src/libbatchlas_backends.so \
  "$W"/build/src/libbatchlas_extensions_eigen.so "$W"/build/src/libbatchlas_extensions_factorization.so \
  "$W"/build/src/libbatchlas_extensions_symmetric.so "$W"/build/src/libbatchlas_extensions_tridiag.so \
  "$W"/build/src/libbatchlas_extensions_sytrd.so "$W"/build/src/libbatchlas_extensions_latrd.so \
  "$W"/build/src/libbatchlas_extensions_stedc.so "$W"/build/src/libbatchlas_extensions_cta.so \
  "$W"/build/src/libbatchlas_util.so "$W"/build/src/libbatchlas_extra.so \
  "$W"/build/src/libbatchlas_sycl.so "$W"/build/src/libbatchlas_backends_cuda.so
echo "built $D/gemvab_v"

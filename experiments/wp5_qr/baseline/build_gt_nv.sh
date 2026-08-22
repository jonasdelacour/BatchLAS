#!/usr/bin/env bash
# Same program, linked against the VENDOR-FREE build. geqrf/orgqr will throw
# "no route" here (that is the point of the WP5 burn-down); ormqr will not.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/baseline"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$W/build-novendor/include" -I"$W" \
  -Wl,--no-as-needed \
  "$D/gemmtrail.cpp" -o "$D/gemmtrail_nv" \
  -Wl,-rpath,"$W/build-novendor/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$W"/build-novendor/src/libbatchlas_core.so "$W"/build-novendor/src/libbatchlas_backends.so \
  "$W"/build-novendor/src/libbatchlas_extensions_eigen.so "$W"/build-novendor/src/libbatchlas_extensions_factorization.so \
  "$W"/build-novendor/src/libbatchlas_extensions_symmetric.so "$W"/build-novendor/src/libbatchlas_extensions_tridiag.so \
  "$W"/build-novendor/src/libbatchlas_extensions_sytrd.so "$W"/build-novendor/src/libbatchlas_extensions_latrd.so \
  "$W"/build-novendor/src/libbatchlas_extensions_stedc.so "$W"/build-novendor/src/libbatchlas_extensions_cta.so \
  "$W"/build-novendor/src/libbatchlas_util.so "$W"/build-novendor/src/libbatchlas_extra.so \
  "$W"/build-novendor/src/libbatchlas_sycl.so
echo "built $D/gemmtrail_nv"

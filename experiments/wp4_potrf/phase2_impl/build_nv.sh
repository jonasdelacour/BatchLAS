#!/usr/bin/env bash
# The same harness, linked against the VENDOR-FREE build. No
# libbatchlas_backends_cuda.so exists there, so the `vendorcmp` mode is unusable;
# `facade` and `direct` are the point.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
B="$W/build-novendor"
D="$W/experiments/wp4_potrf/phase2_impl"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -DNOVENDOR=1 \
  -I"$W/include" -I"$B/include" -I"$W" \
  -Wl,--no-as-needed \
  "$D/verify.cpp" -o "$D/verify_nv" \
  -Wl,-rpath,"$B/src" \
  /usr/lib/x86_64-linux-gnu/liblapacke.so /usr/lib/x86_64-linux-gnu/libblas.so \
  "$B"/src/libbatchlas_core.so "$B"/src/libbatchlas_backends.so \
  "$B"/src/libbatchlas_extensions_eigen.so "$B"/src/libbatchlas_extensions_factorization.so \
  "$B"/src/libbatchlas_extensions_symmetric.so "$B"/src/libbatchlas_extensions_tridiag.so \
  "$B"/src/libbatchlas_extensions_sytrd.so "$B"/src/libbatchlas_extensions_latrd.so \
  "$B"/src/libbatchlas_extensions_stedc.so "$B"/src/libbatchlas_extensions_cta.so \
  "$B"/src/libbatchlas_util.so "$B"/src/libbatchlas_extra.so \
  "$B"/src/libbatchlas_sycl.so
echo "built $D/verify_nv"

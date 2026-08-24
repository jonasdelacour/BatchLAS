#!/usr/bin/env bash
# Build the fused-getrs probe against an ALREADY-BUILT BatchLAS.
#   build.sh nv   -> build-novendor/  (comp arm = the NATIVE trsm composition)
#   build.sh v    -> build/           (comp arm = the vendor trsm composition)
# The `vendor` arm links libcublas DIRECTLY in both, so cublas?getrsBatched is
# the same reference in both links.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp6_getrs/proto"
WHICH="${1:-nv}"
if [ "$WHICH" = "nv" ]; then BD="$W/build-novendor"; OUT="$D/fusedrs_nv"; EXTRA=""; else BD="$W/build"; OUT="$D/fusedrs_v"; EXTRA="$BD/src/libbatchlas_backends_cuda.so"; fi
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$BD/include" -I"$W" -I/usr/local/cuda-13.2/include \
  -Wl,--no-as-needed \
  "$D/fusedrs.cpp" -o "$OUT" \
  -Wl,-rpath,"$BD/src" \
  "$BD"/src/libbatchlas_core.so "$BD"/src/libbatchlas_backends.so \
  "$BD"/src/libbatchlas_extensions_eigen.so "$BD"/src/libbatchlas_extensions_factorization.so \
  "$BD"/src/libbatchlas_extensions_symmetric.so "$BD"/src/libbatchlas_extensions_tridiag.so \
  "$BD"/src/libbatchlas_extensions_sytrd.so "$BD"/src/libbatchlas_extensions_latrd.so \
  "$BD"/src/libbatchlas_extensions_stedc.so "$BD"/src/libbatchlas_extensions_cta.so \
  "$BD"/src/libbatchlas_util.so "$BD"/src/libbatchlas_extra.so \
  "$BD"/src/libbatchlas_sycl.so $EXTRA \
  -L/usr/local/cuda-13.2/lib64 -lcublas -lcudart
echo "built $OUT"

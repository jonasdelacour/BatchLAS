#!/bin/sh
# Build and install the AdaptiveCpp release BatchLAS is A/B-tested against, GPU-less.
#
#   .github/ci/build_adaptivecpp.sh <install-prefix> [<work-dir>]
#
# AdaptiveCpp has no binary release, so the acpp-compile job builds it from the tag against
# the distro LLVM. Only the generic (SSCP) compiler and the OpenMP host runtime are built:
# --acpp-targets=generic compiles device code to target-independent IR, so a BatchLAS compile
# needs no CUDA toolkit and no GPU. Design: docs/design/sycl-implementations.md section 8 (P6).
#
# Environment (all optional):
#   ACPP_REPO   clone URL (default https://github.com/AdaptiveCpp/AdaptiveCpp.git; a local
#               clone works, as file:///path)
#   ACPP_TAG    release tag (default v25.10.0)
#   ACPP_SHA    commit the tag must resolve to; a moved tag is an error
#   LLVM_ROOT   LLVM install (default /usr/lib/llvm-20; 20 is AdaptiveCpp 25.10's ceiling)
#   JOBS        build parallelism (default: nproc)
set -eu

prefix=${1:?usage: build_adaptivecpp.sh <install-prefix> [<work-dir>]}
work=${2:-${TMPDIR:-/tmp}/adaptivecpp-build}
repo=${ACPP_REPO:-https://github.com/AdaptiveCpp/AdaptiveCpp.git}
tag=${ACPP_TAG:-v25.10.0}
sha=${ACPP_SHA:-9f842c701a599107cc6d117d3539f971036363a1}
llvm=${LLVM_ROOT:-/usr/lib/llvm-20}
jobs=${JOBS:-$(nproc)}

for f in "$llvm/bin/clang++" "$llvm/lib/cmake/llvm/LLVMConfig.cmake" \
         "$llvm/include/clang/AST/ASTContext.h" "$llvm/lib/libomp.so"; do
    [ -e "$f" ] || { echo "build_adaptivecpp: missing $f (clang-20 llvm-20-dev libclang-20-dev libomp-20-dev)" >&2; exit 1; }
done

rm -rf "$work"
mkdir -p "$work"
git clone --quiet --depth 1 --branch "$tag" "$repo" "$work/src"
got=$(git -C "$work/src" rev-parse HEAD)
if [ "$got" != "$sha" ]; then
    echo "build_adaptivecpp: $tag resolves to $got, expected $sha" >&2
    exit 1
fi

cmake -S "$work/src" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_C_COMPILER="$llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="$llvm/bin/clang++" \
    -DLLVM_DIR="$llvm/lib/cmake/llvm" \
    -DCLANG_EXECUTABLE_PATH="$llvm/bin/clang++" \
    -DACPP_COMPILER_FEATURE_PROFILE=full \
    -DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF \
    -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF
cmake --build "$work/build" -j "$jobs"
cmake --install "$work/build"

"$prefix/bin/acpp" --acpp-version | head -n 12
grep -q '"plugin-with-sscp-compiler" : "true"' "$prefix/etc/AdaptiveCpp/acpp-core.json" || {
    echo "build_adaptivecpp: the install has no SSCP compiler; --acpp-targets=generic would fail" >&2
    exit 1
}

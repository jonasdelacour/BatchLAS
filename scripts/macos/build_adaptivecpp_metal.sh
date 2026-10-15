#!/bin/bash
#
# Build and install the AdaptiveCpp that the BatchLAS Metal build uses.
#
# This reproduces the toolchain on the Apple-silicon box: AdaptiveCpp develop at
# ACPP_BASE plus the local patch series in scripts/macos/adaptivecpp-patches/,
# configured the way upstream CI does (.github/workflows/macos-metal.yml in the
# AdaptiveCpp tree), with Homebrew opt/ paths so a `brew upgrade` of a patch
# release does not leave Cellar paths dangling in acpp-core.json.
#
# Usage: scripts/macos/build_adaptivecpp_metal.sh
# Environment (all optional):
#   ACPP_SRC       source checkout            (default ~/opt/src/AdaptiveCpp)
#   ACPP_PREFIX    install prefix             (default ~/opt/adaptivecpp)
#   METAL_CPP_DIR  metal-cpp headers          (default ~/opt/src/metal-cpp)
#   ACPP_BASE      upstream commit to patch   (default 9ec394ef)
#   ACPP_BRANCH    local branch for patches   (default batchlas-metal)
#   ACPP_REAPPLY=1 recreate ACPP_BRANCH from ACPP_BASE and re-apply the patches
#   SKIP_BREW=1    do not run brew install
#
# Never push ACPP_BRANCH: the patches are local; docs/guide/build.md lists them.

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
patch_dir="$here/adaptivecpp-patches"
ACPP_SRC="${ACPP_SRC:-$HOME/opt/src/AdaptiveCpp}"
ACPP_PREFIX="${ACPP_PREFIX:-$HOME/opt/adaptivecpp}"
METAL_CPP_DIR="${METAL_CPP_DIR:-$HOME/opt/src/metal-cpp}"
ACPP_BASE="${ACPP_BASE:-9ec394ef}"
ACPP_BRANCH="${ACPP_BRANCH:-batchlas-metal}"
build_dir="$ACPP_SRC/build-metal"

if [ "${SKIP_BREW:-0}" != 1 ]; then
  brew install cmake llvm@21 lld@21 libomp boost openblas python
fi
command -v cmake >/dev/null || { echo "error: cmake not found (brew install cmake)" >&2; exit 1; }
LLVM_PREFIX="$(brew --prefix llvm@21)"
LLD_PREFIX="$(brew --prefix lld@21)"
OMP_PREFIX="$(brew --prefix libomp)"
SDK="$(xcrun --show-sdk-path)"

if [ ! -d "$METAL_CPP_DIR" ]; then
  # The archive's top-level directory is always metal-cpp/: unpack aside, then move it into place.
  parent="$(dirname "$METAL_CPP_DIR")"
  mkdir -p "$parent"
  unpack="$(mktemp -d "$parent/metal-cpp-unpack.XXXXXX")"
  curl -fL -o "$unpack/metal-cpp.zip" https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
  unzip -q "$unpack/metal-cpp.zip" -d "$unpack"
  mv "$unpack/metal-cpp" "$METAL_CPP_DIR"
  rm -rf "$unpack"
fi
if [ ! -f "$METAL_CPP_DIR/Metal/Metal.hpp" ]; then
  echo "error: $METAL_CPP_DIR/Metal/Metal.hpp not found; METAL_CPP_DIR must hold the metal-cpp headers" >&2
  exit 1
fi

if [ ! -d "$ACPP_SRC/.git" ]; then
  git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git "$ACPP_SRC"
fi
git -C "$ACPP_SRC" cat-file -e "$ACPP_BASE^{commit}" 2>/dev/null || git -C "$ACPP_SRC" fetch origin
if [ "${ACPP_REAPPLY:-0}" = 1 ] || ! git -C "$ACPP_SRC" rev-parse --verify -q "$ACPP_BRANCH" >/dev/null; then
  git -C "$ACPP_SRC" checkout -B "$ACPP_BRANCH" "$ACPP_BASE"
  git -C "$ACPP_SRC" am "$patch_dir"/*.patch
else
  git -C "$ACPP_SRC" checkout "$ACPP_BRANCH"
fi
# An existing branch is reused as is, so a patch added or edited here later would silently not be
# built: require the branch to sit on ACPP_BASE and carry exactly the patches' diffs (patch-ids).
base_sha=$(git -C "$ACPP_SRC" rev-parse "$ACPP_BASE^{commit}")
want_ids=$(cat "$patch_dir"/*.patch | git -C "$ACPP_SRC" patch-id --stable | cut -d' ' -f1 | sort)
have_ids=$(git -C "$ACPP_SRC" format-patch --stdout "$ACPP_BASE..HEAD" | git -C "$ACPP_SRC" patch-id --stable |
  cut -d' ' -f1 | sort)
if [ "$(git -C "$ACPP_SRC" merge-base "$ACPP_BASE" HEAD)" != "$base_sha" ] || [ "$want_ids" != "$have_ids" ]; then
  echo "error: $ACPP_BRANCH does not carry exactly the patches in $patch_dir on top of $ACPP_BASE;" \
    "re-run with ACPP_REAPPLY=1 (or update $patch_dir from the branch)" >&2
  exit 1
fi
git -C "$ACPP_SRC" log --oneline "$ACPP_BASE~1..HEAD"

cmake -S "$ACPP_SRC" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCLANG_EXECUTABLE_PATH="$LLVM_PREFIX/bin/clang++" \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DCMAKE_OSX_SYSROOT="$SDK" \
  -DCMAKE_INSTALL_PREFIX="$ACPP_PREFIX" \
  -DWITH_OPENCL_BACKEND=OFF \
  -DWITH_LEVEL_ZERO_BACKEND=OFF \
  -DWITH_CUDA_BACKEND=OFF \
  -DWITH_ROCM_BACKEND=OFF \
  -DWITH_METAL_BACKEND=ON \
  -DMETAL_INCLUDE_DIR="$METAL_CPP_DIR" \
  -DACPP_COMPILER_FEATURE_PROFILE=full \
  -DACPP_LLD_PATH="$LLD_PREFIX/bin/ld64.lld" \
  -DSEQUENTIAL_LINK_LINE="$OMP_PREFIX/lib/libomp.dylib" \
  "-DOMP_LINK_LINE=-fopenmp $OMP_PREFIX/lib/libomp.dylib -L$OMP_PREFIX/lib/" \
  "-DOMP_CXX_FLAGS=-fopenmp -D_ENABLE_EXTENDED_ALIGNED_STORAGE -I$OMP_PREFIX/include"
cmake --build "$build_dir" -j"$(sysctl -n hw.ncpu)"
cmake --install "$build_dir"

# The drivers start with `#!/usr/bin/env python3`. When /usr/bin precedes Homebrew on PATH that is
# the Command Line Tools xcrun shim, which exports CPATH=/usr/local/include and
# LIBRARY_PATH=/usr/local/lib to every compile: a stray /usr/local gtest then shadows the
# project's own. Pin the drivers to Homebrew's python3.
brew_python="$(brew --prefix)/bin/python3"
if [ -x "$brew_python" ]; then
  for drv in acpp syclcc syclcc-clang; do
    [ -f "$ACPP_PREFIX/bin/$drv" ] && sed -i '' "1s|^#!/usr/bin/env python3\$|#!$brew_python|" "$ACPP_PREFIX/bin/$drv"
  done
else
  echo "warning: $brew_python not found; acpp keeps /usr/bin/env python3 (CPATH/LIBRARY_PATH leak)" >&2
fi

# CLANG_EXECUTABLE_PATH is an opt/ path, but the clang include path is derived
# from the resolved binary; report any Cellar path left in the installed config.
if grep -n Cellar "$ACPP_PREFIX/etc/AdaptiveCpp/acpp-core.json"; then
  echo "warning: acpp-core.json pins Cellar paths; pin llvm@21 and libomp (brew pin)" >&2
fi
"$ACPP_PREFIX/bin/acpp-info" -l

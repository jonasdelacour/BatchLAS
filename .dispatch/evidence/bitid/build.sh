#!/usr/bin/env bash
# Build the bit-identity harness against val-base headers; link the .so files WITHOUT an rpath,
# so LD_LIBRARY_PATH alone decides which tree's libraries are loaded at run time.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-$HOME/BatchLAS/.claude/worktrees/val-base}"
LIBS="$BASE/build/src"
CXX=/opt/dpcpp-cuda/bin/clang++
FLAGS=(-O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored -Wno-unused-value
       --cuda-path=/usr/local/cuda-13.2 -fsycl-unnamed-lambda -fsycl-dead-args-optimization
       -fsycl-targets=spir64_x86_64,nvidia_gpu_sm_89)
INCS=(-I"$BASE" -I"$BASE/include" -I"$BASE/build/include")
SOS=()
for l in core backends extensions_eigen extensions_factorization extensions_tridiag extensions_sytrd \
         extensions_latrd extensions_stedc extensions_cta extensions_posv extensions_gesv extensions_level3 \
         util extra sycl backends_cuda; do
    SOS+=("$LIBS/libbatchlas_$l.so.0")
done
"$CXX" "${FLAGS[@]}" "${INCS[@]}" -c "$HERE/bitid.cc" -o "$HERE/bitid.o"
"$CXX" -O2 -fsycl -fsycl-targets=spir64_x86_64,nvidia_gpu_sm_89 --cuda-path=/usr/local/cuda-13.2 \
    -Wl,--no-as-needed "$HERE/bitid.o" -o "$HERE/bitid" "${SOS[@]}"
echo "built $HERE/bitid"
readelf -d "$HERE/bitid" | grep -E 'RPATH|RUNPATH' || echo "no RPATH/RUNPATH (good)"

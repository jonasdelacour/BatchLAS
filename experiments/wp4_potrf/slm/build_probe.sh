#!/usr/bin/env bash
# Standalone SYCL build -- no library link needed, this probe uses only sycl/sycl.hpp.
# Compiler + target flags copied from experiments/wp4_complex/gpu1/build_bench.sh,
# which in turn copied them from build/benchmarks/CMakeFiles/gemm_benchmark.dir/flags.make.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp4_potrf/slm"
SRC="${1:-$D/slm_probe.cpp}"
OUT="${2:-${SRC%.cpp}}"
/opt/dpcpp-cuda/bin/clang++ \
  -O2 -DNDEBUG -std=c++20 -fsycl -Wno-c++20-extensions -Wno-option-ignored \
  -fsycl-unnamed-lambda --cuda-path=/usr/local/cuda-13.2 -fsycl-dead-args-optimization \
  -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz \
  -fsycl-targets=nvidia_gpu_sm_89 \
  -I"$W/include" -I"$W/build/include" \
  "$SRC" -o "$OUT"
echo "built $OUT"

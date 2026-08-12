#!/usr/bin/env bash
# Rebuild every candidate with -Xcuda-ptxas -v into measure/, recording
# registers and spills. Nothing here touches the GPU.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(dirname "$HERE")"
D=/opt/dpcpp-cuda/bin/clang++
CUDA=/usr/local/cuda-13.2
: > "$HERE/status.txt"

build() {
    local T="$1"
    "$D" -O3 -std=c++20 -fsycl -fsycl-targets=nvidia_gpu_sm_89 \
        --cuda-path="$CUDA" -Xcuda-ptxas -v \
        "$SRC/$T.cpp" -o "$HERE/$T" > "$HERE/$T.build.log" 2>&1
    echo "$T exit=$?" >> "$HERE/status.txt"
}

for T in tile-128x128-k8-t8x4 tile-128x64-k8-t8x4 tile-64x64-k16-t4x4 tile-complex-split; do
    build "$T" &
done
wait
cat "$HERE/status.txt"

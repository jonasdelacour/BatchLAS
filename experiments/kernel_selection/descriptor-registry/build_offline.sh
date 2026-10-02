#!/bin/sh
# Offline checks of the descriptor registry (g++, no GPU, no SYCL compiler):
#   1. getrs on the same op-agnostic core (getrs_offline.cc);
#   2. the compile-time guards of a potrf tier (mutate_compile.py).
D=$(cd "$(dirname "$0")" && pwd)
R=$D/../../..
OUT=${TMPDIR:-/tmp}/ks_getrs_offline
g++ -std=c++20 -O1 -Wall -Wextra -I"$R/include" -I"$R/build/include" -I"$R/src" \
    "$D/getrs_offline.cc" -o "$OUT" || exit 1
"$OUT" || exit 1
python3 "$D/mutate_compile.py"

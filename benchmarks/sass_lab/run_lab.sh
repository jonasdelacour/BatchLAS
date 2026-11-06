#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build the SASS lab with the library's sm_89 device flags, dump SASS and resource usage, tabulate.
#   benchmarks/sass_lab/run_lab.sh [build|sass|time|all]   (default: all)
#   benchmarks/sass_lab/run_lab.sh intree LIB.so REGEX      (per-kernel stats of a built library)
# Env: OUT (build dir), CXX, CUDA, GEN_INC (generated headers of a configured tree),
#      LAB_DEFS (extra -D flags for experiments), TIME_ARGS (passed to sass_lab).
# Writes $OUT/sass_table.txt, $OUT/sass.csv, $OUT/timing.csv; ends its log with LAB_EXIT=<rc>.
set -uo pipefail
LAB_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$LAB_DIR/../.." && pwd)
OUT=${OUT:-$REPO/build-sass-lab}
CXX=${CXX:-/opt/dpcpp-cuda/bin/clang++}
CUDA=${CUDA:-/usr/local/cuda-13.2}
GEN_INC=${GEN_INC:-$HOME/BatchLAS/build/include}
MODE=${1:-all}
export LD_LIBRARY_PATH="$(dirname "$CXX")/../lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT/img"

# Device flags as in build/src/CMakeFiles/*/flags.make and link.txt (RelWithDebInfo, sm_89).
CFLAGS=(-O2 -DNDEBUG -std=c++20 -fPIC -gline-tables-only -fno-omit-frame-pointer -fsycl
        -fsycl-unnamed-lambda -fsycl-dead-args-optimization -fsycl-targets=nvidia_gpu_sm_89
        "--cuda-path=$CUDA" -Wno-c++20-extensions -Wno-option-ignored
        "-I$REPO" "-I$REPO/include" "-I$GEN_INC" ${LAB_DEFS:-})
LFLAGS=(-O2 -fsycl -fsycl-targets=nvidia_gpu_sm_89 "--cuda-path=$CUDA"
        -Xsycl-target-backend=nvptx64-nvidia-cuda --ftz=false -mllvm -enable-global-offset=false)

SRCS="main ops_v2a ops_v2b"

build() {
    local pids=() rc=0 src
    for src in $SRCS; do
        "$CXX" "${CFLAGS[@]}" -c "$LAB_DIR/$src.cc" -o "$OUT/$src.o" > "$OUT/$src.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    if [ "$rc" != 0 ]; then tail -n 30 "$OUT"/*.log; return 1; fi
    local objs=(); for src in $SRCS; do objs+=("$OUT/$src.o"); done
    "$CXX" "${LFLAGS[@]}" "${objs[@]}" -o "$OUT/sass_lab" > "$OUT/link.log" 2>&1 \
        || { tail -n 30 "$OUT/link.log"; return 1; }
}

sass() {
    rm -f "$OUT"/img/*
    /opt/dpcpp-cuda/bin/clang-offload-extract --stem="$OUT/img/img" "$OUT/sass_lab" > /dev/null
    : > "$OUT/all.sass"; : > "$OUT/all.res"
    for f in "$OUT"/img/img.*; do
        "$CUDA/bin/cuobjdump" -sass "$f" >> "$OUT/all.sass"
        "$CUDA/bin/cuobjdump" -res-usage "$f" >> "$OUT/all.res"
    done
    python3 "$LAB_DIR/sass_stats.py" --sass "$OUT/all.sass" --res "$OUT/all.res" --csv "$OUT/sass.csv" \
        > "$OUT/sass_table.txt"
}

timing() {
    "$OUT/sass_lab" ${TIME_ARGS:-} > "$OUT/timing.csv"
}

intree() {
    local dir="$OUT/intree"
    mkdir -p "$dir" && rm -f "$dir"/img.* && : > "$dir/all.sass" && : > "$dir/all.res"
    /opt/dpcpp-cuda/bin/clang-offload-extract --stem="$dir/img" "$1" > /dev/null
    for f in "$dir"/img.*; do
        "$CUDA/bin/cuobjdump" -sass "$f" >> "$dir/all.sass" 2> /dev/null
        "$CUDA/bin/cuobjdump" -res-usage "$f" >> "$dir/all.res" 2> /dev/null
    done
    python3 "$LAB_DIR/sass_stats.py" --sass "$dir/all.sass" --res "$dir/all.res" --match "$2"
}

rc=0
case "$MODE" in
    intree) intree "$2" "$3" || rc=$? ;;
    build) build || rc=$? ;;
    sass) sass || rc=$? ;;
    time) timing || rc=$? ;;
    all) { build && sass && timing; } || rc=$? ;;
esac
echo "LAB_EXIT=$rc"
exit "$rc"

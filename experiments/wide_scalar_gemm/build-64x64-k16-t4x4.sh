#!/usr/bin/env bash
# Build tile-64x64-k16-t4x4, report ptxas register/spill counts, and tabulate
# the memory/FMA mix of each generated kernel from the dumped PTX.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG=64x64-k16-t4x4
DPCPP=${DPCPP:-/opt/dpcpp-cuda/bin/clang++}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
ARCH=${ARCH:-sm_89}
DUMP="$HERE/devdump-$TAG"
LOG="$HERE/build-$TAG.log"

rm -rf "$DUMP"
mkdir -p "$DUMP"

"$DPCPP" -O3 -std=c++20 -fsycl \
    -fsycl-targets="nvidia_gpu_${ARCH}" \
    --cuda-path="$CUDA_HOME" \
    -Xcuda-ptxas -v \
    -save-offload-code="$DUMP" \
    "$HERE/tile-$TAG.cpp" -o "$HERE/tile-$TAG" 2> "$LOG"
rc=$?
if [ $rc -ne 0 ]; then
    echo "BUILD FAILED"; cat "$LOG"; exit $rc
fi
echo "BUILD OK -> $HERE/tile-$TAG"
echo
echo "=== ptxas: registers / spills (per kernel) ==="
grep -E "Compiling entry|Used [0-9]+ registers|spill" "$LOG" \
    | paste - - - | sed 's/ptxas info    : //g' | grep -v with_offset

# --- PTX instruction mix -------------------------------------------------
slice() {  # slice <ptx> <mangled>
    local F="$1" NAME="$2"
    local START END SL
    START=$(grep -n "\.weak \.entry ${NAME}(" "$F" | head -1 | cut -d: -f1)
    [ -z "$START" ] && return 1
    END=$(awk -v s="$START" 'NR>s && /^\.weak \.entry/ {print NR-1; exit}' "$F")
    [ -z "$END" ] && END=$(wc -l < "$F")
    SL=$(mktemp)
    sed -n "${START},${END}p" "$F" > "$SL"
    echo "--- $NAME"
    grep -oE "ld\.(shared|global)(\.nc)?\.[a-z0-9.]*|st\.(shared|global)\.[a-z0-9.]*|fma\.rn\.f(32|64)|__mul[sd]c3|call\.uni|bar\.sync" "$SL" \
        | sed 's/bar\.sync.*/bar.sync/' | sort | uniq -c | sort -rn | sed 's/^/    /'
    rm -f "$SL"
    return 0
}

echo
echo "=== PTX instruction mix ==="
for name in \
    _ZTSN3wsg20WideScalarGemmKernelIfLb1EEE \
    _ZTSN3wsg20WideScalarGemmKernelIdLb1EEE \
    _ZTSN3wsg20WideScalarGemmKernelINS_2CxIfEELb1EEE \
    _ZTSN3wsg20WideScalarGemmKernelINS_2CxIdEELb1EEE
do
    for f in "$DUMP"/*.s; do slice "$f" "$name" && break; done
done

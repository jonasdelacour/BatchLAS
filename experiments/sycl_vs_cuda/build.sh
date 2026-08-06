#!/usr/bin/env bash
# Build both halves of the SYCL-vs-CUDA head-to-head and dump SASS for each,
# so the generated code can be diffed instruction-for-instruction.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/build}"
mkdir -p "$OUT"

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
DPCPP=${DPCPP:-/opt/dpcpp-cuda/bin/clang++}
ARCH=${ARCH:-sm_89}

echo "=== nvcc build ==="
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch="$ARCH" -lineinfo \
    -Xptxas -v \
    "$HERE/sgemm_cuda.cu" -o "$OUT/sgemm_cuda" -lcublas 2> "$OUT/nvcc_ptxas.log"
cat "$OUT/nvcc_ptxas.log"

echo
echo "=== dpcpp build ==="
# -fsycl-targets picks AOT for sm_89 so we compare compiled SASS, not JIT.
"$DPCPP" -O3 -std=c++20 -fsycl \
    -fsycl-targets="nvidia_gpu_${ARCH}" \
    --cuda-path="$CUDA_HOME" \
    -Xcuda-ptxas -v \
    "$HERE/sgemm_sycl.cpp" -o "$OUT/sgemm_sycl" 2> "$OUT/dpcpp.log" || {
        echo "primary dpcpp invocation failed; log:"; cat "$OUT/dpcpp.log"; exit 1; }
grep -iE "register|spill|smem|used" "$OUT/dpcpp.log" || true

echo
echo "=== SASS dump ==="
"$CUDA_HOME/bin/cuobjdump" -sass "$OUT/sgemm_cuda" > "$OUT/sgemm_cuda.sass" 2>/dev/null || true
"$CUDA_HOME/bin/cuobjdump" -sass "$OUT/sgemm_sycl" > "$OUT/sgemm_sycl.sass" 2>/dev/null || true

for f in "$OUT/sgemm_cuda.sass" "$OUT/sgemm_sycl.sass"; do
    [ -s "$f" ] || { echo "  (no SASS in $(basename "$f"))"; continue; }
    echo "--- $(basename "$f")"
    printf '    %-12s %s\n' \
        FFMA   "$(grep -c 'FFMA'    "$f" || true)" \
        LDS.128 "$(grep -c 'LDS.128' "$f" || true)" \
        LDS.64  "$(grep -c 'LDS.64'  "$f" || true)" \
        "LDS (all)" "$(grep -cE '\bLDS\b|LDS\.' "$f" || true)" \
        LDG.128 "$(grep -c 'LDG.E.128' "$f" || true)" \
        STS     "$(grep -cE '\bSTS' "$f" || true)" \
        total   "$(grep -cE '^\s+/\*[0-9a-f]{4}\*/' "$f" || true)"
done

echo
echo "binaries in $OUT"

#!/usr/bin/env bash
# CORRECTNESS FIRST. Every candidate's own reference check, every dtype, both
# beta=0 and beta=1, serialized through gpu_guard on GPU 0. A candidate whose
# maxrelerr is not at round-off is BROKEN and its speed is not a result.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="$HERE/../../gpu_guard.sh"
GPU=${GPU:-0}
OUT="$HERE/check.log"
: > "$OUT"

say() { echo "$@" | tee -a "$OUT"; }

for BETA in 0 1; do
for DT in double cfloat cdouble float; do
    for T in tile-128x128-k8-t8x4 tile-128x64-k8-t8x4 tile-64x64-k16-t4x4; do
        say "### $T dtype=$DT beta=$BETA"
        "$GUARD" "$GPU" "$HERE/$T" --dtype "$DT" --beta "$BETA" --check-only 2>&1 | tee -a "$OUT"
        say "   (exit ${PIPESTATUS[0]})"
    done
done
done

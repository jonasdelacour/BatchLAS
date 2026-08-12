#!/usr/bin/env bash
# Timed sweep. Everything serialized through gpu_guard on one GPU; never two at
# once. Clocks are warmed with a throwaway run before the first timed launch of
# each binary (an idle 4090 sits at ~210 MHz, and a cold first iteration once
# fabricated a 3.7x regression in this repo).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="$HERE/../../gpu_guard.sh"
GPU=${GPU:-0}
OUT=${OUT:-$HERE/bench.log}
ITERS=${ITERS:-30}
WARMUP=${WARMUP:-10}

# shape triples: m n k batch  -- large batch only, per the library's actual use
SHAPES=("256 256 256 512" "512 512 512 128" "1024 1024 1024 32")
DTYPES=${DTYPES:-"double cfloat cdouble float"}
BETAS=${BETAS:-"0 1"}

: > "$OUT"

warm() {  # burn a few hundred ms on the card so clocks are up
    "$GUARD" "$GPU" "$HERE/cublas_baseline" --dtype double --m 512 --n 512 --k 512 \
        --batch 32 --iters 20 --warmup 5 >/dev/null 2>&1
}

run() {  # run <label> <binary> <extra args...>
    local label="$1"; shift
    local bin="$1"; shift
    # Save the extra args NOW: the shape loop below uses `set --`, which would
    # otherwise overwrite "$@" with the shape and pass it as junk to the binary.
    local extra=("$@")
    for s in "${SHAPES[@]}"; do
        local M N K B
        read -r M N K B <<< "$s"
        for dt in $DTYPES; do
            for beta in $BETAS; do
                local line
                line=$("$GUARD" "$GPU" "$bin" --dtype "$dt" --m "$M" --n "$N" --k "$K" \
                        --batch "$B" --beta "$beta" --iters "$ITERS" --warmup "$WARMUP" \
                        "${extra[@]}" 2>/dev/null | grep '^RESULT')
                if [ -z "$line" ]; then
                    line="RESULT dtype=$dt m=$M n=$N k=$K batch=$B beta=$beta ms=FAILED tflops=FAILED"
                fi
                echo "CAND=$label $line" | tee -a "$OUT"
            done
        done
    done
}

warm
run cublas          "$HERE/cublas_baseline"
run 128x128-t8x4    "$HERE/tile-128x128-k8-t8x4"
run 128x64-t8x4     "$HERE/tile-128x64-k8-t8x4"
run 64x64-k16-t4x4  "$HERE/tile-64x64-k16-t4x4" --skip-check
echo "done -> $OUT"

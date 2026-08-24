#!/usr/bin/env bash
# Correctness of the INTEGRATED tier, through the PUBLIC API, all three transA
# modes, against the host oracle in lubench6.cpp. The resolved route is printed on
# every row, so a cell that silently fell back to another tier is visible rather
# than passing green over code nothing executed.
#
# $1 selects the binary: nv (vendor-free -> native:cta) or v (vendor-present).
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/bench
BIN="$D/lubench6_${1:-nv}"
OUT="${OUT:-/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_getrs/correct_${1:-nv}.csv}"
: > "$OUT"
for t in float double cfloat cdouble; do
  for n in 1 2 3 7 16 17 31 32 33 63 64 65 127 128 129 255 511 512; do
    for r in 1 2 3 4 5 8; do
      CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0.05 NTRANS=3 NPROBE=3 \
        "$BIN" getrs "$t" "$n" "$r" 64 3 2>/dev/null >> "$OUT"
    done
  done
done
echo "--- rows: $(wc -l < "$OUT")"
echo "--- BAD rows:"; grep -c ',BAD$' "$OUT" || true
grep ',BAD$' "$OUT" || echo "(none)"
echo "--- distinct resolved routes:"
cut -d, -f12 "$OUT" | sort | uniq -c

#!/usr/bin/env bash
# The LARGE-n half of the correctness sweep, at small batch so the memory fits.
#
# It exists as its own file because it is the half that reaches wg = 1024 and the
# widest instantiations, i.e. the exact corner where the register gate fires --
# float n=2048 nrhs=8 transA=Trans aborted with "Exceeded the number of registers
# available on the hardware" before getrs_fused_wg gained its cap, and the NoTrans
# arm of the very same call had already printed a green residual first.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_lu/bench
BIN="$D/lubench6_${1:-nv}"
OUT="${OUT:-/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_getrs/correct_large_${1:-nv}.csv}"
: > "$OUT"
for t in float double cfloat cdouble; do
  for n in 513 1023 1024 1025 2048; do
    for r in 1 2 3 4 5 6 7 8; do
      CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0.05 NTRANS=3 NPROBE=2 \
        "$BIN" getrs "$t" "$n" "$r" 8 2 2>/dev/null >> "$OUT"
    done
  done
done
echo "--- rows: $(wc -l < "$OUT")"
echo "--- BAD / THREW rows:"
grep -E ',BAD$|THREW' "$OUT" || echo "(none)"
echo "--- distinct resolved routes:"
cut -d, -f12 "$OUT" | sort | uniq -c

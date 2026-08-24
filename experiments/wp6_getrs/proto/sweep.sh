#!/usr/bin/env bash
# The A/B grid. ONE process per cell; the four arms are interleaved INSIDE it,
# rep by rep, so the vendor, the shipped composition and the fused kernel see the
# same clocks. Batches are WP6's saturating ones (experiments/wp6_lu/bench).
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_getrs/proto
OUT="${OUT:-$D/grid.csv}"
: > "$OUT"
REPS="${REPS:-9}"
NS="${NS:-64 128 512 2048}"
RHS="${RHS:-1 2 4 8 16}"
TYPES="${TYPES:-float double cfloat cdouble}"
for t in $TYPES; do
  for n in $NS; do
    case $n in
      32|64) b=8192;;
      128) b=4096;;
      256) b=1024;;
      512) b=512;;
      1024) b=128;;
      2048) b=32;;
      *) b=512;;
    esac
    for r in $RHS; do
      CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S="${WARM_S:-0.5}" \
        "$D/fusedrs_${BUILD:-nv}" "$t" "$n" "$r" "$b" "$REPS" 2>/dev/null >> "$OUT"
    done
  done
done
column -s, -t < "$OUT"

#!/usr/bin/env bash
# pivman (the shipped scheme's geometry: one matrix per work-group, SLM-tree
# argmax, work-group barriers) versus pivsg (one SUB-GROUP per matrix, G packed
# per work-group, shuffle argmax, no work-group barrier).
#
# INTERLEAVED IN ONE SESSION, arm by arm within each cell, because this campaign
# has fabricated results by comparing runs from different sessions.
# 21 reps: at 5 the pivsg arm reported 12.4% relative sd, above the 10% discard
# rule, and a ratio taken there would not have been quotable.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_perf/proto
cd "$D"
OUT="$D/ab.csv"
: > "$OUT"
REPS="${REPS:-21}"
for t in float double cfloat cdouble; do
  for n in 16 24 32 48 64 96 128; do
    b=8192
    if [ "$n" -ge 96 ]; then b=4096; fi
    for v in pivman pivsg; do
      CUDA_VISIBLE_DEVICES="${GPU:-0}" ./pivsg "$v" "$t" "$n" "$b" "$REPS" 2>&1 | tail -1 >> "$OUT"
    done
  done
done
column -s, -t < "$OUT"

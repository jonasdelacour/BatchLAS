#!/usr/bin/env bash
# SETTLE THE CELLS THAT APPEARED TO WIN.
#
# The 21-rep sweep showed pivsg ahead at n <= 32 for float and cfloat -- but the
# pivman BASELINE in exactly those cells was the noisiest thing in the file
# (float n=16 relative sd 0.4389, n=24 0.1091; cfloat n=16 0.2874, n=24 0.1244,
# n=32 0.1276), all above the 10% discard rule. A ratio whose denominator is
# unusable is not a result, and the apparent win is precisely where the noise is,
# which is the shape of an artefact.
#
# 3 passes x 61 reps, arms interleaved WITHIN each pass, so a drift in clocks
# cannot be mistaken for a difference between arms.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_perf/proto
cd "$D"
OUT="$D/settle.csv"
: > "$OUT"
for pass in 1 2 3; do
  for t in float cfloat; do
    for n in 16 24 32; do
      for v in pivman pivsg; do
        CUDA_VISIBLE_DEVICES="${GPU:-0}" ./pivsg "$v" "$t" "$n" 8192 61 2>&1 | tail -1 \
          | sed "s/^/pass$pass,/" >> "$OUT"
      done
    done
  done
done
column -s, -t < "$OUT"

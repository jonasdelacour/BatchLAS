#!/usr/bin/env bash
# The ONE cell flat4 discarded for vendor relative sd > 10%, re-run in three
# passes of nine reps. The campaign's rule: a heavy-tailed rep distribution can
# inflate the relative sd while the MEDIAN reproduces to three or four significant
# figures, so the evidence for such a cell is the CROSS-PASS MEDIAN SPREAD, not
# the within-pass sd -- and a cell is neither quoted at an unstable number nor
# discarded when it is in fact stable.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" NPROBE=1 NTRANS=1 WARM_S=1.5 REPS=9
echo "getrs:float:32:1:4096" > "$D/noisy4_cells.txt"
for p in 1 2 3; do
  CELLFILE="$D/noisy4_cells.txt" bash "$D/run_cells.sh" "$D/noisy4_p${p}_vendor.csv" lubench6_v  vendor
  CELLFILE="$D/noisy4_cells.txt" bash "$D/run_cells.sh" "$D/noisy4_p${p}_cta.csv"    lubench6_nv native:cta
done
echo noisy4-DONE

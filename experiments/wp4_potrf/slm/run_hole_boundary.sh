#!/usr/bin/env bash
# The 48 KB hole: slm_probe found 49152 FAILING at wg=32 while 45056 and 65536 both
# passed. This locates the exact hole boundary at every work-group size.
set -euo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/slm
G=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/gpu_guard.sh
out="$D/scan_hole_boundary.csv"
echo "wg,bytes,rep,ok,err" > "$out"
for W in 32 64 128 256 1024; do
    "$G" 0 "$D/slm_scan" "$W" 48880 49168 8 1 2>/dev/null | tail -n +2 >> "$out"
done
echo "wrote $out"

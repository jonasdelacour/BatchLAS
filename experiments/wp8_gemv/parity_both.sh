#!/usr/bin/env bash
# Re-run the WP7 audit's PARITY GATE, unchanged, against the repaired kernel.
# Two passes, device 1 (idle and display-free; see the device note in
# experiments/wp8_gemv/README-ish header of parity_gpu0_note).
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
export GPU="${GPU:-1}"
export REPS=11
OUT="$D/parity_w8_p1.csv" bash "$W/experiments/wp7_gemv/audit/parity.sh"
OUT="$D/parity_w8_p2.csv" bash "$W/experiments/wp7_gemv/audit/parity.sh"
echo DONE

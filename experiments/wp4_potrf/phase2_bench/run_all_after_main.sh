#!/usr/bin/env bash
# Everything after the main grid, in one sequence so only one process ever holds
# GPU 1.  Order is by value: the recheck settles two suspect cells in main.csv,
# the novendor run settles whether forcing `native` is the same thing as a
# vendor-free build resolving to it, nsys says where the time goes, and the
# nb/W sweep is the tuning question the driver constants depend on.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_bench
cd "$D"
echo "=== recheck ==="   ; bash run_recheck.sh
echo "=== novendor ===" ; bash run_novendor.sh
echo "=== nsys ==="     ; bash run_nsys.sh
echo "=== nbsweep ==="  ; bash run_nb.sh
echo "=== ALL DONE ==="

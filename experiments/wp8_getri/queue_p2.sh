#!/usr/bin/env bash
# LU pass 2, on DEVICE 0, started once device 0's gemv queue has drained.
#
# THE TWO PASSES RUN ON DIFFERENT CARDS, AND THAT IS DELIBERATE RATHER THAN A
# COMPROMISE. Both are RTX 4090s with the same clocks and the same power limit;
# device 0 additionally drives the display (Xorg, gnome-shell, firefox), which
# WP8-I3 measured as depressing cuBLAS by up to 1.8x on L2-RESIDENT cells while
# leaving a DRAM-streaming kernel untouched. Every cell here is an LU
# factorisation or inversion at batch >= 128, i.e. hundreds of megabytes, so the
# prediction is that the two passes agree -- and the cross-pass spread over 161
# cells IS the test of that prediction. A systematic disagreement shows up as a
# spread, and GATE-B's rule is to quote the WORSE pass, which is the conservative
# direction whichever card is the honest one.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
cd "$W"
while pgrep -f 'gemvab_v' >/dev/null; do sleep 20; done
OUT=$W/experiments/wp8_getri/lu_p2.csv \
CELLFILE=$W/experiments/wp8_getri/cells.txt \
GPU=0 REPS=11 WARM_S=1.0 bash "$W/experiments/wp8_getri/pair_cells.sh"
echo LU_P2_DONE

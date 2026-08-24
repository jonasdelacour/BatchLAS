#!/usr/bin/env bash
# THE TWO CELLS THE DISCARD RULE REJECTED, RE-RUN IN THREE PASSES.
#
# grid/nrhs dropped `float n=64 nrhs=1 b=8192` and flat dropped
# `float n=512 nrhs=1 b=64`, both because the VENDOR arm's relative sd was above
# 10 % (0.143 and 0.120). The brief's rule for exactly this case: with a
# heavy-tailed rep distribution the relative sd can exceed 10 % while the MEDIAN
# reproduces to several significant figures, so the evidence to report is the
# CROSS-PASS MEDIAN SPREAD, not either the discarded cell or the unstable sd.
#
# Three passes, nine reps each, both arms interleaved pass by pass on the same
# pinned GPU -- interleaved so a drift in the machine hits both arms, which is
# the whole reason the A and B of an A/B are not run as two blocks.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" WARM_S=1.0 REPS=9 NPROBE=1 NTRANS=1
CELLS="getrs:float:64:1:8192 getrs:float:512:1:64"
for p in 1 2 3; do
  bash "$D/run_cells.sh" "$D/noisy_p${p}_vendor.csv" lubench6_v  vendor      $CELLS
  bash "$D/run_cells.sh" "$D/noisy_p${p}_cta.csv"    lubench6_nv native:cta  $CELLS
done
echo noisy-DONE

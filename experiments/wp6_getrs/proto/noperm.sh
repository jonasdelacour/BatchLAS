#!/usr/bin/env bash
# WHAT THE FOLDED-IN PERMUTATION COSTS, priced by removing it.
#
# This is a TIMING-ONLY BREAK: with the interchange walk gone the answers are
# wrong by construction (the residual column goes to O(1) and that is the proof
# the break took, not a failure). It exists because the walk is the one part of
# the fused kernel that CANNOT be parallelised -- LAPACK's ipiv is a sequence of
# transpositions, so column c of the RHS is walked by a single work-item over n
# dependent local-memory swaps -- and a number is the only way to know whether it
# is worth attacking next.
#
# For contrast: as a SEPARATE LAUNCH in the composed tier the same permutation was
# 26.4% of the call (nsys, float n=512 nrhs=1 batch=512).
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_getrs/proto
OUT="${OUT:-$D/noperm.csv}"
: > "$OUT"
run() {
  CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0.5 NOPERM=1 NOVENDOR=1 NOCOMP=1 NOSTREAM=1 \
    SLMBUDGET=97280 "$D/fusedrs_nv" "$@" 9 2>/dev/null >> "$OUT"
}
for t in float cdouble; do
  run "$t" 64 1 8192
  run "$t" 128 1 4096
  run "$t" 512 1 512
  run "$t" 2048 1 32
done
column -s, -t < "$OUT"

#!/usr/bin/env bash
# (wg, nb) tuning for the fused arm. ONE cell, both fused arms only, so the
# vendor and composition arms do not perturb the clocks between geometries.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_getrs/proto
T="${T:-float}"; N="${N:-512}"; R="${R:-1}"; B="${B:-512}"; REPS="${REPS:-9}"
for wg in 32 64 128 256 512 1024; do
  for nb in 4 8 16 32; do
    CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S="${WARM_S:-0.3}" NOVENDOR=1 NOCOMP=1 WG=$wg NB=$nb \
      "$D/fusedrs_nv" "$T" "$N" "$R" "$B" "$REPS" 2>&1 | grep '^fblock'
  done
done

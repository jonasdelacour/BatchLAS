#!/usr/bin/env bash
# Confirm (a) synthI agrees with ormqrI on time in the vendor build -- i.e. the
# synthetic reflectors measure the same work; (b) the vendor-free BUILD really
# has no geqrf/orgqr route, so the burn-down claim is a measurement not a quote;
# (c) synthI runs and is correct in the vendor-free build.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/baseline
export CUDA_VISIBLE_DEVICES=1 WARM_S=0.5
echo "== vendor build: ormqrI vs synthI, same shape =="
for t in float cdouble; do
  "$D/wp5qr_v" ormqrI "$t" 256 512 5
  "$D/wp5qr_v" synthI "$t" 256 512 5
done
echo "== vendor-FREE build: geqrf / orgqr must throw, synthI must work =="
for t in float double cfloat cdouble; do
  "$D/wp5qr_nv" geqrf "$t" 256 64 2
  "$D/wp5qr_nv" orgqr "$t" 256 64 2
done
for t in float double cfloat cdouble; do
  "$D/wp5qr_nv" synthI "$t" 256 512 5
done

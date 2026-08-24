#!/usr/bin/env bash
# THE getrf / getri REGRESSION CHECK.
#
# The change under measurement is getrs-only by inspection (the diff touches
# route_getrs.hh, getrs_route.hh, getrs_fused.cc, getrs_native.hh, the getrs arm
# of factorization.cc and getrs_buffer_size). Inspection is not the check: getrf
# and getri share the LuLaswp kernel family and a tier table, and this campaign
# has already shipped one change whose only effect was somewhere it was not
# looked for. So both ops are RE-RUN on wp6_lu/bench's own order32 and order1024
# cells, at wp6_lu's own WARM_S and REPS, and diffed cell by cell against the
# recorded medians in experiments/wp6_lu/bench/order{32,1024}_{vendor,native}.csv.
#
# PIN. wp6_lu's runner exports one value into all three LU variables; this one
# pins per op, so the getrf/getri arms here carry PINF/PINI and leave
# BATCHLAS_GETRS_ROUTE unset -- getrs is not called by these cells at all.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export GPU="${GPU:-1}" NPROBE=1 NTRANS=1 WARM_S=0.5 REPS=3
PINF=vendor PINI=vendor CELLFILE="$D/lu_cells.txt" bash "$D/run_cells.sh" "$D/lu_vendor.csv" lubench6_v none
CELLFILE="$D/lu_cells.txt" bash "$D/run_cells.sh" "$D/lu_native.csv" lubench6_nv none
echo lu-DONE

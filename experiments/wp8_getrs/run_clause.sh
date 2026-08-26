#!/usr/bin/env bash
# The clause ladder AFTER the gather: composition (vendor-free build, default
# spelling) vs cuBLAS (vendor-present build, no pin -- preferred() is all-false
# at these widths so it resolves vendor:auto).
#
# Two native passes and two vendor passes. The arms are two BUILDS, so they
# cannot share a process; the substitute for GATE-B's within-session interleave
# is the cross-pass median spread, which experiments/wp6_perf/regcap/run.sh uses
# for the same reason.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
export GPU=0 REPS=11 WARM_S=1.0 NPROBE=1 NTRANS=1
export CELLFILE="$D/clause_cells.txt"
# THREE sweeps, not four. The VENDOR arm is untouched by this pass, and a
# complete vendor measurement of every one of these cells already exists at
# REPS=11 with zero foreign processes from the WALK ladder earlier in this
# session (lad_v_p1.csv + hi_v_p1.csv). That is vendor pass 1; cl_v_p2 below is an
# independent vendor pass 2. Re-running a sweep that cannot have moved would cost
# 40 minutes of card time and buy nothing.
bash "$D/run_cells.sh" "$D/cl_nv_p1.csv" lubench6_nv none
bash "$D/run_cells.sh" "$D/cl_nv_p2.csv" lubench6_nv none
bash "$D/run_cells.sh" "$D/cl_v_p2.csv"  lubench6_v  none
echo "CLAUSE DONE"

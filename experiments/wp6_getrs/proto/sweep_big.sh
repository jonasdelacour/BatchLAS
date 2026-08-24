#!/usr/bin/env bash
# The cells the default probe budget (46,080 B) refused. This device's real
# per-work-group budget is local_mem_size - 4096 = 97,280 B, which is what the
# shipped tier uses, so these cells are inside the tier's capacity and have to be
# measured rather than left as a hole in the table.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp6_getrs/proto
OUT="${OUT:-$D/grid_big.csv}"
: > "$OUT"
run() { CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0.5 SLMBUDGET=97280 "$D/fusedrs_nv" "$@" 9 2>/dev/null >> "$OUT"; }
run cdouble 2048 1 32
run cdouble 2048 2 32
run cdouble 512 8 512
run cfloat 2048 4 32
run cfloat 2048 8 32
run cfloat 512 16 512
run double 2048 4 32
run double 2048 8 32
run double 512 16 512
run float 2048 8 32
run float 2048 16 32
column -s, -t < "$OUT"

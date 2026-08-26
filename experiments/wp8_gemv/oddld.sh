#!/usr/bin/env bash
# WP8/I3 -- THE ODD-ld CELL. The named risk against body 5 is that a run starts
# at element (b*stride + j*ld + s), so an ld that is not a multiple of the run
# length misaligns every column and the L*sizeof(T)-byte run straddles an extra
# 32-byte sector. Body 3's 512-byte runs pay ~6% for that; body 5's 16-to-64
# byte runs are predicted to pay 25-100%.
#
# tests/gemv_tests.cc already exercises ld = 79 at m = 70, so this is not a
# hypothetical layout. Each cell is run PACKED (ld = red_len) and ODD (ld =
# red_len + 1 or a prime-ish pad) so the two are directly comparable.
set -uo pipefail
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENBLAS_CORETYPE=SKYLAKEX
export WARM_S="${WARM_S:-1.0}"
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp8_gemv"
BIN="$D/gemvsegab_v"
OUT="${OUT:-$D/oddld.csv}"
REPS="${REPS:-11}"

# red_len out_len batch ld
CELLS="
8 2048 512 8
8 2048 512 9
16 2048 512 16
16 2048 512 17
32 2048 512 32
32 2048 512 33
48 2048 512 48
48 2048 512 49
64 2048 512 64
64 2048 512 79
"
echo "type,m,n,batch,transA,arm,wA,wB,med_a_ms,med_b_ms,relsd_a,relsd_b,GBs_a,GBs_b,ratio,relerr_a,relerr_b,ld" > "$OUT"
while read -r rl ol b ldv; do
  [ -z "$rl" ] && continue
  for ty in cdouble double cfloat float; do
    LD=$ldv "$BIN" "$ty" "$rl" "$ol" "$b" T "$REPS" auto >> "$OUT" 2>>"$D/oddld_err.txt"
  done
done <<< "$CELLS"
echo "wrote $OUT"

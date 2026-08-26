#!/usr/bin/env bash
# WP8/I3 -- body 5 correctness spot-check, through the harness's own in-process
# host oracle (relerr, printed as field 12). Exercises the shapes a break would
# hide in: an ODD ld, a PARTIAL TAIL sub-group (out_len*batch not a multiple of
# W), red_len < L, ConjTrans, and every W including `off` (body 3, the control).
set -uo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES=$GPU
export OPENBLAS_CORETYPE=SKYLAKEX
export BATCHLAS_GEMV_ROUTE=native:cta
export WARM_S="${WARM_S:-0.05}"
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN="$W/experiments/wp7_gemv/ab/gemvab_v"

# m n batch ld-override -- under T/C, m is red_len and n is out_len.
SHAPES="
33 70 5 79
33 70 5 0
1 7 3 0
2 33 7 0
64 129 9 0
7 1 1 0
64 1024 4 0
3 5 1 71
"
fail=0
echo "type transA W ld m n batch relerr"
while read -r m n b ldv; do
  [ -z "$m" ] && continue
  for ty in double cdouble float cfloat; do
    for tr in T C; do
      for w in off auto 2 4 8; do
        out=$(LD=$ldv BATCHLAS_GEMV_SEGT=$w "$BIN" "$ty" "$m" "$n" "$b" "$tr" 3 2>&1)
        re=$(echo "$out" | awk -F, '{print $12}')
        echo "$ty $tr $w $ldv $m $n $b $re"
        case "$re" in 0.00e+00) ;; *) fail=$((fail+1)); echo "   ^^ NONZERO relerr";; esac
      done
    done
  done
done <<< "$SHAPES"
echo "nonzero-relerr rows: $fail"

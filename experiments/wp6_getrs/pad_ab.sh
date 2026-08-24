#!/usr/bin/env bash
# WHAT THE +1 BANK-CONFLICT PAD ON THE DIAGONAL BLOCK IS WORTH, measured on the
# TRANSPOSED path -- the only path whose block recurrence reads blk[s + t*bld] at
# stride bld across lanes. The NoTrans recurrence reads at stride 1 and cannot
# care either way, which is why timing it (as every other row in this experiment
# does) would have reported 1.00x and proved nothing.
#
# Break B5 removed the pad and turned NOTHING red -- correct, it is a performance
# choice -- so it needs a number or the comment claiming it is "load-bearing"
# is an unmeasured claim.
#
# usage: pad_ab.sh          (run once with the pad, once after breaks.sh B5's edit)
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN="$W/experiments/wp6_getrs/lubench6_tr_nv"
for t in float cdouble; do
  for n in 512 2048; do
    b=512; [ "$n" = 2048 ] && b=32
    for r in 1 8; do
      CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0.5 NTRANS=2 NPROBE=1 TRONLY=1 \
        BATCHLAS_GETRS_ROUTE=native:cta \
        "$BIN" getrs "$t" "$n" "$r" "$b" 9 2>/dev/null
    done
  done
done

#!/usr/bin/env bash
# Apply one deliberate break, REBUILD THE .so (the harness picks it up through
# its rpath, so no relink is needed), re-run the verification sweep, and revert.
#
# The output is kept in its own file: a break run must never be mistaken for a
# measurement run.
set -u
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp6_lu/kernels"
NAME="${1:?usage: run_break.sh <break-name> [pin]}"
PIN="${2:-native:blocked}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

python3 "$D/break.py" apply "$NAME" || exit 2
if ! cmake --build "$W/build" -j 32 > "$D/break_build.log" 2>&1; then
  echo "BUILD FAILED"; tail -20 "$D/break_build.log"
  python3 "$D/break.py" revert "$NAME"; exit 3
fi
echo "=== BREAK=$NAME pin=$PIN"
NS="${NS:-31 33 64 96 100}" NB="${NB:-4}" WARM_S=0.05 \
  bash "$D/run_verify.sh" luverify_v "$PIN" native:blocked 2>"$D/break_${NAME}_err.txt" | tee "$D/break_${NAME}.txt"
python3 "$D/break.py" revert "$NAME"

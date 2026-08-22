#!/usr/bin/env bash
# The KERNEL breaks. Each one DELETES or INVERTS the exact thing a check is meant
# to guard, rebuilds the CTA device-code cluster (~2 min), reruns the harness and
# reverts. Nothing is left in the tree.
#
# A break that does NOT turn a row red is the most valuable outcome here and is
# reported, not hidden.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp5_qr/kernels"
PY=/home/jonaslacour/.claude/jobs/20812aa0/tmp/kbreak.py
CELLS='^(cta|blocked),(float|double|cfloat|cdouble),(66,66|300,200|128,128),'

for k in "$@"; do
  echo "=== KERNEL BREAK $k ==="
  python3 "$PY" "$k" apply || exit 2
  if ! cmake --build "$W/build" --target batchlas_extensions_cta -j 32 > /tmp/kb_$k.log 2>&1; then
     echo "BUILD FAILED"; tail -20 /tmp/kb_$k.log
  else
     CUDA_VISIBLE_DEVICES=1 timeout 900 "$D/qrcheck_v" 2>&1 | grep -E "$CELLS"
  fi
  python3 "$PY" "$k" revert || exit 2
done
echo "=== restoring the unbroken build ==="
cmake --build "$W/build" --target batchlas_extensions_cta -j 32 > /tmp/kb_restore.log 2>&1 && echo restored

#!/usr/bin/env bash
# The REFERENCE breaks. BREAK=<n> damages the checker's own reference, never the
# kernel, so a green control (BREAK unset) and a red break together prove the
# probes can discriminate. A break that does NOT turn red is reported, not
# hidden: BREAK=4 (conjugate tau) is expected to be a NULL result for float and
# double, because conjugation is the identity on a real scalar.
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp5_qr/kernels
BIN=${BIN:-$D/qrcheck_v}
CELLS=${CELLS:-'^(cta|blocked),(float|double|cfloat|cdouble),(66,66|300,200),'}
for b in 0 1 2 3 4 5; do
  echo "=== BREAK=$b ==="
  BREAK=$b CUDA_VISIBLE_DEVICES=1 timeout 900 "$BIN" 2>/dev/null | grep -E "$CELLS"
done

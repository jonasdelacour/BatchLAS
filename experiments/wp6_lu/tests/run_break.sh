#!/usr/bin/env bash
# Apply one break, REBUILD THE .so, run tests/getrf_tests, revert, and record the
# full output. The rebuild is not optional: a break that is never compiled is a
# break that proves nothing, and this repository has six recorded instances of a
# guard that could not fail.
#
#   ./run_break.sh <break-name>
#
# THE BINARY IS RUN ONCE PER SCALAR TYPE, not once. A corrupted kernel can abort
# the process (the `short_final` break writes garbage pivots and takes the run
# down at float), and a single run then reports NOTHING about the three types
# that never executed. Four filtered runs cost nothing beside the rebuild and
# make a crash a per-type result rather than a black hole.
#
# Output goes to break_<name>.txt beside this script. FULL output is captured,
# not just the failure list, so a row the break should NOT have moved can be
# read -- which is how WP6's kernel-side campaign caught its own corrupted tree.
set -u
cd "$(dirname "$0")/../../.." || exit 1
HERE=experiments/wp6_lu/tests
NAME="$1"; shift || true
OUT="$HERE/break_${NAME}.txt"

python3 "$HERE/break.py" "$NAME" || exit 1
{
  echo "=== BREAK: $NAME ==="
  python3 - "$NAME" <<'PY'
import sys, importlib.util
spec = importlib.util.spec_from_file_location("bk", "experiments/wp6_lu/tests/break.py")
bk = importlib.util.module_from_spec(spec); spec.loader.exec_module(bk)
for nm in bk.GROUPS.get(sys.argv[1], [sys.argv[1]]):
    rel, old, new = bk.BREAKS[nm]
    print("  file: %s\n  from: %s\n  to  : %s" % (rel, old.strip()[:110], new.strip()[:110]))
PY
  echo "=== build ==="
  cmake --build build -j 32 2>&1 | tail -4
  for t in 4 5 6 7; do
    echo "=== run type $t ==="
    CUDA_VISIBLE_DEVICES=1 ./build/tests/getrf_tests --gtest_filter="LuTest/$t.*" 2>&1
    rc=$?
    echo "--- exit $rc ---"
  done
} > "$OUT" 2>&1
python3 "$HERE/break.py" "$NAME" --revert || echo "REVERT FAILED -- TREE IS DIRTY"
echo "wrote $OUT"

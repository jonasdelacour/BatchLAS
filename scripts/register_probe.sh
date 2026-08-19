#!/usr/bin/env bash
# Register/spill probe for the SYCL device link.
#
# WHY THIS SHAPE. Device code here is AOT-compiled to an sm_89 cubin by ptxas at the
# SHARED-LIBRARY DEVICE LINK, not per translation unit, so `-Xcuda-ptxas -v` on a compile
# is reported "argument unused" and produces nothing. WP3_TRSM_SPEC.md section 8 step 2
# makes register residency a hard gate before any other code is written; this is the
# recipe that actually satisfies it.
#
# It replays build/src/CMakeFiles/batchlas_sycl.dir/link.txt verbatim with one extra
# -Xsycl-target-backend pair appended and the output redirected, so no cmake reconfigure
# is needed and the flags stay exactly what the real build uses.
#
# THE GATE. Use these two conditions, NOT "stack frame == 0":
#   * `0 bytes spill stores, 0 bytes spill loads` on the kernel's lines, and
#   * `Used N registers` x work-group size <= 65536, the per-BLOCK limit that
#     src/sycl/gemm_kernels.cc:725-735 records as the real failure mode (a launch
#     abort, not a slowdown).
# Stack frame is the WRONG gate: measured on this tree, 220 of 376 entry functions
# carry a non-zero stack frame with zero spills, so gating on it rejects healthy
# kernels -- and a grep for "spill" that finds nothing reads as "no spill" whether
# or not the flag ever took effect.
#
# Each kernel appears TWICE, as `<name>` and `<name>_with_offset`; they can differ
# by a couple of registers. Take the max. Grep by mangled name.
#
# Verified against WP2's wide-scalar tile, which reproduces
# WP2_WIDE_SCALAR_GEMM_VERDICT.md's standalone measurements exactly:
#   float 56, double 76, complex<float> 80, complex<double> 132, all zero spill.
#
# Baseline on this branch with no TRSM code: 43.4 s link, 376 entry functions,
# 0 kernels with non-zero spill. WP3's link-time budget is a DELTA against that,
# not against the spec's stale ~30 s figure.
#
# Usage: scripts/register_probe.sh <out.log> [grep-pattern]
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/build/src
OUT="${1:-/home/jonaslacour/.claude/jobs/20812aa0/tmp/regprobe.log}"
PAT="${2:-}"

LINE=$(cat CMakeFiles/batchlas_sycl.dir/link.txt)
# Redirect the artifact away from the real build tree and add the ptxas verbosity.
LINE=${LINE/-o libbatchlas_sycl.so/-o \/tmp\/claude-1000\/regprobe.so}
LINE="$LINE -Xsycl-target-backend=nvptx64-nvidia-cuda -Xcuda-ptxas -v"

/usr/bin/time -f 'LINK %e s real, %U s user' bash -c "$LINE" > "$OUT" 2>&1
rc=$?
echo "exit=$rc  log=$OUT"
grep -c 'Compiling entry function' "$OUT" | sed 's/^/entry functions: /'
echo "kernels with non-zero spill:"
grep -E 'spill (stores|loads)' "$OUT" | grep -vE '0 bytes spill stores, 0 bytes spill loads' | wc -l
if [ -n "$PAT" ]; then
  echo "=== matching '$PAT' ==="
  grep -B1 -A2 "$PAT" "$OUT" | grep -E "Compiling entry|Used [0-9]+ registers|spill" | paste - - - 2>/dev/null | sed 's/ptxas info    : //g'
fi
grep 'LINK ' "$OUT" || true

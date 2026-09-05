#!/usr/bin/env bash
# Register/spill probe for the SYCL device link.
#
# WHY THIS SHAPE. Device code here is AOT-compiled to an sm_89 cubin by ptxas at the
# SHARED-LIBRARY DEVICE LINK, not per translation unit, so `-Xcuda-ptxas -v` on a compile
# is reported "argument unused" and produces nothing. docs/perf/trsm.md section 8 step 2
# makes register residency a hard gate before any other code is written; this is the
# recipe that actually satisfies it.
#
# It replays a target's link.txt verbatim with one extra -Xsycl-target-backend pair
# appended and the output redirected, so no cmake reconfigure is needed and the flags
# stay exactly what the real build uses.
#
# WHICH LIBRARY. This defaults to batchlas_sycl, and for a long time it could probe
# NOTHING ELSE -- which made it silently blind to every kernel in src/extensions/.
# Found during WP4: potrf_cta.cc links into libbatchlas_extensions_cta.so, so the
# default probe reported "424 entry functions, 0 with spill" over a set that did not
# contain PotrfCtaKernel at all. A clean report from the wrong library reads exactly
# like a clean report from the right one. Pass the target as $3 (or
# BATCHLAS_PROBE_TARGET) to probe a different one:
#
#   scripts/register_probe.sh out.log ''            # batchlas_sycl (default)
#   scripts/register_probe.sh out.log '' batchlas_extensions_cta
#
# The script now FAILS LOUDLY if the named target has no link.txt, rather than
# probing the default and reporting a healthy-looking result for the wrong thing.
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
# docs/perf/gemm.md's standalone measurements exactly:
#   float 56, double 76, complex<float> 80, complex<double> 132, all zero spill.
#
# Baseline on this branch with no TRSM code: 43.4 s link, 376 entry functions,
# 0 kernels with non-zero spill. WP3's link-time budget is a DELTA against that,
# not against the spec's stale ~30 s figure.
#
# Usage: scripts/register_probe.sh <out.log> [grep-pattern] [cmake-target]
set -uo pipefail
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/build/src
OUT="${1:-/home/jonaslacour/.claude/jobs/20812aa0/tmp/regprobe.log}"
PAT="${2:-}"

TARGET="${3:-${BATCHLAS_PROBE_TARGET:-batchlas_sycl}}"
LINKTXT="CMakeFiles/${TARGET}.dir/link.txt"
if [[ ! -f "$LINKTXT" ]]; then
    echo "register_probe: no link.txt for target '$TARGET' ($PWD/$LINKTXT)." >&2
    echo "  Available targets:" >&2
    ls -d CMakeFiles/*.dir 2>/dev/null | sed 's#CMakeFiles/##; s#\.dir$##; s/^/    /' >&2
    exit 2
fi
LINE=$(cat "$LINKTXT")
# Redirect the artifact away from the real build tree and add the ptxas verbosity.
LINE=${LINE/-o lib${TARGET}.so/-o \/tmp\/claude-1000\/regprobe.so}
LINE="$LINE -Xsycl-target-backend=nvptx64-nvidia-cuda -Xcuda-ptxas -v"

/usr/bin/time -f 'LINK %e s real, %U s user' bash -c "$LINE" > "$OUT" 2>&1
rc=$?
echo "exit=$rc  target=$TARGET  log=$OUT"
grep -c 'Compiling entry function' "$OUT" | sed 's/^/entry functions: /'
# ENTRY FUNCTIONS AND EVERYTHING ELSE, COUNTED SEPARATELY. ptxas emits
# "Function properties" for non-inlined DEVICE functions as well as for entry
# functions, and the two used to be summed into one number. On
# batchlas_extensions_cta that reads "16 kernels with non-zero spill" when every
# entry function is clean and all 16 belong to gesvdj_cta_impl<complex<double>>,
# a pre-existing 255-register kernel -- i.e. it reports a regression that is not
# there, on a library where nothing changed. The gate is the ENTRY-FUNCTION line.
echo "entry functions with non-zero spill (THIS IS THE GATE):"
awk '/Compiling entry function/ {e=1; next}
     /Function properties for/ {e=0}
     e && /spill stores/ && !/0 bytes spill stores, 0 bytes spill loads/ {n++}
     END {print n+0}' "$OUT"
echo "all functions (entry + non-inlined device) with non-zero spill:"
grep -E 'spill (stores|loads)' "$OUT" | grep -vE '0 bytes spill stores, 0 bytes spill loads' | wc -l
if [ -n "$PAT" ]; then
  echo "=== matching '$PAT' ==="
  grep -B1 -A2 "$PAT" "$OUT" | grep -E "Compiling entry|Used [0-9]+ registers|spill" | paste - - - 2>/dev/null | sed 's/ptxas info    : //g'
fi
grep 'LINK ' "$OUT" || true

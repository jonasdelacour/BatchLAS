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
# THE GATE, AND IT DEPENDS ON WHAT YOU ARE CLAIMING. Two different questions read two
# different columns, and the wrong pairing is silent in both directions.
#
# (1) A SPILL HUNT -- "is this kernel's working set too big for its registers?" Use:
#   * `0 bytes spill stores, 0 bytes spill loads` on the kernel's lines, and
#   * `Used N registers` against `resident::sm89_fits(regs, work_group)` in
#     src/util/resident_capacity.hh -- the one spelling of the ceiling -- and NOT against
#     the per-BLOCK form `regs x work_group <= 65536` this header used to give. That form
#     is OPTIMISTIC: the SM's 65,536 registers are FOUR SUB-PARTITIONS of 16,384, a
#     block's warps are dealt over them round-robin, and ptxas allocates in banks of
#     eight, so what decides the launch abort (an abort, not a slowdown) is
#         ceil(warps/4) * 32 * ((regs + 7) & ~7) <= 16384.
#     The two agree only when the warp count is a multiple of four; otherwise the
#     per-block form accepts launches the driver refuses.
#     evidence: docs/perf/lu.md#the-register-cap-that-binds-is-per-sub-partition
#     Symbols, not line numbers, because the line reference this header carried --
#     `src/sycl/gemm_kernels.cc:725-735` -- had DRIFTED: that range is a KernelVariant
#     dispatch switch today, so the reference no longer lands on the rule.
#     An earlier version of this comment went further and called the reference
#     fabricated -- "a pickaxe for 65536 over that file's whole history returns nothing,
#     so it has never held the rule in any revision". That is REFUTED, and refuted in an
#     instructive way: the file spells the constant WITH A THOUSANDS SEPARATOR, so the
#     pickaxe was run on a string the source has never contained.
#     `git log -S'65,536' -- src/sycl/gemm_kernels.cc` returns e56c6d8 ("WP2: GEMM --
#     close the vendor-free envelope, flip the default to Auto"), which introduced it,
#     and the rule is in that file RIGHT NOW: the Tiled128x128RegisterK8 case at lines
#     745-751, "Wider scalars overrun the hard 65,536 registers-per-block limit at this
#     64-accumulator tile (a launch failure, not spilling)". The line numbers moved; the
#     rule did not. Prefer symbols anyway -- but a search that finds nothing is evidence
#     about the search string, not about the tree.
# Here stack frame is the WRONG gate: a large fraction of perfectly healthy entry
# functions carry a non-zero stack frame with zero spills, so gating on it rejects them
# -- and a grep for "spill" that finds nothing reads as "no spill" whether or not the
# flag ever took effect.
# The figure this comment used to give -- "220 of 376 entry functions" -- was stale AND
# in the wrong unit. 376 is the raw `entry functions:` baseline recorded further down,
# i.e. an UNFOLDED ptxas line count on a pre-TRSM tree, whereas the stack-frame line this
# script prints counts DISTINCT kernels with the `_with_offset` twins folded. The two
# were never comparable; a reader diffing them would see a halving that is pure units.
# Re-measured on the probe logs this branch was developed against, both 2026-09-11:
#   batchlas_sycl            576 raw lines, 288 distinct kernels, 114 with a non-zero frame
#   batchlas_extensions_cta  974 raw lines, 487 distinct kernels,  96 with a non-zero frame
# Compare any new run against the DISTINCT column -- that is the one printed below.
#
# (2) A RESIDENCY CLAIM -- "is this array actually IN registers?", which is what the
# register-resident tiny tiers (src/extensions/*_tiny.cc) assert. Here the gate is
# STACK FRAME == 0, and the spill columns are the ones that mislead: an unroll the
# compiler declined makes the array dynamically indexed, ptxas relocates the WHOLE
# array to local memory, and because nothing was ever in a register to evict there is
# ZERO SPILL. Measured: `break` instead of `continue` inside the getrf tiny tier's
# unrolled j loop gave float N=32 a 128-byte frame -- exactly 32 floats -- at 40
# registers, 0 spill, and a green test suite.
# evidence: docs/perf/lu.md#the-register-probe-and-the-unroll-that-decides-it
#
# BOTH columns are now reported, on their own output lines, so choosing the wrong one is
# at least a visible choice. Neither is subtracted from the other: read the one your
# claim needs.
#
# Each kernel appears TWICE, as `<name>` and `<name>_with_offset`; they can differ
# by a couple of registers. Take the max. Grep by mangled name. The two counters below
# FOLD the twins onto the base name and count each kernel ONCE -- they are two codegen
# variants of one source kernel, a frame or a spill appears in both, and counting lines
# would double every number against a claim that is always made about one kernel. The
# raw `entry functions:` line above is left UN-folded, because the baselines recorded
# here are raw ptxas line counts.
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
#   BATCHLAS_BUILD_DIR overrides the build tree (default: <repo>/build).
# Paths are derived, not hardcoded: this script implements a correctness gate, and
# a gate nobody else can run is not a gate. `set -e` so a failed cd cannot silently
# replay link.txt from whatever directory happens to be current.
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
BUILD="${BATCHLAS_BUILD_DIR:-$REPO/build}"
cd "$BUILD/src"
OUT="${1:-$(mktemp -t regprobe.XXXXXX.log)}"
PAT="${2:-}"
ARTIFACT="$(mktemp -t regprobe.XXXXXX.so)"

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
LINE=${LINE/-o lib${TARGET}.so/-o $ARTIFACT}
LINE="$LINE -Xsycl-target-backend=nvptx64-nvidia-cuda -Xcuda-ptxas -v"

rc=0
/usr/bin/time -f 'LINK %e s real, %U s user' bash -c "$LINE" > "$OUT" 2>&1 || rc=$?
echo "exit=$rc  target=$TARGET  log=$OUT"
ENTRIES=$(grep -c 'Compiling entry function' "$OUT" || true)
echo "entry functions: $ENTRIES"
# A SILENT ZERO IS HOW THE BROKEN GATE SURVIVED A WHOLE CAMPAIGN. An empty log, a link
# that died before ptxas ran, and a -v that never reached the backend all render as
# "0 with spill", which reads exactly like a clean probe. Refuse to report instead.
if [ "${ENTRIES:-0}" -eq 0 ]; then
    echo "register_probe: no 'Compiling entry function' lines in $OUT (link exit=$rc)." >&2
    echo "  This is NOT a clean report -- ptxas produced no parseable output. Log tail:" >&2
    tail -n 20 "$OUT" >&2
    exit 3
fi
# ENTRY FUNCTIONS AND EVERYTHING ELSE, COUNTED SEPARATELY. ptxas emits
# "Function properties" for non-inlined DEVICE functions as well as for entry
# functions, and the two used to be summed into one number. On
# batchlas_extensions_cta that reads "16 kernels with non-zero spill" when every
# entry function is clean -- i.e. it reports a regression that is not there, on a
# library where nothing changed. The gate is the ENTRY-FUNCTION line.
# Those 16 are NOT one kernel. This comment used to attribute all of them to
# gesvdj_cta_impl<complex<double>>; attributing each spill line to its owning
# `Function properties for` name in /tmp/regprobe_cta.log (2026-09-11) gives a mix of
# eight distinct non-inlined device functions, each emitted twice because both the base
# entry and its `_with_offset` twin pull the same callee in:
#     6  gesvdj_cta_impl<complex<double>, 32, {64,false | 32,true | 32,false}>
#     8  ormqx_cta_impl<complex<double>, 32, ...>   (four instantiations)
#     2  syev_cta_fused_impl<complex<double>, 32, 1>
# All three are pre-existing complex<double> CTA kernels. The point stands -- none of
# them is an entry function -- but "all 16 are gesvdj" was simply not what the log says.
# AN ENTRY WHOSE NAME DOES NOT PARSE IS A HOLE IN THE GATE, NOT A PASS. Every awk below
# keys on the `_Z[A-Za-z0-9_]+` mangled name. If an entry line does not match it, `raw`
# comes out empty, `want` never becomes true, and that kernel is silently never tested
# for spill or frame -- while `seen[""]` still occupied a slot in the denominator, so the
# kernel COUNT included a kernel the gate had skipped. That is the same silent-zero shape
# this whole rewrite exists to kill, one level down. Count them separately, print them,
# and fail: a kernel that cannot be gated must be reported, never quietly passed.
UNPARSEABLE=$(awk '/Compiling entry function/ && !/_Z[A-Za-z0-9_]+/' "$OUT")
NUNPARSEABLE=$(printf %s "$UNPARSEABLE" | grep -c . || true)
GATE_RC=0
if [ "$NUNPARSEABLE" -gt 0 ]; then
    GATE_RC=4
    echo "register_probe: $NUNPARSEABLE entry function(s) with an unparseable mangled name." >&2
    echo "  These are NOT covered by any count below -- neither gate looked at them:" >&2
    printf '%s\n' "$UNPARSEABLE" | sed -n '1,20 s/^/    /p' >&2
    if [ "$NUNPARSEABLE" -gt 20 ]; then echo "    ... and $((NUNPARSEABLE - 20)) more" >&2; fi
fi
echo "entry functions with an unparseable name (NOT GATED): $NUNPARSEABLE"
echo "entry functions with non-zero spill (THIS IS THE GATE):"
# ptxas emits, in this order: `Compiling entry function`, `Function properties for <the
# same name>`, the stack-frame+spill data line, `Used N registers` -- and only THEN the
# properties blocks of any non-inlined device functions it called. The previous form
# cleared its state on `Function properties`, i.e. on the line that ALWAYS sits between
# the entry line and the data, so the spill test was never reached and this gate printed
# 0 for every input it was ever given. Match the properties block by NAME against the
# entry just seen, then consume it; that also keeps the trailing device-function blocks
# out of the entry-function count by construction rather than by luck.
awk '/Compiling entry function/ {
         match($0, /_Z[A-Za-z0-9_]+/); raw = substr($0, RSTART, RLENGTH);
         key = raw; sub(/_with_offset$/, "", key); next }
     /Function properties for/ {
         match($0, /_Z[A-Za-z0-9_]+/);
         want = (raw != "" && substr($0, RSTART, RLENGTH) == raw); next }
     want && /bytes stack frame/ {
         want = 0; raw = "";
         if ($0 !~ /0 bytes spill stores, 0 bytes spill loads/) bad[key] = 1 }
     END {for (x in bad) n++; print n+0}' "$OUT"
echo "all functions (entry + non-inlined device) with non-zero spill:"
# `|| true` on the pipeline, not on a stage: pipefail turns a CLEAN log -- where the
# second grep matches nothing -- into a set -e abort with no message.
grep -E 'spill (stores|loads)' "$OUT" | grep -vE '0 bytes spill stores, 0 bytes spill loads' | wc -l || true
# STACK FRAME, COUNTED SEPARATELY -- case (2) in the header. A kernel with zero spill and
# a non-zero frame is not register-resident, and nothing above this line would say so.
KERNELS=$(awk '/Compiling entry function/ {
         match($0, /_Z[A-Za-z0-9_]+/); k = substr($0, RSTART, RLENGTH);
         sub(/_with_offset$/, "", k); if (k != "") seen[k] = 1 }
     END {for (x in seen) n++; print n+0}' "$OUT")
FRAMES=$(awk '/Compiling entry function/ {
         match($0, /_Z[A-Za-z0-9_]+/); raw = substr($0, RSTART, RLENGTH);
         key = raw; sub(/_with_offset$/, "", key); next }
     /Function properties for/ {
         match($0, /_Z[A-Za-z0-9_]+/);
         want = (raw != "" && substr($0, RSTART, RLENGTH) == raw); next }
     want && /bytes stack frame/ {
         want = 0; raw = "";
         if ($1 + 0 > frame[key]) frame[key] = $1 + 0 }
     END {for (x in frame) if (frame[x] > 0) print frame[x], x}' "$OUT" | sort -k1,1nr)
NFRAMES=$(printf %s "$FRAMES" | grep -c . || true)
echo "entry functions with non-zero STACK FRAME (THE GATE FOR A RESIDENCY CLAIM): $NFRAMES of $KERNELS distinct kernels"
if [ "$NFRAMES" -gt 0 ]; then
    # TRUNCATE INSIDE sed, NOT WITH `head` ON A PIPE. `printf ... | head -n 20 | sed`
    # deadlocks its own error handling once $FRAMES exceeds the 64 KiB pipe buffer: head
    # exits after 20 lines, printf takes EPIPE/SIGPIPE on the rest, pipefail propagates
    # 141 and `set -e` kills the script HERE -- silently truncating the very report this
    # script exists to produce. Measured on a 500-line/209 KB fixture: the `head` form
    # exits 141 and never reaches the next statement; this form exits 0. sed without `q`
    # consumes its whole input, so the writer never sees a closed pipe.
    printf '%s\n' "$FRAMES" | sed -n '1,20 s/^/    /p'
    if [ "$NFRAMES" -gt 20 ]; then echo "    ... and $((NFRAMES - 20)) more"; fi
fi
if [ -n "$PAT" ]; then
  echo "=== matching '$PAT' ==="
  grep -B1 -A2 "$PAT" "$OUT" | grep -E "Compiling entry|Used [0-9]+ registers|spill" | paste - - - 2>/dev/null | sed 's/ptxas info    : //g'
fi
grep 'LINK ' "$OUT" || true
# Deferred, so the whole report still prints before the failure. An unparseable entry
# name means the numbers above are reported over a SUBSET of the kernels and the script
# cannot say which ones were missed -- that is a failed probe, not a clean one.
if [ "$GATE_RC" -ne 0 ]; then
    echo "register_probe: FAILED -- $NUNPARSEABLE entry function(s) could not be gated (see above)." >&2
    exit "$GATE_RC"
fi

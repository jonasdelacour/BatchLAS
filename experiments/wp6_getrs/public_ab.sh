#!/usr/bin/env bash
# THE PUBLIC-API A/B. This is the number that counts; the prototype's is reported
# separately because the two will differ.
#
# ONE BINARY PER BUILD, THREE PINNED ROUTES, INTERLEAVED CELL BY CELL. Arm-by-arm
# interleaving across processes within a cell is the shape run_ab.sh used for the
# pivsg probe -- separate processes, but the same session and the same clocks.
#
# EVERY PIN IS VERIFIED. lubench6 prints the RESOLVED route on every row
# (getrf_route|getrs_route, field 12); route_resolve.hh:165 falls through to
# automatic() at :175 when a forced route is unsupported, so an unverified pin can
# silently measure cuBLAS and pass green. analyse_public.py refuses any row whose
# printed route does not match the pin.
#
#   $1 = nv  : the VENDOR-FREE build. native:cta (fused) vs native:blocked
#              (composition over the NATIVE trsm). This is the build the campaign
#              exists for.
#   $1 = v   : the VENDOR-PRESENT build. native:cta vs native:blocked (over the
#              VENDOR trsm) vs vendor (cublas?getrsBatched).
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp6_lu/bench"
WHICH="${1:-nv}"
BIN="$D/lubench6_${WHICH}"
OUT="${OUT:-$W/experiments/wp6_getrs/public_${WHICH}.csv}"
: > "$OUT"
REPS="${REPS:-9}"
ARMS="${ARMS:-native:cta native:blocked vendor}"
if [ "$WHICH" = "nv" ]; then ARMS="${ARMS_NV:-native:cta native:blocked}"; fi

# NTRANS=1: the transposed modes are a CORRECTNESS question (correctness.sh and
# correctness_large.sh cover all three), and paying three warm-ups per cell in a
# timing run buys nothing.
for t in float double cfloat cdouble; do
  for n in 64 128 512 2048; do
    case $n in
      64) b=8192;; 128) b=4096;; 512) b=512;; 2048) b=32;; *) b=512;;
    esac
    for r in 1 2 4 8; do
      for a in $ARMS; do
        CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S="${WARM_S:-0.5}" NTRANS=1 NPROBE=1 \
          BATCHLAS_GETRS_ROUTE="$a" \
          "$BIN" getrs "$t" "$n" "$r" "$b" "$REPS" 2>/dev/null \
          | sed "s/^/${a},/" >> "$OUT"
      done
    done
  done
done
echo "rows: $(wc -l < "$OUT")"

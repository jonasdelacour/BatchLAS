#!/usr/bin/env bash
# Every break, in order, each one rebuilt and run and reverted. Prints one
# summary block per break: the per-type pass/fail counts, whether the process
# ABORTED (which is a red, not an absence), and the set of test NAMES that went
# red -- which is the part that matters. A break that turns the WRONG tests red,
# or none at all, is as informative as one that behaves.
set -u
cd "$(dirname "$0")/../../.." || exit 1
HERE=experiments/wp6_lu/tests
SUM="$HERE/breaks.txt"
: > "$SUM"
TYPES="4=float 5=double 6=cfloat 7=cdouble"
for nm in "$@"; do
  bash "$HERE/run_break.sh" "$nm" > /dev/null 2>&1
  f="$HERE/break_${nm}.txt"
  {
    echo "=== $nm"
    for t in 4 5 6 7; do
      lab=$(echo "$TYPES" | tr ' ' '\n' | grep "^$t=" | cut -d= -f2)
      blk=$(awk -v t="$t" '/^=== run type /{on=($4==t)} on' "$f")
      p=$(echo "$blk" | grep -oE '^\[  PASSED  \] [0-9]+' | tail -1 | grep -oE '[0-9]+')
      fl=$(echo "$blk" | grep -oE '^\[  FAILED  \] [0-9]+ tests' | head -1 | grep -oE '[0-9]+')
      rc=$(echo "$blk" | grep -oE '^--- exit [0-9]+' | grep -oE '[0-9]+')
      names=$(echo "$blk" | grep -oE 'FAILED  \] LuTest/[0-9]+\.[A-Za-z0-9]+' \
              | sed -E 's#.*\.##' | sort -u | tr '\n' ' ')
      crash=""
      [ -z "${p:-}" ] && crash=" ABORTED"
      printf "    %-8s pass=%-3s fail=%-3s exit=%-3s%s %s\n" \
             "$lab" "${p:-?}" "${fl:-0}" "${rc:-?}" "$crash" "$names"
    done
  } >> "$SUM"
done
cat "$SUM"

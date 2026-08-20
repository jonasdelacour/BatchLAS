#!/usr/bin/env bash
# Per-binary coverage attribution: run each non-slow test binary on its own,
# with its own BATCHLAS_COVERAGE_OUT, so a coverage row can be traced to the
# suite that issued it.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
S=/home/jonaslacour/.claude/jobs/20812aa0/tmp/attrib
mkdir -p "$S"
cd "$W"
mapfile -t TESTS < <(ctest --test-dir build -N -LE slow 2>/dev/null | sed -n 's/^  Test *#[0-9]*: //p')
for t in "${TESTS[@]}"; do
    bin="build/tests/$t"
    [[ -x "$bin" ]] || { echo "skip $t (no binary)"; continue; }
    rm -f "$S/$t.csv" "$S/$t.csv".[0-9]*
    BATCHLAS_COVERAGE_OUT="$S/$t.csv" timeout 900 "./$bin" > "$S/$t.log" 2>&1
    rc=$?
    "$W/scripts/coverage_merge.sh" "$S/$t.csv" >> "$S/$t.log" 2>&1
    n=$(grep -c '^reached,gemm,complex' "$S/$t.csv" 2>/dev/null || echo 0)
    echo "$t rc=$rc complex_gemm_rows=$n"
done
echo ATTRIB_DONE

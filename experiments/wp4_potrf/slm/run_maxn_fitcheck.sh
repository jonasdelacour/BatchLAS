#!/usr/bin/env bash
# Every per-type max_n candidate, launched COLD in its own process at the exact
# byte count section 4.1 (+ the W9 off[] term) asks for. A fresh process per size
# matters: the CUDA opt-in attribute is sticky per kernel function, so probing them
# in one process would let an earlier large launch mask a hole.
#
# Columns: budget, type, n, bytes, launched?
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp4_potrf/slm"
out="$D/maxn_fitcheck.csv"
echo "budget,type,n,bytes,wg,launched" > "$out"
# budget|type|n|bytes  -- bytes from derive_max_n.py
rows="
45056|float|105|44320
45056|double|74|44652
45056|complex<float>|74|44588
45056|complex<double>|52|44312
97280|float|155|96368
97280|double|109|95336
97280|complex<float>|109|95272
97280|complex<double>|77|95132
101120|float|158|100760
101120|double|111|98856
101120|complex<float>|111|98792
101120|complex<double>|79|100128
hole|float|110|49064
hole-padded|float|110|49408
"
for r in $rows; do
    IFS='|' read -r b t n bytes <<< "$r"
    if "$W/experiments/gpu_guard.sh" 0 "$D/slm_occ" "$bytes" 128 512 >/dev/null 2>&1; then ok=1; else ok=0; fi
    echo "$b,$t,$n,$bytes,128,$ok" >> "$out"
done
column -s, -t "$out"

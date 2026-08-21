#!/usr/bin/env bash
# Host-only, no SYCL and no library link: route_gemm.hh / route_trsm.hh are pure
# headers. Their one external reference is the coverage recorder, stubbed in
# routeq.cpp so the answer cannot depend on a coverage build.
set -euo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp4_potrf/phase2_ab"
g++ -O1 -std=c++20 -I "$W/include" -I "$W/build/include" \
    "$D/routeq.cpp" -o "$D/routeq"
"$D/routeq" > "$D/routeq.txt" 2>&1
cat "$D/routeq.txt"

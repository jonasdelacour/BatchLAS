#!/usr/bin/env bash
# The route_diff health checks the script itself does NOT do. The script only
# dies on ZERO reached rows; these catch the rest, including defect 4's
# signature (a vendor-free capture with ZERO miss rows means the gate-declined
# half went unrecorded, which hides exactly the transition a route diff is for).
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
S="$W/.route-diff"
echo "header: $(head -1 "$S/wp8i3-after-v.csv")"
for l in "$@"; do
  printf '%-18s reached=%-5s linked=%-3s miss=%-3s decisions=%-5s gemv=%-4s getrf=%-3s getrs=%-3s getri=%-3s shards=%s\n' \
    "$l" \
    "$(grep -c '^reached,' "$S/$l.csv")" \
    "$(grep -c '^linked,' "$S/$l.csv")" \
    "$(grep -c '^miss,' "$S/$l.csv")" \
    "$(wc -l < "$S/$l.routes")" \
    "$(grep -c ',gemv,' "$S/$l.routes")" \
    "$(grep -c ',getrf,' "$S/$l.routes")" \
    "$(grep -c ',getrs,' "$S/$l.routes")" \
    "$(grep -c ',getri,' "$S/$l.routes")" \
    "$(grep -o 'merged [0-9]* shards' "$S/$l.ctest.log" | tail -1)"
done

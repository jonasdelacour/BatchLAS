#!/usr/bin/env bash
# route_diff HEALTH CHECKS. The script itself only dies on ZERO 'reached' rows;
# D4 part C4 lists the six other ways a capture can look clean and be broken.
# Run this on EVERY capture.
set -u
S=.route-diff
for l in "$@"; do
  echo "=== $l ==="
  printf '  reached rows   : %s\n'   "$(grep -c '^reached,' $S/$l.csv)"
  printf '  distinct dec.  : %s\n'   "$(wc -l < $S/$l.routes)"
  printf '  miss rows      : %s   (0 in build/ is normal; a 0 in build-novendor is defect 4)\n' \
                                      "$(grep -c '^miss,' $S/$l.csv || true)"
  printf '  linked rows    : %s   (expect 40)\n' "$(grep -c '^linked,' $S/$l.csv || true)"
  printf '  ctest selected : %s   (expect 56; trap 1)\n' \
                                      "$(grep -cE 'Test +#[0-9]+:' $S/$l.ctest.log)"
  printf '  merged shards  : %s\n'   "$(grep -o 'merged [0-9]* shards' $S/$l.ctest.log | tail -1)"
  for op in getri getrf getrs gemv; do
    printf '  rows for %-6s: %s\n' "$op" "$(grep -c ",$op," $S/$l.routes || true)"
  done
  printf '  KNOWN GAP getri/double/NETLIB declined : %s\n' \
      "$(grep -c '^reached,getri,double,NETLIB' $S/$l.csv || true)"
  printf '  ctest verdict  : %s\n' "$(grep -E 'tests passed|tests failed' $S/$l.ctest.log | tail -1)"
done

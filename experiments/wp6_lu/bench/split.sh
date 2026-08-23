#!/usr/bin/env bash
# Reduce nsys's cuda_gpu_kern_sum tables to the split this write-up quotes:
# percent of GPU kernel time, instance count, and the demangled kernel name.
# The captures themselves are not committed; this is what turns them into a
# table that is.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
for f in "$@"; do
  echo "##### $f"
  awk -F'|' 'NF>3 && $2+0>0 {
      name=$10; gsub(/^ +| +$/,"",name); sub(/^Typeinfo name for /,"",name);
      inst=$4; gsub(/ /,"",inst);
      printf "  %6s%%  %6s launches  %s\n", $2, inst, substr(name,1,100)
  }' "$D/kernsum/${f}_kern.txt"
  echo
done

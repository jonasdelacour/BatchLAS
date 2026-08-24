#!/usr/bin/env bash
# A STRUCTURAL profile, never a timing one -- this campaign has fabricated results
# by timing under a trace. The only question asked here is HOW MANY KERNELS one
# public getrs call launches, which is the mechanism the fused tier exists to
# change: the composition ran 4 kernel families and ~75,000 launches at
# float n=512 nrhs=1 batch=512.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
D="$W/experiments/wp6_getrs/nsys"
mkdir -p "$D"
BIN="$W/experiments/wp6_lu/bench/lubench6_nv"
run() {
  local arm="$1"; local route="$2"
  rm -f "$D/${arm}.nsys-rep" "$D/${arm}.sqlite"
  CUDA_VISIBLE_DEVICES="${GPU:-1}" WARM_S=0 NTRANS=1 NPROBE=1 BATCHLAS_GETRS_ROUTE="$route" \
    nsys profile -t cuda -o "$D/$arm" --force-overwrite true \
      "$BIN" getrs float 512 1 512 1 > /dev/null 2>&1
  echo "=== $arm ($route) ==="
  nsys stats --report cuda_gpu_kern_sum --format csv "$D/${arm}.nsys-rep" 2>/dev/null \
    | awk -F, 'NR<=8 {print}'
}
run fused native:cta
run composed native:blocked
# The .nsys-rep / .sqlite files are DELETED: this branch was caught twice
# committing traces, once with 13 files making up 83% of the PR diff.
rm -f "$D"/*.nsys-rep "$D"/*.sqlite
rmdir "$D" 2>/dev/null || true

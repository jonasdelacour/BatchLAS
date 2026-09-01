#!/usr/bin/env bash
# Does the reported spread shrink with warm-up? Cell L, vendor route, one process
# per setting, on device 1. Parsed with csv (the name column contains a comma).
set -eu
ROOT=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=$ROOT/build/benchmarks/spmm_benchmark
export CUDA_VISIBLE_DEVICES=1
export BATCHLAS_SPMM_ROUTE=vendor
for w in 2 20 60 200; do
  for wi in 1 10; do
    out=/tmp/wp_${w}_${wi}.csv
    $BIN --name=BM_SPMM_Cells --type=float --warmup=$w --warmup_internal=$wi \
         --min_time=400 --csv=$out 1024 3 2 512 0 0 1 0 >/dev/null 2>&1
    python3 - "$out" "$w" "$wi" <<'EOF'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[-1]
a=float(r['avg_ms']); s=float(r['stddev_ms'])
print(f"warmup={sys.argv[2]} internal={sys.argv[3]} it={r['iterations']} avg_ms={a:.6f} sd={s:.6f} relsd={s/a:.4f} GBs={float(r['GB/s']):.1f}")
EOF
  done
done

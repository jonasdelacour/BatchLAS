#!/usr/bin/env bash
# Is a LOW warm-up enough once the process has already run one row -- i.e. is the
# clock ramp a per-PROCESS cost or a per-ROW cost? Two rows (pattern 0 then 1);
# compare the SECOND row against the converged 250-warm-call value 0.161747 ms.
set -eu
ROOT=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
BIN=$ROOT/build/benchmarks/spmm_benchmark
export CUDA_VISIBLE_DEVICES=1 BATCHLAS_SPMM_ROUTE=vendor
for w in 2 5 25; do
  for wi in 1 2 10; do
    out=/tmp/op_${w}_${wi}.csv
    $BIN --name=BM_SPMM_Grid --type=float --warmup=$w --warmup_internal=$wi \
         --min_time=400 --csv=$out 1024 3 2 512 0 0 0,1 0 >/dev/null 2>&1
    python3 - "$out" "$w" "$wi" <<'EOF'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
out=[]
for i,r in enumerate(rows):
    a=float(r['avg_ms']); s=float(r['stddev_ms'])
    out.append(f"row{i}(pat={r['arg6']}) avg={a:.6f} relsd={s/a:.4f}")
print(f"warmup={sys.argv[2]}x{sys.argv[3]} ({int(sys.argv[2])*int(sys.argv[3])} calls): "+" | ".join(out))
EOF
  done
done

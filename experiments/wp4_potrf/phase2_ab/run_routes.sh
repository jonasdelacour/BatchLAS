#!/usr/bin/env bash
# WHICH KERNEL ACTUALLY RUNS. Never run while timing -- BATCHLAS_KERNEL_TRACE
# inflates ~60%. One short blocked run per (type, pin), kernel names counted.
#
# THE TRACE CANNOT SEE cuBLAS. It is a SYCL-side scope, so a vendor call shows
# up only as an ABSENCE. Read "no gemm_* row" as "the gemm went to cuBLAS".
set -uo pipefail
D=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_potrf/phase2_ab
cd "$D"
OUT="$D/routes.txt"
: > "$OUT"
for t in float double cfloat cdouble; do
  for pin in default vendorgemm nativegemm vendortrsm nativetrsm; do
    unset BATCHLAS_TRSM_ROUTE BATCHLAS_GEMM_ROUTE
    case "$pin" in
      vendorgemm) export BATCHLAS_GEMM_ROUTE=vendor ;;
      nativegemm) export BATCHLAS_GEMM_ROUTE=native ;;
      vendortrsm) export BATCHLAS_TRSM_ROUTE=vendor ;;
      nativetrsm) export BATCHLAS_TRSM_ROUTE=native ;;
    esac
    echo "===== type=$t n=256 nb=64 W=128 batch=32 pin=$pin =====" >> "$OUT"
    rm -f "$D/trace.json"
    BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$D/trace.json" BENCH_WARM_S=0.001 \
      ./phase2 blocked "$t" 256 64 128 32 1 >> "$OUT" 2>&1
    python3 - >> "$OUT" <<'PY'
import json, collections, os
p = os.path.join(os.path.dirname(os.path.abspath("trace.json")), "trace.json")
try:
    d = json.load(open("trace.json"))
except Exception as e:
    print("  (no trace: %s)" % e); raise SystemExit
c = collections.Counter()
dur = collections.Counter()
for e in d.get("traceEvents", []):
    c[e["name"]] += 1
    dur[e["name"]] += e.get("dur", 0.0)
for k, v in sorted(c.items(), key=lambda x: -dur[x[0]]):
    print("  n=%-6d us_total=%-12.1f %s" % (v, dur[k], k))
PY
  done
done
unset BATCHLAS_TRSM_ROUTE BATCHLAS_GEMM_ROUTE
cat "$OUT"

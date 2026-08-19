#!/usr/bin/env bash
# WP3 step 12 -- re-measure Side::Left with the staging tile in place.
#
# Both legs on ONE card so the ratio is internally consistent.
#
# TWO GUARDS BEYOND gpu_guard.sh, both added because the first attempt produced
# 22 of 180 native cells with 10-103% relative standard deviation while
# gpu_guard reported the run clean.
#
#   1. A LOCK. gpu_guard checks the card, not the operator. Two copies of this
#      script were queued behind the same busy GPU and both started when it
#      freed, so the run contended with ITSELF -- and since both were mine and
#      both were waiting politely, nothing in the guard could notice.
#
#   2. A NOISE GATE. gpu_guard samples foreign processes before and after the
#      run, so a disturbance that starts and ends inside the run is invisible to
#      it. Relative standard deviation per cell is not: a clean leg here has 0
#      cells above 10%, and the contended one had 22. A leg that fails this
#      check is DELETED rather than reported, because a cell at 98% sd is not a
#      slow measurement, it is not a measurement.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="experiments/wp3_s12"
GPU="${GPU:-1}"

exec 9>"$OUT/.sweep.lock"
flock -n 9 || { echo "another sweep holds the lock; refusing to run a second"; exit 3; }

for route in native vendor; do
    echo "=== Side::Left, route=$route, GPU $GPU ==="
    BATCHLAS_TRSM_ROUTE="$route" GPU_GUARD_MAX_WAIT="${GPU_GUARD_MAX_WAIT:-5400}" \
        ./experiments/gpu_guard.sh "$GPU" ./build/benchmarks/trsm_benchmark \
            --backend=CUDA --min_time=200 --min_iters=10 --max_iters=200 \
            --name=BM_TRSM_OrthoLeft --csv="$OUT/left-$route.csv" \
        > "$OUT/left-$route.log" 2>&1
    echo "  $route exit=$?"
    if grep -q "WARNING -- foreign process" "$OUT/left-$route.log"; then
        echo "  *** FOREIGN PROCESS -- discarding $route"; rm -f "$OUT/left-$route.csv"; continue
    fi
    python3 - "$OUT/left-$route.csv" "$route" <<'PY'
import csv, sys, os
path, route = sys.argv[1], sys.argv[2]
bad = tot = 0
for r in csv.DictReader(open(path)):
    ms, sd = float(r['avg_ms']), float(r['stddev_ms'])
    tot += 1
    if ms > 0 and sd / ms > 0.10:
        bad += 1
print(f'  {route}: {bad}/{tot} cells above 10% relative sd')
if bad:
    print(f'  *** NOISY -- discarding {route}; re-run when the box is quiet')
    os.remove(path)
PY
done
echo "left sweep finished"

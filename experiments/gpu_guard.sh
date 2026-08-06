#!/usr/bin/env bash
# Run a benchmark on one dedicated GPU, refusing to start if anything else is
# using it.
#
#     experiments/gpu_guard.sh 0 ./build/benchmarks/syrk_benchmark --backend=CUDA ...
#
# Two agents benchmarking at once is only safe because each owns a different
# physical GPU -- that is what makes it race-free, not the idle check. The idle
# check exists for the other hazard: something OUTSIDE this workflow (another
# session, a stray process) touching the card mid-measurement, which would
# silently inflate the numbers rather than fail loudly.
#
# Exits non-zero rather than measuring anything it cannot trust.
set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <gpu-index> <command> [args...]" >&2
    exit 2
fi

GPU="$1"; shift
SELF_PID=$$
MAX_WAIT=${GPU_GUARD_MAX_WAIT:-300}
UTIL_CEILING=${GPU_GUARD_UTIL_CEILING:-5}

query() { nvidia-smi --id="$GPU" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null; }

# Compute processes on this GPU that are not us. A GPU with someone else's
# kernels on it is not idle no matter what the utilisation counter says, since
# utilisation is sampled and can read 0 between another process's launches.
foreign_procs() {
    nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -vE "^\s*$" | grep -vx "$SELF_PID" || true
}

waited=0
while :; do
    util="$(query utilization.gpu)"
    procs="$(foreign_procs)"
    [ -z "${util:-}" ] && { echo "gpu_guard: cannot query GPU $GPU" >&2; exit 3; }
    if [ "$util" -le "$UTIL_CEILING" ] && [ -z "$procs" ]; then
        break
    fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        echo "gpu_guard: GPU $GPU still busy after ${MAX_WAIT}s (util=${util}%, pids=[${procs//$'\n'/,}]); refusing to benchmark" >&2
        exit 4
    fi
    sleep 5
    waited=$((waited + 5))
done

before_clock="$(query clocks.sm)"
CUDA_VISIBLE_DEVICES="$GPU" "$@"
rc=$?

# If someone else landed on the card while we were measuring, the numbers we
# just printed are not trustworthy -- say so loudly rather than let them be
# quoted as a result.
after_procs="$(foreign_procs)"
if [ -n "$after_procs" ]; then
    echo "gpu_guard: WARNING -- foreign process(es) [${after_procs//$'\n'/,}] appeared on GPU $GPU during the run; DISCARD these numbers and re-run" >&2
    exit 5
fi
echo "gpu_guard: GPU $GPU exclusive for the whole run (SM clock was ${before_clock} MHz at start)" >&2
exit $rc

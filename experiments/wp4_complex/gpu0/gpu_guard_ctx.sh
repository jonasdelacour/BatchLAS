#!/usr/bin/env bash
# A narrower gpu_guard for one specific, verifiable situation.
#
# experiments/gpu_guard.sh refuses to start while ANY foreign process appears in
# nvidia-smi's compute-apps list for the GPU. That is the right default. It
# blocks indefinitely, though, on a process whose real work is on the OTHER
# card but which has initialised a CONTEXT on this one -- it shows up in the
# list, holds a few hundred MiB, and never runs a kernel here. That is what is
# on GPU 0 right now:
#
#   1697188  386 MiB  .../bench_sycl ... --gpu 1     (29+ minutes, util 4%)
#
# This variant accepts exactly that case and nothing else:
#   * utilisation must be <= UTIL_CEILING before AND after the run;
#   * every foreign process must hold LESS than CTX_MIB (default 512 MiB) --
#     a bare context, not a working set -- before AND after;
#   * the set of foreign pids must be UNCHANGED across the run. A process that
#     arrived or left mid-run means something happened that this check cannot
#     see, and the numbers are discarded.
#
# It is strictly weaker than gpu_guard.sh, so cells measured with it are
# cross-checked: control_ctx.sh re-measures cells that sweep.sh already has
# under the strict guard and the two must agree.
set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <gpu-index> <command> [args...]" >&2
    exit 2
fi

GPU="$1"; shift
SELF_PID=$$
MAX_WAIT=${GPU_GUARD_MAX_WAIT:-300}
UTIL_CEILING=${GPU_GUARD_UTIL_CEILING:-8}
CTX_MIB=${GPU_GUARD_CTX_MIB:-512}

query() { nvidia-smi --id="$GPU" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null; }

# pid:mib for every compute app on this GPU that is not us.
foreign_list() {
    nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -vE '^$' | grep -v "^${SELF_PID}," || true
}

# Any foreign process holding a real working set?
heavy_foreign() {
    foreign_list | awk -F, -v lim="$CTX_MIB" '$2 >= lim {print $1}'
}

foreign_pids() { foreign_list | cut -d, -f1 | sort | tr '\n' ' '; }

waited=0
while :; do
    util="$(query utilization.gpu)"
    [ -z "${util:-}" ] && { echo "gpu_guard_ctx: cannot query GPU $GPU" >&2; exit 3; }
    heavy="$(heavy_foreign)"
    if [ "$util" -le "$UTIL_CEILING" ] && [ -z "$heavy" ]; then
        break
    fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        echo "gpu_guard_ctx: GPU $GPU still busy after ${MAX_WAIT}s (util=${util}%, heavy=[${heavy//$'\n'/,}]); refusing" >&2
        exit 4
    fi
    sleep 5
    waited=$((waited + 5))
done

before_pids="$(foreign_pids)"
before_clock="$(query clocks.sm)"
CUDA_VISIBLE_DEVICES="$GPU" "$@"
rc=$?

after_pids="$(foreign_pids)"
after_heavy="$(heavy_foreign)"
after_util="$(query utilization.gpu)"
if [ "$before_pids" != "$after_pids" ]; then
    echo "gpu_guard_ctx: WARNING -- foreign pid set changed during the run ([$before_pids] -> [$after_pids]); DISCARD" >&2
    exit 5
fi
if [ -n "$after_heavy" ]; then
    echo "gpu_guard_ctx: WARNING -- foreign process grew past ${CTX_MIB}MiB during the run; DISCARD" >&2
    exit 5
fi
echo "gpu_guard_ctx: GPU $GPU had only idle foreign contexts [$before_pids] for the whole run (SM clock ${before_clock} MHz at start, util now ${after_util}%)" >&2
exit $rc

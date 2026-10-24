#!/bin/sh
# ctest, one test per GPU slot of <build>/ctest_resources.json (tests/README.md):
#   [CUDA_VISIBLE_DEVICES=1] scripts/ctest_gpus.sh [--test-dir build] -LE slow
# A pre-set CUDA_VISIBLE_DEVICES trims the spec to its length. A later -j wins.
#
# Guards (the 2x4090 box has no swap and is shared; docs/developer/agent-guide.md section 8):
#   - refuses while MemAvailable < BATCHLAS_CTEST_MIN_MEM_GB (default 20) or while a build
#     (make, ninja, cmake, clang, icpx, acpp, ptxas, sycl-post-link) is running;
#   - without a pre-set CUDA_VISIBLE_DEVICES, drops GPUs whose utilization is above
#     BATCHLAS_GPU_BUSY_UTIL % (default 30) or whose used memory is above
#     BATCHLAS_GPU_BUSY_MEM_MB (default 4096); BATCHLAS_GPU_BUSY_SKIP=0 keeps them.
# BATCHLAS_CTEST_FORCE=1 runs anyway (and on every GPU if all of them look busy).
set -eu

build=${BATCHLAS_BUILD_DIR:-build}
if [ "${1:-}" = "--test-dir" ] && [ "$#" -ge 2 ]; then
    build=$2
    shift 2
fi
# Absolute: ctest resolves a relative spec path after changing into --test-dir.
dir=$(cd "$build" 2>/dev/null && pwd) || dir=$build
spec="$dir/ctest_resources.json"
if [ ! -f "$spec" ]; then
    echo "ctest_gpus.sh: $spec not found (no CUDA build, no nvidia-smi GPUs, or" >&2
    echo "BATCHLAS_TEST_GPUS=0). Run plain serial ctest instead: ctest --test-dir $build $*" >&2
    exit 2
fi

force=${BATCHLAS_CTEST_FORCE:-0}
refuse() {
    echo "ctest_gpus.sh: refusing to start: $1" >&2
    echo "  (override with BATCHLAS_CTEST_FORCE=1 if you know the box can take it)" >&2
    exit 3
}

min_gb=${BATCHLAS_CTEST_MIN_MEM_GB:-20}
if [ "$force" != 1 ] && [ -r /proc/meminfo ]; then
    avail_kb=$(grep '^MemAvailable:' /proc/meminfo | tr -s ' ' | cut -d' ' -f2)
    if [ -n "$avail_kb" ] && [ "$avail_kb" -lt $((min_gb * 1024 * 1024)) ]; then
        refuse "MemAvailable is $((avail_kb / 1024 / 1024)) GB, under ${min_gb} GB (no swap: an OOM kill takes the run and its caller with it)"
    fi
fi

if [ "$force" != 1 ]; then
    builds=$(ps -A -o comm= 2>/dev/null | sed 's|.*/||' |
        grep -E '^(make|gmake|ninja|cmake|clang|clang\+\+|clang-[0-9a-z-]*|icpx|acpp|ptxas|sycl-post-link)$' |
        sort | uniq -c | tr -s ' ' | tr '\n' ',' | sed 's/,$//') || true
    if [ -n "$builds" ]; then
        refuse "a build is running (process counts:$builds); one build or test run at a time on this box"
    fi
fi

slots=$(grep -o '"slots": *[0-9]*' "$spec" | head -n 1 | grep -o '[0-9]*$')
gpus=$(grep -c '"id":' "$spec")

# Busy GPUs: nvidia-smi index is PCI order, which tests/ctest_gpu_env.sh also uses for spec ids.
busy_util=${BATCHLAS_GPU_BUSY_UTIL:-30}
busy_mem=${BATCHLAS_GPU_BUSY_MEM_MB:-4096}
if command -v nvidia-smi >/dev/null 2>&1; then
    query=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || query=""
else
    query=""
fi
is_busy() {
    printf '%s\n' "$query" | while IFS=, read -r idx util mem; do
        [ "$idx" = "$1" ] || continue
        case "$util$mem" in *[!0-9]*|"") continue ;; esac
        if [ "$util" -gt "$busy_util" ] || [ "$mem" -gt "$busy_mem" ]; then
            echo "GPU $idx: ${util}% busy, ${mem} MiB used"
        fi
    done
}

if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then
    for dev in $(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
        why=$(is_busy "$dev")
        [ -z "$why" ] || echo "ctest_gpus.sh: warning: CUDA_VISIBLE_DEVICES names a busy GPU ($why)" >&2
    done
elif [ -n "$query" ] && [ "${BATCHLAS_GPU_BUSY_SKIP:-1}" != 0 ]; then
    free=""
    i=0
    while [ "$i" -lt "$gpus" ]; do
        why=$(is_busy "$i")
        if [ -n "$why" ]; then
            echo "ctest_gpus.sh: skipping $why (BATCHLAS_GPU_BUSY_SKIP=0 keeps it)" >&2
        else
            free="$free${free:+,}$i"
        fi
        i=$((i + 1))
    done
    if [ -z "$free" ]; then
        [ "$force" = 1 ] || refuse "every GPU in $spec is busy"
    elif [ "$free" != "$(seq -s, 0 $((gpus - 1)))" ]; then
        export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
        export CUDA_VISIBLE_DEVICES="$free"
        echo "ctest_gpus.sh: running on CUDA_VISIBLE_DEVICES=$free" >&2
    fi
fi

if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then
    visible=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . || true)
    if [ "$visible" -eq 0 ]; then
        echo "ctest_gpus.sh: CUDA_VISIBLE_DEVICES is empty, so no GPU is visible." >&2
        exit 2
    fi
    if [ "$visible" -lt "$gpus" ]; then
        gpus=$visible
        tmp=$(mktemp "${TMPDIR:-/tmp}/batchlas_ctest_resources.XXXXXX")
        trap 'rm -f "$tmp"' EXIT INT TERM
        {
            printf '{\n  "version": { "major": 1, "minor": 0 },\n  "local": [\n    {\n      "gpus": [\n'
            i=0
            while [ "$i" -lt "$gpus" ]; do
                sep=","
                [ "$i" -eq $((gpus - 1)) ] && sep=""
                printf '        { "id": "%s", "slots": %s }%s\n' "$i" "$slots" "$sep"
                i=$((i + 1))
            done
            printf '      ]\n    }\n  ]\n}\n'
        } > "$tmp"
        spec=$tmp
    fi
fi

jobs=$((gpus * slots))
if [ "$jobs" -le 0 ]; then
    echo "ctest_gpus.sh: no GPU slots in $spec" >&2
    exit 2
fi

# Not exec: the trap must remove a trimmed spec after ctest has read it.
ctest --test-dir "$build" -j"$jobs" --resource-spec-file "$spec" "$@"

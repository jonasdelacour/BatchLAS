#!/bin/sh
# ctest, one test per GPU slot of <build>/ctest_resources.json (tests/README.md):
#   [CUDA_VISIBLE_DEVICES=1] scripts/ctest_gpus.sh [--test-dir build] -LE slow
# A pre-set CUDA_VISIBLE_DEVICES trims the spec to its length. A later -j wins.
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

slots=$(grep -o '"slots": *[0-9]*' "$spec" | head -n 1 | grep -o '[0-9]*$')
gpus=$(grep -c '"id":' "$spec")

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

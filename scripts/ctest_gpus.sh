#!/bin/sh
# Run ctest with one test per GPU slot, using the build's resource spec.
#
#   scripts/ctest_gpus.sh -LE slow                     # build/ (or $BATCHLAS_BUILD_DIR)
#   scripts/ctest_gpus.sh --test-dir build-vf -L eig
#
# <build>/ctest_resources.json is written at configure from nvidia-smi
# (BATCHLAS_TEST_GPUS / BATCHLAS_TEST_GPU_SLOTS). Each GPU test gets one slot and
# sees only its GPU (tests/ctest_gpu_env.sh). A later -j in the arguments wins.
set -eu

build=${BATCHLAS_BUILD_DIR:-build}
if [ "${1:-}" = "--test-dir" ] && [ "$#" -ge 2 ]; then
    build=$2
    shift 2
fi
spec="$build/ctest_resources.json"
if [ ! -f "$spec" ]; then
    echo "ctest_gpus.sh: $spec not found. Configure with tests enabled on a box where" >&2
    echo "nvidia-smi lists GPUs, or set -DBATCHLAS_TEST_GPUS=<n>." >&2
    exit 2
fi

jobs=0
for n in $(grep -o '"slots": *[0-9]*' "$spec" | grep -o '[0-9]*$'); do
    jobs=$((jobs + n))
done
if [ "$jobs" -le 0 ]; then
    echo "ctest_gpus.sh: no GPU slots in $spec" >&2
    exit 2
fi

exec ctest --test-dir "$build" -j"$jobs" --resource-spec-file "$spec" "$@"

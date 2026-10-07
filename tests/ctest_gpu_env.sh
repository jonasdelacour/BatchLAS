#!/bin/sh
# CTest launcher for GPU tests: maps the resource allocation to CUDA_VISIBLE_DEVICES.
#
# Under `ctest --resource-spec-file` (scripts/ctest_gpus.sh) CTest exports
# CTEST_RESOURCE_GROUP_0_GPUS=id:<N>,slots:1. Without an allocation this is a
# pass-through. A caller's own CUDA_VISIBLE_DEVICES list is honoured: id N picks
# its N-th entry. HIP/ONEAPI selectors are left alone.
set -eu

if [ -n "${CTEST_RESOURCE_GROUP_COUNT:-}" ] && [ -n "${CTEST_RESOURCE_GROUP_0_GPUS:-}" ]; then
    id=${CTEST_RESOURCE_GROUP_0_GPUS#id:}
    id=${id%%,*}
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        dev=$(printf '%s,\n' "$CUDA_VISIBLE_DEVICES" | cut -d, -f"$((id + 1))")
        if [ -z "$dev" ]; then
            echo "ctest_gpu_env.sh: resource id $id has no entry in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
            exit 1
        fi
    else
        dev=$id
        # nvidia-smi --list-gpus numbers devices in PCI order; CUDA's default does not.
        export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
    fi
    export CUDA_VISIBLE_DEVICES="$dev"
fi

exec "$@"

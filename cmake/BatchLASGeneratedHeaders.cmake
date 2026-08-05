file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/batchlas")

set(BATCHLAS_DEVICE_LIMIT_ENTRY_LINES "")
foreach(_device_limit_entry IN LISTS BATCHLAS_DETECTED_DEVICE_LIMIT_ENTRIES)
    string(REPLACE "|" ";" _device_limit_fields "${_device_limit_entry}")
    list(LENGTH _device_limit_fields _device_limit_field_count)
    if(_device_limit_field_count EQUAL 7)
        list(GET _device_limit_fields 0 _device_limit_platform_name)
        list(GET _device_limit_fields 1 _device_limit_type)
        list(GET _device_limit_fields 2 _device_limit_name)
        list(GET _device_limit_fields 3 _device_limit_architecture)
        list(GET _device_limit_fields 4 _device_limit_subgroup_size)
        list(GET _device_limit_fields 5 _device_limit_local_mem_bytes)
        list(GET _device_limit_fields 6 _device_limit_workspace_budget_bytes)
        string(APPEND BATCHLAS_DEVICE_LIMIT_ENTRY_LINES
            "    DetectedDeviceLimit{\"${_device_limit_platform_name}\", \"${_device_limit_type}\", \"${_device_limit_name}\", \"${_device_limit_architecture}\", ${_device_limit_subgroup_size}u, ${_device_limit_local_mem_bytes}ull, ${_device_limit_workspace_budget_bytes}ull},\n")
    endif()
endforeach()

list(LENGTH BATCHLAS_DETECTED_DEVICE_LIMIT_ENTRIES BATCHLAS_DEVICE_LIMIT_ENTRY_COUNT)
if(NOT DEFINED BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_LOCAL_MEM_BYTES)
    set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_LOCAL_MEM_BYTES 32768)
endif()
if(NOT DEFINED BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES)
    set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES 28672)
endif()
if(NOT DEFINED BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SAFE_SUBGROUPS_PER_WORKGROUP)
    set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SAFE_SUBGROUPS_PER_WORKGROUP 2)
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/backend_config.h.in"
    "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
)

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/device_limits.h.in"
    "${PROJECT_BINARY_DIR}/include/batchlas/device_limits.hh"
)

set(BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_TINY 16)
set(BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_SMALL 32)
set(BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_MEDIUM 64)
set(BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_LARGE 128)
set(BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_XLARGE 128)
set(BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_TINY 8)
set(BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_SMALL 16)
set(BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_MEDIUM 32)
set(BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_LARGE 64)
set(BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_XLARGE 64)
set(BATCHLAS_TUNE_DEFAULT_LATRD_LOWER_PANEL_WG_HINT 0)
set(BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_TINY 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_SMALL 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_MEDIUM 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_LARGE 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_XLARGE 32)
# StedcMergeVariant: 1 = Fused, 2 = FusedCta. These were 2; FusedCta currently
# fails stedc_tests (NaN under heavy deflation) and is 2-12% slower than Fused
# on an RTX 4090. See include/batchlas/tuning_params.hh for the full note.
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_TINY 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_SMALL 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_MEDIUM 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_LARGE 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_XLARGE 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_TINY 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_SMALL 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_MEDIUM 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_LARGE 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_XLARGE 32)
set(BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_TINY 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_SMALL 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_MEDIUM 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_LARGE 1)
set(BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_XLARGE 1)

set(BATCHLAS_TUNING_HEADER_FALLBACK_ARGS
    --fallback-ormqr-block-size-tiny "${BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_TINY}"
    --fallback-ormqr-block-size-small "${BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_SMALL}"
    --fallback-ormqr-block-size-medium "${BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_MEDIUM}"
    --fallback-ormqr-block-size-large "${BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_LARGE}"
    --fallback-ormqr-block-size-xlarge "${BATCHLAS_TUNE_DEFAULT_ORMQR_BLOCK_SIZE_XLARGE}"
    --fallback-sytrd-block-size-tiny "${BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_TINY}"
    --fallback-sytrd-block-size-small "${BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_SMALL}"
    --fallback-sytrd-block-size-medium "${BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_MEDIUM}"
    --fallback-sytrd-block-size-large "${BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_LARGE}"
    --fallback-sytrd-block-size-xlarge "${BATCHLAS_TUNE_DEFAULT_SYTRD_BLOCK_SIZE_XLARGE}"
    --fallback-latrd-lower-panel-wg-hint "${BATCHLAS_TUNE_DEFAULT_LATRD_LOWER_PANEL_WG_HINT}"
    --fallback-stedc-recursion-threshold-tiny "${BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_TINY}"
    --fallback-stedc-recursion-threshold-small "${BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_SMALL}"
    --fallback-stedc-recursion-threshold-medium "${BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_MEDIUM}"
    --fallback-stedc-recursion-threshold-large "${BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_LARGE}"
    --fallback-stedc-recursion-threshold-xlarge "${BATCHLAS_TUNE_DEFAULT_STEDC_RECURSION_THRESHOLD_XLARGE}"
    --fallback-stedc-merge-variant-tiny "${BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_TINY}"
    --fallback-stedc-merge-variant-small "${BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_SMALL}"
    --fallback-stedc-merge-variant-medium "${BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_MEDIUM}"
    --fallback-stedc-merge-variant-large "${BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_LARGE}"
    --fallback-stedc-merge-variant-xlarge "${BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_XLARGE}"
    --fallback-stedc-threads-per-root-tiny "${BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_TINY}"
    --fallback-stedc-threads-per-root-small "${BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_SMALL}"
    --fallback-stedc-threads-per-root-medium "${BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_MEDIUM}"
    --fallback-stedc-threads-per-root-large "${BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_LARGE}"
    --fallback-stedc-threads-per-root-xlarge "${BATCHLAS_TUNE_DEFAULT_STEDC_THREADS_PER_ROOT_XLARGE}"
    --fallback-stedc-wg-multiplier-tiny "${BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_TINY}"
    --fallback-stedc-wg-multiplier-small "${BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_SMALL}"
    --fallback-stedc-wg-multiplier-medium "${BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_MEDIUM}"
    --fallback-stedc-wg-multiplier-large "${BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_LARGE}"
    --fallback-stedc-wg-multiplier-xlarge "${BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_XLARGE}")

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/tuning_params.h.in"
    "${PROJECT_BINARY_DIR}/include/batchlas/tuning_params.hh"
)

if(BATCHLAS_TUNING_PROFILE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_target(batchlas_tuning_header
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/evaluation/tuning/generate_tuning_header.py"
            --profile "${BATCHLAS_TUNING_PROFILE}"
            --out "${PROJECT_BINARY_DIR}/include/batchlas/tuning_params.hh"
            ${BATCHLAS_TUNING_HEADER_FALLBACK_ARGS}
        DEPENDS "${BATCHLAS_TUNING_PROFILE}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating tuning constants header from ${BATCHLAS_TUNING_PROFILE}"
        VERBATIM
    )
endif()

function(batchlas_enable_tuning_targets)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(_BATCHLAS_TUNING_SPACE "${PROJECT_SOURCE_DIR}/evaluation/tuning/spaces/default.json")
    set(_BATCHLAS_TUNING_OUT "${PROJECT_BINARY_DIR}/tuning/profile.json")

    set(BATCHLAS_TUNE_BACKEND "CUDA" CACHE STRING "Backend to pass to tuning benchmarks (e.g., CUDA/ROCM/NETLIB/MKL)")
    set(BATCHLAS_TUNE_TYPE "float" CACHE STRING "Type to pass to tuning benchmarks (e.g., float/double)")

    add_custom_target(batchlas_tune
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${PROJECT_BINARY_DIR}/tuning"
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/evaluation/tuning/tune.py"
            --build-dir "${PROJECT_BINARY_DIR}"
            --space "${_BATCHLAS_TUNING_SPACE}"
            --backend "${BATCHLAS_TUNE_BACKEND}"
            --type "${BATCHLAS_TUNE_TYPE}"
            --out "${_BATCHLAS_TUNING_OUT}"
            --skip-missing
            --skip-failed
        DEPENDS
            stedc_benchmark
            steqr_benchmark
            sytrd_blocked_benchmark
            ormqr_blocked_benchmark
            syev_blocked_benchmark
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Running BatchLAS tuning harness (writes ${_BATCHLAS_TUNING_OUT})"
        VERBATIM
    )

    if(NOT TARGET batchlas_tuning_header)
        add_custom_target(batchlas_tuning_header
            COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/evaluation/tuning/generate_tuning_header.py"
                --profile "${_BATCHLAS_TUNING_OUT}"
                --out "${PROJECT_BINARY_DIR}/include/batchlas/tuning_params.hh"
                ${BATCHLAS_TUNING_HEADER_FALLBACK_ARGS}
            DEPENDS batchlas_tune
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            COMMENT "Generating tuning constants header from ${_BATCHLAS_TUNING_OUT}"
            VERBATIM
        )
    endif()
endfunction()

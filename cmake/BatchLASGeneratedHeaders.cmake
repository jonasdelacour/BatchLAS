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

function(batchlas_enable_tuning_targets)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(_BATCHLAS_TUNING_SPACE "${PROJECT_SOURCE_DIR}/evaluation/tuning/spaces/default.json")
    set(_BATCHLAS_TUNING_OUT "${PROJECT_BINARY_DIR}/tuning/profile.json")

    set(BATCHLAS_TUNE_BACKEND "CUDA" CACHE STRING "Backend to pass to tuning benchmarks (e.g., CUDA/ROCM/NETLIB)")
    set(BATCHLAS_TUNE_TYPE "float" CACHE STRING "Type to pass to tuning benchmarks (e.g., float/double)")

    add_custom_target(batchlas_tune_constants
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
endfunction()

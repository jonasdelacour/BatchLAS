file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/batchlas")

if(NOT DEFINED BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES)
    set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES 28672)
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

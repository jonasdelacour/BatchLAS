file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/batchlas")

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/backend_config.h.in"
    "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
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
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_TINY 2)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_SMALL 2)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_MEDIUM 2)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_LARGE 2)
set(BATCHLAS_TUNE_DEFAULT_STEDC_MERGE_VARIANT_XLARGE 2)
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
            --fallback-stedc-wg-multiplier-xlarge "${BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_XLARGE}"
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
                --fallback-stedc-wg-multiplier-xlarge "${BATCHLAS_TUNE_DEFAULT_STEDC_WG_MULTIPLIER_XLARGE}"
            DEPENDS batchlas_tune
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            COMMENT "Generating tuning constants header from ${_BATCHLAS_TUNING_OUT}"
            VERBATIM
        )
    endif()
endfunction()

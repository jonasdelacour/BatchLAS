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

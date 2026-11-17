file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/batchlas")

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/backend_config.h.in"
    "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
)

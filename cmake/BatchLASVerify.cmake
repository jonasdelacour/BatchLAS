# batchlas::verify (docs/design/verification.md): the tolerance policy and the host LAPACKE references
# (the checks themselves are MatrixView statics). Host code only; not installed (no install rule, and
# cmake/BatchLASPackaging.cmake excludes include/batchlas/verify).
add_library(batchlas_verify STATIC "${PROJECT_SOURCE_DIR}/src/verify/reference.cc")
add_library(batchlas::verify ALIAS batchlas_verify)
target_include_directories(batchlas_verify PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
target_compile_features(batchlas_verify PUBLIC cxx_std_20)
target_link_libraries(batchlas_verify PUBLIC batchlas)

if(BATCHLAS_HAS_HOST_BACKEND)
    target_link_libraries(batchlas_verify PUBLIC "$<BUILD_INTERFACE:${BATCHLAS_NETLIB_LINK_LIBRARIES}>")
    target_include_directories(batchlas_verify SYSTEM PUBLIC ${BATCHLAS_NETLIB_INCLUDE_DIRS})
    target_compile_definitions(batchlas_verify PUBLIC BATCHLAS_VERIFY_HAVE_LAPACKE=1)
else()
    target_compile_definitions(batchlas_verify PUBLIC BATCHLAS_VERIFY_HAVE_LAPACKE=0)
endif()

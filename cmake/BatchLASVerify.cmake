# batchlas::verify: the header-only host verification library (docs/design/verification.md).
# Not installed (cmake/BatchLASPackaging.cmake excludes include/batchlas/verify).
add_library(batchlas_verify INTERFACE)
add_library(batchlas::verify ALIAS batchlas_verify)
target_include_directories(batchlas_verify INTERFACE "${PROJECT_SOURCE_DIR}/include")

if(BATCHLAS_HAS_HOST_BACKEND)
    target_link_libraries(batchlas_verify INTERFACE ${BATCHLAS_NETLIB_LINK_LIBRARIES})
    target_include_directories(batchlas_verify SYSTEM INTERFACE ${BATCHLAS_NETLIB_INCLUDE_DIRS})
    target_compile_definitions(batchlas_verify INTERFACE BATCHLAS_VERIFY_HAVE_LAPACKE=1)
else()
    target_compile_definitions(batchlas_verify INTERFACE BATCHLAS_VERIFY_HAVE_LAPACKE=0)
endif()

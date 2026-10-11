# batchlas::verify (docs/design/verification.md): headers in include/batchlas/verify, the dense
# checks as SYCL kernels in libbatchlas_verify (src/verify/engine.cc), device-linked once here
# rather than in every test binary. Not installed: no install rule, and
# cmake/BatchLASPackaging.cmake excludes include/batchlas/verify.
add_library(batchlas_verify SHARED "${PROJECT_SOURCE_DIR}/src/verify/engine.cc")
add_library(batchlas::verify ALIAS batchlas_verify)
batchlas_configure_component(batchlas_verify)

if(BATCHLAS_HAS_HOST_BACKEND)
    target_link_libraries(batchlas_verify PUBLIC "$<BUILD_INTERFACE:${BATCHLAS_NETLIB_LINK_LIBRARIES}>")
    target_include_directories(batchlas_verify SYSTEM PUBLIC ${BATCHLAS_NETLIB_INCLUDE_DIRS})
    target_compile_definitions(batchlas_verify PUBLIC BATCHLAS_VERIFY_HAVE_LAPACKE=1)
else()
    target_compile_definitions(batchlas_verify PUBLIC BATCHLAS_VERIFY_HAVE_LAPACKE=0)
endif()

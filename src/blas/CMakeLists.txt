set(BLAS_SOURCES
    # Add your BLAS implementation source files here
    # Example:
    # gemm.cc
    # gemv.cc
)

if(NOT BLAS_SOURCES)
    set(_BATCHLAS_BLAS_DUMMY "${CMAKE_CURRENT_BINARY_DIR}/batchlas_blas_dummy.cc")
    file(WRITE "${_BATCHLAS_BLAS_DUMMY}" "// Dummy source for empty BLAS object library\n")
    list(APPEND BLAS_SOURCES "${_BATCHLAS_BLAS_DUMMY}")
endif()

target_sources(batchlas_blas_obj PRIVATE ${BLAS_SOURCES})
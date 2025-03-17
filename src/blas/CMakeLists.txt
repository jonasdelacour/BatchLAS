set(BLAS_SOURCES
    # Add your BLAS implementation source files here
    # Example:
    # gemm.cc
    # gemv.cc
)

target_sources(batchlas PRIVATE ${BLAS_SOURCES})
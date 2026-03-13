set(SYCL_SOURCES
    gemm_kernels.cc
)

target_sources(batchlas_sycl_obj PRIVATE ${SYCL_SOURCES})

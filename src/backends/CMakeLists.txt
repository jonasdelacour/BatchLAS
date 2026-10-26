set(BACKEND_COMMON_SOURCES
    backend_handle_instantiations.cc
)

set(BACKEND_HOST_SOURCES)
set(BACKEND_CUDA_SOURCES)
set(BACKEND_ROCM_SOURCES)

# Each vendor translation unit is gated on its own library, not on the device
# family: netlib_lapack.cc needs LAPACKE and CBLAS, the CUDA TUs need cuBLAS,
# cuSOLVER and cuSPARSE, and each ROCm TU its roc* library. The public op
# definitions live in src/ops/, so dropping a vendor TU drops none of them.
if(BATCHLAS_HAS_LAPACKE AND BATCHLAS_HAS_CBLAS)
    list(APPEND BACKEND_HOST_SOURCES
        netlib_lapack.cc
    )
endif()

# The three CUDA TUs are one group: cublas.cc makes cusolverDn* calls through the
# LinalgHandle<Backend::CUDA> that needs all three libraries (linalg-impl.hh), and
# cusparse.cc borrows cuBLAS's compute-type enum. BatchLASDependencies.cmake rejects a
# partial set, so BATCHLAS_HAS_CUDA_VENDOR_TUS is the one gate.
if(BATCHLAS_HAS_CUDA_VENDOR_TUS)
    list(APPEND BACKEND_CUDA_SOURCES cublas.cc cusolver.cc cusparse.cc)
    list(APPEND BACKEND_CUDA_SOURCES cusolver_sygvd.cc)
endif()

if(BATCHLAS_HAS_ROCBLAS)
    list(APPEND BACKEND_ROCM_SOURCES rocblas.cc)
endif()
if(BATCHLAS_HAS_ROCSOLVER)
    list(APPEND BACKEND_ROCM_SOURCES rocsolver.cc)
endif()
if(BATCHLAS_HAS_ROCSPARSE)
    list(APPEND BACKEND_ROCM_SOURCES rocsparse.cc)
endif()

target_sources(batchlas_backends_obj PRIVATE ${BACKEND_COMMON_SOURCES} ${BACKEND_HOST_SOURCES})

if(TARGET batchlas_backends_cuda_obj)
    target_sources(batchlas_backends_cuda_obj PRIVATE ${BACKEND_CUDA_SOURCES})
endif()

if(TARGET batchlas_backends_rocm_obj)
    target_sources(batchlas_backends_rocm_obj PRIVATE ${BACKEND_ROCM_SOURCES})
endif()
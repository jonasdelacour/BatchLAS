set(BACKEND_COMMON_SOURCES
    backend_handle_instantiations.cc
)

set(BACKEND_HOST_SOURCES)
set(BACKEND_CUDA_SOURCES)
set(BACKEND_ROCM_SOURCES)

if(BATCHLAS_HAS_HOST_BACKEND)
    list(APPEND BACKEND_HOST_SOURCES
        netlib_lapack.cc
    )
endif()

if(BATCHLAS_HAS_MKL_BACKEND)
    #list(APPEND BACKEND_SOURCES
    #    mkl.cc
    #)
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    list(APPEND BACKEND_CUDA_SOURCES
        cublas.cc
        gemm_cublasdx.cu
        gemm_cublasdx_dispatch.cc
        symm_cublasdx_fused.cu
        symm_custom_dispatch.cc
        syrk_cublasdx_fused.cu
        syrk_custom_dispatch.cc
        syr2k_cublasdx_fused.cu
        syr2k_custom_dispatch.cc
        trmm_cublasdx_fused.cu
        trmm_custom_dispatch.cc
        cusparse.cc
        cusolver.cc
        cusolverdx.cc
        cusolverdx_kernels.cu
    )
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    list(APPEND BACKEND_ROCM_SOURCES
        rocblas.cc
        rocsparse.cc
        rocsolver.cc
    )
endif()

target_sources(batchlas_backends_obj PRIVATE ${BACKEND_COMMON_SOURCES} ${BACKEND_HOST_SOURCES})

if(TARGET batchlas_backends_cuda_obj)
    target_sources(batchlas_backends_cuda_obj PRIVATE ${BACKEND_CUDA_SOURCES})
endif()

if(TARGET batchlas_backends_rocm_obj)
    target_sources(batchlas_backends_rocm_obj PRIVATE ${BACKEND_ROCM_SOURCES})
endif()
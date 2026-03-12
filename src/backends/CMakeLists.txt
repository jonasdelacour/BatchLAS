set(BACKEND_SOURCES
    backend_handle_instantiations.cc
)

if(BATCHLAS_HAS_HOST_BACKEND)
    list(APPEND BACKEND_SOURCES
        netlib_lapack.cc
    )
endif()

if(BATCHLAS_HAS_MKL_BACKEND)
    #list(APPEND BACKEND_SOURCES
    #    mkl.cc
    #)
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    list(APPEND BACKEND_SOURCES
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
    list(APPEND BACKEND_SOURCES
        rocblas.cc
        rocsparse.cc
        rocsolver.cc
    )
endif()

target_sources(batchlas_backends_obj PRIVATE ${BACKEND_SOURCES})
set(BACKEND_SOURCES)

if(BATCHLAS_HAS_HOST_BACKEND)
    list(APPEND BACKEND_SOURCES
        netlib_lapack.cc
    )
endif()

if(BATCHLAS_HAS_MKL_BACKEND)
    list(APPEND BACKEND_SOURCES
        mkl.cc
    )
endif()

if(BATCHLAS_ENABLE_CUDA)
    list(APPEND BACKEND_SOURCES
        cublas.cc
        cusparse.cc
        cusolver.cc
    )
endif()

target_sources(batchlas PRIVATE ${BACKEND_SOURCES})
set(BACKEND_SOURCES
    netlib_lapack.cc
)

if(BATCHLAS_ENABLE_CUDA)
    list(APPEND BACKEND_SOURCES
        cublas.cc
        cusolver.cc
        cusparse.cc
    )
endif()

target_sources(batchlas PRIVATE ${BACKEND_SOURCES})
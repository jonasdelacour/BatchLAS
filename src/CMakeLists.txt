# Define the main library target
add_library(batchlas SHARED
    linalg-impl.hh
    matrix.cc
    #util/sycl-util-impl.cc
)

# Add source files from subdirectories
add_subdirectory(backends)
add_subdirectory(blas)
add_subdirectory(extensions)
add_subdirectory(util)
add_subdirectory(extra)

if(BATCHLAS_ENABLE_SYCL)
    add_subdirectory(sycl)
endif()

# Setup target properties
target_include_directories(batchlas 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)

# Export the target
install(TARGETS batchlas
    EXPORT BatchLASTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

set(BATCHLAS_LINK_LIBRARIES)

if(BATCHLAS_HAS_HOST_BACKEND)
    list(APPEND BATCHLAS_LINK_LIBRARIES
        ${BATCHLAS_LAPACKE_LIBRARY}
        ${BATCHLAS_CBLAS_LIBRARY}
    )
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    find_package(CUDAToolkit REQUIRED)

    list(APPEND BATCHLAS_LINK_LIBRARIES
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
    )
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    list(APPEND BATCHLAS_LINK_LIBRARIES
        rocblas
        rocsparse
        rocsolver
    )
endif()

#if(BATCHLAS_HAS_MKL_BACKEND)
 #   list(APPEND BATCHLAS_LINK_LIBRARIES
        #batchlas::mkl
#  )
#endif()

target_link_libraries(batchlas
    PUBLIC 
    # Specify any additional libraries here
    ${BATCHLAS_LINK_LIBRARIES}
    tbb # Intel Threading Building Blocks, needed for oneapi parallel primitives
)

# Create namespaced alias
add_library(BatchLAS::batchlas ALIAS batchlas)
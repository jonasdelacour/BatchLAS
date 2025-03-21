# Define the main library target
add_library(batchlas SHARED
    linalg-impl.hh
)

# Add source files from subdirectories
add_subdirectory(backends)
add_subdirectory(blas)
add_subdirectory(extensions)
add_subdirectory(util)

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

# Find LAPACKE and CBLAS libraries
find_library(LAPACKE_LIBRARY NAMES lapacke PATHS ${LD_LIBRARY_PATH})
find_library(CBLAS_LIBRARY NAMES cblas PATHS ${LD_LIBRARY_PATH})

if(NOT LAPACKE_LIBRARY)
  message(FATAL_ERROR "LAPACKE library not found")
endif()
if(NOT CBLAS_LIBRARY)
  message(FATAL_ERROR "CBLAS library not found")
endif()

set(BATCHLAS_LINK_LIBRARIES
    ${LAPACKE_LIBRARY}
    ${CBLAS_LIBRARY}
)

if(BATCHLAS_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    list(APPEND BATCHLAS_LINK_LIBRARIES
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
    )
endif()

target_link_libraries(batchlas
    PUBLIC 
    # Specify any additional libraries here
    ${BATCHLAS_LINK_LIBRARIES}
)

# Create namespaced alias
add_library(BatchLAS::batchlas ALIAS batchlas)
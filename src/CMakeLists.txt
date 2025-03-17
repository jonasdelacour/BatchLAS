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

set(BATCHLAS_LINK_LIBRARIES
    lapack
)

if(BATCHLAS_ENABLE_CUDA)
    list(APPEND BATCHLAS_LINK_LIBRARIES
        cudart
        cublas
        cusolver
        cusparse
    )
endif()

target_link_libraries(batchlas
    PUBLIC 
    # Specify any additional libraries here
    ${BATCHLAS_LINK_LIBRARIES}
)

# Create namespaced alias
add_library(BatchLAS::batchlas ALIAS batchlas)
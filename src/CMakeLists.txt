# Partition the library into small, independent chunks (OBJECT libs)
# to speed up incremental compilation.
add_library(batchlas_core_obj OBJECT
    linalg-impl.hh
    matrix.cc
    #util/sycl-util-impl.cc
)
add_library(batchlas_backends_obj OBJECT)
add_library(batchlas_blas_obj OBJECT)
add_library(batchlas_extensions_obj OBJECT)
add_library(batchlas_extensions_cta_obj OBJECT)
add_library(batchlas_util_obj OBJECT)
add_library(batchlas_extra_obj OBJECT)
set(_BATCHLAS_SYCL_OBJ_DUMMY "${CMAKE_CURRENT_BINARY_DIR}/batchlas_sycl_obj_dummy.cc")
file(WRITE "${_BATCHLAS_SYCL_OBJ_DUMMY}" "// Dummy source for SYCL object library\n")
add_library(batchlas_sycl_obj OBJECT "${_BATCHLAS_SYCL_OBJ_DUMMY}")

set(BATCHLAS_OBJECT_LIBS
    batchlas_core_obj
    batchlas_backends_obj
    batchlas_blas_obj
    batchlas_extensions_obj
    batchlas_extensions_cta_obj
    batchlas_util_obj
    batchlas_extra_obj
    batchlas_sycl_obj
)

set_target_properties(
    ${BATCHLAS_OBJECT_LIBS}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
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

# Create the shared library to ensure SYCL link happens once
add_library(batchlas SHARED
    $<TARGET_OBJECTS:batchlas_core_obj>
    $<TARGET_OBJECTS:batchlas_backends_obj>
    $<TARGET_OBJECTS:batchlas_blas_obj>
    $<TARGET_OBJECTS:batchlas_extensions_obj>
    $<TARGET_OBJECTS:batchlas_extensions_cta_obj>
    $<TARGET_OBJECTS:batchlas_util_obj>
    $<TARGET_OBJECTS:batchlas_extra_obj>
    $<TARGET_OBJECTS:batchlas_sycl_obj>
)

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

if(BATCHLAS_ENABLE_SYCL)
    if(BATCHLAS_SYCL_TARGETS_STRING)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING}")
        foreach(_lib ${BATCHLAS_OBJECT_LIBS})
            if(_lib STREQUAL "batchlas_extensions_cta_obj" AND BATCHLAS_SYCL_TARGETS_NO_CPU_STRING)
                target_compile_options(${_lib} PRIVATE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_NO_CPU_STRING})
            else()
                target_compile_options(${_lib} PRIVATE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
            endif()
        endforeach()
        target_link_options(batchlas PRIVATE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
        set_property(TARGET batchlas APPEND_STRING PROPERTY LINK_FLAGS " -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING}")
        target_compile_options(batchlas INTERFACE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
        target_link_options(batchlas INTERFACE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
    endif()
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    find_package(CUDAToolkit REQUIRED)

    foreach(_obj ${BATCHLAS_OBJECT_LIBS})
        target_include_directories(${_obj} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endforeach()

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
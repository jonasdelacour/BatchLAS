# Partition the library into small, independent chunks (OBJECT libs)
# to speed up incremental compilation.
add_library(batchlas_core_obj OBJECT
    csr_generators.cc
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

foreach(_obj ${BATCHLAS_OBJECT_LIBS})
    if(_obj STREQUAL "batchlas_extensions_cta_obj")
        batchlas_apply_object_options(${_obj} NO_CPU_TARGETS)
    else()
        batchlas_apply_object_options(${_obj})
    endif()
endforeach()

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
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(batchlas PUBLIC cxx_std_17)
target_link_libraries(batchlas
    PRIVATE
        batchlas_build_options
        batchlas_dep_options
    PUBLIC
        batchlas_sycl_options
)

set(BATCHLAS_LINK_LIBRARIES)

if(BATCHLAS_HAS_HOST_BACKEND)
    list(APPEND BATCHLAS_LINK_LIBRARIES
        ${BATCHLAS_NETLIB_LINK_LIBRARIES}
    )
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    if(BATCHLAS_CUDA_ARCHITECTURES)
        set_property(TARGET batchlas_backends_obj PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
        set_property(TARGET batchlas PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
    endif()

    if(BATCHLAS_MATHDX_TARGETS)
        target_link_libraries(batchlas_backends_obj PRIVATE ${BATCHLAS_MATHDX_TARGETS})
        target_link_libraries(batchlas PRIVATE ${BATCHLAS_MATHDX_TARGETS})
    endif()
    if(BATCHLAS_MATHDX_INCLUDE_DIRS)
        target_include_directories(batchlas_backends_obj PRIVATE ${BATCHLAS_MATHDX_INCLUDE_DIRS})
    endif()
    if(BATCHLAS_KEEP_CUDA_INTERMEDIATES)
        target_compile_options(batchlas_backends_obj PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--keep>
            $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${CMAKE_BINARY_DIR}/cuda-intermediates>
        )
    endif()

    if(BATCHLAS_ENABLE_CUBLASDX_WRAPPER)
        target_compile_definitions(batchlas_backends_obj PRIVATE BATCHLAS_ENABLE_CUBLASDX_WRAPPER=1)
        target_compile_definitions(batchlas PRIVATE BATCHLAS_ENABLE_CUBLASDX_WRAPPER=1)
        target_compile_options(batchlas_backends_obj PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()

    list(APPEND BATCHLAS_LINK_LIBRARIES
        ${BATCHLAS_CUDA_LINK_LIBRARIES}
    )
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    list(APPEND BATCHLAS_LINK_LIBRARIES
        ${BATCHLAS_ROCM_LINK_LIBRARIES}
    )
endif()

#if(BATCHLAS_HAS_MKL_BACKEND)
 #   list(APPEND BATCHLAS_LINK_LIBRARIES
        #batchlas::mkl
#  )
#endif()

target_link_libraries(batchlas
    PUBLIC 
    ${BATCHLAS_LINK_LIBRARIES}
    tbb # Intel Threading Building Blocks, needed for oneapi parallel primitives
)

# Create namespaced alias
add_library(BatchLAS::batchlas ALIAS batchlas)

# Partition the library into small, independent chunks (OBJECT libs)
# to speed up incremental compilation.
add_library(batchlas_core_obj OBJECT
    csr_generators.cc
    linalg-impl.hh
    matrix.cc
    #util/sycl-util-impl.cc
)
add_library(batchlas_backends_obj OBJECT)
if(BATCHLAS_HAS_CUDA_BACKEND)
    add_library(batchlas_backends_cuda_obj OBJECT)
endif()
if(BATCHLAS_HAS_ROCM_BACKEND)
    add_library(batchlas_backends_rocm_obj OBJECT)
endif()
add_library(batchlas_blas_obj OBJECT)
add_library(batchlas_extensions_eigen_obj OBJECT)
add_library(batchlas_extensions_factorization_obj OBJECT)
add_library(batchlas_extensions_symmetric_obj OBJECT)
add_library(batchlas_extensions_tridiag_obj OBJECT)
add_library(batchlas_extensions_cta_obj OBJECT)
add_library(batchlas_util_obj OBJECT)
add_library(batchlas_extra_obj OBJECT)
set(_BATCHLAS_SYCL_OBJ_DUMMY "${CMAKE_CURRENT_BINARY_DIR}/batchlas_sycl_obj_dummy.cc")
file(WRITE "${_BATCHLAS_SYCL_OBJ_DUMMY}" "// Dummy source for SYCL object library\n")
add_library(batchlas_sycl_obj OBJECT "${_BATCHLAS_SYCL_OBJ_DUMMY}")

set(BATCHLAS_OBJECT_LIBS
    batchlas_core_obj
    batchlas_backends_obj
    batchlas_extensions_eigen_obj
    batchlas_extensions_factorization_obj
    batchlas_extensions_symmetric_obj
    batchlas_extensions_tridiag_obj
    batchlas_extensions_cta_obj
    batchlas_util_obj
    batchlas_extra_obj
    batchlas_sycl_obj
    batchlas_blas_obj
)

if(TARGET batchlas_backends_cuda_obj)
    list(APPEND BATCHLAS_OBJECT_LIBS batchlas_backends_cuda_obj)
endif()

if(TARGET batchlas_backends_rocm_obj)
    list(APPEND BATCHLAS_OBJECT_LIBS batchlas_backends_rocm_obj)
endif()

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

function(batchlas_configure_component target)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_link_libraries(${target}
        PRIVATE
            batchlas_build_options
            batchlas_dep_options
            batchlas_sycl_options
            tbb
    )
endfunction()

add_library(batchlas_core SHARED
    $<TARGET_OBJECTS:batchlas_core_obj>
)
batchlas_configure_component(batchlas_core)

add_library(batchlas_backends SHARED
    $<TARGET_OBJECTS:batchlas_backends_obj>
)
batchlas_configure_component(batchlas_backends)

add_library(batchlas_blas SHARED
    $<TARGET_OBJECTS:batchlas_blas_obj>
)
batchlas_configure_component(batchlas_blas)

add_library(batchlas_extensions_eigen SHARED
    $<TARGET_OBJECTS:batchlas_extensions_eigen_obj>
)
batchlas_configure_component(batchlas_extensions_eigen)

add_library(batchlas_extensions_factorization SHARED
    $<TARGET_OBJECTS:batchlas_extensions_factorization_obj>
)
batchlas_configure_component(batchlas_extensions_factorization)

add_library(batchlas_extensions_symmetric SHARED
    $<TARGET_OBJECTS:batchlas_extensions_symmetric_obj>
)
batchlas_configure_component(batchlas_extensions_symmetric)

add_library(batchlas_extensions_tridiag SHARED
    $<TARGET_OBJECTS:batchlas_extensions_tridiag_obj>
)
batchlas_configure_component(batchlas_extensions_tridiag)

add_library(batchlas_extensions_cta SHARED
    $<TARGET_OBJECTS:batchlas_extensions_cta_obj>
)
batchlas_configure_component(batchlas_extensions_cta)

add_library(batchlas_util SHARED
    $<TARGET_OBJECTS:batchlas_util_obj>
)
batchlas_configure_component(batchlas_util)

add_library(batchlas_extra SHARED
    $<TARGET_OBJECTS:batchlas_extra_obj>
)
batchlas_configure_component(batchlas_extra)

add_library(batchlas_sycl SHARED
    $<TARGET_OBJECTS:batchlas_sycl_obj>
)
batchlas_configure_component(batchlas_sycl)

set(BATCHLAS_COMPONENT_TARGETS
    batchlas_core
    batchlas_backends
    batchlas_blas
    batchlas_extensions_eigen
    batchlas_extensions_factorization
    batchlas_extensions_symmetric
    batchlas_extensions_tridiag
    batchlas_extensions_cta
    batchlas_util
    batchlas_extra
    batchlas_sycl
)

if(BATCHLAS_HAS_CUDA_BACKEND)
    add_library(batchlas_backends_cuda SHARED
        $<TARGET_OBJECTS:batchlas_backends_cuda_obj>
    )
    batchlas_configure_component(batchlas_backends_cuda)
    list(APPEND BATCHLAS_COMPONENT_TARGETS batchlas_backends_cuda)
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    add_library(batchlas_backends_rocm SHARED
        $<TARGET_OBJECTS:batchlas_backends_rocm_obj>
    )
    batchlas_configure_component(batchlas_backends_rocm)
    list(APPEND BATCHLAS_COMPONENT_TARGETS batchlas_backends_rocm)
endif()

set(BATCHLAS_LINK_LIBRARIES)

if(BATCHLAS_HAS_HOST_BACKEND)
    target_link_libraries(batchlas_backends
        PUBLIC
        ${BATCHLAS_NETLIB_LINK_LIBRARIES}
    )
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    if(BATCHLAS_CUDA_ARCHITECTURES)
        set_property(TARGET batchlas_backends_obj PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
        set_property(TARGET batchlas_backends_cuda_obj PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
        set_property(TARGET batchlas_backends_cuda PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
    endif()

    if(BATCHLAS_MATHDX_TARGETS)
        target_link_libraries(batchlas_backends_cuda PRIVATE ${BATCHLAS_MATHDX_TARGETS})
    endif()
    if(BATCHLAS_MATHDX_INCLUDE_DIRS)
        target_include_directories(batchlas_backends_cuda_obj PRIVATE ${BATCHLAS_MATHDX_INCLUDE_DIRS})
    endif()
    if(BATCHLAS_KEEP_CUDA_INTERMEDIATES)
        target_compile_options(batchlas_backends_cuda_obj PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--keep>
            $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=${CMAKE_BINARY_DIR}/cuda-intermediates>
        )
    endif()

    if(BATCHLAS_ENABLE_CUBLASDX_WRAPPER)
        target_compile_definitions(batchlas_backends_cuda_obj PRIVATE BATCHLAS_ENABLE_CUBLASDX_WRAPPER=1)
        target_compile_options(batchlas_backends_cuda_obj PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()

    target_link_libraries(batchlas_backends_cuda PUBLIC ${BATCHLAS_CUDA_LINK_LIBRARIES})
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    target_link_libraries(batchlas_backends_rocm PUBLIC ${BATCHLAS_ROCM_LINK_LIBRARIES})
endif()

#if(BATCHLAS_HAS_MKL_BACKEND)
 #   list(APPEND BATCHLAS_LINK_LIBRARIES
        #batchlas::mkl
#  )
#endif()

add_library(batchlas INTERFACE)
target_include_directories(batchlas
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(batchlas INTERFACE cxx_std_17)
target_link_libraries(batchlas
    INTERFACE
        batchlas_sycl_options
        ${BATCHLAS_COMPONENT_TARGETS}
)

add_custom_target(batchlas_components DEPENDS ${BATCHLAS_COMPONENT_TARGETS})

if(BATCHLAS_ENABLE_CUBLASDX_WRAPPER)
    target_compile_definitions(batchlas INTERFACE BATCHLAS_ENABLE_CUBLASDX_WRAPPER=1)
endif()

# Create namespaced alias
add_library(BatchLAS::batchlas ALIAS batchlas)

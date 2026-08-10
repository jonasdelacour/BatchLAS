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
add_library(batchlas_extensions_eigen_obj OBJECT)
add_library(batchlas_extensions_factorization_obj OBJECT)
add_library(batchlas_extensions_symmetric_obj OBJECT)
add_library(batchlas_extensions_tridiag_obj OBJECT)
add_library(batchlas_extensions_sytrd_obj OBJECT)
add_library(batchlas_extensions_latrd_obj OBJECT)
add_library(batchlas_extensions_stedc_obj OBJECT)
add_library(batchlas_extensions_cta_obj OBJECT)
add_library(batchlas_util_obj OBJECT)
add_library(batchlas_extra_obj OBJECT)
# src/sycl/CMakeLists.txt target_sources()s gemm_kernels.cc into this, so the
# generated dummy TU it used to carry was a second object compiled for nothing.
add_library(batchlas_sycl_obj OBJECT)

set(BATCHLAS_OBJECT_LIBS
    batchlas_core_obj
    batchlas_backends_obj
    batchlas_extensions_eigen_obj
    batchlas_extensions_factorization_obj
    batchlas_extensions_symmetric_obj
    batchlas_extensions_tridiag_obj
    batchlas_extensions_sytrd_obj
    batchlas_extensions_latrd_obj
    batchlas_extensions_stedc_obj
    batchlas_extensions_cta_obj
    batchlas_util_obj
    batchlas_extra_obj
    batchlas_sycl_obj
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
            # The headers install under <prefix>/include/batchlas/ so that
            # blas/, util/ and internal/ do not squat the consumer's include
            # root. Every existing #include spelling is unchanged.
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/batchlas>
    )
    target_compile_features(${target} PUBLIC cxx_std_20)
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

# INTERFACE, not SHARED: src/blas/ was never populated, so this used to compile
# a generated dummy TU and run a full SYCL device link to produce an 18 KB
# library exporting nothing but weak runtime shims that every other component
# also defines. INTERFACE keeps BatchLAS::batchlas_blas resolvable for anyone
# who already links it by name.
add_library(batchlas_blas INTERFACE)

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

add_library(batchlas_extensions_sytrd SHARED
    $<TARGET_OBJECTS:batchlas_extensions_sytrd_obj>
)
batchlas_configure_component(batchlas_extensions_sytrd)

add_library(batchlas_extensions_latrd SHARED
    $<TARGET_OBJECTS:batchlas_extensions_latrd_obj>
)
batchlas_configure_component(batchlas_extensions_latrd)

add_library(batchlas_extensions_stedc SHARED
    $<TARGET_OBJECTS:batchlas_extensions_stedc_obj>
)
batchlas_configure_component(batchlas_extensions_stedc)

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
    batchlas_extensions_sytrd
    batchlas_extensions_latrd
    batchlas_extensions_stedc
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
    # PRIVATE: these are absolute -dev symlink paths
    # (/usr/lib/x86_64-linux-gnu/liblapacke.so). PUBLIC put them verbatim into
    # the exported BatchLAS::batchlas_backends interface, so a consumer on a
    # runtime-only image failed with "No rule to make target
    # '.../liblapacke.so'" - a message that never mentions BatchLAS.
    # libbatchlas_backends.so already records DT_NEEDED on liblapacke.so.3 and
    # libblas.so.3, and no public header calls LAPACKE or includes <lapacke.h>,
    # so a consumer needs neither the headers nor the -dev package.
    target_link_libraries(batchlas_backends
        PRIVATE
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

    # PRIVATE: PUBLIC exported CUDA::cudart/cublas/cusolver/cusparse into the
    # package interface, which forced find_dependency(CUDAToolkit REQUIRED) on
    # every consumer - including CPU-only teammates who cannot resolve it.
    # libbatchlas_backends_cuda.so already records DT_NEEDED on libcudart.so.13
    # et al, and no public header includes a CUDA header.
    target_link_libraries(batchlas_backends_cuda PRIVATE ${BATCHLAS_CUDA_LINK_LIBRARIES})
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    # PRIVATE for the same reason as the netlib and CUDA links above: these are
    # absolute paths under ${ROCM_PATH} and must not ride out on the export.
    target_link_libraries(batchlas_backends_rocm PRIVATE ${BATCHLAS_ROCM_LINK_LIBRARIES})
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
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/batchlas>
)
target_compile_features(batchlas INTERFACE cxx_std_20)
target_link_libraries(batchlas
    INTERFACE
        batchlas_sycl_options
        ${BATCHLAS_COMPONENT_TARGETS}
)

# The component .so files reference each other's symbols but record no
# DT_NEEDED on one another: their symbol graph has 14 cycles, which CMake
# refuses to express (cyclic dependencies are only allowed among static
# libraries). A toolchain defaulting to -Wl,--as-needed therefore drops
# libraries the link genuinely needs, and the documented
# `target_link_libraries(app PRIVATE BatchLAS::batchlas)` recipe fails with
# "undefined reference to batchlas::sycl_gemm::gemm_custom<float>(...)".
# This is the documented mitigation until the .so merge lands; it keeps every
# component on the link line. Linker options precede link libraries in CMake's
# link rule, so it covers all of them.
if(UNIX AND NOT APPLE)
    target_link_options(batchlas INTERFACE "-Wl,--no-as-needed")
endif()

add_custom_target(batchlas_components DEPENDS ${BATCHLAS_COMPONENT_TARGETS})

if(BATCHLAS_ENABLE_CUBLASDX_WRAPPER)
    target_compile_definitions(batchlas INTERFACE BATCHLAS_ENABLE_CUBLASDX_WRAPPER=1)
endif()

# Create namespaced alias
add_library(BatchLAS::batchlas ALIAS batchlas)

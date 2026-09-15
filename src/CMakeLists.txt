# Partition the library into small, independent chunks (OBJECT libs)
# to speed up incremental compilation.
add_library(batchlas_core_obj OBJECT
    csr_generators.cc
    linalg-impl.hh
    matrix.cc
    #util/sycl-util-impl.cc
)
add_library(batchlas_backends_obj OBJECT)
# The vendor-independent facade -- see src/dispatch/CMakeLists.txt.
add_library(batchlas_dispatch_obj OBJECT)
# WP0 S6: these carry the VENDOR wrapper TUs, so they exist only when at least
# one vendor library for that family does. "There is a CUDA device" is no longer
# sufficient -- with every CUDA math library off, BACKEND_CUDA_SOURCES is empty
# and an OBJECT library with no sources is a hard CMake error, which is the
# first thing a vendor-free configure used to hit.
if(BATCHLAS_HAS_CUDA_BACKEND AND (BATCHLAS_HAS_CUBLAS OR BATCHLAS_HAS_CUSOLVER OR BATCHLAS_HAS_CUSPARSE))
    add_library(batchlas_backends_cuda_obj OBJECT)
endif()
if(BATCHLAS_HAS_ROCM_BACKEND AND (BATCHLAS_HAS_ROCBLAS OR BATCHLAS_HAS_ROCSOLVER OR BATCHLAS_HAS_ROCSPARSE))
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
    batchlas_dispatch_obj
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

# ---------------------------------------------------------------------------
# WP3c BEGIN -- export header and hidden visibility (owned by the 3c change)
# ---------------------------------------------------------------------------
include(GenerateExportHeader)

# BATCHLAS_API is generated, not hand-written, so the one place that decides
# what "exported" means is CMake's own platform knowledge.
#
# WHY IT IS GENERATED AGAINST AN OBJECT LIBRARY. generate_export_header()
# returns with a warning for an INTERFACE library, and `batchlas` is INTERFACE
# in split mode -- so keying the header off the facade would silently produce
# nothing in the default build. The only thing the target contributes to the
# generated file is the `<target>_EXPORTS` macro the header tests, and on ELF
# BOTH arms of that test expand to __attribute__((visibility("default")))
# (CMake's _DO_SET_MACRO_VALUES, elseif(COMPILER_HAS_HIDDEN_VISIBILITY) branch:
# DEFINE_EXPORT and DEFINE_IMPORT are the same string). One generated header
# therefore serves split mode, monolithic mode and an external consumer
# identically, and batchlas_core_obj is simply the target that exists in every
# one of those configurations.
generate_export_header(batchlas_core_obj
    BASE_NAME BATCHLAS
    EXPORT_MACRO_NAME BATCHLAS_API
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/batchlas/export.hh"
)

# MONOLITHIC MODE ONLY, and that restriction is load-bearing rather than tidy.
# Measured on the 14 split .so in build/presets/dev-tests: 879 symbol
# references cross a component boundary, and four weak DATA objects
# (batchlas_kernel_trace::g_records, batchlas::QueueImpl::device_arrays and
# both guard variables) are defined in all 14 libraries and collapsed to one
# instance today only because they carry default visibility. Hiding them in
# SPLIT mode gives each .so its own copy of the kernel-trace vector and the
# per-device array cache -- a silent behavioural change with no undefined
# reference anywhere to point at it. Only once the objects are in ONE .so are
# those 879 references intra-library relocations and the four globals single.
#
# The properties emit exactly -fvisibility=hidden and
# -fvisibility-inlines-hidden for this compiler; they are used instead of raw
# flags so the same declaration also covers a future non-Clang host build.
if(BATCHLAS_MONOLITHIC_LIBRARY)
    set_target_properties(${BATCHLAS_OBJECT_LIBS} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
# ---------------------------------------------------------------------------
# WP3c END
# ---------------------------------------------------------------------------

# Add source files from subdirectories
add_subdirectory(backends)
add_subdirectory(dispatch)
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
            # Every public header lives under include/batchlas/ in the source
            # tree and installs to <prefix>/include/batchlas/, so the only name
            # BatchLAS claims in a consumer's include root is `batchlas`.
            # The include root itself is therefore what we export.
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_link_libraries(${target}
        PRIVATE
            batchlas_build_options
            batchlas_dep_options
            batchlas_sycl_options
            tbb
    )
    # SONAME discipline. Without these every install wrote a bare
    # libbatchlas_*.so whose DT_SONAME was the same bare name, so two versions
    # could not coexist in one prefix, no distribution could package it, and an
    # ABI break was indistinguishable from a rebuild. VERSION gives the real
    # file (libbatchlas_core.so.0.1.0), SOVERSION the DT_SONAME and the runtime
    # symlink (.so.0), and CMake adds the .so devel symlink itself.
    #
    # PROJECT_VERSION_MAJOR is 0 today, so every library is .so.0: that is the
    # honest statement, not an oversight. SemVer promises nothing across 0.x
    # minors, which is why the package version file is ExactVersion while the
    # major stays 0 (see cmake/BatchLASPackaging.cmake).
    set_target_properties(${target} PROPERTIES
        VERSION "${PROJECT_VERSION}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
    )
endfunction()

# ---------------------------------------------------------------------------
# One public library, or fourteen components. See BATCHLAS_MONOLITHIC_LIBRARY
# in cmake/BatchLASOptions.cmake for why the split is the DEV default and the
# merge is the RELEASE shape.
# ---------------------------------------------------------------------------
if(BATCHLAS_MONOLITHIC_LIBRARY)
    # Every object library, one device link. batchlas_extensions_cta_obj is
    # built with batchlas_sycl_no_cpu_options while the other 14 use
    # batchlas_sycl_options, so this merge puts objects with two different
    # -fsycl-targets sets into one sycl-post-link. MEASURED: it links. The
    # single `Linking CXX shared library libbatchlas.so` step took 575 s
    # (9m35s) on this box for a 273 MiB .so, single-threaded, out of an 11m38s
    # -j16 build of the whole target. That is the release cost; it is also why
    # BATCHLAS_MONOLITHIC_LIBRARY must never become the dev default.
    set(_batchlas_monolithic_objects "")
    foreach(_obj ${BATCHLAS_OBJECT_LIBS})
        list(APPEND _batchlas_monolithic_objects $<TARGET_OBJECTS:${_obj}>)
    endforeach()

    add_library(batchlas SHARED ${_batchlas_monolithic_objects})
    unset(_batchlas_monolithic_objects)

    # $<TARGET_OBJECTS:> carries objects only, never the object library's usage
    # requirements, so the monolithic target has to repeat exactly what every
    # component gets. Calling the same function is the point: an include
    # directory that differs between the two modes silently changes a
    # consumer's include path.
    batchlas_configure_component(batchlas)

    # The split-mode facade links batchlas_sycl_options INTERFACE and
    # batchlas_configure_component() links it PRIVATE; the union is PUBLIC.
    # Every flag in it is $<BUILD_INTERFACE:>-wrapped, so in-tree tests and
    # benchmarks still get -fsycl and external consumers still get nothing.
    target_link_libraries(batchlas INTERFACE batchlas_sycl_options)

    # No component targets exist in this mode: nothing to install beside
    # `batchlas`, and nothing to export.
    set(BATCHLAS_COMPONENT_TARGETS "")

    # tests/CMakeLists.txt links `batchlas_util` by name for util_span_tests and
    # util_vector_tests (NO_FACADE, so they deliberately avoid the whole
    # library). That target does not exist here and the failure would be a
    # configure error, not a link error. The alias keeps those two green; it
    # pulls in the whole .so, which for two tests is cheaper than a second
    # spelling of the same objects.
    add_library(batchlas_util ALIAS batchlas)
else()
# --- split mode: the 14 component .so, deliberately left unindented so the
# --- diff against the pre-BATCHLAS_MONOLITHIC_LIBRARY file stays readable.
add_library(batchlas_core SHARED
    $<TARGET_OBJECTS:batchlas_core_obj>
)
batchlas_configure_component(batchlas_core)

# batchlas_dispatch_obj rides in this component rather than getting a .so of its
# own: it is the public face of exactly these entry points, and a separate
# library would add an install target and an export entry for two dozen
# forwarding functions. The separation that matters is at TU level -- the
# definitions are no longer inside a vendor file -- not at library level.
add_library(batchlas_backends SHARED
    $<TARGET_OBJECTS:batchlas_backends_obj>
    $<TARGET_OBJECTS:batchlas_dispatch_obj>
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

if(TARGET batchlas_backends_cuda_obj)
    add_library(batchlas_backends_cuda SHARED
        $<TARGET_OBJECTS:batchlas_backends_cuda_obj>
    )
    batchlas_configure_component(batchlas_backends_cuda)
    list(APPEND BATCHLAS_COMPONENT_TARGETS batchlas_backends_cuda)
endif()

if(TARGET batchlas_backends_rocm_obj)
    add_library(batchlas_backends_rocm SHARED
        $<TARGET_OBJECTS:batchlas_backends_rocm_obj>
    )
    batchlas_configure_component(batchlas_backends_rocm)
    list(APPEND BATCHLAS_COMPONENT_TARGETS batchlas_backends_rocm)
endif()
endif()

# cmake/BatchLASPackaging.cmake needs this list and runs in the top-level scope,
# so hand it over rather than keeping a second hand-written copy of the same 16
# names in the packaging file (the two drifted apart once already).
set_property(GLOBAL PROPERTY BATCHLAS_COMPONENT_TARGETS "${BATCHLAS_COMPONENT_TARGETS}")

# Where the PRIVATE vendor links and the CUDA architecture properties land.
# In monolithic mode there is no batchlas_backends / batchlas_backends_cuda /
# batchlas_backends_rocm target and every one of them is `batchlas`; dropping
# any of these would either fail the link or, worse, re-export absolute -dev
# paths into the package interface (see the PRIVATE comments below).
if(BATCHLAS_MONOLITHIC_LIBRARY)
    set(_batchlas_backends_lib batchlas)
    set(_batchlas_backends_cuda_lib batchlas)
    set(_batchlas_backends_rocm_lib batchlas)
else()
    set(_batchlas_backends_lib batchlas_backends)
    set(_batchlas_backends_cuda_lib batchlas_backends_cuda)
    set(_batchlas_backends_rocm_lib batchlas_backends_rocm)
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
    target_link_libraries(${_batchlas_backends_lib}
        PRIVATE
        ${BATCHLAS_NETLIB_LINK_LIBRARIES}
    )
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    if(BATCHLAS_CUDA_ARCHITECTURES)
        set_property(TARGET batchlas_backends_obj PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
    endif()
endif()

# Everything below needs the CUDA VENDOR component, which only exists when at
# least one CUDA math library was found.
if(TARGET batchlas_backends_cuda_obj)
    if(BATCHLAS_CUDA_ARCHITECTURES)
        set_property(TARGET batchlas_backends_cuda_obj PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
        set_property(TARGET ${_batchlas_backends_cuda_lib} PROPERTY CUDA_ARCHITECTURES ${BATCHLAS_CUDA_ARCHITECTURES})
    endif()

    if(BATCHLAS_MATHDX_TARGETS)
        target_link_libraries(${_batchlas_backends_cuda_lib} PRIVATE ${BATCHLAS_MATHDX_TARGETS})
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
    target_link_libraries(${_batchlas_backends_cuda_lib} PRIVATE ${BATCHLAS_CUDA_LINK_LIBRARIES})
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    # PRIVATE for the same reason as the netlib and CUDA links above: these are
    # absolute paths under ${ROCM_PATH} and must not ride out on the export.
    target_link_libraries(${_batchlas_backends_rocm_lib} PRIVATE ${BATCHLAS_ROCM_LINK_LIBRARIES})
endif()

#if(BATCHLAS_HAS_MKL_BACKEND)
 #   list(APPEND BATCHLAS_LINK_LIBRARIES
        #batchlas::mkl
#  )
#endif()

# In monolithic mode `batchlas` is the SHARED library itself, already fully
# configured above. In split mode it is the INTERFACE facade over the 14
# components, and it is the facade that carries the --no-as-needed mitigation.
if(NOT BATCHLAS_MONOLITHIC_LIBRARY)
    add_library(batchlas INTERFACE)
    target_include_directories(batchlas
        INTERFACE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            # Must stay in lockstep with batchlas_configure_component() above and
            # with INCLUDES DESTINATION in cmake/BatchLASPackaging.cmake.
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_features(batchlas INTERFACE cxx_std_20)
    target_link_libraries(batchlas
        INTERFACE
            batchlas_sycl_options
            ${BATCHLAS_COMPONENT_TARGETS}
    )

    # The component .so files reference each other's symbols but record no
    # DT_NEEDED on one another: their symbol graph has 56 elementary cycles in
    # one 9-node strongly connected component (backends, backends_cuda,
    # extensions_{cta,eigen,factorization,stedc,sytrd,tridiag}, extra -- measured
    # by intersecting each .so's undefined symbols with every sibling's defined
    # ones; the "14 cycles" this comment used to claim was an undercount).
    # CMake refuses to express that (cyclic dependencies are only allowed among
    # static libraries). A toolchain defaulting to -Wl,--as-needed therefore
    # drops libraries the link genuinely needs, and the documented
    # `target_link_libraries(app PRIVATE BatchLAS::batchlas)` recipe fails with
    # "undefined reference to batchlas::sycl_gemm::gemm_custom<float>(...)".
    # It keeps every component on the link line; linker options precede link
    # libraries in CMake's link rule, so it covers all of them.
    #
    # BATCHLAS_MONOLITHIC_LIBRARY=ON is what retires this: one .so has no
    # sibling edges at all, so there is nothing for --as-needed to drop and the
    # workaround is deliberately absent from that arm.
    if(UNIX AND NOT APPLE)
        target_link_options(batchlas INTERFACE "-Wl,--no-as-needed")
    endif()
endif()

# The build presets name this target. In monolithic mode the one library IS
# every component, so the name keeps meaning the same thing.
if(BATCHLAS_MONOLITHIC_LIBRARY)
    add_custom_target(batchlas_components DEPENDS batchlas)
else()
    add_custom_target(batchlas_components DEPENDS ${BATCHLAS_COMPONENT_TARGETS})
endif()

if(BATCHLAS_ENABLE_CUBLASDX_WRAPPER)
    target_compile_definitions(batchlas INTERFACE BATCHLAS_ENABLE_CUBLASDX_WRAPPER=1)
endif()

# Create namespaced alias
add_library(BatchLAS::batchlas ALIAS batchlas)

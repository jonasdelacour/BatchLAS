include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(BATCHLAS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/BatchLAS")

function(batchlas_install_package)
    set(_batchlas_install_targets
        batchlas
        batchlas_sycl_options
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

    if(TARGET batchlas_backends_cuda)
        list(APPEND _batchlas_install_targets batchlas_backends_cuda)
    endif()

    if(TARGET batchlas_backends_rocm)
        list(APPEND _batchlas_install_targets batchlas_backends_rocm)
    endif()

    install(TARGETS ${_batchlas_install_targets}
        EXPORT BatchLASTargets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        # The include ROOT, not <root>/batchlas: every public header is spelled
        # <batchlas/...>. Must stay in lockstep with the $<INSTALL_INTERFACE:>
        # genexes in src/CMakeLists.txt - if the two disagree, the exported
        # target carries both dirs and the unprefixed spellings resolve again.
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    # The whole public header tree lives in include/batchlas/ and is copied
    # verbatim to <prefix>/include/batchlas/. Only the directory `batchlas` and
    # the single file `batchlas.hh` are ever created in the consumer's include
    # root, so a consumer header named util/workspace.hh, blas/enums.hh or
    # internal/sort.hh can neither be overwritten by us nor shadow us.
    #
    # This installs the directory rather than include/'s contents on purpose:
    # a future include/foo/ then cannot silently re-squat the include root.
    # examples/consumer_test.sh asserts both halves of that (nothing at
    # <prefix>/include/{blas,util,internal}, everything at
    # <prefix>/include/batchlas/...).
    #
    # FILES_MATCHING keeps non-header files out. The EXCLUDEs drop the benchmark
    # harness: batchlas/util/minibench.hh defines MINI_BENCHMARK_MAIN() ->
    # int main(), and none of minibench.hh, minibench_structured.hh,
    # bench_structured.hh is reachable from <batchlas.hh> or from any other
    # installed header (verified by grep: they are only included by each other
    # and by benchmarks/). PATTERN matches the last path component only, which
    # is all these three need.
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/batchlas"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
            PATTERN "*.hh"
            PATTERN "*.h"
            PATTERN "minibench.hh" EXCLUDE
            PATTERN "minibench_structured.hh" EXCLUDE
            PATTERN "bench_structured.hh" EXCLUDE
    )
    # The umbrella header is the one file that legitimately sits in the include
    # root; install(DIRECTORY) above cannot carry it.
    install(FILES "${PROJECT_SOURCE_DIR}/include/batchlas.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    # tuning_params.hh is deliberately NOT listed here. src/CMakeLists.txt and
    # batchlas_dep_options both put ${PROJECT_SOURCE_DIR}/include ahead of
    # ${PROJECT_BINARY_DIR}/include, so the library is compiled against
    # include/batchlas/tuning_params.hh (364 lines, with the BATCHLAS_TUNE_*
    # runtime-override layer) and the configure_file() copy in the binary dir is
    # never compiled by anything - its own header says so. Installing the binary
    # copy shipped consumers different constants than the .so was built with and
    # silently preempted the library's own inline definitions. The source copy is
    # installed by the install(DIRECTORY) above, which is what we want.
    #
    # These two exist ONLY in the binary tree, so they cannot collide with the
    # source-tree copy of include/batchlas/ installed above. The destination is
    # <prefix>/include/batchlas so that <batchlas/backend_config.h> resolves
    # identically in the build tree and the install tree.
    install(FILES
        "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
        "${PROJECT_BINARY_DIR}/include/batchlas/device_limits.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/batchlas"
    )

    # The BLAS health check writes this when it detects an OpenBLAS whose
    # auto-selected coretype produces wrong double-precision results (see
    # cmake/BatchLASBlasHealthCheck.cmake). Installing it means an installed
    # tree carries its own workaround instead of leaving it in a build dir that
    # gets deleted.
    if(EXISTS "${PROJECT_BINARY_DIR}/batchlas-env.sh")
        install(FILES "${PROJECT_BINARY_DIR}/batchlas-env.sh"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/batchlas")
    endif()

    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/BatchLASConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/BatchLASConfig.cmake"
        INSTALL_DESTINATION "${BATCHLAS_INSTALL_CMAKEDIR}"
    )
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/BatchLASConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
    )

    install(EXPORT BatchLASTargets
        FILE BatchLASTargets.cmake
        NAMESPACE BatchLAS::
        DESTINATION "${BATCHLAS_INSTALL_CMAKEDIR}"
    )
    install(FILES
        "${PROJECT_BINARY_DIR}/BatchLASConfig.cmake"
        "${PROJECT_BINARY_DIR}/BatchLASConfigVersion.cmake"
        DESTINATION "${BATCHLAS_INSTALL_CMAKEDIR}"
    )
endfunction()

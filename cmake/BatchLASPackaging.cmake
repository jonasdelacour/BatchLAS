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
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/batchlas"
    )

    # Headers go under <prefix>/include/batchlas/, not straight into
    # <prefix>/include/. The old destination wrote blas/, util/ and internal/
    # into the consumer's include root - under the default /usr/local prefix
    # that means /usr/local/include/util/env.hh, which can overwrite another
    # package's files and shadows any same-named header the consumer owns.
    # The exported include dir is <prefix>/include/batchlas (see
    # src/CMakeLists.txt), so every existing #include spelling still resolves.
    #
    # FILES_MATCHING keeps non-header files out. The EXCLUDEs drop the benchmark
    # harness: util/minibench.hh defines MINI_BENCHMARK_MAIN() -> int main(),
    # and none of util/minibench.hh, util/minibench_structured.hh,
    # util/bench_structured.hh is reachable from <batchlas.hh> or from any
    # installed header under blas/, util/ or internal/ (verified by grep: they
    # are only included by each other and by benchmarks/).
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/batchlas"
        FILES_MATCHING
            PATTERN "*.hh"
            PATTERN "*.h"
            PATTERN "minibench.hh" EXCLUDE
            PATTERN "minibench_structured.hh" EXCLUDE
            PATTERN "bench_structured.hh" EXCLUDE
    )
    # tuning_params.hh is deliberately NOT listed here. src/CMakeLists.txt and
    # batchlas_dep_options both put ${PROJECT_SOURCE_DIR}/include ahead of
    # ${PROJECT_BINARY_DIR}/include, so the library is compiled against
    # include/batchlas/tuning_params.hh (364 lines, with the BATCHLAS_TUNE_*
    # runtime-override layer) and the configure_file() copy in the binary dir is
    # never compiled by anything - its own header says so. Installing the binary
    # copy shipped consumers different constants than the .so was built with and
    # silently preempted the library's own inline definitions. The source copy is
    # installed by the install(DIRECTORY) above, which is what we want.
    install(FILES
        "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
        "${PROJECT_BINARY_DIR}/include/batchlas/device_limits.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/batchlas/batchlas"
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

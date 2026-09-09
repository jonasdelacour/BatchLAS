include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(BATCHLAS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/BatchLAS")

function(batchlas_install_package)
    # Set by src/CMakeLists.txt: the 14 component .so plus the two conditional
    # vendor ones, or EMPTY when BATCHLAS_MONOLITHIC_LIBRARY is ON. Read rather
    # than re-listed, because the hand-written second copy of these names that
    # used to live here is exactly the thing that drifts.
    get_property(_batchlas_component_targets GLOBAL PROPERTY BATCHLAS_COMPONENT_TARGETS)

    # THE public export set: the one name a consumer should ever spell, plus
    # the INTERFACE target carrying the (all $<BUILD_INTERFACE:>-wrapped) SYCL
    # flags that BatchLAS::batchlas references.
    install(TARGETS batchlas batchlas_sycl_options
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

    # In split mode the component .so ARE the runtime artifacts, so they must
    # still be installed; what changes is that they land in a SECOND export set.
    # BatchLASConfig.cmake includes BatchLASComponentTargets.cmake first and
    # BatchLASTargets.cmake second, because BatchLAS::batchlas' link interface
    # names them and CMake generates an existence check for imported targets
    # that live in another export set. Keeping them out of BatchLASTargets is
    # the point: BatchLAS::batchlas is the only supported name, and the
    # components resolve only as an implementation detail of that one target.
    # None of them is independently usable anyway - every component .so has
    # unresolved symbols in its siblings (56 measured cycles; see
    # src/CMakeLists.txt).
    if(_batchlas_component_targets)
        install(TARGETS ${_batchlas_component_targets}
            EXPORT BatchLASComponentTargets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
    endif()

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
    # WP3c adds export.hh to this list: it is generated by
    # generate_export_header() in src/CMakeLists.txt, every installed public
    # header that carries BATCHLAS_API includes it, and a consumer that cannot
    # find it fails at the FIRST #include rather than at link.
    install(FILES
        "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
        "${PROJECT_BINARY_DIR}/include/batchlas/device_limits.hh"
        "${PROJECT_BINARY_DIR}/include/batchlas/export.hh"
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
    # ExactVersion, not SameMajorVersion, while the major is 0. SemVer promises
    # nothing across 0.x minors, so SameMajorVersion made find_package(BatchLAS
    # 0.1) accept a 0.9 install whose ABI is unrelated - a false promise the
    # SOVERSION (also 0, see batchlas_configure_component) cannot catch either,
    # because every 0.x ships as .so.0. Revisit at 1.0: SameMajorVersion becomes
    # the right answer the moment the major number carries a guarantee.
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/BatchLASConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY ExactVersion
    )

    install(EXPORT BatchLASTargets
        FILE BatchLASTargets.cmake
        NAMESPACE BatchLAS::
        DESTINATION "${BATCHLAS_INSTALL_CMAKEDIR}"
    )
    if(_batchlas_component_targets)
        install(EXPORT BatchLASComponentTargets
            FILE BatchLASComponentTargets.cmake
            NAMESPACE BatchLAS::
            DESTINATION "${BATCHLAS_INSTALL_CMAKEDIR}"
        )
    endif()
    install(FILES
        "${PROJECT_BINARY_DIR}/BatchLASConfig.cmake"
        "${PROJECT_BINARY_DIR}/BatchLASConfigVersion.cmake"
        DESTINATION "${BATCHLAS_INSTALL_CMAKEDIR}"
    )
endfunction()

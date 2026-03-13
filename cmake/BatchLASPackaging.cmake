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
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        PATTERN "batchlas/tuning_params.hh" EXCLUDE
    )
    install(FILES
        "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
        "${PROJECT_BINARY_DIR}/include/batchlas/tuning_params.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/batchlas"
    )

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

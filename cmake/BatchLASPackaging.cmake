include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(BATCHLAS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/BatchLAS")

function(batchlas_install_package)
    install(TARGETS
        batchlas
        batchlas_sycl_options
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

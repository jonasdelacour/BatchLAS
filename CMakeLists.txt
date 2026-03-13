cmake_minimum_required(VERSION 3.14)

include(cmake/BatchLASCompilerBootstrap.cmake)

project(BatchLAS VERSION 0.1.0 LANGUAGES CXX)

include(cmake/BatchLASOptions.cmake)
include(cmake/BatchLASDetectSYCL.cmake)
include(cmake/BatchLASDependencies.cmake)
include(cmake/BatchLASGeneratedHeaders.cmake)
include(cmake/BatchLASTargetHelpers.cmake)
include(cmake/BatchLASPackaging.cmake)

add_subdirectory(src)

if(BATCHLAS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BATCHLAS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
    if(BATCHLAS_ENABLE_TUNING)
        batchlas_enable_tuning_targets()
    endif()
endif()

if(BATCHLAS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

batchlas_install_package()

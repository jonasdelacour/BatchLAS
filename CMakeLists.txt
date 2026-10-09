cmake_minimum_required(VERSION 3.14)

include(cmake/BatchLASCompilerBootstrap.cmake)

project(BatchLAS VERSION 0.1.0 LANGUAGES CXX)

include(cmake/BatchLASCcache.cmake)
include(cmake/BatchLASOptions.cmake)
if(BATCHLAS_SYCL_IMPL_ACPP)
    include(cmake/BatchLASSyclAcpp.cmake)
else()
    include(cmake/BatchLASDetectSYCL.cmake)
endif()
include(cmake/BatchLASDependencies.cmake)
include(cmake/BatchLASGeneratedHeaders.cmake)
include(cmake/BatchLASTargetHelpers.cmake)
include(cmake/BatchLASPackaging.cmake)
include(cmake/BatchLASDocs.cmake)

add_subdirectory(src)

if(BATCHLAS_BUILD_TESTS)
    enable_testing()
    # Before tests/ and python/: both register GPU tests with its helpers.
    include(tests/cmake/BatchLASTestGpus.cmake)
    add_subdirectory(tests)
endif()

if(BATCHLAS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
    add_subdirectory(tools/tune)
    if(BATCHLAS_ENABLE_TUNING)
        batchlas_enable_tuning_targets()
    endif()
endif()

if(BATCHLAS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BATCHLAS_BUILD_DOCS)
    batchlas_add_docs_target()
endif()

batchlas_install_package()

# C++ examples.
#
# This file works two ways:
#
#   * as part of the BatchLAS build     (cmake -DBATCHLAS_BUILD_EXAMPLES=ON ...)
#   * standalone against an installed BatchLAS
#       cmake -S examples/cpp -B build-examples -DCMAKE_PREFIX_PATH=/path/to/install
#
# The standalone path is the one to copy when starting your own project.

cmake_minimum_required(VERSION 3.14)

set(BATCHLAS_EXAMPLE_TARGETS
    01_getting_started
    02_dense_blas
    03_linear_solvers
    04_qr_and_orthogonalization
    05_svd
    06_symmetric_eigensolvers
    07_tridiagonal_reduction
    08_tridiagonal_eigensolvers
    09_sparse_and_iterative
    10_jacobi_relative_accuracy
    11_generators_and_utilities
    12_choosing_a_variant
)

if(TARGET batchlas)
    # In-tree: reuse the project's SYCL/compiler configuration.
    foreach(example ${BATCHLAS_EXAMPLE_TARGETS})
        add_executable(${example} ${example}.cc)
        batchlas_configure_binary_target(${example})
        target_include_directories(${example} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()

    if(BATCHLAS_BUILD_TESTS)
        # The examples check every result they compute, so they are also tests.
        foreach(example ${BATCHLAS_EXAMPLE_TARGETS})
            add_test(NAME example_${example} COMMAND ${example})
            if(BATCHLAS_OPENBLAS_CORETYPE)
                set_tests_properties(example_${example} PROPERTIES
                    ENVIRONMENT "OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
            endif()
        endforeach()
    endif()
else()
    # Standalone: BatchLAS comes from an install tree.
    project(BatchLASExamples LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(BatchLAS REQUIRED)

    foreach(example ${BATCHLAS_EXAMPLE_TARGETS})
        add_executable(${example} ${example}.cc)
        target_link_libraries(${example} PRIVATE BatchLAS::batchlas)
        target_include_directories(${example} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
endif()

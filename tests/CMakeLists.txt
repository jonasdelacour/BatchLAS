# Find a testing framework (like GTest)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Include FetchContent for downloading dependencies
    include(FetchContent)
    
    # Fetch GoogleTest
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )

    FetchContent_MakeAvailable(googletest)
endif()

# Define test executables
set(TEST_TARGETS
    util_span_tests
    util_vector_tests
    util_device_queue_tests
    mempool_tests
    gemm_tests
    syevx_tests
    lanczos_tests
    gemv_tests
    matrix_tests
    matrix_view_tests
    trsm_tests
    ortho_tests
    inverse_tests
    norm_tests
    cond_tests
    ormqr_tests
    ormqr_cta_tests
    ormqr_blocked_tests
    orgqr_tests
    transpose_tests
    syev_tests
    steqr_tests
    trmm_tests
    stedc_tests
    ritz_values_tests
    sytrd_sy2sb_tests
    sytrd_sb2st_tests
    sytrd_cta_tests
    sytrd_blocked_tests
    syev_cta_tests
    syev_blocked_tests
)

# Tests that require CPU SYCL kernel compilation (not just NETLIB)
set(CPU_SYCL_DEPENDENT_TESTS
    minibench_cli_tests
)

# Add CPU SYCL-dependent tests only if CPU kernels will be compiled
if(BATCHLAS_HAS_CPU_TARGET)
    list(APPEND TEST_TARGETS ${CPU_SYCL_DEPENDENT_TESTS})
    message(STATUS "Including CPU SYCL-dependent tests: ${CPU_SYCL_DEPENDENT_TESTS}")
else()
    message(STATUS "Skipping CPU SYCL-dependent tests (no CPU target in fsycl-targets): ${CPU_SYCL_DEPENDENT_TESTS}")
endif()

# Create test executables
foreach(test ${TEST_TARGETS})
    # Add executable
    add_executable(${test} ${test}.cc)
    
    # Link dependencies
    target_link_libraries(${test} PRIVATE 
        batchlas
        GTest::gtest 
        GTest::gtest_main
    )

    if(BATCHLAS_SYCL_TARGETS_STRING)
        target_compile_options(${test} PRIVATE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
        target_link_options(${test} PRIVATE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
    endif()
    
    # Add to CTest
    add_test(NAME ${test} COMMAND ${test})
endforeach()


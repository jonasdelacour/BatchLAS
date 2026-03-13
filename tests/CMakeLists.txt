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
    iluk_tests
    symm_tests
    syrk_tests
    syr2k_tests
    syev_tests
    gesvd_tests
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

function(batchlas_add_test_target test_name)
    add_executable(${test_name} ${test_name}.cc)
    batchlas_configure_binary_target(${test_name}
        GTest::gtest
        GTest::gtest_main
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

foreach(test ${TEST_TARGETS})
    batchlas_add_test_target(${test})
endforeach()

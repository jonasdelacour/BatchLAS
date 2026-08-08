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
    sg_compat_tests
    util_span_tests
    util_vector_tests
    util_device_queue_tests
    device_blas_tests
    mempool_tests
    backend_dispatch_tests
    options_api_tests
    linalg_layer_tests
    gemm_tests
    syevx_tests
    syevx_range_tests
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
    hemm_tests
    herk_tests
    her2k_tests
    syrk_tests
    syr2k_tests
    syev_tests
    gesvd_tests
    gesvdj_cta_tests
    bdsqr_tests
    bdsdc_tests
    steqr_tests
    stebz_tests
    stein_tests
    trmm_tests
    stedc_tests
    ritz_values_tests
    sytrd_sy2sb_tests
    sytrd_sb2st_tests
    sytrd_sb2st_hh_tests
    sytrd_cta_tests
    sytrd_blocked_tests
    syev_cta_tests
    syev_jacobi_cta_tests
    syev_jacobi_blocked_tests
    syev_cta_fused_tests
    syev_blocked_tests
    syev_two_stage_tests
)

set(BATCHLAS_SMOKE_TEST_TARGETS
    util_span_tests
    util_vector_tests
    matrix_tests
    # Workspace sizing is derived from the same allocator arithmetic that the
    # run path uses, so a sizing regression is a silent heap overflow. Keep it
    # in the fast gate rather than only in the full suite.
    mempool_tests
    # Runtime backend dispatch sits under every concise-API call; if it routes
    # to the wrong backend or drops a defaulted argument, everything downstream
    # is wrong in a way that is hard to trace back here.
    backend_dispatch_tests
    options_api_tests
    linalg_layer_tests
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

if(BATCHLAS_TEST_TARGET_SET STREQUAL "smoke")
    set(TEST_TARGETS ${BATCHLAS_SMOKE_TEST_TARGETS})
    message(STATUS "Configuring smoke-test subset: ${TEST_TARGETS}")
elseif(NOT BATCHLAS_TEST_TARGET_SET STREQUAL "all")
    message(FATAL_ERROR "Unsupported BATCHLAS_TEST_TARGET_SET='${BATCHLAS_TEST_TARGET_SET}'. Expected one of: all, smoke")
endif()

# ---------------------------------------------------------------------------
# Test labels, for selective runs during iteration.
#
#   ctest -L eig            # only the eigensolver tests
#   ctest -LE slow          # everything except the known-expensive binaries
#   ctest -L tridiag -LE slow
#
# COMPONENT: which part of the library a test covers. Pick the label matching
# the subsystem you changed; if you touched a shared low-level piece (Queue,
# Matrix, memory pool) run the whole suite.
#
# SLOW: binaries that dominate wall-clock. Keep this list honest — if a test
# grows past ~15 s, label it here rather than letting it bloat the default run.
# ---------------------------------------------------------------------------
set(BATCHLAS_TEST_LABELS_util
    sg_compat_tests util_span_tests util_vector_tests util_device_queue_tests
    mempool_tests matrix_tests matrix_view_tests minibench_cli_tests
    backend_dispatch_tests options_api_tests linalg_layer_tests)
set(BATCHLAS_TEST_LABELS_blas
    device_blas_tests gemm_tests gemv_tests symm_tests hemm_tests herk_tests her2k_tests
    syrk_tests syr2k_tests
    trmm_tests trsm_tests transpose_tests norm_tests cond_tests inverse_tests)
set(BATCHLAS_TEST_LABELS_ortho
    ortho_tests ormqr_tests ormqr_cta_tests ormqr_blocked_tests orgqr_tests)
set(BATCHLAS_TEST_LABELS_tridiag
    steqr_tests stedc_tests stebz_tests stein_tests
    sytrd_sy2sb_tests sytrd_sb2st_tests sytrd_sb2st_hh_tests
    sytrd_cta_tests sytrd_blocked_tests)
set(BATCHLAS_TEST_LABELS_eig
    syev_tests syevx_tests syevx_range_tests syev_cta_tests syev_jacobi_cta_tests
    syev_jacobi_blocked_tests
    syev_cta_fused_tests syev_blocked_tests syev_two_stage_tests
    lanczos_tests ritz_values_tests gesvd_tests gesvdj_cta_tests bdsqr_tests bdsdc_tests)
set(BATCHLAS_TEST_LABELS_sparse
    iluk_tests)

# Binaries that dominate the suite. Measured serially on this machine.
set(BATCHLAS_SLOW_TESTS
    stedc_tests            # heaviest: dense reference solves at n=512/1024
    steqr_tests            # ~135 s, almost all in the NETLIB instantiations
    sytrd_sb2st_tests      # ~70 s, launch-latency bound bulge chase
    sytrd_sb2st_hh_tests   # 48-combo (n, kd) cross product
    syevx_tests
    iluk_tests
    gesvd_tests)

# Map a target to its component label.
function(batchlas_test_component out_var test_name)
    foreach(component util blas ortho tridiag eig sparse)
        list(FIND BATCHLAS_TEST_LABELS_${component} ${test_name} _idx)
        if(NOT _idx EQUAL -1)
            set(${out_var} ${component} PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "unlabelled" PARENT_SCOPE)
endfunction()

function(batchlas_add_test_target test_name)
    add_executable(${test_name} ${test_name}.cc)
    if(test_name STREQUAL "sg_compat_tests")
        # sg_compat_tests uses sg_compat.hh directly — no full batchlas library
        # needed, only the SYCL compiler and include paths.
        batchlas_configure_binary_target(${test_name}
            NO_FACADE
            LIBRARIES
                GTest::gtest
                GTest::gtest_main
        )
    elseif(test_name STREQUAL "util_span_tests")
        batchlas_configure_binary_target(${test_name}
            NO_FACADE
            LIBRARIES
                GTest::gtest
                GTest::gtest_main
                batchlas_util
        )
    elseif(test_name STREQUAL "util_vector_tests")
        batchlas_configure_binary_target(${test_name}
            NO_FACADE
            LIBRARIES
                GTest::gtest
                GTest::gtest_main
                batchlas_util
        )
    elseif(test_name STREQUAL "matrix_tests")
        batchlas_configure_binary_target(${test_name}
            LIBRARIES
                GTest::gtest
                GTest::gtest_main
        )
    else()
        batchlas_configure_binary_target(${test_name}
            LIBRARIES
                GTest::gtest
                GTest::gtest_main
        )
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})

    batchlas_test_component(_component ${test_name})
    set(_labels ${_component})
    list(FIND BATCHLAS_SLOW_TESTS ${test_name} _slow_idx)
    if(NOT _slow_idx EQUAL -1)
        list(APPEND _labels slow)
    endif()
    set_tests_properties(${test_name} PROPERTIES LABELS "${_labels}")

    if(BATCHLAS_OPENBLAS_CORETYPE)
        # The host BLAS was found to compute dgemm incorrectly with its
        # auto-detected kernel. OpenBLAS picks the kernel in its library
        # constructor, before main(), so this has to come from the environment.
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT "OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
    endif()
endfunction()

foreach(test ${TEST_TARGETS})
    batchlas_add_test_target(${test})
endforeach()

if(BATCHLAS_SMOKE_TEST_TARGETS)
    add_custom_target(batchlas_smoke_tests DEPENDS ${BATCHLAS_SMOKE_TEST_TARGETS})
endif()

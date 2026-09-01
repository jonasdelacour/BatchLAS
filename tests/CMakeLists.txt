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
    route_vocabulary_tests
    route_gemm_equivalence_tests
    linalg_layer_tests
    gemm_tests
    syevx_tests
    syevx_range_tests
    lanczos_tests
    gemv_tests
    matrix_tests
    matrix_view_tests
    trsm_tests
    potrf_tests
    geqrf_tests
    getrf_tests
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
    spmm_tests
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
    # The route vocabulary pins the legacy BATCHLAS_*_VARIANT / _PROVIDER
    # spellings onto the new Route type. Those spellings appear in committed
    # benchmark scripts and in the provenance of recorded results, so a silent
    # change of meaning would invalidate measurements rather than fail loudly.
    route_vocabulary_tests
    route_gemm_equivalence_tests
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
    backend_dispatch_tests options_api_tests route_vocabulary_tests
    route_gemm_equivalence_tests linalg_layer_tests)
set(BATCHLAS_TEST_LABELS_blas
    device_blas_tests gemm_tests gemv_tests symm_tests hemm_tests herk_tests her2k_tests
    syrk_tests syr2k_tests
    trmm_tests trsm_tests potrf_tests geqrf_tests getrf_tests transpose_tests norm_tests cond_tests inverse_tests)
set(BATCHLAS_TEST_LABELS_ortho
    ortho_tests ormqr_tests ormqr_cta_tests ormqr_blocked_tests orgqr_tests)
set(BATCHLAS_TEST_LABELS_tridiag
    steqr_tests stedc_tests stebz_tests stein_tests
    sytrd_sy2sb_tests sytrd_sb2st_tests sytrd_sb2st_hh_tests
    sytrd_cta_tests sytrd_blocked_tests)
set(BATCHLAS_TEST_LABELS_eig
    syev_tests syevx_tests syevx_range_tests syev_cta_tests syev_jacobi_cta_tests
    syev_cta_fused_tests syev_blocked_tests syev_two_stage_tests
    lanczos_tests ritz_values_tests gesvd_tests gesvdj_cta_tests bdsqr_tests bdsdc_tests)
set(BATCHLAS_TEST_LABELS_sparse
    iluk_tests spmm_tests)

# Binaries that dominate the suite. Measured serially on this machine.
#
# A test that does NOT dominate must not be in here: `-LE slow` is the iteration
# run, and everything listed is invisible to it. iluk_tests (2.5 s) sat here long
# enough for an API-breaking regression to survive a `-LE slow` verification.
set(BATCHLAS_SLOW_TESTS
    stedc_tests            # 27 s: dense reference solves at n=512/1024
    steqr_tests            # 13 s here with NETLIB compiled in and nothing skipped,
                           # against a ~135 s figure recorded earlier on this box --
                           # kept until the discrepancy is understood
    sytrd_sb2st_tests      # 69 s, launch-latency bound bulge chase
    gesvd_tests)           # 1.8 s here, but 37 of its 92 cases skip in this build
                           # config; not timed in a configuration that runs them
# Dropped after timing (serial, this build, all cases running -- none skipped):
#   iluk_tests             2.5 s, 17 tests
#   syevx_tests            2.7 s, 102 tests
#   sytrd_sb2st_hh_tests   5.8 s, 9 tests

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
    # A few tests reach past BatchLAS' public API, so they have to ask for the
    # requirements BatchLAS deliberately no longer propagates. The backend
    # libraries are now linked PRIVATE into the component .so files (see
    # src/CMakeLists.txt) because PUBLIC pushed absolute -dev paths and a
    # find_dependency(CUDAToolkit) onto every installed consumer.
    #
    # LAPACKE: these two call LAPACKE_* directly for their reference results.
    if(BATCHLAS_HAS_HOST_BACKEND AND test_name MATCHES "^(gesvd_tests|sytrd_sb2st_tests)$")
        target_link_libraries(${test_name} PRIVATE ${BATCHLAS_NETLIB_LINK_LIBRARIES})
    endif()
    # CUDA headers: these #include src/backends/*cublasdx*.hh, which reaches
    # <cuda_runtime_api.h>. CUDA::cudart is wanted here for its include dir.
    if(BATCHLAS_HAS_CUDA_BACKEND AND TARGET CUDA::cudart
       AND test_name MATCHES "^(gemm_tests|syr2k_tests|trmm_tests)$")
        target_link_libraries(${test_name} PRIVATE CUDA::cudart)
    endif()
    # options_api_tests reads options.hh back to check that no backend-deducing
    # overload was added without its pointer check -- a rule no runtime test can
    # enforce for an entry point that does not exist yet.
    if(test_name STREQUAL "options_api_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_OPTIONS_HH_PATH="${PROJECT_SOURCE_DIR}/include/batchlas/blas/options.hh")
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

# ---------------------------------------------------------------------------
# batchlas_coverage: what this build can do, and what a run reached.
#
#   cmake --build build --target batchlas_coverage
#
# Writes build/coverage.csv. Two kinds of row:
#
#   linked,<op>,,<backend>,...   -- STATIC. What is compiled in. Exact, needs no
#                                   GPU and no test run, and is the planning
#                                   input: it names the ops with no native
#                                   kernel at all.
#   miss,<op>,<scalar>,...       -- a call that found no route, with the library
#                                   that would have served it.
#   reached,<op>,...             -- DYNAMIC, per shape_class: the route the
#                                   resolver actually returned. Recorded by any
#                                   build whenever BATCHLAS_COVERAGE_OUT is set,
#                                   which this target sets. No special configure
#                                   step -- see dispatch/coverage.hh for why the
#                                   compile-time option was removed.
#
# scripts/route_diff.sh captures and diffs the `reached` rows, which is how a
# change is held to "no dispatch decision moved".
#
# Run it against a vendor-free build to turn VENDOR_FREE_BASELINE.md's list of
# failing SUITES into a list of missing OPS -- one missing trsm fails a dozen
# suites, and the suite list cannot tell you that.
add_custom_target(batchlas_coverage
    COMMAND ${CMAKE_COMMAND} -E env
            BATCHLAS_COVERAGE_OUT=${CMAKE_BINARY_DIR}/coverage.csv
            ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} -LE slow || ${CMAKE_COMMAND} -E true
    # Each test BINARY writes its own shard. Merging is not optional: a single
    # shared file meant every process truncated the last, so this target used to
    # report the rows of whichever binary finished last.
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/coverage_merge.sh ${CMAKE_BINARY_DIR}/coverage.csv
    COMMAND ${CMAKE_COMMAND} -E echo "coverage table: ${CMAKE_BINARY_DIR}/coverage.csv"
    DEPENDS ${TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)

# ---------------------------------------------------------------------------
# consumer_package_tests: the outside-in guard.
#
# Everything above tests the library from inside its own build tree, where the
# include paths, compile flags and library paths are all handed to it. This one
# installs the just-built library to a throwaway prefix and consumes it the way
# an outside project does -- its own project(), find_package(BatchLAS CONFIG
# REQUIRED), nothing else -- then runs the result and checks the numbers. It is
# the only test that can see a packaging defect: a flag baked into the exported
# interface, an absolute path in the export, a header that does not exist after
# install, or an include root squatted with generic directory names.
#
# The work is in examples/consumer_test.sh so it stays readable and runnable by
# hand; see that file and examples/consumer/README.md.
#
# Labels: 'packaging' (component) and 'slow' (it installs ~250 MB and drives a
# second CMake configure+build), so `ctest -LE slow` -- the default iteration
# loop -- does not pay for it. Not part of the smoke subset either. It skips
# rather than fails when a prerequisite is missing (no SYCL compiler, nothing
# built yet, no device).
# ---------------------------------------------------------------------------
if(BATCHLAS_TEST_TARGET_SET STREQUAL "all")
    find_program(BATCHLAS_BASH_PROGRAM bash)
    if(BATCHLAS_BASH_PROGRAM AND EXISTS "${PROJECT_SOURCE_DIR}/examples/consumer_test.sh")
        add_test(NAME consumer_package_tests
            COMMAND "${BATCHLAS_BASH_PROGRAM}" "${PROJECT_SOURCE_DIR}/examples/consumer_test.sh"
                    --build-dir "${PROJECT_BINARY_DIR}"
                    --example-dir "${PROJECT_SOURCE_DIR}/examples/consumer"
                    --compiler "${CMAKE_CXX_COMPILER}"
                    --cmake "${CMAKE_COMMAND}")
        set_tests_properties(consumer_package_tests PROPERTIES
            LABELS "packaging;slow"
            SKIP_RETURN_CODE 77
            TIMEOUT 1200)
        if(BATCHLAS_OPENBLAS_CORETYPE)
            set_tests_properties(consumer_package_tests PROPERTIES
                ENVIRONMENT "OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
        endif()
    else()
        message(STATUS "Skipping consumer_package_tests (bash or examples/consumer_test.sh not found)")
    endif()
endif()

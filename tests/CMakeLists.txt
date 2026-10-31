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

# Host-only (no GPU slot, tests/cmake/BatchLASTestGpus.cmake); RUN_SERIAL under
# any -j: syev_cta_tests has a concurrency wrong-answer flake (docs/developer/agent-guide.md section 8).
set(BATCHLAS_HOST_ONLY_TESTS tune_tests)
set(BATCHLAS_RUN_SERIAL_TESTS syev_cta_tests)

# Define test executables
set(TEST_TARGETS
    sg_partition_tests
    util_span_tests
    util_vector_tests
    philox_tests
    util_device_queue_tests
    device_blas_tests
    mempool_tests
    backend_dispatch_tests
    options_api_tests
    resident_capacity_tests
    settings_tests
    select_tests
    tuned_tables_tests
    tune_tests
    verify_tests
    tune_race_gpu_tests
    error_model_tests
    linalg_layer_tests
    gemm_tests
    gemm_mask_tests
    iluk_mask_tests
    getrf_mask_tests
    getrs_mask_tests
    syev_mask_tests
    sygvd_mask_tests
    geqrf_mask_tests
    ormqr_mask_tests
    trsm_mask_tests
    level3_mask_tests
    spmm_mask_tests
    sytrf_mask_tests
    potrf_mask_tests
    gesvd_mask_tests
    norm_mask_tests
    ortho_mask_tests
    syevx_mask_tests
    steqr_mask_tests
    stedc_mask_tests
    tridiag_mask_tests
    syevx_tests
    syevx_range_tests
    lanczos_tests
    gemv_tests
    matrix_tests
    matrix_view_tests
    batch_mask_view_tests
    batch_mask_entries_tests
    masked_contract_tests
    trsm_tests
    potrf_tests
    potrf_candidates_tests
    posv_candidates_tests
    trsm_candidates_tests
    symm_candidates_tests
    symm_mirror_tests
    gemm_candidates_tests
    gemv_candidates_tests
    getrf_candidates_tests
    getrs_candidates_tests
    getri_candidates_tests
    gesv_candidates_tests
    geqrf_tests
    geqrf_candidates_tests
    getrf_tests
    vendor_ptr_array_tests
    gesv_tests
    posv_tests
    potri_tests
    potrs_tests
    ortho_tests
    inverse_tests
    norm_tests
    cond_tests
    ormqr_tests
    ormqr_cta_tests
    ormqr_blocked_tests
    ormqr_candidates_tests
    orgqr_tests
    orgqr_candidates_tests
    transpose_tests
    iluk_tests
    spmm_tests
    spmm_candidates_tests
    symm_tests
    hemm_tests
    herk_tests
    her2k_tests
    hermitian_candidates_tests
    syrk_tests
    syr2k_tests
    syr2k_candidates_tests
    syev_tests
    gesvd_tests
    gesvdj_cta_tests
    gesvd_candidates_tests
    bdsqr_tests
    bdsdc_tests
    steqr_tests
    stebz_tests
    stein_tests
    trmm_tests
    trmm_candidates_tests
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
    syev_candidates_tests
    syrk_candidates_tests
    sytrf_tests
    syev_align_tests
    syfn_tests
    sygvd_tests
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
    # settings() decides which kernel runs and whether arguments are checked, for
    # every op at once, and it is read once per process -- so a regression here is
    # not one suite going red, it is the whole suite silently measuring the default
    # arm. Cheap (no kernel launches on the arm that runs by default), so it belongs
    # in the fast gate.
    settings_tests
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

# The tuner's specs time a GPU vendor arm (tools/tune/cell_runner.hh); without a GPU backend
# they reference instantiations that do not exist, so neither the tuner nor this test is built.
if(NOT BATCHLAS_BUILD_TUNE_TOOL)
    list(REMOVE_ITEM TEST_TARGETS tune_race_gpu_tests)
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
#   ctest -L heavy          # only the <name>_heavy case tiers (BATCHLAS_HEAVY_TESTS below)
#   ctest -LE 'slow|heavy'  # the quick iteration tier: -LE slow without the heavy cases
#
# COMPONENT: which part of the library a test covers. Pick the label matching
# the subsystem you changed; if you touched a shared low-level piece (Queue,
# Matrix, memory pool) run the whole suite.
#
# SLOW: binaries that dominate wall-clock. Keep this list honest — if a test
# grows past ~15 s, label it here rather than letting it bloat the default run.
# ---------------------------------------------------------------------------
set(BATCHLAS_TEST_LABELS_util
    sg_partition_tests util_span_tests util_vector_tests util_device_queue_tests philox_tests
    mempool_tests matrix_tests matrix_view_tests batch_mask_view_tests batch_mask_entries_tests masked_contract_tests minibench_cli_tests
    backend_dispatch_tests options_api_tests
    settings_tests error_model_tests
    linalg_layer_tests select_tests tuned_tables_tests tune_tests verify_tests tune_race_gpu_tests)
set(BATCHLAS_TEST_LABELS_blas
    device_blas_tests gemm_tests gemm_mask_tests getrf_mask_tests getrs_mask_tests geqrf_mask_tests ormqr_mask_tests trsm_mask_tests level3_mask_tests potrf_mask_tests gemv_tests symm_tests hemm_tests herk_tests her2k_tests hermitian_candidates_tests
    syrk_tests syr2k_tests symm_candidates_tests symm_mirror_tests syrk_candidates_tests syr2k_candidates_tests
    trmm_tests trmm_candidates_tests trsm_tests potrf_tests potrf_candidates_tests geqrf_tests getrf_tests vendor_ptr_array_tests resident_capacity_tests
    geqrf_candidates_tests
    gesv_tests posv_tests potri_tests potrs_tests posv_candidates_tests trsm_candidates_tests gemm_candidates_tests
    gemv_candidates_tests getrf_candidates_tests getrs_candidates_tests getri_candidates_tests gesv_candidates_tests
    transpose_tests norm_tests cond_tests inverse_tests norm_mask_tests
    sytrf_tests sytrf_mask_tests)
set(BATCHLAS_TEST_LABELS_ortho
    ortho_tests ortho_mask_tests ormqr_tests ormqr_cta_tests ormqr_blocked_tests orgqr_tests
    orgqr_candidates_tests ormqr_candidates_tests)
set(BATCHLAS_TEST_LABELS_tridiag
    steqr_tests stedc_tests stebz_tests stein_tests
    steqr_mask_tests stedc_mask_tests tridiag_mask_tests
    sytrd_sy2sb_tests sytrd_sb2st_tests sytrd_sb2st_hh_tests
    sytrd_cta_tests sytrd_blocked_tests)
set(BATCHLAS_TEST_LABELS_eig
    syev_tests syevx_tests syevx_range_tests syev_cta_tests syev_jacobi_cta_tests
    syev_cta_fused_tests syev_blocked_tests syev_two_stage_tests
    lanczos_tests ritz_values_tests gesvd_tests gesvdj_cta_tests bdsqr_tests bdsdc_tests
    gesvd_candidates_tests syev_candidates_tests
    syev_align_tests syfn_tests sygvd_tests syev_mask_tests sygvd_mask_tests gesvd_mask_tests syevx_mask_tests)
set(BATCHLAS_TEST_LABELS_sparse
    iluk_tests spmm_tests spmm_candidates_tests spmm_mask_tests)

# Binaries that dominate the suite. Measured serially on this machine.
#
# A test that does NOT dominate must not be in here: `-LE slow` is the iteration
# run, and everything listed is invisible to it. iluk_tests (2.5 s) sat here long
# enough for an API-breaking regression to survive a `-LE slow` verification.
set(BATCHLAS_SLOW_TESTS
    steqr_tests            # 13 s here with NETLIB compiled in and nothing skipped,
                           # against a ~135 s figure recorded earlier on this box --
                           # kept until the discrepancy is understood
    )
# Per-case slow tier: <name>_slow runs these cases (label slow), <name> the rest.
# BANDR1 at n=256: the launch-latency bound bulge chase behind most of the binary's 69 s (a
# profile on the 4090 box, not re-timed per case since the split); n=96 stays quick.
set(BATCHLAS_SLOW_CASES_sytrd_sb2st_tests "*BandReductionMatchesDenseSyevSpectrumN256*")
# Dropped after timing (serial, this build, all cases running -- none skipped):
#   iluk_tests             2.5 s, 17 tests
#   syevx_tests            2.7 s, 102 tests
#   sytrd_sb2st_hh_tests   5.8 s, 9 tests
#   stedc_tests            10.7 s (2x4090 gate junit, 2026-10-31; under the 15 s rule)
#   gesvd_tests            3.8 s + 3.9 s for gesvd_tests_heavy (same run)

# Case-level tiers. A binary in BATCHLAS_HEAVY_TESTS is registered twice: <name> runs
# --gtest_filter=-<HEAVY> and <name>_heavy (label heavy) runs --gtest_filter=<HEAVY>.
# One pattern list feeds both filters, so the two are exact complements by construction.
# The heavy tier holds the saturating-batch race guards (agent guide section 8 rule 5) and
# the AutoReads table-routing guards: it is part of the pre-push gate, never optional.
# Deliberately NOT labelled slow: every `-LE slow` run (PR CI, the vendor-free trees,
# route_diff.sh, batchlas_coverage) keeps the cases it ran before the split, and only a
# run that opts in with `-LE 'slow|heavy'` drops them.
# Measured (2x RTX 4090, serial, 2026-10-23 full run): these patterns move 1463 s of case
# time out of the quick tier, which is ~430 s against ~1990 s for `-LE slow` (~335 s
# without the route-native reruns); gemm_candidates_tests keeps 43 s of its 1057 s.
set(BATCHLAS_HEAVY_TEST_PATTERNS
    *Saturating*
    *AutoReads*
    *PinnedCandidatesEveryFormShapeAndLayout*
    *PinnedCandidatesStraddleTheirLimits*
    *CoverageRow*
    *AtEveryK*
    *WideV2ComplexDoubleElementLeg*
    *WideV2TileAndSlabEdgesUnderNaNPoison*
    *BareNativePicksTheBestRunnableNonVendor*
    *ComplexSymmetricSyr*kFamilies*
    *SameAsInOrder*
    *FreshChildAgreesWithAWorkerAfterALargerLaunch*
    *EigenvaluesOnlyLowerMatchesNetlib*
    *GradedHermitianRunsAtN512*
    *PastTheGrid*
    *DoubleBandEvery*
    # Mask-contract cases over every blocked size or composite (sygvd/syev/gesvd mask).
    *AtBlockedSizes*
    *Composites*
    *BlockedOverEach*
    *SygstDeviceFamilies*
    *FactoredCompose*)
string(REPLACE ";" ":" BATCHLAS_HEAVY_TEST_FILTER "${BATCHLAS_HEAVY_TEST_PATTERNS}")
# Binaries with at least one matching case (checked with --gtest_list_tests at c960d7b1). A
# binary missing here runs its heavy cases in its single entry (slower, never lost). A _heavy
# entry that selects no case (a stale line, or cases compiled out in this configuration)
# reports SKIPPED rather than a silent pass.
set(BATCHLAS_HEAVY_TESTS
    gemm_candidates_tests gemm_mask_tests gemm_tests gemv_candidates_tests level3_mask_tests spmm_mask_tests trsm_mask_tests geqrf_candidates_tests geqrf_mask_tests potrf_mask_tests
    gesv_candidates_tests gesvd_candidates_tests getrf_candidates_tests
    getri_candidates_tests getrs_candidates_tests hermitian_candidates_tests matrix_tests
    orgqr_candidates_tests ormqr_candidates_tests ormqr_mask_tests gesvd_mask_tests posv_candidates_tests
    potrf_candidates_tests potrf_tests potri_tests potrs_tests spmm_candidates_tests spmm_tests syev_blocked_tests
    syev_candidates_tests syev_cta_tests symm_candidates_tests syr2k_candidates_tests
    syrk_candidates_tests trmm_candidates_tests trsm_candidates_tests trsm_tests
    tune_race_gpu_tests verify_tests geqrf_tests getrf_tests gesvd_tests ormqr_blocked_tests
    syev_tests sygvd_tests)
list(APPEND BATCHLAS_HEAVY_TESTS sytrf_tests sytrf_mask_tests)
list(APPEND BATCHLAS_HEAVY_TESTS norm_mask_tests ortho_mask_tests)
list(APPEND BATCHLAS_HEAVY_TESTS steqr_mask_tests tridiag_mask_tests)
list(APPEND BATCHLAS_HEAVY_TESTS stedc_mask_tests)
list(APPEND BATCHLAS_HEAVY_TESTS sygvd_mask_tests syev_mask_tests syevx_mask_tests getrs_mask_tests
    getrf_mask_tests)

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
    if(test_name STREQUAL "sg_partition_tests")
        # Header-only: compiles src/kernels/common/sg_partition/ directly and links
        # no batchlas library, only the SYCL compiler and include paths.
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
    elseif(test_name STREQUAL "tune_tests")
        # Host-only: the tuner core (tools/tune/tune_core.cc) and no batchlas library. The
        # round-trip case runs scripts/sweep_to_table.py, so it needs the interpreter's path.
        target_sources(${test_name} PRIVATE "${PROJECT_SOURCE_DIR}/tools/tune/tune_core.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/race.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/grid.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/ledger.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/replay_core.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/schedule.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/spec.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/tiered_driver.cc"
                                            "${PROJECT_SOURCE_DIR}/tools/tune/worker.cc")
        batchlas_configure_binary_target(${test_name} NO_FACADE LIBRARIES GTest::gtest GTest::gtest_main)
        find_package(Python3 QUIET COMPONENTS Interpreter)
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_TUNE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            BATCHLAS_TUNE_PYTHON="$<$<BOOL:${Python3_Interpreter_FOUND}>:${Python3_EXECUTABLE}>")
    elseif(test_name STREQUAL "verify_tests")
        # libbatchlas_verify's kernels run on a GPU slot; MatrixView's constructor lives in the library.
        target_sources(${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/verify_residuals_tests.cc")
        batchlas_configure_binary_target(${test_name} LIBRARIES GTest::gtest GTest::gtest_main batchlas::verify)
    elseif(test_name STREQUAL "tune_race_gpu_tests")
        # The tuner's GPU side: the race and the worker against real kernels (potrf, posv, trsm specs).
        # The worker cases run batchlas_tune_impl, built with the benchmarks.
        # gemm, ormqr and the level-3 four: their batch-ladder hooks (TuneSpecLadder).
        foreach(_src tune_core race grid ledger schedule spec worker nvml_guard potrf_spec posv_spec trsm_spec gemm_spec ormqr_spec
                     syrk_spec symm_spec syr2k_spec trmm_spec)
            target_sources(${test_name} PRIVATE "${PROJECT_SOURCE_DIR}/tools/tune/${_src}.cc")
        endforeach()
        batchlas_configure_binary_target(${test_name} LIBRARIES GTest::gtest GTest::gtest_main batchlas::verify)
        target_link_libraries(${test_name} PRIVATE ${CMAKE_DL_LIBS})  # nvml_guard.cc dlopens libnvidia-ml
        if(BATCHLAS_BUILD_BENCHMARKS)
            target_compile_definitions(${test_name} PRIVATE BATCHLAS_TUNE_IMPL="$<TARGET_FILE:batchlas_tune_impl>")
            add_dependencies(${test_name} batchlas_tune_impl)
        endif()
    elseif(test_name STREQUAL "matrix_tests")
        batchlas_configure_binary_target(${test_name}
            LIBRARIES
                GTest::gtest
                GTest::gtest_main
                batchlas::verify
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
    # batchlas::verify (libbatchlas_verify) brings LAPACKE and BATCHLAS_VERIFY_HAVE_LAPACKE=0|1 to every test.
    # No test calls LAPACKE_* directly: references go through batchlas::verify (reference.hh), and
    # geqrf_tests and getrf_tests use it as an independent oracle (rationale below).
    target_link_libraries(${test_name} PRIVATE batchlas::verify)
    # geqrf_tests compares tau ELEMENTWISE against LAPACKE. NOT because the residual and
    # orthogonality probes are blind to tau -- that was the original rationale and it is
    # measured FALSE: v(j) = 1 is implicit in LAPACK's storage, so for a given stored
    # factor tau is uniquely determined, and armed breaks (store 2v with tau/4; flip
    # LAPACK's beta sign) each turned the RESIDUAL red as well as tau. What LAPACKE adds
    # is an INDEPENDENT ORACLE: every other numerical check in the file measures the
    # kernel against batchlas::verify::form_q, so a convention error shared by the
    # kernel and that oracle passes all of them. evidence: docs/perf/qr.md, "Breaks (h)
    # and (i) correct the rationale D9 was written from". Where no host backend is present
    # BATCHLAS_VERIFY_HAVE_LAPACKE is 0 and the test compiles and skips.
    # getrf_tests compares the tiny tier's ipiv ELEMENTWISE against LAPACKE on
    # UNSTRUCTURED data. Every other pivot oracle in that file is BatchLAS code: the
    # combinatorial expect_piv, or the CTA tier -- which shares getrf_cta_device.hh's
    # lu_cabs1 with the tier under test, so a defect in the pivot METRIC moves both
    # and leaves the tiny-vs-cta comparisons green.
    # util_device_queue_tests checks Device::cuda_compute_capability() against the
    # CUDA runtime's own answer, an oracle independent of the SYCL string it parses.
    if(BATCHLAS_HAS_CUDA_BACKEND AND TARGET CUDA::cudart
       AND test_name STREQUAL "util_device_queue_tests")
        target_link_libraries(${test_name} PRIVATE CUDA::cudart)
        target_compile_definitions(${test_name} PRIVATE BATCHLAS_UTIL_TESTS_HAVE_CUDART=1)
    endif()
    # options_api_tests reads options.hh back to check that no backend-deducing
    # overload was added without its pointer check -- a rule no runtime test can
    # enforce for an entry point that does not exist yet.
    if(test_name STREQUAL "options_api_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_OPTIONS_HH_PATH="${PROJECT_SOURCE_DIR}/include/batchlas/blas/options.hh")
    endif()
    # error_model_tests reads the installed public headers back to check that
    # none of them terminates the host process. cmake/BatchLASPackaging.cmake
    # installs include/batchlas wholesale, so "public" is the whole directory
    # and there is no per-file allowlist a runtime test could consult instead.
    # potrf_tests reads potrf_tiny.cc back to check that the tier names no
    # local_accessor and no group_barrier. Both properties are design invariants
    # with no observable behaviour on this device -- a work-group barrier in a
    # partition kernel is a race and a local_accessor is an occupancy cost, and
    # both stay green in a functional test -- so a source check is the only guard
    # that can fail.
    if(test_name STREQUAL "potrf_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_POTRF_TINY_CC_PATH="${PROJECT_SOURCE_DIR}/src/ops/potrf/kernels/potrf_tiny.cc")
    endif()
    # geqrf's tiny kernel DOES carry a local_accessor -- its reflector apply needs the
    # product tile -- so its source check is the narrower one: no WORK-GROUP barrier and
    # no reduce_over_group. Both would be silently green here (the first is a race, the
    # second adds the static shared that re-opens the 48 KB hole for a NEIGHBOURING
    # kernel), which is why this is a source check and not a behavioural one.
    if(test_name STREQUAL "geqrf_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_GEQRF_TINY_CC_PATH="${PROJECT_SOURCE_DIR}/src/ops/geqrf/kernels/geqrf_tiny.cc")
    endif()
    # getrf_tests reads getrf_tiny.cc back for the same reason potrf_tests reads
    # potrf_tiny.cc: the register-resident tier declares no local memory and no
    # barrier of any scope, and both properties are silent on this device -- a
    # work-group barrier in a partition kernel is a race, and static shared
    # re-opens the 48 KB hole for a NEIGHBOURING kernel.
    if(test_name STREQUAL "getrf_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_GETRF_TINY_CC_PATH="${PROJECT_SOURCE_DIR}/src/ops/getrf/kernels/getrf_tiny.cc")
    endif()
    if(test_name STREQUAL "gesvd_mask_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_GEBRD_CTA_CC_PATH="${PROJECT_SOURCE_DIR}/src/ops/gesvd/gebrd/gebrd_cta.cc")
    endif()
    if(test_name STREQUAL "ormqr_mask_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_ORMQR_CTA_CC_PATH="${PROJECT_SOURCE_DIR}/src/ops/ormqr/kernels/ormqr_cta.cc")
    endif()
    if(test_name STREQUAL "error_model_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_INCLUDE_DIR_PATH="${PROJECT_SOURCE_DIR}/include/batchlas")
    endif()
    # tuned_tables_tests compares the embedded set against tuned/*.txt by name and bytes,
    # so a broken embed step fails instead of leaving nothing to check.
    if(test_name STREQUAL "tuned_tables_tests")
        target_compile_definitions(${test_name} PRIVATE
            BATCHLAS_TUNED_SOURCE_DIR="${PROJECT_SOURCE_DIR}/tuned")
    endif()

    list(FIND BATCHLAS_HOST_ONLY_TESTS ${test_name} _host_only_idx)
    if(_host_only_idx EQUAL -1)
        batchlas_gpu_test_command(_cmd ${test_name})
    else()
        set(_cmd ${test_name})
    endif()
    set(_entries ${test_name})
    set(_heavy OFF)
    if(test_name IN_LIST BATCHLAS_HEAVY_TESTS)
        set(_heavy ON)
    endif()
    if(_heavy)
        add_test(NAME ${test_name} COMMAND ${_cmd} "--gtest_filter=-${BATCHLAS_HEAVY_TEST_FILTER}")
        # The _heavy XML must not collide with <name>.xml under GTEST_OUTPUT=xml:<dir>/:
        # tests/ctest_gpu_env.sh renames it from BATCHLAS_GTEST_XML_STEM, so it needs the
        # launcher even without GPU slots. Not on Apple: SIP strips DYLD_* through /bin/sh.
        set(_heavy_cmd ${_cmd})
        set(_heavy_env_mod "")
        if(NOT BATCHLAS_TEST_GPU_LAUNCHER AND BATCHLAS_TEST_SH AND NOT APPLE)
            set(_heavy_cmd "${BATCHLAS_TEST_SH}" "${PROJECT_SOURCE_DIR}/tests/ctest_gpu_env.sh" ${_cmd})
        elseif(NOT BATCHLAS_TEST_GPU_LAUNCHER AND NOT CMAKE_VERSION VERSION_LESS 3.22)
            # No sh: append the stem to a directory-mode GTEST_OUTPUT in ctest itself. A
            # single-file or unset GTEST_OUTPUT gets a value gtest rejects with a warning.
            set(_heavy_env_mod "GTEST_OUTPUT=string_append:${test_name}_heavy.xml")
        endif()
        add_test(NAME ${test_name}_heavy COMMAND ${_heavy_cmd} "--gtest_filter=${BATCHLAS_HEAVY_TEST_FILTER}")
        list(APPEND _entries ${test_name}_heavy)
    elseif(BATCHLAS_SLOW_CASES_${test_name})
        # Its XML is renamed from BATCHLAS_GTEST_XML_STEM as _heavy's is.
        set(_slow_cmd ${_cmd})
        if(NOT BATCHLAS_TEST_GPU_LAUNCHER AND BATCHLAS_TEST_SH AND NOT APPLE)
            set(_slow_cmd "${BATCHLAS_TEST_SH}" "${PROJECT_SOURCE_DIR}/tests/ctest_gpu_env.sh" ${_cmd})
        endif()
        add_test(NAME ${test_name} COMMAND ${_cmd} "--gtest_filter=-${BATCHLAS_SLOW_CASES_${test_name}}")
        add_test(NAME ${test_name}_slow COMMAND ${_slow_cmd} "--gtest_filter=${BATCHLAS_SLOW_CASES_${test_name}}")
        list(APPEND _entries ${test_name}_slow)
    else()
        add_test(NAME ${test_name} COMMAND ${_cmd})
    endif()

    batchlas_test_component(_component ${test_name})
    set(_labels ${_component})
    list(FIND BATCHLAS_SLOW_TESTS ${test_name} _slow_idx)
    if(NOT _slow_idx EQUAL -1)
        list(APPEND _labels slow)
    endif()
    set(_env "")
    if(BATCHLAS_OPENBLAS_CORETYPE)
        # The host BLAS was found to compute dgemm incorrectly with its
        # auto-detected kernel. OpenBLAS picks the kernel in its library
        # constructor, before main(), so this has to come from the environment.
        list(APPEND _env "OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
    endif()
    foreach(_entry IN LISTS _entries)
        if(_host_only_idx EQUAL -1)
            batchlas_test_needs_gpu(${_entry})
        endif()
        if(test_name IN_LIST BATCHLAS_RUN_SERIAL_TESTS)
            set_tests_properties(${_entry} PROPERTIES RUN_SERIAL TRUE)
        endif()
        if(test_name STREQUAL "sg_partition_tests")
            # A backend that breaks a divergent collective deadlocks rather than
            # failing; the suite needs ~60 s even under an OpenCL CPU JIT.
            set_tests_properties(${_entry} PROPERTIES TIMEOUT 300)
        endif()
        if(test_name STREQUAL "sytrf_tests")
            # NonFiniteAndAliasedItemsTerminate: a lane-divergent pivot decision deadlocks, not fails.
            set_tests_properties(${_entry} PROPERTIES TIMEOUT 1800)
        endif()
    endforeach()
    set_tests_properties(${test_name} PROPERTIES LABELS "${_labels}")
    if(_env)
        set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "${_env}")
    endif()
    if(_heavy)
        set(_heavy_labels ${_labels} heavy)
        list(REMOVE_DUPLICATES _heavy_labels)
        list(APPEND _env "BATCHLAS_GTEST_XML_STEM=${test_name}_heavy")
        set_tests_properties(${test_name}_heavy PROPERTIES
            LABELS "${_heavy_labels}"
            ENVIRONMENT "${_env}"
            SKIP_REGULAR_EXPRESSION "Running 0 tests from 0 test suites")
        if(_heavy_env_mod)
            set_tests_properties(${test_name}_heavy PROPERTIES ENVIRONMENT_MODIFICATION "${_heavy_env_mod}")
        endif()
    elseif(BATCHLAS_SLOW_CASES_${test_name})
        set(_slow_env ${_env} "BATCHLAS_GTEST_XML_STEM=${test_name}_slow")
        set_tests_properties(${test_name}_slow PROPERTIES
            LABELS "${_component};slow"
            ENVIRONMENT "${_slow_env}"
            SKIP_REGULAR_EXPRESSION "Running 0 tests from 0 test suites")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Route-pinned reruns.
#
# The tuned table decides the DEFAULT kernel, so on a vendor-present box a suite for
# an op whose tables rank the vendor first measures the vendor and validates none of
# the native kernel it was written for. That is how spmm's whole transposed arm (two of its three kernel bodies)
# and every one of gemv's 264 cases came to run cuSPARSE/cuBLAS under ctest while
# the coverage table reported a native route available: LINKED IS NOT REACHABLE.
#
# Each entry below reruns one suite with its op pinned native. `native` is select's
# class word: the table walk restricted to native candidates, so an op with several
# native kernels never names one that cannot serve the shape. Labelled route-native
# so `ctest -LE route-native` gets the old, shorter run.
set(BATCHLAS_ROUTE_PINNED_TESTS
    gemm_tests   "BATCHLAS_GEMM_ROUTE=native"
    gemv_tests   "BATCHLAS_GEMV_ROUTE=native"
    potrf_tests  "BATCHLAS_POTRF_ROUTE=native"
    # '|' separates several pins for one suite, NOT ';': a semicolon inside a
    # quoted entry is a list separator, so it flattens into this list and breaks
    # the name/value pairing below.
    # No ORGQR pin on geqrf_tests: its only orgqr facade call holds a ScopedPin,
    # which wins over BATCHLAS_ORGQR_ROUTE; orgqr routing is in orgqr_candidates_tests.
    geqrf_tests  "BATCHLAS_GEQRF_ROUTE=native|BATCHLAS_ORMQR_ROUTE=native"
    orgqr_tests  "BATCHLAS_ORGQR_ROUTE=native|BATCHLAS_ORMQR_ROUTE=native"
    getrf_tests  "BATCHLAS_GETRF_ROUTE=native|BATCHLAS_GETRS_ROUTE=native|BATCHLAS_GETRI_ROUTE=native"
    spmm_tests   "BATCHLAS_SPMM_ROUTE=native"
    symm_tests   "BATCHLAS_SYMM_ROUTE=native"
    syrk_tests   "BATCHLAS_SYRK_ROUTE=native"
    syr2k_tests  "BATCHLAS_SYR2K_ROUTE=native"
    hemm_tests   "BATCHLAS_HEMM_ROUTE=native"
    herk_tests   "BATCHLAS_HERK_ROUTE=native"
    her2k_tests  "BATCHLAS_HER2K_ROUTE=native"
    # No trsm_tests/trmm_tests rerun: their sm_89 and apple_m5_max tables rank a native first
    # in every row (tuned_tables_tests TrmmTrsmRowsRankANativeFirstOnSm89AndAppleM5Max).
)

function(batchlas_add_route_pinned_test test_name env)
    if(NOT TARGET ${test_name})
        return()
    endif()
    set(_pinned "${test_name}_native")
    batchlas_gpu_test_command(_cmd ${test_name})
    add_test(NAME ${_pinned} COMMAND ${_cmd})
    batchlas_test_needs_gpu(${_pinned})
    batchlas_test_component(_component ${test_name})
    string(REPLACE "|" ";" _env "${env}")
    if(BATCHLAS_OPENBLAS_CORETYPE)
        list(APPEND _env "OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
    endif()
    set_tests_properties(${_pinned} PROPERTIES
        LABELS "${_component};route-native"
        ENVIRONMENT "${_env}")
endfunction()

foreach(test ${TEST_TARGETS})
    batchlas_add_test_target(${test})
endforeach()

list(LENGTH BATCHLAS_ROUTE_PINNED_TESTS _rp_len)
math(EXPR _rp_last "${_rp_len} / 2 - 1")
foreach(_i RANGE ${_rp_last})
    math(EXPR _name_i "${_i} * 2")
    math(EXPR _env_i "${_i} * 2 + 1")
    list(GET BATCHLAS_ROUTE_PINNED_TESTS ${_name_i} _rp_name)
    list(GET BATCHLAS_ROUTE_PINNED_TESTS ${_env_i} _rp_env)
    batchlas_add_route_pinned_test(${_rp_name} "${_rp_env}")
endforeach()

# An AdaptiveCpp Metal tree: ACPP_ADAPTIVITY_LEVEL=0 must make a GPU Queue throw (it would JIT
# a TU's double kernels with its float ones). The unset arm runs inside backend_dispatch_tests.
if(BATCHLAS_GPU_FAMILY_RESOLVED STREQUAL "METAL" AND TARGET backend_dispatch_tests)
    batchlas_gpu_test_command(_cmd backend_dispatch_tests)
    add_test(NAME backend_dispatch_tests_adaptivity0
             COMMAND ${_cmd} --gtest_filter=MetalAdaptivityGuard.*)
    batchlas_test_needs_gpu(backend_dispatch_tests_adaptivity0)
    set_tests_properties(backend_dispatch_tests_adaptivity0 PROPERTIES
        LABELS "util" ENVIRONMENT "ACPP_ADAPTIVITY_LEVEL=0")
endif()

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
#                                   selection actually took. Recorded by any
#                                   build whenever BATCHLAS_COVERAGE_OUT is set,
#                                   which this target sets. No special configure
#                                   step -- see src/select/coverage.hh for why the
#                                   compile-time option was removed.
#
# scripts/route_diff.sh captures and diffs the `reached` rows, which is how a
# change is held to "no dispatch decision moved".
#
# Run it against a vendor-free build to turn docs/design/vendor-independence.md#vendor-free-status's list of
# failing SUITES into a list of missing OPS -- one missing trsm fails a dozen
# suites, and the suite list cannot tell you that.
add_custom_target(batchlas_coverage
    # One sh -c word: the `|| true` must not rest on CMake leaving `||` unquoted under VERBATIM.
    COMMAND sh -c "BATCHLAS_COVERAGE_OUT='${CMAKE_BINARY_DIR}/coverage.csv' '${CMAKE_CTEST_COMMAND}' --test-dir '${CMAKE_BINARY_DIR}' -LE slow || true"
    # Each test BINARY writes its own shard. Merging is not optional: a single
    # shared file meant every process truncated the last, so this target used to
    # report the rows of whichever binary finished last.
    COMMAND ${PROJECT_SOURCE_DIR}/scripts/coverage_merge.sh ${CMAKE_BINARY_DIR}/coverage.csv
    COMMAND ${CMAKE_COMMAND} -E echo "coverage table: ${CMAKE_BINARY_DIR}/coverage.csv"
    DEPENDS ${TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)

# Script-mode checks of the NVIDIA architecture recovery in BatchLASDetectSYCL.cmake
# (sycl-ls "Architecture: unknown" -> nvidia_gpu_sm_<cc>, generic-triple rewrite).
# No compiler or device needed.
add_test(NAME cmake_detect_sycl_tests
    COMMAND "${CMAKE_COMMAND}" "-DBATCHLAS_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/detect_sycl_arch_tests.cmake")
set_tests_properties(cmake_detect_sycl_tests PROPERTIES LABELS "util")

# device_calls_tests: scans the built libraries' NVPTX images and fails if any
# kernel calls a SPIR-V builtin (sycl::fma, barriers, id queries) out of line --
# what an icpx build without -ffp-model=precise produces, 6-15x slower and green
# in every functional test. Skips (77) when there is no NVPTX image or no cuobjdump.
# evidence: docs/perf/blackwell.md#device-call-guard
# Not for acpp multipass CUDA: its fat binaries carry cubin only, no PTX to scan.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND NOT BATCHLAS_ACPP_CUDA_AOT_ARCH)
    set(_device_calls_args "${PROJECT_BINARY_DIR}")
    if(CUDAToolkit_BIN_DIR AND EXISTS "${CUDAToolkit_BIN_DIR}/cuobjdump")
        list(APPEND _device_calls_args --cuobjdump "${CUDAToolkit_BIN_DIR}/cuobjdump")
    endif()
    add_test(NAME device_calls_tests
        COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/check_device_calls.py"
                ${_device_calls_args})
    set_tests_properties(device_calls_tests PROPERTIES
        LABELS "util" SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()

# no_raw_subgroup_queries: every sub-group capability check goes through
# Device::supports_sub_group_size, which carries the acpp "exactly {32}" rule.
add_test(NAME no_raw_subgroup_queries
    COMMAND sh -c "! grep -rn 'info::device::sub_group_sizes' '${PROJECT_SOURCE_DIR}/src' | grep -v 'util/queue-impl.cc'")
set_tests_properties(no_raw_subgroup_queries PROPERTIES LABELS "util")

# acpp_jit_tests: the AdaptiveCpp counterpart. acpp JIT-compiles kernels at first launch, so
# scripts/check_acpp_jit.py runs these CUDA cases against a fresh ACPP_APPDB_DIR, then scans the
# PTX the runtime cached: no out-of-line call, every required family reached, no stack array in a
# register-resident family beyond its allowlist. Skips (77) when no PTX appears (no CUDA device).
# evidence: docs/design/sycl-implementations.md#sycl-impl-jit-ptx-check
set(_acpp_jit_bins gemm_candidates_tests gesv_tests potrf_tests trsm_tests syev_cta_fused_tests getrf_tests)
set(_acpp_jit_ok ${BATCHLAS_SYCL_IMPL_ACPP})
foreach(_bin IN LISTS _acpp_jit_bins)
    if(NOT TARGET ${_bin})
        set(_acpp_jit_ok OFF)
    endif()
endforeach()
# Multipass CUDA (cuda:sm_XX) compiles ahead of time: nothing reaches the JIT cache it scans.
if(BATCHLAS_ACPP_CUDA_AOT_ARCH)
    set(_acpp_jit_ok OFF)
endif()
if(Python3_Interpreter_FOUND AND BATCHLAS_HAS_CUDA_BACKEND AND _acpp_jit_ok)
    set(_cuda "BATCHLAS_TEST_BACKEND=CUDA")
    add_test(NAME acpp_jit_tests
        COMMAND ${BATCHLAS_TEST_GPU_LAUNCHER} "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/scripts/check_acpp_jit.py" --build-dir "${PROJECT_BINARY_DIR}"
                --run "${_cuda} BATCHLAS_TEST_FLOAT_TYPE=float '$<TARGET_FILE:gemm_candidates_tests>' --gtest_filter=GemmCandidates/*.PinnedChoiceLaunchesItsOwnKernel"
                --run "${_cuda} BATCHLAS_TEST_FLOAT_TYPE=double '$<TARGET_FILE:gesv_tests>' --gtest_filter=GesvTest/*"
                --run "${_cuda} BATCHLAS_TEST_FLOAT_TYPE=complex '$<TARGET_FILE:potrf_tests>' --gtest_filter=PotrfCtaTest/*:PotrfLpanelTest/*"
                --run "'$<TARGET_FILE:trsm_tests>' --gtest_filter=TrsmNativeSgLeft.*"
                --run "${_cuda} BATCHLAS_TEST_FLOAT_TYPE=complex '$<TARGET_FILE:syev_cta_fused_tests>' --gtest_filter=SyevCtaFusedTest/*"
                --run "${_cuda} '$<TARGET_FILE:getrf_tests>' --gtest_filter=LuTest/*.RegPanel*")
    set_tests_properties(acpp_jit_tests PROPERTIES
        LABELS "util" SKIP_RETURN_CODE 77 TIMEOUT 900)
    batchlas_test_needs_gpu(acpp_jit_tests)
endif()

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
            COMMAND ${BATCHLAS_TEST_GPU_LAUNCHER}
                    "${BATCHLAS_BASH_PROGRAM}" "${PROJECT_SOURCE_DIR}/examples/consumer_test.sh"
                    --build-dir "${PROJECT_BINARY_DIR}"
                    --example-dir "${PROJECT_SOURCE_DIR}/examples/consumer"
                    --compiler "${CMAKE_CXX_COMPILER}"
                    --cmake "${CMAKE_COMMAND}")
        set_tests_properties(consumer_package_tests PROPERTIES
            LABELS "packaging;slow"
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE
            TIMEOUT 1200)
        # RUN_SERIAL: three parallel cmake builds beside GPU tests (OOM, docs/ci.md).
        batchlas_test_needs_gpu(consumer_package_tests)
        if(BATCHLAS_OPENBLAS_CORETYPE)
            set_tests_properties(consumer_package_tests PROPERTIES
                ENVIRONMENT "OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
        endif()
    else()
        message(STATUS "Skipping consumer_package_tests (bash or examples/consumer_test.sh not found)")
    endif()
endif()

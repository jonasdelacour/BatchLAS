# batchlas_tune (docs/design/flat-kernel-selection.md §6). Built with the benchmarks
# (BATCHLAS_BUILD_BENCHMARKS): it is a measuring tool like factor_bench, not part of the library.
# One spec per op in <op>_spec.cc.
add_executable(batchlas_tune_impl
    batchlas_tune.cc
    tune_core.cc
    race.cc
    grid.cc
    ledger.cc
    schedule.cc
    tiered_driver.cc
    worker.cc
    spec.cc
    potrf_spec.cc
    posv_spec.cc
    trsm_spec.cc
    gemm_spec.cc
    syev_spec.cc
    gesvd_spec.cc
    spmm_spec.cc)
batchlas_configure_binary_target(batchlas_tune_impl)
# syev and gesvd check eigen- and singular values against host LAPACKE (host_reference.hh);
# libbatchlas_backends links it PRIVATE, so a direct caller asks for it itself (as the benchmarks do).
if(BATCHLAS_HAS_HOST_BACKEND)
    target_link_libraries(batchlas_tune_impl PRIVATE ${BATCHLAS_NETLIB_LINK_LIBRARIES})
    target_include_directories(batchlas_tune_impl SYSTEM PRIVATE ${BATCHLAS_NETLIB_INCLUDE_DIRS})
    target_compile_definitions(batchlas_tune_impl PRIVATE BATCHLAS_TUNE_HAVE_LAPACKE=1)
endif()
# The §6.4 kernel hash and the spec-list check read sources relative to the repository.
target_compile_definitions(batchlas_tune_impl PRIVATE BATCHLAS_TUNE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# The command users run. It must stay SYCL-free (no batchlas_configure_binary_target): linking
# libbatchlas or libsycl would put a CUDA context on every visible GPU (see launcher.cc).
add_executable(batchlas_tune launcher.cc)
add_dependencies(batchlas_tune batchlas_tune_impl)

# Offline replay of the tiers against a raw sweep (docs/design/tiered-tuning.md). Host-only, like the tests:
# no batchlas_configure_binary_target, no SYCL.
add_executable(tune_replay replay.cc replay_core.cc race.cc grid.cc tune_core.cc)
target_compile_features(tune_replay PRIVATE cxx_std_20)

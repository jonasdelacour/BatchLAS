# batchlas_tune (docs/design/flat-kernel-selection.md §6). Built with the benchmarks
# (BATCHLAS_BUILD_BENCHMARKS) and only with a CUDA or ROCm backend (BATCHLAS_BUILD_TUNE_TOOL in
# the top-level CMakeLists.txt), so a CPU-only or acpp generic configure does not compile it.
# One spec per op in <op>_spec.cc.
add_executable(batchlas_tune_impl
    batchlas_tune.cc
    tune_core.cc
    race.cc
    grid.cc
    ledger.cc
    schedule.cc
    tiered_driver.cc
    replay_core.cc
    worker.cc
    nvml_guard.cc
    spec.cc
    potrf_spec.cc
    posv_spec.cc
    trsm_spec.cc
    gemm_spec.cc
    getrf_spec.cc
    getrs_spec.cc
    getri_spec.cc
    potri_spec.cc
    sygst_spec.cc
    sygvd_spec.cc
    gesv_spec.cc
    gemv_spec.cc
    trmm_spec.cc
    symm_spec.cc
    syrk_spec.cc
    syr2k_spec.cc
    hemm_spec.cc
    herk_spec.cc
    her2k_spec.cc
    syev_spec.cc
    gesvd_spec.cc
    spmm_spec.cc
    geqrf_spec.cc
    orgqr_spec.cc
    ormqr_spec.cc)
# Every spec verifies through batchlas::verify; syev's and gesvd's LAPACKE references come with it.
batchlas_configure_binary_target(batchlas_tune_impl LIBRARIES batchlas::verify)
# The §6.4 kernel hash and the spec-list check read sources relative to the repository.
target_compile_definitions(batchlas_tune_impl PRIVATE BATCHLAS_TUNE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# The guard dlopens libnvidia-ml (nvml_guard.cc); no link-time NVML dependency.
target_link_libraries(batchlas_tune_impl PRIVATE ${CMAKE_DL_LIBS})

# The command users run. It must stay SYCL-free (no batchlas_configure_binary_target): linking
# libbatchlas or libsycl would put a CUDA context on every visible GPU (see launcher.cc).
add_executable(batchlas_tune launcher.cc)
target_include_directories(batchlas_tune PRIVATE "${PROJECT_SOURCE_DIR}")
add_dependencies(batchlas_tune batchlas_tune_impl)
target_compile_definitions(batchlas_tune PRIVATE BATCHLAS_TUNE_OPENBLAS_CORETYPE="${BATCHLAS_OPENBLAS_CORETYPE}")

# Offline replay of the tiers against a raw sweep (docs/design/tiered-tuning.md). Host-only, like the tests:
# no batchlas_configure_binary_target, no SYCL.
add_executable(tune_replay replay.cc replay_core.cc race.cc grid.cc ledger.cc tune_core.cc)
target_compile_features(tune_replay PRIVATE cxx_std_20)
# <batchlas/error.hh> is header-only; export.hh is generated into the build tree.
target_include_directories(tune_replay PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include"
                                               "${PROJECT_BINARY_DIR}/include")

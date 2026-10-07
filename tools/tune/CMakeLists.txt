# batchlas_tune (docs/design/flat-kernel-selection.md §6). Built with the benchmarks
# (BATCHLAS_BUILD_BENCHMARKS): it is a measuring tool like factor_bench, not part of the library.
# One spec per op in <op>_spec.cc.
add_executable(batchlas_tune_impl
    batchlas_tune.cc
    tune_core.cc
    race.cc
    spec.cc
    potrf_spec.cc
    posv_spec.cc
    trsm_spec.cc
    gemm_spec.cc)
batchlas_configure_binary_target(batchlas_tune_impl)
# The §6.4 kernel hash and the spec-list check read sources relative to the repository.
target_compile_definitions(batchlas_tune_impl PRIVATE BATCHLAS_TUNE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# The command users run. It must stay SYCL-free (no batchlas_configure_binary_target): linking
# libbatchlas or libsycl would put a CUDA context on every visible GPU (see launcher.cc).
add_executable(batchlas_tune launcher.cc)
add_dependencies(batchlas_tune batchlas_tune_impl)

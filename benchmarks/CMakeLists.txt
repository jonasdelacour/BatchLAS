set(BENCHMARK_TARGETS
    iluk_convergence_stats
    syevx_iluk_acc
    orthogonality_miniacc
    steqr_acc
    steqr_cta_acc
    stedc_acc
    syev_cta_acc
    syev_blocked_acc
    gesvd_cta_acc
    gesvd_blocked_acc
    gesvd_relacc
    syevx_acc
    band_reduction_benchmark
    bandr1_dump
    geqrf_benchmark
    factor_bench
    gemm_benchmark
    gemm_heterogeneous_benchmark
    gemm_steady_benchmark
    gemm_128x32x32_family_benchmark
    gemm_128x64x32_family_benchmark
    gemm_transpose_benchmark
    gemm_vs_level3_benchmark
    symm_benchmark
    hemm_benchmark
    herk_benchmark
    her2k_benchmark
    syrk_benchmark
    syr2k_benchmark
    sygst_benchmark
    trmm_benchmark
    gemv_benchmark
    spmm_benchmark
    trsm_benchmark
    potrs_benchmark
    ormqr_benchmark
    ormqr_blocked_benchmark
    ormqr_cta_benchmark
    orgqr_benchmark
    syev_benchmark
    syev_cta_benchmark
    syev_jacobi_cta_benchmark
    syev_cta_fused_benchmark
    syev_blocked_benchmark
    gesvd_cta_benchmark
    gesvd_blocked_benchmark
    gesvd_vendor_benchmark
    syevx_benchmark
    lanczos_benchmark
    ortho_benchmark
    steqr_benchmark
    steqr_accuracy
    orthogonality_accuracy
    eigensolver_accuracy
    transpose_benchmark
    permuted_copy_benchmark
    matrix_copy_benchmark
    stehr_benchmark
    stedc_benchmark
    latrd_lower_panel_benchmark
    sytrd_cta_benchmark
    sytrd_blocked_benchmark
    sytrd_sy2sb_benchmark
    sytrd_sb2st_benchmark
    sb2st_hh_benchmark
    syev_two_stage_benchmark
    htev_compare_benchmark
    sytrs_benchmark
)

# Benchmarks ARE part of the default ALL target. BATCHLAS_BUILD_BENCHMARKS
# (default OFF, ON only in the `benchmarks` preset) is the opt-in; marking the
# targets EXCLUDE_FROM_ALL as well meant switching the option on in ccmake
# defined 61 targets that a plain `cmake --build` / `make` never built. To
# iterate on one benchmark, build it by name; `batchlas_benchmarks` still
# builds exactly the benchmark set.
function(batchlas_register_benchmark target_name)
    set_property(GLOBAL APPEND PROPERTY BATCHLAS_ALL_BENCHMARK_TARGETS ${target_name})
endfunction()

function(batchlas_add_benchmark_target target_name)
    add_executable(${target_name} ${target_name}.cc)
    batchlas_configure_binary_target(${target_name})
    # batchlas::verify carries LAPACKE for the accuracy benchmarks' references (libbatchlas_backends
    # links Netlib PRIVATE, so a direct caller must ask for it) and the include path.
    target_link_libraries(${target_name} PRIVATE batchlas::verify)
    batchlas_register_benchmark(${target_name})
endfunction()

foreach(test ${BENCHMARK_TARGETS})
    batchlas_add_benchmark_target(${test})
endforeach()

# Collected via a global property rather than ${BENCHMARK_TARGETS}: every
# batchlas_register_benchmark() target is reachable, whatever list created it.
get_property(_batchlas_all_benchmarks GLOBAL PROPERTY BATCHLAS_ALL_BENCHMARK_TARGETS)
add_custom_target(batchlas_benchmarks DEPENDS ${_batchlas_all_benchmarks})

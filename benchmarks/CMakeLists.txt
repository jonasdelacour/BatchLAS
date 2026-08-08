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
    trmm_benchmark
    gemv_benchmark
    spmm_benchmark
    trsm_benchmark
    ormqr_benchmark
    ormqr_blocked_benchmark
    ormqr_cta_benchmark
    orgqr_benchmark
    syev_benchmark
    syev_cta_benchmark
    syev_jacobi_cta_benchmark
    syev_jacobi_blocked_benchmark
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
    vector_benchmark
    stehr_benchmark
    stedc_benchmark
    latrd_lower_panel_benchmark
    sytrd_cta_benchmark
    sytrd_blocked_benchmark
    sytrd_sy2sb_benchmark
    sytrd_sb2st_benchmark
    sb2st_hh_benchmark
    syev_two_stage_benchmark
    cusolverdx_heev_benchmark
    cusolverdx_htev_benchmark
    htev_compare_benchmark
)

# Benchmarks are kept out of the default ALL target: they are 61 heavy
# translation units (~710 s of compile) that no dev loop and no test run needs,
# and there is a hard ~12 s floor per TU regardless of how trivial the benchmark
# is. Build them with `--target batchlas_benchmarks`, by name, or via the
# `benchmarks` build preset, which names the aggregate explicitly.
function(batchlas_register_benchmark target_name)
    set_target_properties(${target_name} PROPERTIES EXCLUDE_FROM_ALL ON)
    set_property(GLOBAL APPEND PROPERTY BATCHLAS_ALL_BENCHMARK_TARGETS ${target_name})
endfunction()

function(batchlas_add_benchmark_target target_name)
    add_executable(${target_name} ${target_name}.cc)
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

foreach(test ${BENCHMARK_TARGETS})
    batchlas_add_benchmark_target(${test})
endforeach()

function(batchlas_add_device_blas_variant_target target_name source)
    add_executable(${target_name} ${source})
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

function(batchlas_add_device_blas_vector_target target_name mode policy)
    add_executable(${target_name} device_blas_level2_benchmark.cc)
    target_compile_definitions(${target_name} PRIVATE DEVICE_BLAS_VECTOR_MODE=${mode} DEVICE_BLAS_POLICY=${policy} DEVICE_BLAS_BENCHMARK_NAME="${target_name}")
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

function(batchlas_add_device_blas_trmv_variant_target target_name policy uplo trans diag)
    add_executable(${target_name} device_blas_level2_benchmark.cc)
    target_compile_definitions(${target_name} PRIVATE
        DEVICE_BLAS_VECTOR_MODE=2
        DEVICE_BLAS_POLICY=${policy}
        DEVICE_BLAS_BENCHMARK_NAME="${target_name}"
        DEVICE_BLAS_TRMV_UPLO=${uplo}
        DEVICE_BLAS_TRMV_TRANS=${trans}
        DEVICE_BLAS_TRMV_DIAG=${diag})
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

function(batchlas_add_device_blas_matrix_target target_name mode policy)
    add_executable(${target_name} device_blas_level3_benchmark.cc)
    target_compile_definitions(${target_name} PRIVATE DEVICE_BLAS_MATRIX_MODE=${mode} DEVICE_BLAS_POLICY=${policy} DEVICE_BLAS_BENCHMARK_NAME="${target_name}")
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

function(batchlas_add_device_blas_complex_matrix_target target_name mode policy)
    add_executable(${target_name} device_blas_level3_benchmark.cc)
    target_compile_definitions(${target_name} PRIVATE
        DEVICE_BLAS_MATRIX_MODE=${mode}
        DEVICE_BLAS_POLICY=${policy}
        DEVICE_BLAS_MATRIX_COMPLEX=1
        DEVICE_BLAS_BENCHMARK_NAME="${target_name}")
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

function(batchlas_add_device_blas_rank2k_target target_name policy trans)
    add_executable(${target_name} device_blas_level3_benchmark.cc)
    target_compile_definitions(${target_name} PRIVATE
        DEVICE_BLAS_MATRIX_MODE=5
        DEVICE_BLAS_POLICY=${policy}
        DEVICE_BLAS_MATRIX_TRANS=${trans}
        DEVICE_BLAS_BENCHMARK_NAME="${target_name}")
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

function(batchlas_add_device_blas_her2k_target target_name policy trans)
    add_executable(${target_name} device_blas_level3_benchmark.cc)
    target_compile_definitions(${target_name} PRIVATE
        DEVICE_BLAS_MATRIX_MODE=5
        DEVICE_BLAS_POLICY=${policy}
        DEVICE_BLAS_MATRIX_TRANS=${trans}
        DEVICE_BLAS_MATRIX_COMPLEX=1
        DEVICE_BLAS_MATRIX_HERMITIAN=1
        DEVICE_BLAS_BENCHMARK_NAME="${target_name}")
    batchlas_configure_binary_target(${target_name})
    batchlas_register_benchmark(${target_name})
endfunction()

if(BATCHLAS_BUILD_DEVICE_BLAS_BENCHMARKS)
    batchlas_add_device_blas_vector_target(device_blas_gemv_auto_benchmark 1 0)
    batchlas_add_device_blas_vector_target(device_blas_gemv_generic_benchmark 1 1)
    batchlas_add_device_blas_vector_target(device_blas_gemv_sg16_benchmark 1 2)
    batchlas_add_device_blas_vector_target(device_blas_gemv_sg32_benchmark 1 3)
    batchlas_add_device_blas_vector_target(device_blas_trmv_auto_benchmark 2 0)
    batchlas_add_device_blas_trmv_variant_target(device_blas_trmv_trans_auto_benchmark 0 Lower Trans NonUnit)
    batchlas_add_device_blas_trmv_variant_target(device_blas_trmv_trans_upper_auto_benchmark 0 Upper Trans NonUnit)
    batchlas_add_device_blas_trmv_variant_target(device_blas_trmv_trans_upper_unit_auto_benchmark 0 Upper Trans Unit)
    batchlas_add_device_blas_vector_target(device_blas_symv_auto_benchmark 3 0)

    batchlas_add_device_blas_matrix_target(device_blas_trmm_auto_benchmark 1 0)
    batchlas_add_device_blas_matrix_target(device_blas_trmm_generic_benchmark 1 1)
    batchlas_add_device_blas_matrix_target(device_blas_symm_auto_benchmark 2 0)
    batchlas_add_device_blas_matrix_target(device_blas_symm_generic_benchmark 2 1)
    batchlas_add_device_blas_complex_matrix_target(device_blas_symm_cfloat_auto_benchmark 2 0)
    batchlas_add_device_blas_complex_matrix_target(device_blas_symm_cfloat_generic_benchmark 2 1)
    batchlas_add_device_blas_matrix_target(device_blas_gemm_auto_benchmark 3 0)
    batchlas_add_device_blas_matrix_target(device_blas_gemm_generic_benchmark 3 1)
    batchlas_add_device_blas_complex_matrix_target(device_blas_gemm_cfloat_auto_benchmark 3 0)
    batchlas_add_device_blas_complex_matrix_target(device_blas_gemm_cfloat_generic_benchmark 3 1)
    batchlas_add_device_blas_matrix_target(device_blas_ger_auto_benchmark 4 0)
    batchlas_add_device_blas_matrix_target(device_blas_ger_generic_benchmark 4 1)
    batchlas_add_device_blas_matrix_target(device_blas_ger_sg16_benchmark 4 2)
    batchlas_add_device_blas_matrix_target(device_blas_ger_sg32_benchmark 4 3)
    batchlas_add_device_blas_rank2k_target(device_blas_syr2k_auto_benchmark 0 0)
    batchlas_add_device_blas_rank2k_target(device_blas_syr2k_generic_benchmark 1 0)
    batchlas_add_device_blas_rank2k_target(device_blas_syr2k_sg16_benchmark 2 0)
    batchlas_add_device_blas_rank2k_target(device_blas_syr2k_sg32_benchmark 3 0)
    batchlas_add_device_blas_rank2k_target(device_blas_syr2k_trans_auto_benchmark 0 1)
    batchlas_add_device_blas_rank2k_target(device_blas_syr2k_trans_generic_benchmark 1 1)
    batchlas_add_device_blas_her2k_target(device_blas_her2k_auto_benchmark 0 0)
    batchlas_add_device_blas_her2k_target(device_blas_her2k_generic_benchmark 1 0)
    batchlas_add_device_blas_her2k_target(device_blas_her2k_conjtrans_auto_benchmark 0 2)
    batchlas_add_device_blas_her2k_target(device_blas_her2k_conjtrans_generic_benchmark 1 2)
endif()

# Collected via a global property rather than ${BENCHMARK_TARGETS} so the
# device_blas_* targets above, which are created after the main foreach and only
# under BATCHLAS_BUILD_DEVICE_BLAS_BENCHMARKS, are reachable too.
get_property(_batchlas_all_benchmarks GLOBAL PROPERTY BATCHLAS_ALL_BENCHMARK_TARGETS)
add_custom_target(batchlas_benchmarks DEPENDS ${_batchlas_all_benchmarks})

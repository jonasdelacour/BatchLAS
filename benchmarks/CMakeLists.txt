set(BENCHMARK_TARGETS
    bandr1_dump
    geqrf_benchmark
    gemm_benchmark
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
    syev_blocked_benchmark
    syevx_benchmark
    lanczos_benchmark
    ortho_benchmark
    steqr_benchmark
    steqr_cta_benchmark
    steqr_accuracy
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
)

foreach(test ${BENCHMARK_TARGETS})
    add_executable(${test} ${test}.cc)
    target_link_libraries(${test} PRIVATE batchlas)
    if(BATCHLAS_SYCL_TARGETS_STRING)
        target_compile_options(${test} PRIVATE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
        target_link_options(${test} PRIVATE -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING})
    endif()
    #add_test(NAME ${test} COMMAND ${test})
endforeach()


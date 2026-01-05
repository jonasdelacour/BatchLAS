set(BENCHMARK_TARGETS
    geqrf_benchmark
    gemm_benchmark
    trmm_benchmark
    gemv_benchmark
    spmm_benchmark
    trsm_benchmark
    ormqr_benchmark
    orgqr_benchmark
    syev_benchmark
    syev_cta_benchmark
    syevx_benchmark
    lanczos_benchmark
    ortho_benchmark
    steqr_benchmark
    transpose_benchmark
    permuted_copy_benchmark
    matrix_copy_benchmark
    vector_benchmark
    stehr_benchmark
    stedc_benchmark
    steqr_cta_benchmark
    sytrd_cta_benchmark
    ormqr_cta_benchmark
)

foreach(test ${BENCHMARK_TARGETS})
    add_executable(${test} ${test}.cc)
    target_link_libraries(${test} PRIVATE batchlas)
    #add_test(NAME ${test} COMMAND ${test})
endforeach()


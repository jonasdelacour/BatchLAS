set(BENCHMARK_TARGETS
    geqrf_benchmark
    gemm_benchmark
    gemv_benchmark
    spmm_benchmark
    trsm_benchmark
    ormqr_benchmark
    orgqr_benchmark
    syev_benchmark
    syevx_benchmark
    lanczos_benchmark
    gemm_custom
)

foreach(test ${BENCHMARK_TARGETS})
    add_executable(${test} ${test}.cc)
    target_link_libraries(${test} PRIVATE batchlas)
    #add_test(NAME ${test} COMMAND ${test})
endforeach()


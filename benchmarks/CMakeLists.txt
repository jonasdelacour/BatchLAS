# Find a testing framework (like GTest)
find_package(GTest QUIET)
find_package(benchmark QUIET)
if(NOT GTest_FOUND)
    # Include FetchContent for downloading dependencies
    include(FetchContent)
    
    # Fetch GoogleTest
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    
    # Configure GoogleTest to build shared libraries
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Define test executables
set(BENCHMARK_TARGETS
    geqrf_benchmark
    gemm_benchmark
    gemv_benchmark
    spmm_benchmark
    trsm_benchmark
    ormqr_benchmark
)

set(MINI_BENCH_TARGETS
    gemm_custom
)

# Create test executables
foreach(test ${BENCHMARK_TARGETS})
    # Add executable
    add_executable(${test} ${test}.cc)
    
    # Link dependencies
    target_link_libraries(${test} PRIVATE 
        batchlas
        GTest::gtest 
        GTest::gtest_main
        benchmark::benchmark
    )
    
    # Add to CTest
    add_test(NAME ${test} COMMAND ${test})
endforeach()

foreach(test ${MINI_BENCH_TARGETS})
    add_executable(${test} ${test}.cc)
    target_link_libraries(${test} PRIVATE
        batchlas
    )
    add_test(NAME ${test} COMMAND ${test})
endforeach()


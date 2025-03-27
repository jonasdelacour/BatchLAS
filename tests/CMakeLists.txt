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
    
    # Configure GoogleTest to build shared libraries
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Define test executables
set(TEST_TARGETS
    util_span_tests
    util_vector_tests
    util_device_queue_tests
    blas_operations_tests
    syevx_tests
)

# Create test executables
foreach(test ${TEST_TARGETS})
    # Add executable
    add_executable(${test} ${test}.cc)
    
    # Link dependencies
    target_link_libraries(${test} PRIVATE 
        batchlas
        GTest::gtest 
        GTest::gtest_main
    )
    
    # Add to CTest
    add_test(NAME ${test} COMMAND ${test})
endforeach()
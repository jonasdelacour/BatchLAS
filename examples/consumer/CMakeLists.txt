# A STANDALONE consumer of an installed BatchLAS.
#
# This project is deliberately NOT add_subdirectory()'d into the BatchLAS build:
# it has its own project() and finds BatchLAS the way an outside team would, via
# find_package(BatchLAS CONFIG REQUIRED) against an install prefix. That is the
# whole point -- everything it needs must come out of the installed package, so
# a packaging regression fails here instead of in someone's evaluation.
#
# Configure it like this (see README.md for why each line is load-bearing):
#
#   cmake -S examples/consumer -B build-consumer \
#         -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
#         -DCMAKE_PREFIX_PATH=<batchlas-install-prefix>
#   cmake --build build-consumer
#   LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:<prefix>/lib ./build-consumer/hello_batched_gemm
#
# consumer_test.sh (one directory up) does exactly that, from CTest.

cmake_minimum_required(VERSION 3.21)
project(batchlas_consumer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BatchLAS CONFIG REQUIRED)

add_executable(hello_batched_gemm main.cc)
target_link_libraries(hello_batched_gemm PRIVATE BatchLAS::batchlas)

# BatchLAS's SYCL compile/link flags are build-tree-only ($<BUILD_INTERFACE:>),
# so a consumer that submits its own kernels has to ask for -fsycl itself. This
# example submits none -- it only calls into the library -- and compiles clean
# without the flag, but it is enabled by default because that is the recipe a
# real consumer needs, and because it is what keeps this example honest about
# the SYCL runtime being on the link line.
option(BATCHLAS_CONSUMER_USE_FSYCL "Compile and link the example with -fsycl" ON)
if(BATCHLAS_CONSUMER_USE_FSYCL)
    target_compile_options(hello_batched_gemm PRIVATE -fsycl)
    target_link_options(hello_batched_gemm PRIVATE -fsycl)
endif()

# The include-root squatting guard. decoy_include/ holds three headers named
# blas/enums.hh, util/workspace.hh and internal/ormqr_blocked.hh, placed where a
# consumer's own headers go, i.e. AHEAD of anything an imported target propagates
# (a target's own include dirs are ordered before propagated ones, and imported
# targets propagate theirs as -isystem). Each is an #error, so reaching any of
# them fails the build with a named sentinel.
#
# BatchLAS' public headers live under include/batchlas/ and are spelled
# <batchlas/...>, so this build succeeds. consumer_test.sh builds the example a
# second time with this ON and FAILS if a sentinel fires. It stays off by
# default so the primary build stays a plain, copyable example.
option(BATCHLAS_CONSUMER_DECOY "Put colliding blas/, util/ and internal/ headers on the include path" OFF)
if(BATCHLAS_CONSUMER_DECOY)
    target_include_directories(hello_batched_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/decoy_include")
endif()

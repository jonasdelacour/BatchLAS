include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
    message(WARNING "CMAKE_CXX_COMPILER does not appear to be a SYCL compiler (icpx/clang++). Build may fail.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BATCHLAS_BUILD_TESTS "Build BatchLAS tests" ON)
option(BATCHLAS_BUILD_BENCHMARKS "Build BatchLAS benchmarks" OFF)
option(BATCHLAS_BUILD_EXAMPLES "Build BatchLAS examples" OFF)
option(BATCHLAS_BUILD_DOCS "Build BatchLAS documentation" OFF)
option(BATCHLAS_ENABLE_CUDA "Enable CUDA support" OFF)
option(BATCHLAS_DISABLE_CUDA_FTZ "Disable flush-to-zero (FTZ) for CUDA device code" ON)
option(BATCHLAS_ENABLE_OPENMP "Enable OpenMP support" OFF)
option(BATCHLAS_ENABLE_ROCM "Enable ROCm support even when no AMD GPU is detected" OFF)
option(BATCHLAS_BUILD_PYTHON "Build Python bindings" OFF)
option(BATCHLAS_ENABLE_NETLIB "Enable Netlib BLAS/LAPACK backend" ON)
option(BATCHLAS_ENABLE_MKL "Enable oneMKL backend" OFF)
option(BATCHLAS_ENABLE_CPU_TESTS "Enable tests/benchmarks requiring CPU SYCL kernel compilation" ON)
option(BATCHLAS_KEEP_CUDA_INTERMEDIATES "Preserve CUDA and SYCL CUDA device compilation intermediates for PTX/SASS analysis" OFF)
option(BATCHLAS_SANITIZER_FRIENDLY_DEBUG "Use more unwind-friendly debug flags for sanitizer runs (may slow down builds/runs)" OFF)
option(BATCHLAS_ENABLE_TUNING "Enable BatchLAS tuning targets (requires Python and benchmarks)" OFF)

set(BATCHLAS_MATHDX_ROOT "" CACHE PATH "Path to an unpacked NVIDIA MathDx package root")
set(BATCHLAS_CPU_TARGET "auto" CACHE STRING "CPU SYCL target override: auto|native_cpu|spir64_x86_64|none")
set(BATCHLAS_TEST_TARGET_SET "all" CACHE STRING "Subset of tests to generate: all|smoke")
set(BATCHLAS_TUNING_PROFILE "" CACHE FILEPATH "Optional tuning profile JSON to generate compile-time tuning constants")
set(BATCHLAS_AMD_ARCH "amd_gpu_gfx942" CACHE STRING "AMD GPU architecture for ROCm")
set(BATCHLAS_NVIDIA_ARCH "sm_50" CACHE STRING "NVIDIA GPU architecture for CUDA")

set_property(CACHE BATCHLAS_CPU_TARGET PROPERTY STRINGS auto native_cpu spir64_x86_64 none)
set_property(CACHE BATCHLAS_TEST_TARGET_SET PROPERTY STRINGS all smoke)

set(_BATCHLAS_DEFAULT_NATIVE_CPU_DISABLE_VECZ OFF)
if(CMAKE_CXX_COMPILER MATCHES "/opt/dpcpp-cuda")
    set(_BATCHLAS_DEFAULT_NATIVE_CPU_DISABLE_VECZ ON)
endif()
option(BATCHLAS_NATIVE_CPU_DISABLE_VECZ "Disable vecz for native_cpu backend (workaround for clang crash)" ${_BATCHLAS_DEFAULT_NATIVE_CPU_DISABLE_VECZ})

set(BATCHLAS_HAS_HOST_BACKEND FALSE)
set(BATCHLAS_HAS_MKL_BACKEND FALSE)
set(BATCHLAS_HAS_CUDA_BACKEND FALSE)
set(BATCHLAS_HAS_ROCM_BACKEND FALSE)
set(BATCHLAS_HAS_CPU_TARGET FALSE)
set(BATCHLAS_ENABLE_SYCL ON)

message(STATUS "SYCL support is mandatory for BatchLAS")

add_library(batchlas_build_options INTERFACE)
add_library(batchlas_sycl_options INTERFACE)
add_library(batchlas_sycl_no_cpu_options INTERFACE)
add_library(batchlas_dep_options INTERFACE)

if(BATCHLAS_SANITIZER_FRIENDLY_DEBUG)
    message(STATUS "Enabling sanitizer-friendly debug flags (-g, no-inline, keep frame pointers)")
    set(_BATCHLAS_DEBUG_COMPILE_OPTIONS
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-fno-omit-frame-pointer>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-fno-optimize-sibling-calls>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-fno-inline>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O1>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-fno-omit-frame-pointer>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-fno-optimize-sibling-calls>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-fno-inline>
    )
else()
    set(_BATCHLAS_DEBUG_COMPILE_OPTIONS
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-gline-tables-only>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-fno-omit-frame-pointer>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-gline-tables-only>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-fno-omit-frame-pointer>
    )
endif()

target_compile_options(batchlas_build_options INTERFACE
    ${_BATCHLAS_DEBUG_COMPILE_OPTIONS}
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:-Os>
)
target_compile_definitions(batchlas_build_options INTERFACE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:NDEBUG>
    $<$<CONFIG:MinSizeRel>:NDEBUG>
)

target_include_directories(batchlas_dep_options INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:/opt/include>
)

set(ONEDPL_ROOT "/opt/intel/oneapi/dpl/latest")
if(EXISTS "${ONEDPL_ROOT}/include/oneapi/dpl")
    message(STATUS "Adding oneDPL include dir: ${ONEDPL_ROOT}/include")
    target_include_directories(batchlas_dep_options INTERFACE
        $<BUILD_INTERFACE:${ONEDPL_ROOT}/include>
    )
endif()

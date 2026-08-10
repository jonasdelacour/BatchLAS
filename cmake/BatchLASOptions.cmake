include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
    message(WARNING "CMAKE_CXX_COMPILER does not appear to be a SYCL compiler (icpx/clang++). Build may fail.")
endif()

# Is BatchLAS the top-level project, or was it pulled in with add_subdirectory()
# / FetchContent? Anything that writes global state has to be conditional on
# this. PROJECT_IS_TOP_LEVEL would be the idiomatic spelling but needs CMake
# 3.21 and this project declares 3.14.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(BATCHLAS_IS_TOP_LEVEL ON)
else()
    set(BATCHLAS_IS_TOP_LEVEL OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# batchlas_build_options appends -gline-tables-only for RelWithDebInfo, which is
# what BatchLAS actually wants; the toolchain default '-g' that precedes it on
# the command line only inflates the device-code debug sections, and full DWARF
# in the SYCL device images has been observed to fail CUDA JIT program builds.
# This used to be gated on the compiler living at a hardcoded /opt/dpcpp-cuda,
# which made build correctness depend on one developer's install path.
option(BATCHLAS_STRIP_RELWITHDEBINFO_G
    "Drop '-g' from CMAKE_CXX_FLAGS_RELWITHDEBINFO for BatchLAS' own sources (-gline-tables-only is added instead)"
    ON)
if(BATCHLAS_STRIP_RELWITHDEBINFO_G AND CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES "(^| )-g($| )")
    string(REGEX REPLACE "(^| )-g($| )" " " _batchlas_relwithdebinfo_flags "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REGEX REPLACE " +" " " _batchlas_relwithdebinfo_flags "${_batchlas_relwithdebinfo_flags}")
    string(STRIP "${_batchlas_relwithdebinfo_flags}" _batchlas_relwithdebinfo_flags)
    # Deliberately a directory-scoped set(), not CACHE ... FORCE: under
    # add_subdirectory()/FetchContent a cache write here rewrites the
    # *consuming* project's global flags. Directory scope still covers
    # src/, tests/, benchmarks/ and python/.
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${_batchlas_relwithdebinfo_flags}")
    message(STATUS "Removed '-g' from RelWithDebInfo flags (BATCHLAS_STRIP_RELWITHDEBINFO_G=ON)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default OFF when BatchLAS is a subproject: 55 GoogleTest binaries in the
# consuming project's default target is never what an integrator asked for, and
# without a system GTest the FetchContent clone makes an offline configure fail.
option(BATCHLAS_BUILD_TESTS "Build BatchLAS tests" ${BATCHLAS_IS_TOP_LEVEL})
option(BATCHLAS_BUILD_BENCHMARKS "Build BatchLAS benchmarks" OFF)
option(BATCHLAS_BUILD_DEVICE_BLAS_BENCHMARKS "Build device_blas_* benchmark targets" OFF)
option(BATCHLAS_BUILD_EXAMPLES "Build BatchLAS examples" OFF)
option(BATCHLAS_BUILD_DOCS "Build BatchLAS documentation" OFF)
# Tri-state on purpose. This used to be option(... OFF), and
# BatchLASDetectSYCL.cmake then force-promoted it to ON in PARENT_SCOPE whenever
# sycl-ls reported a CUDA device - a normal variable shadowing the cache entry,
# so -DBATCHLAS_ENABLE_CUDA=OFF was silently ignored while CMakeCache.txt kept
# reporting OFF forever. AUTO is the honest spelling of the old behaviour; ON now
# means "require it" and errors out instead of quietly configuring for sm_50.
#
# Migrate the legacy entry first. Every build directory configured before this
# change holds BATCHLAS_ENABLE_CUDA:BOOL=OFF -- that was the option() default,
# and it was never written back even on a CUDA box, because the promotion to ON
# happened in a normal variable. Left alone, the set() below would not touch an
# existing entry, so re-configuring such a tree would normalise that stale OFF
# into a genuine "no CUDA backend": no cuBLAS/cuSOLVER, no error, no warning,
# and a CMakeCache.txt reading exactly what it read before. The cache *type* is
# the discriminator: only option() writes BOOL here, while -DBATCHLAS_ENABLE_CUDA=...
# on a fresh configure lands as UNINITIALIZED and becomes STRING below, so this
# runs exactly once per build tree.
get_property(_batchlas_cuda_cache_type CACHE BATCHLAS_ENABLE_CUDA PROPERTY TYPE)
if(_batchlas_cuda_cache_type STREQUAL "BOOL")
    set(_batchlas_cuda_legacy "${BATCHLAS_ENABLE_CUDA}")
    if(_batchlas_cuda_legacy)
        set(_batchlas_cuda_migrated "ON")
    else()
        set(_batchlas_cuda_migrated "AUTO")
    endif()
    set(BATCHLAS_ENABLE_CUDA "${_batchlas_cuda_migrated}" CACHE STRING
        "CUDA backend: AUTO (enable when the SYCL runtime exposes a CUDA device), ON (require it), OFF"
        FORCE)
    message(STATUS
        "Migrated the pre-tri-state BATCHLAS_ENABLE_CUDA:BOOL=${_batchlas_cuda_legacy} cache entry to "
        "${_batchlas_cuda_migrated}; pass -DBATCHLAS_ENABLE_CUDA=OFF to keep the CUDA backend off")
    unset(_batchlas_cuda_migrated)
    unset(_batchlas_cuda_legacy)
endif()
unset(_batchlas_cuda_cache_type)
set(BATCHLAS_ENABLE_CUDA "AUTO" CACHE STRING
    "CUDA backend: AUTO (enable when the SYCL runtime exposes a CUDA device), ON (require it), OFF")
set_property(CACHE BATCHLAS_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
option(BATCHLAS_DISABLE_CUDA_FTZ "Disable flush-to-zero (FTZ) for CUDA device code" ON)
option(BATCHLAS_CUDA_DEVICE_LINE_INFO
    "Pass --generate-line-info to the NVPTX backend in Debug/RelWithDebInfo builds (useful for ncu/nsight, has been observed to fail CUDA JIT program builds)"
    OFF)
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
set(BATCHLAS_SYCL_LINK_JOBS "4" CACHE STRING
    "llvm-foreach parallelism for the SYCL device link (-fsycl-max-parallel-link-jobs); 1 disables")

set_property(CACHE BATCHLAS_CPU_TARGET PROPERTY STRINGS auto native_cpu spir64_x86_64 none)
set_property(CACHE BATCHLAS_TEST_TARGET_SET PROPERTY STRINGS all smoke)

# Normalise BATCHLAS_ENABLE_CUDA into BATCHLAS_ENABLE_CUDA_MODE (AUTO|ON|OFF).
# Plain booleans keep working, so -DBATCHLAS_ENABLE_CUDA=ON/OFF/1/0/TRUE/FALSE
# and any pre-existing BOOL cache entry still mean what they used to.
string(TOUPPER "${BATCHLAS_ENABLE_CUDA}" BATCHLAS_ENABLE_CUDA_MODE)
if(BATCHLAS_ENABLE_CUDA_MODE STREQUAL "")
    set(BATCHLAS_ENABLE_CUDA_MODE "AUTO")
elseif(BATCHLAS_ENABLE_CUDA_MODE MATCHES "^(1|TRUE|YES|Y)$")
    set(BATCHLAS_ENABLE_CUDA_MODE "ON")
elseif(BATCHLAS_ENABLE_CUDA_MODE MATCHES "^(0|FALSE|NO|N|IGNORE|NOTFOUND|.*-NOTFOUND)$")
    set(BATCHLAS_ENABLE_CUDA_MODE "OFF")
endif()
if(NOT BATCHLAS_ENABLE_CUDA_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "BATCHLAS_ENABLE_CUDA must be AUTO, ON or OFF (got '${BATCHLAS_ENABLE_CUDA}')")
endif()
# Resolved by detect_sycl_gpu_architectures() in BatchLASDetectSYCL.cmake; this
# is the value that survives if sycl-ls is missing and detection bails out early.
if(BATCHLAS_ENABLE_CUDA_MODE STREQUAL "ON")
    set(BATCHLAS_CUDA_ENABLED ON)
else()
    set(BATCHLAS_CUDA_ENABLED OFF)
endif()

# Workaround for a vecz crash in the native_cpu backend. Only ever reaches the
# command line when native_cpu is among the SYCL targets. This used to default
# from a hardcoded /opt/dpcpp-cuda path match.
option(BATCHLAS_NATIVE_CPU_DISABLE_VECZ "Disable vecz for native_cpu backend (workaround for clang crash)" ON)

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

# oneDPL is a hard dependency (src/matrix.cc, src/extensions/lanczos.cc,
# src/extensions/tridiag_solver.cc and src/extensions/syevx_lobpcg.cc all
# include <oneapi/dpl/...> unconditionally). This used to be a plain, non-cache
# set() to one absolute path, so -DONEDPL_ROOT=... was silently ignored.
# The actual search lives in BatchLASDependencies.cmake.
set(ONEDPL_ROOT "" CACHE PATH
    "Root of a oneDPL installation; the headers are expected at <ONEDPL_ROOT>/include/oneapi/dpl")

include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID}) is not a SYCL compiler. "
        "Configure a fresh build directory with -DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++ (or icpx), "
        "or -DBATCHLAS_SYCL_IMPL=ACPP for AdaptiveCpp (docs/guide/build.md).")
endif()

# The resolved implementation is BATCHLAS_SYCL_IMPL_RESOLVED plus the 0/1 pair
# BATCHLAS_SYCL_IMPL_DPCPP / _ACPP; the cache entry keeps what the user asked for.
# Design: docs/design/sycl-implementations.md#sycl-impl-build
set(BATCHLAS_SYCL_IMPL "AUTO" CACHE STRING
    "SYCL implementation: AUTO (ACPP when CMAKE_CXX_COMPILER is the acpp driver), DPCPP or ACPP")
set_property(CACHE BATCHLAS_SYCL_IMPL PROPERTY STRINGS AUTO DPCPP ACPP)
string(TOUPPER "${BATCHLAS_SYCL_IMPL}" _batchlas_sycl_impl)
if(NOT _batchlas_sycl_impl MATCHES "^(AUTO|DPCPP|ACPP)$")
    message(FATAL_ERROR "BATCHLAS_SYCL_IMPL must be AUTO, DPCPP or ACPP (got '${BATCHLAS_SYCL_IMPL}')")
endif()
get_filename_component(_batchlas_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
set(_batchlas_cxx_is_acpp FALSE)
if(_batchlas_cxx_name MATCHES "^(acpp|syclcc|syclcc-clang)$")
    set(_batchlas_cxx_is_acpp TRUE)
elseif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --acpp-version
        OUTPUT_VARIABLE _batchlas_acpp_version ERROR_QUIET RESULT_VARIABLE _batchlas_acpp_rc TIMEOUT 60)
    if(_batchlas_acpp_rc EQUAL 0 AND _batchlas_acpp_version MATCHES "AdaptiveCpp")
        set(_batchlas_cxx_is_acpp TRUE)
    endif()
endif()
if(_batchlas_sycl_impl STREQUAL "AUTO")
    if(_batchlas_cxx_is_acpp)
        set(_batchlas_sycl_impl ACPP)
    else()
        set(_batchlas_sycl_impl DPCPP)
    endif()
elseif(_batchlas_sycl_impl STREQUAL "ACPP" AND NOT _batchlas_cxx_is_acpp AND BATCHLAS_ACPP_SWAP_PENDING)
    message(STATUS "BATCHLAS_SYCL_IMPL=ACPP: switching to the acpp driver; CMake resets the cache and reconfigures")
    set(_batchlas_sycl_impl DPCPP)
elseif(_batchlas_sycl_impl STREQUAL "ACPP" AND NOT _batchlas_cxx_is_acpp)
    message(FATAL_ERROR "BATCHLAS_SYCL_IMPL=ACPP needs the AdaptiveCpp driver as the compiler "
        "(-DCMAKE_CXX_COMPILER=<prefix>/bin/acpp); got ${CMAKE_CXX_COMPILER}")
elseif(_batchlas_sycl_impl STREQUAL "DPCPP" AND _batchlas_cxx_is_acpp)
    message(FATAL_ERROR "BATCHLAS_SYCL_IMPL=DPCPP but ${CMAKE_CXX_COMPILER} is the AdaptiveCpp driver")
endif()
# Cached so tools can read it off the tree: compare_failures.py --build-dir picks the ledger by it.
set(BATCHLAS_SYCL_IMPL_RESOLVED "${_batchlas_sycl_impl}" CACHE INTERNAL "Resolved BATCHLAS_SYCL_IMPL")
set(BATCHLAS_SYCL_IMPL_DPCPP OFF)
set(BATCHLAS_SYCL_IMPL_ACPP OFF)
set(BATCHLAS_SYCL_IMPL_${_batchlas_sycl_impl} ON)
message(STATUS "SYCL implementation: ${BATCHLAS_SYCL_IMPL_RESOLVED} (BATCHLAS_SYCL_IMPL=${BATCHLAS_SYCL_IMPL})")

# acpp + CUDA launches at most 48 KiB of local memory per work-group. ON builds a cuLaunchKernel
# interposer that opts kernels in (src/sycl/acpp_slm_optin.cc) and budgets the opt-in maximum:
# a recorded A/B variant, not the default. evidence: docs/design/sycl-implementations.md#sycl-impl-slm-budget
option(BATCHLAS_ACPP_SLM_OPTIN "AdaptiveCpp + CUDA: opt launches into >48 KiB local memory (interposer)" OFF)
if(BATCHLAS_ACPP_SLM_OPTIN AND NOT BATCHLAS_SYCL_IMPL_ACPP)
    message(FATAL_ERROR "BATCHLAS_ACPP_SLM_OPTIN needs BATCHLAS_SYCL_IMPL=ACPP (DPC++ opts in by itself)")
endif()
if(BATCHLAS_ACPP_SLM_OPTIN AND APPLE)
    message(FATAL_ERROR "BATCHLAS_ACPP_SLM_OPTIN is a CUDA cuLaunchKernel interposer; macOS has no CUDA")
endif()

# Is BatchLAS the top-level project, or was it pulled in with add_subdirectory()
# / FetchContent? Anything that writes global state has to be conditional on
# this. PROJECT_IS_TOP_LEVEL would be the idiomatic spelling but needs CMake
# 3.21 and this project declares 3.17.
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

# Installed components find each other ($ORIGIN) and their vendor libraries (link path).
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN")
if(NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()
# Build tree only: RPATH entries inside the build tree become $ORIGIN-relative, so a copied
# tree (cp -a build seed) loads its own .so instead of the original's. Entries outside the
# tree (the SYCL runtime, CUDA) stay absolute, and the install RPATH above is untouched.
if(NOT DEFINED CMAKE_BUILD_RPATH_USE_ORIGIN)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
endif()

# A test or benchmark executable is not relinked when only a BatchLAS .so it links changed:
# every library edit used to relink ~150 executables. The library still builds first.
option(BATCHLAS_LINK_DEPENDS_NO_SHARED
    "Do not relink executables when only a shared library they link changed (faster dev loop). Trade-off: a symbol removed or renamed in a .so leaves a stale executable, which fails at load time ('undefined symbol') instead of relinking (in-tree executables are linked -z now for that); scripts/gate.sh turns it OFF"
    ON)
if(BATCHLAS_LINK_DEPENDS_NO_SHARED)
    set(CMAKE_LINK_DEPENDS_NO_SHARED ON)
    # Lazy binding would find a symbol missing from a .so only when the call runs, and a
    # case that never runs it (skipped, filtered, another tier) would stay green. BIND_NOW
    # makes every executable resolve all of its own references when it is loaded.
    if(UNIX AND NOT APPLE)
        add_link_options("$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-Wl,-z,now>")
    endif()
else()
    set(CMAKE_LINK_DEPENDS_NO_SHARED OFF)
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
option(BATCHLAS_BUILD_DOCS "Build BatchLAS documentation" OFF)
# Tri-state on purpose. This used to be option(... OFF), and
# BatchLASDetectSYCL.cmake then force-promoted it to ON in PARENT_SCOPE whenever
# sycl-ls reported a CUDA device - a normal variable shadowing the cache entry,
# so -DBATCHLAS_ENABLE_CUDA=OFF was silently ignored while CMakeCache.txt kept
# reporting OFF forever. AUTO is the honest spelling of the old behaviour; ON now
# means "require it" and errors out instead of quietly configuring for sm_50.
set(BATCHLAS_ENABLE_CUDA "AUTO" CACHE STRING
    "CUDA backend: AUTO (enable when the SYCL runtime exposes a CUDA device), ON (require it), OFF")
set_property(CACHE BATCHLAS_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
# Apple GPU through AdaptiveCpp's Metal backend, dispatched as Backend::SYCL (native kernels,
# no vendor library). Same AUTO|ON|OFF contract as BATCHLAS_ENABLE_CUDA; resolved in
# BatchLASSyclAcpp.cmake. Design: docs/design/sycl-implementations.md.
set(BATCHLAS_ENABLE_METAL "AUTO" CACHE STRING
    "Metal backend (AdaptiveCpp only): AUTO (enable when acpp-info lists a Metal GPU), ON (require it), OFF")
set_property(CACHE BATCHLAS_ENABLE_METAL PROPERTY STRINGS AUTO ON OFF)
option(BATCHLAS_DISABLE_CUDA_FTZ
    "Pass --ftz=false to the CUDA image's runtime JIT options; the AOT ptxas step never sees it" ON)
option(BATCHLAS_ENABLE_ROCM "Enable ROCm support even when no AMD GPU is detected" OFF)
option(BATCHLAS_BUILD_PYTHON "Build Python bindings" OFF)
option(BATCHLAS_KEEP_CUDA_INTERMEDIATES "Preserve CUDA and SYCL CUDA device compilation intermediates for PTX/SASS analysis" OFF)
option(BATCHLAS_SANITIZER_FRIENDLY_DEBUG "Use more unwind-friendly debug flags for sanitizer runs (may slow down builds/runs)" OFF)

# Whether the `unsafe` group in batchlas::Settings is allowed to be set from the
# process environment.
#
# Most of the ~106 BATCHLAS_* knobs pick a route, a launch geometry or a dump
# path: setting one by accident costs a measurement, not a result. A few are
# different in kind, because they remove a check rather than change one.
# BATCHLAS_SKIP_POINTER_CHECKS turns off the USM reachability test that stands
# between an ordinary std::vector and CUDA_ERROR_ILLEGAL_ADDRESS followed by a
# SIGABRT no catch block can stop; BATCHLAS_LATRD_GRID_FORCE_UNSAFE lets the
# grid latrd path launch more work-groups per matrix than are guaranteed
# co-resident, and its spin barrier then HANGS rather than returning a wrong
# answer; BATCHLAS_BLAS_HEALTH=off skips the host-dgemm probe, after which a
# known-bad OpenBLAS kernel makes every double result silently wrong with no
# diagnostic. All three are worth having while debugging. None should be
# reachable by exporting a shell variable and forgetting it, which is how the
# readiness audit's A-3 finding describes the failure: a benchmark exports one,
# the shell keeps it, and every later run in that process has different
# semantics with nothing on stderr to say so.
#
# So the default is OFF, and OFF is not a no-op: settings() holds those fields
# at their safe defaults whatever the environment says, and warns once on
# stderr naming both the variable and this option, so the knob fails loudly
# instead of silently. An embedding application gets a build whose argument
# checking cannot be disarmed from outside.
#
# ON in the dev*, fast-dev and benchmarks presets, which exist to measure and to
# debug; deliberately OFF in the cuda preset, which is the pre-push gate and has
# to run the arm a release build runs. See docs/cpp-api.md#configuration.
option(BATCHLAS_ALLOW_UNSAFE_ENV
    "Honour the BATCHLAS_* environment overrides that disable a safety check (batchlas::Settings::unsafe)"
    OFF)

# There is deliberately NO BATCHLAS_ENABLE_COVERAGE option.
#
# The per-call half of the coverage instrument (src/select/coverage.hh) is gated
# at runtime on $BATCHLAS_COVERAGE_OUT instead. A compile-time switch on an inline
# header function is only sound when every TU in the process agrees on it, which a
# library cannot enforce: a consumer TU compiled without the macro interposes its
# own uninstrumented copy and recording silently stops.

set(BATCHLAS_CPU_TARGET "auto" CACHE STRING "CPU SYCL target override: auto|native_cpu|spir64_x86_64|none")
set(BATCHLAS_TEST_TARGET_SET "all" CACHE STRING "Subset of tests to generate: all|smoke")
set(BATCHLAS_AMD_ARCH "" CACHE STRING
    "AMD architecture (gfxNNN) to build instead of the detected GPU; empty = detect")
# amd_gpu_gfx942 was the old default and meant "detect"; an existing cache still holds it.
if(BATCHLAS_AMD_ARCH STREQUAL "amd_gpu_gfx942")
    set_property(CACHE BATCHLAS_AMD_ARCH PROPERTY VALUE "")
    set(BATCHLAS_AMD_ARCH "")
endif()
set(BATCHLAS_NVIDIA_ARCH "" CACHE STRING
    "NVIDIA architecture (sm_XX) to build instead of the detected GPUs; empty = detect")
# sm_50 was the old default and meant "detect"; an existing cache still holds it.
if(BATCHLAS_NVIDIA_ARCH STREQUAL "sm_50")
    set_property(CACHE BATCHLAS_NVIDIA_ARCH PROPERTY VALUE "")
    set(BATCHLAS_NVIDIA_ARCH "")
endif()
set(BATCHLAS_SYCL_LINK_JOBS "auto" CACHE STRING
    "llvm-foreach parallelism for the SYCL device link (-fsycl-max-parallel-link-jobs); 1 disables, auto = the logical cores, 4..32")
# 4 was the old fixed default; an existing cache still holds it, so it moves to auto once.
if(BATCHLAS_SYCL_LINK_JOBS STREQUAL "4" AND NOT BATCHLAS_SYCL_LINK_JOBS_MIGRATED)
    set_property(CACHE BATCHLAS_SYCL_LINK_JOBS PROPERTY VALUE "auto")
    set(BATCHLAS_SYCL_LINK_JOBS "auto")
endif()
set(BATCHLAS_SYCL_LINK_JOBS_MIGRATED ON CACHE INTERNAL "BATCHLAS_SYCL_LINK_JOBS left its old default once")
# DPC++ only (BatchLASDetectSYCL.cmake). OFF passes -mllvm -enable-global-offset=false to the
# device link, so no kernel gets a `<name>_with_offset` twin: about half of every NVPTX device
# link. Nothing in the tree launches with a SYCL 1.2.1 global offset (parallel_for(range, id,
# k), get_offset()); a kernel that did would no longer see its offset.
option(BATCHLAS_SYCL_GLOBAL_OFFSET
    "DPC++: keep the _with_offset twin of every kernel (SYCL 1.2.1 global offsets). OFF (default) drops them and roughly halves NVPTX device-link time"
    OFF)

# ---------------------------------------------------------------------------
# One .so, or sixteen.
#
# OFF (the DEVELOPMENT default) builds the 16 component shared objects. That
# split exists for one measured reason: the unit of SYCL device linking is the
# shared library, not the object file, so one changed object redoes
# sycl-post-link + ptxas + AOT for every object in the same .so. Before the
# split, a one-line edit cost 216 s of which 198 s was a single relink, and the
# device link is single-threaded so -j cannot shorten it.
#
# ON builds one `batchlas` .so from every object library. It is the RELEASE
# shape, never the dev default: it makes the whole library the relink unit
# again. What it buys is the packaging story - one SONAME, one export set, no
# cross-.so symbol cycles, and therefore no -Wl,--no-as-needed workaround and a
# bounded ABI that -fvisibility=hidden can actually enforce.
#
# Measured on this box (2x RTX 4090, DPC++ clang-22 at /opt/dpcpp-cuda,
# RelWithDebInfo, CUDA on, -fsycl-targets=nvidia_gpu_sm_89,native_cpu):
# `cmake --build <dir> --target batchlas -j16` from a scratch tree took
# 11m38s wall, of which the single `Linking CXX shared library libbatchlas.so`
# step was 9m35s (575 s) on its own. That step is single-threaded, so -j does
# not shorten it. The product is a 273 MiB libbatchlas.so.0.1.0. Usable for a
# release build, ruinous as a dev loop - which is the whole reason this
# defaults OFF.
# macOS can only build the monolithic shape (check below), so it is the default there.
set(_batchlas_monolithic_default OFF)
if(APPLE)
    set(_batchlas_monolithic_default ON)
endif()
option(BATCHLAS_MONOLITHIC_LIBRARY
    "Build ONE batchlas shared library instead of the 16 component .so (release shape; makes every incremental edit relink the world)"
    ${_batchlas_monolithic_default})
# Monolithic mode used to hide every symbol unconditionally. The in-tree tests reach ~1,000
# unannotated internal symbols, so a monolithic tree that builds tests defaults it OFF.
set(_batchlas_hidden_default ON)
if(BATCHLAS_BUILD_TESTS)
    set(_batchlas_hidden_default OFF)
endif()
cmake_dependent_option(BATCHLAS_HIDDEN_VISIBILITY
    "-fvisibility=hidden on the monolithic library (OFF to link the in-tree tests against it)"
    ${_batchlas_hidden_default} "BATCHLAS_MONOLITHIC_LIBRARY" OFF)
# ld64 rejects undefined symbols in a dylib, and the split components reference each other in
# cycles with no recorded dependency (src/CMakeLists.txt, the --no-as-needed note).
if(APPLE AND NOT BATCHLAS_MONOLITHIC_LIBRARY)
    message(FATAL_ERROR "macOS needs -DBATCHLAS_MONOLITHIC_LIBRARY=ON: the split component dylibs "
        "have cyclic undefined symbols, which ld64 rejects")
endif()

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
string(TOUPPER "${BATCHLAS_ENABLE_METAL}" BATCHLAS_ENABLE_METAL_MODE)
if(BATCHLAS_ENABLE_METAL_MODE MATCHES "^(1|TRUE|YES|Y)$")
    set(BATCHLAS_ENABLE_METAL_MODE "ON")
elseif(BATCHLAS_ENABLE_METAL_MODE MATCHES "^(0|FALSE|NO|N)$")
    set(BATCHLAS_ENABLE_METAL_MODE "OFF")
endif()
if(NOT BATCHLAS_ENABLE_METAL_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "BATCHLAS_ENABLE_METAL must be AUTO, ON or OFF (got '${BATCHLAS_ENABLE_METAL}')")
endif()
if(BATCHLAS_ENABLE_METAL_MODE STREQUAL "ON" AND NOT BATCHLAS_SYCL_IMPL_ACPP)
    message(FATAL_ERROR "BATCHLAS_ENABLE_METAL=ON needs BATCHLAS_SYCL_IMPL=ACPP (DPC++ has no Metal backend)")
endif()
# Resolved by detect_sycl_gpu_architectures() in BatchLASDetectSYCL.cmake; this
# is the value that survives if sycl-ls is missing and detection bails out early.
if(BATCHLAS_ENABLE_CUDA_MODE STREQUAL "ON")
    set(BATCHLAS_CUDA_ENABLED ON)
else()
    set(BATCHLAS_CUDA_ENABLED OFF)
endif()

set(BATCHLAS_HAS_HOST_BACKEND FALSE)
set(BATCHLAS_HAS_CUDA_BACKEND FALSE)
set(BATCHLAS_HAS_ROCM_BACKEND FALSE)
# Backend::SYCL: a GPU with no vendor library (AdaptiveCpp Metal). Set by BatchLASSyclAcpp.cmake.
set(BATCHLAS_HAS_SYCL_BACKEND FALSE)
set(BATCHLAS_DEVICE_NO_VECTOR_TYPES FALSE)
set(BATCHLAS_HAS_CPU_TARGET FALSE)

# double/complex<double> instantiation for Backend::SYCL only; CUDA, ROCm and the host keep all
# four types. AUTO is OFF on Metal (no fp64 in MSL) and ON elsewhere; ON with Metal is an error.
set(BATCHLAS_DEVICE_FP64 "AUTO" CACHE STRING
    "Instantiate double and complex<double> for the native-SYCL GPU backend: AUTO, ON, OFF")
set_property(CACHE BATCHLAS_DEVICE_FP64 PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${BATCHLAS_DEVICE_FP64}" BATCHLAS_DEVICE_FP64_MODE)
if(BATCHLAS_DEVICE_FP64_MODE MATCHES "^(1|TRUE|YES|Y)$")
    set(BATCHLAS_DEVICE_FP64_MODE "ON")
elseif(BATCHLAS_DEVICE_FP64_MODE MATCHES "^(0|FALSE|NO|N)$")
    set(BATCHLAS_DEVICE_FP64_MODE "OFF")
endif()
if(NOT BATCHLAS_DEVICE_FP64_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "BATCHLAS_DEVICE_FP64 must be AUTO, ON or OFF (got '${BATCHLAS_DEVICE_FP64}')")
endif()
if(BATCHLAS_DEVICE_FP64_MODE STREQUAL "OFF")
    set(BATCHLAS_HAS_DEVICE_FP64 OFF)
else()
    set(BATCHLAS_HAS_DEVICE_FP64 ON)
endif()

# ---------------------------------------------------------------------------
# Vendor math libraries, separate from the BATCHLAS_HAS_<FAMILY>_BACKEND device families above:
# a family says which SYCL target a queue runs on, a library flag whether someone else's math
# code can be called (src/select/vendor.hh). OFF everywhere is the vendor-free build.
option(BATCHLAS_ENABLE_VENDOR_BLAS
    "Build against vendor math libraries (cuBLAS/cuSOLVER/cuSPARSE, roc*, netlib)"
    ON)

# Re-read on every configure, so turning it OFF in an existing tree does give a vendor-free
# build. BATCHLAS_HAS_<LIB> ("found AND allowed") is set by BatchLASDependencies.cmake.
foreach(_batchlas_lib CUBLAS CUSOLVER CUSPARSE ROCBLAS ROCSOLVER ROCSPARSE LAPACKE CBLAS)
    set(BATCHLAS_HAS_${_batchlas_lib} FALSE)
endforeach()
unset(_batchlas_lib)

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

# icpx defaults to -fp-model=fast, which tags device code with a flush-to-zero
# denormal mode that the linked libspirv builtins do not share. LLVM then refuses
# to inline sycl::fma, barriers and id queries, and ~98% of kernels call them out
# of line (cfloat gemm 8x, potrf cfloat 6x slower on sm_120). clang++/dpcpp
# already defaults to precise; this makes both compilers agree.
# evidence: docs/perf/blackwell.md#icpx-fast-fp-model
target_compile_options(batchlas_build_options INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:IntelLLVM>,$<BOOL:${BATCHLAS_SYCL_IMPL_DPCPP}>>:-ffp-model=precise>
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
)

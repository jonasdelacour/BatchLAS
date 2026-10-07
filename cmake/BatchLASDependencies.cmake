# Probe the NVIDIA math libraries through FindCUDAToolkit: its CUDA:: targets
# are what the backends link (see BATCHLAS_CUDA_LINK_LIBRARIES below), so the
# probe and the link cannot disagree. A separate find_library() search used to:
# on the NVIDIA HPC SDK, cuSOLVER and cuSPARSE live in
# <sdk>/math_libs/<cuda-version>/lib64 rather than in the toolkit, the probe
# missed them, cublas.cc was silently dropped, and the link failed on
# undefined *_vendor<Backend::CUDA> symbols. Override a library with
# -DCUDA_<name>_LIBRARY=<path>, e.g. -DCUDA_cusolver_LIBRARY=...
function(find_nvidia_libs)
    if(NOT BATCHLAS_CUDA_ENABLED)
        return()
    endif()

    message(STATUS "Searching for NVIDIA CUDA libraries...")
    find_package(CUDAToolkit QUIET)

    if(TARGET CUDA::cublas)
        message(STATUS "Found cuBLAS: ${CUDA_cublas_LIBRARY}")
        # NOTE (WP0 S1): this line is the family/library conflation itself --
        # "we found cuBLAS" is being used to answer "is there a CUDA backend".
        # S2 replaces it with a derivation from the hardware.
        set(BATCHLAS_HAS_CUDA_BACKEND TRUE PARENT_SCOPE)
        if(BATCHLAS_ENABLE_CUBLAS)
            set(BATCHLAS_HAS_CUBLAS TRUE PARENT_SCOPE)
        endif()
    else()
        message(WARNING "NVIDIA GPU detected but cuBLAS library not found. "
            "Point CMake at the toolkit with -DCUDAToolkit_ROOT=<path> or at the "
            "library with -DCUDA_cublas_LIBRARY=<path>.")
    endif()

    if(TARGET CUDA::cusolver AND BATCHLAS_ENABLE_CUSOLVER)
        message(STATUS "Found cuSOLVER: ${CUDA_cusolver_LIBRARY}")
        set(BATCHLAS_HAS_CUSOLVER TRUE PARENT_SCOPE)
    endif()
    if(TARGET CUDA::cusparse AND BATCHLAS_ENABLE_CUSPARSE)
        message(STATUS "Found cuSPARSE: ${CUDA_cusparse_LIBRARY}")
        set(BATCHLAS_HAS_CUSPARSE TRUE PARENT_SCOPE)
    endif()
endfunction()

function(find_rocm_libs)
    set(ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    elseif(EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    endif()

    if(NOT ROCM_PATH)
        message(STATUS "ROCm path not found, skipping ROCm backend detection")
        return()
    endif()

    message(STATUS "Searching for ROCm libraries in: ${ROCM_PATH}")

    find_library(HIPBLAS_LIBRARY
        NAMES hipblas
        PATHS "${ROCM_PATH}"
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
        DOC "AMD hipBLAS library"
    )

    if(HIPBLAS_LIBRARY)
        message(STATUS "Found hipBLAS: ${HIPBLAS_LIBRARY}")
        get_filename_component(HIPBLAS_LIBRARY_DIR "${HIPBLAS_LIBRARY}" DIRECTORY)
        find_library(ROCBLAS_LIBRARY rocblas PATHS "${HIPBLAS_LIBRARY_DIR}" NO_DEFAULT_PATH)
        find_library(HIPSPARSE_LIBRARY hipsparse PATHS "${HIPBLAS_LIBRARY_DIR}" NO_DEFAULT_PATH)
        find_library(ROCSOLVER_LIBRARY rocsolver PATHS "${HIPBLAS_LIBRARY_DIR}" NO_DEFAULT_PATH)

        if(ROCBLAS_LIBRARY)
            message(STATUS "Found rocBLAS: ${ROCBLAS_LIBRARY}")
        endif()
        if(HIPSPARSE_LIBRARY)
            message(STATUS "Found hipSPARSE: ${HIPSPARSE_LIBRARY}")
        endif()
        if(ROCSOLVER_LIBRARY)
            message(STATUS "Found rocSOLVER: ${ROCSOLVER_LIBRARY}")
        endif()

        # ---- axis 3: which ROCm math libraries are present -----------------
        if(BATCHLAS_ENABLE_ROCBLAS AND ROCBLAS_LIBRARY)
            set(BATCHLAS_HAS_ROCBLAS TRUE PARENT_SCOPE)
        endif()
        if(BATCHLAS_ENABLE_ROCSOLVER AND ROCSOLVER_LIBRARY)
            set(BATCHLAS_HAS_ROCSOLVER TRUE PARENT_SCOPE)
        endif()
        if(BATCHLAS_ENABLE_ROCSPARSE AND HIPSPARSE_LIBRARY)
            set(BATCHLAS_HAS_ROCSPARSE TRUE PARENT_SCOPE)
        endif()

        # find_library() leaves <VAR>-NOTFOUND in the cache variable when it
        # fails, so an unqualified ${ROCSOLVER_LIBRARY} here appended the
        # literal string "ROCSOLVER_LIBRARY-NOTFOUND" to the link line and
        # turned a missing optional library into a link error. Only append the
        # ones that were actually found. (Cannot be verified on this machine --
        # there is no AMD GPU here -- but the failure mode is unambiguous.)
        set(_rocm_link_libs)
        foreach(_rocm_lib ROCBLAS_LIBRARY HIPSPARSE_LIBRARY ROCSOLVER_LIBRARY)
            if(${_rocm_lib})
                list(APPEND _rocm_link_libs "${${_rocm_lib}}")
            endif()
        endforeach()
        unset(_rocm_lib)

        set(BATCHLAS_ROCM_LINK_LIBRARIES ${_rocm_link_libs} PARENT_SCOPE)
        set(BATCHLAS_HAS_ROCM_BACKEND TRUE PARENT_SCOPE)
        set(BATCHLAS_ROCM_INCLUDE_DIR "${ROCM_PATH}/include" PARENT_SCOPE)
        message(STATUS "ROCm backend will be enabled")
    else()
        message(STATUS "hipBLAS library not found in ROCm installation")
    endif()
endfunction()

function(find_netlib_libs)
    if(NOT BATCHLAS_ENABLE_NETLIB)
        return()
    endif()

    message(STATUS "Searching for Netlib BLAS/LAPACK libraries")

    find_library(LAPACKE_LIBRARY NAMES lapacke
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
        NO_DEFAULT_PATH)
    if(NOT LAPACKE_LIBRARY)
        find_library(LAPACKE_LIBRARY NAMES lapacke)
    endif()

    # CBLAS must come from LAPACKE's own install. A netlib LAPACKE in /opt/lib
    # calls dgemm_64_ etc. from its sibling libblas.so.3; pairing it with the
    # distro OpenBLAS libblas.so (same SONAME, no _64_ symbols) shadows that
    # sibling and the link fails on every *_64_ BLAS routine.
    if(LAPACKE_LIBRARY)
        get_filename_component(_lapacke_dir "${LAPACKE_LIBRARY}" DIRECTORY)
        find_library(CBLAS_LIBRARY NAMES cblas blas
            PATHS "${_lapacke_dir}"
            NO_DEFAULT_PATH)
    endif()
    find_library(CBLAS_LIBRARY NAMES cblas blas
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
        NO_DEFAULT_PATH)
    if(NOT CBLAS_LIBRARY)
        find_library(CBLAS_LIBRARY NAMES cblas blas)
    endif()

    # ---- axis 3: LAPACKE and CBLAS are independent libraries ---------------
    # They are found separately above, so record them separately. The host
    # DEVICE family is a different question -- a CPU SYCL device exists whether
    # or not netlib is installed -- but that decoupling is S2; here the family
    # flag keeps its current derivation so the build stays bit-identical.
    if(BATCHLAS_ENABLE_LAPACKE AND LAPACKE_LIBRARY)
        set(BATCHLAS_HAS_LAPACKE TRUE PARENT_SCOPE)
    endif()
    if(BATCHLAS_ENABLE_CBLAS AND CBLAS_LIBRARY)
        set(BATCHLAS_HAS_CBLAS TRUE PARENT_SCOPE)
    endif()

    if(LAPACKE_LIBRARY AND CBLAS_LIBRARY)
        message(STATUS "Found LAPACKE: ${LAPACKE_LIBRARY}")
        message(STATUS "Found CBLAS: ${CBLAS_LIBRARY}")
        set(BATCHLAS_NETLIB_LINK_LIBRARIES "${LAPACKE_LIBRARY};${CBLAS_LIBRARY}" PARENT_SCOPE)
        set(BATCHLAS_HAS_HOST_BACKEND TRUE PARENT_SCOPE)
        # Headers are searched next to the libraries actually found, so a
        # netlib install in <prefix>/lib (not the -dev package) still gives
        # <prefix>/include to the tests and benchmarks that call LAPACKE_*.
        get_filename_component(_lapacke_prefix "${LAPACKE_LIBRARY}" DIRECTORY)
        get_filename_component(_cblas_prefix "${CBLAS_LIBRARY}" DIRECTORY)
        find_path(LAPACKE_INCLUDE_DIR lapacke.h
            HINTS "${_lapacke_prefix}/../include"
            PATH_SUFFIXES lapacke)
        find_path(CBLAS_INCLUDE_DIR cblas.h
            HINTS "${_cblas_prefix}/../include"
            PATH_SUFFIXES cblas openblas)
        set(_netlib_includes)
        foreach(_dir IN ITEMS "${LAPACKE_INCLUDE_DIR}" "${CBLAS_INCLUDE_DIR}")
            if(_dir)
                list(APPEND _netlib_includes "${_dir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _netlib_includes)
        set(BATCHLAS_NETLIB_INCLUDE_DIRS "${_netlib_includes}" PARENT_SCOPE)
    else()
        message(WARNING "LAPACKE/CBLAS libraries not found - disabling host backend")
        set(BATCHLAS_HAS_HOST_BACKEND FALSE PARENT_SCOPE)
    endif()
endfunction()

# oneDPL is a hard dependency: src/matrix.cc, src/extensions/lanczos.cc,
# src/extensions/tridiag_solver.cc and src/extensions/syevx_lobpcg.cc all
# include <oneapi/dpl/{algorithm,execution,random}> unconditionally. DPC++ does
# not bundle it, so without this the build dies on a missing header with no hint
# about which knob to turn.
find_path(BATCHLAS_ONEDPL_INCLUDE_DIR
    NAMES oneapi/dpl/algorithm
    HINTS
        "${ONEDPL_ROOT}/include"
        "$ENV{ONEDPL_ROOT}/include"
        "$ENV{DPL_ROOT}/include"
        "$ENV{DPLROOT}/include"
        "/opt/intel/oneapi/dpl/latest/include"
    DOC "Directory containing oneapi/dpl (oneDPL headers)"
)
if(NOT BATCHLAS_ONEDPL_INCLUDE_DIR)
    message(FATAL_ERROR
        "oneDPL headers not found. BatchLAS requires them unconditionally "
        "(<oneapi/dpl/algorithm>, <oneapi/dpl/execution>, <oneapi/dpl/random>). "
        "Configure with -DONEDPL_ROOT=<prefix>, where <prefix>/include/oneapi/dpl exists, "
        "or set the ONEDPL_ROOT / DPL_ROOT environment variable "
        "(oneAPI's setvars.sh sets DPL_ROOT for you).")
endif()
message(STATUS "Found oneDPL headers: ${BATCHLAS_ONEDPL_INCLUDE_DIR}")
# Skip the -I when the headers are already on the default search path; adding
# /usr/include explicitly reorders the system include search and breaks builds.
if(NOT BATCHLAS_ONEDPL_INCLUDE_DIR STREQUAL "/usr/include")
    target_include_directories(batchlas_dep_options INTERFACE
        $<BUILD_INTERFACE:${BATCHLAS_ONEDPL_INCLUDE_DIR}>
    )
endif()

if(BATCHLAS_CUDA_ENABLED)
    enable_language(CUDA)
    find_nvidia_libs()
endif()

if(BATCHLAS_DETECTED_AMD_GPU OR BATCHLAS_ENABLE_ROCM)
    find_rocm_libs()
endif()

find_netlib_libs()

# Some BLAS builds dispatch to CPU kernels that compute wrong results; find out
# now rather than through mysterious numerical test failures later.
include(${CMAKE_CURRENT_LIST_DIR}/BatchLASBlasHealthCheck.cmake)
if(BATCHLAS_HAS_HOST_BACKEND)
    batchlas_check_blas_health("${BATCHLAS_NETLIB_LINK_LIBRARIES}")
endif()
batchlas_write_env_script()

if(BATCHLAS_HAS_CUDA_BACKEND)
    find_package(CUDAToolkit REQUIRED)
    set(BATCHLAS_CUDA_LINK_LIBRARIES
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
    )

    set(BATCHLAS_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    foreach(_cuda_target IN LISTS BATCHLAS_CUDA_LINK_LIBRARIES)
        if(TARGET ${_cuda_target})
            get_target_property(_cuda_target_include_dirs ${_cuda_target} INTERFACE_INCLUDE_DIRECTORIES)
            if(_cuda_target_include_dirs)
                list(APPEND BATCHLAS_CUDA_INCLUDE_DIRS ${_cuda_target_include_dirs})
            endif()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES BATCHLAS_CUDA_INCLUDE_DIRS)

    target_include_directories(batchlas_dep_options INTERFACE
        ${BATCHLAS_CUDA_INCLUDE_DIRS}
    )

    set(BATCHLAS_CUDA_ARCHITECTURES "")
    if(DETECTED_NVIDIA_ARCH MATCHES "nvidia_gpu_sm_([0-9]+)")
        set(BATCHLAS_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}")
    elseif(BATCHLAS_NVIDIA_ARCH MATCHES "sm_([0-9]+)")
        set(BATCHLAS_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}")
    endif()
endif()

if(BATCHLAS_HAS_HOST_BACKEND)
    target_compile_definitions(batchlas_dep_options INTERFACE BATCHLAS_HAS_HOST_BACKEND=1)
endif()
if(BATCHLAS_HAS_CUDA_BACKEND)
    target_compile_definitions(batchlas_dep_options INTERFACE BATCHLAS_HAS_CUDA_BACKEND=1)
endif()
if(BATCHLAS_HAS_ROCM_BACKEND)
    target_compile_definitions(batchlas_dep_options INTERFACE BATCHLAS_HAS_ROCM_BACKEND=1)
    if(BATCHLAS_ROCM_INCLUDE_DIR)
        target_include_directories(batchlas_dep_options INTERFACE
            $<BUILD_INTERFACE:${BATCHLAS_ROCM_INCLUDE_DIR}>
        )
    endif()
endif()

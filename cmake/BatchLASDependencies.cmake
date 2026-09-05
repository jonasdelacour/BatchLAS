find_package(OpenMP QUIET)

if(BATCHLAS_ENABLE_MKL)
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        if(TARGET MKL::MKL_DPCPP)
            set(_MKL_SYCL_TARGET MKL::MKL_DPCPP)
        elseif(TARGET MKL::MKL_SYCL)
            set(_MKL_SYCL_TARGET MKL::MKL_SYCL)
        else()
            message(WARNING "oneMKL was found but provides no SYCL interface target (MKL::MKL_DPCPP or MKL::MKL_SYCL). The MKL backend will be disabled.")
        endif()

        if(DEFINED _MKL_SYCL_TARGET)
            message(STATUS "Found oneMKL SYCL target: ${_MKL_SYCL_TARGET}")
            add_library(batchlas_mkl INTERFACE)
            add_library(batchlas::mkl ALIAS batchlas_mkl)
            target_link_libraries(batchlas_mkl INTERFACE "${_MKL_SYCL_TARGET}")
            target_compile_definitions(batchlas_mkl INTERFACE MKL_ILP64)
            set(BATCHLAS_HAS_MKL_BACKEND TRUE)
            # axis 3: oneMKL the LIBRARY, as distinct from Backend::MKL the
            # device family. linalg-impl.hh:20 already uses the family flag to
            # mean "oneMKL supplies cblas.h", which is a library question.
            if(BATCHLAS_ENABLE_ONEMKL)
                set(BATCHLAS_HAS_ONEMKL TRUE)
            endif()
        endif()
    else()
        message(STATUS "MKL not found via CMake package, falling back to manual search")
    endif()
else()
    message(STATUS "MKL backend disabled")
endif()

function(find_nvidia_libs)
    if(NOT BATCHLAS_CUDA_ENABLED)
        return()
    endif()

    message(STATUS "Searching for NVIDIA CUDA libraries...")

    set(NVIDIA_HPC_SDK_BASE "")
    if(BATCHLAS_CUDA_PATH)
        if(BATCHLAS_CUDA_PATH MATCHES ".*/nvidia/hpc_sdk/.*" OR BATCHLAS_CUDA_PATH MATCHES ".*/nvhpc/.*")
            string(REGEX REPLACE "(.*nvidia/hpc_sdk)/.*" "\\1" POTENTIAL_HPC_SDK_BASE "${BATCHLAS_CUDA_PATH}")
            if(EXISTS "${POTENTIAL_HPC_SDK_BASE}")
                set(NVIDIA_HPC_SDK_BASE "${POTENTIAL_HPC_SDK_BASE}")
                message(STATUS "Detected NVIDIA HPC SDK installation at: ${NVIDIA_HPC_SDK_BASE}")
            endif()

            if(NOT NVIDIA_HPC_SDK_BASE)
                string(REGEX REPLACE "(.*nvhpc)/.*" "\\1" POTENTIAL_HPC_SDK_BASE "${BATCHLAS_CUDA_PATH}")
                if(EXISTS "${POTENTIAL_HPC_SDK_BASE}")
                    set(NVIDIA_HPC_SDK_BASE "${POTENTIAL_HPC_SDK_BASE}")
                    message(STATUS "Detected NVIDIA HPC SDK installation at: ${POTENTIAL_HPC_SDK_BASE}")
                endif()
            endif()

            if(NVIDIA_HPC_SDK_BASE)
                string(REGEX REPLACE "${NVIDIA_HPC_SDK_BASE}/(.*)/cuda.*" "\\1" HPC_SDK_PLATFORM_VERSION "${BATCHLAS_CUDA_PATH}")
                if(BATCHLAS_CUDA_PATH MATCHES ".*/([0-9]+\\.[0-9]+)/cuda.*")
                    string(REGEX REPLACE ".*/([0-9]+\\.[0-9]+)/cuda.*" "\\1" HPC_SDK_VERSION "${BATCHLAS_CUDA_PATH}")
                    message(STATUS "HPC SDK version: ${HPC_SDK_VERSION}")
                    set(POTENTIAL_MATH_LIBS_DIR "${NVIDIA_HPC_SDK_BASE}/${HPC_SDK_PLATFORM_VERSION}/math_libs")
                    if(EXISTS "${POTENTIAL_MATH_LIBS_DIR}")
                        message(STATUS "Found HPC SDK math_libs directory: ${POTENTIAL_MATH_LIBS_DIR}")
                        file(GLOB MATH_LIBS_VERSIONS "${POTENTIAL_MATH_LIBS_DIR}/*")
                        list(SORT MATH_LIBS_VERSIONS)
                        list(REVERSE MATH_LIBS_VERSIONS)
                        foreach(VERSION_DIR ${MATH_LIBS_VERSIONS})
                            if(IS_DIRECTORY "${VERSION_DIR}")
                                set(MATH_LIBS_DIR "${VERSION_DIR}")
                                get_filename_component(MATH_LIBS_VERSION "${VERSION_DIR}" NAME)
                                message(STATUS "Using math_libs version: ${MATH_LIBS_VERSION}")
                                break()
                            endif()
                        endforeach()
                    endif()
                endif()
            endif()
        endif()
    endif()

    set(NVIDIA_HPC_SDK_PATHS
        "${NVIDIA_HPC_SDK_BASE}"
        "/opt/nvidia/hpc_sdk"
        "/usr/local/nvidia/hpc_sdk"
        "$ENV{NVHPC_ROOT}"
    )

    if(DEFINED MATH_LIBS_DIR)
        find_library(CUBLAS_LIBRARY
            NAMES cublas
            PATHS "${MATH_LIBS_DIR}"
            PATH_SUFFIXES targets/x86_64-linux/lib lib64 lib
            NO_DEFAULT_PATH
            DOC "NVIDIA cuBLAS library"
        )
    endif()

    if(NOT CUBLAS_LIBRARY)
        find_library(CUBLAS_LIBRARY
            NAMES cublas
            PATHS
                ${BATCHLAS_CUDA_PATH}/lib64
                ${BATCHLAS_CUDA_PATH}/lib
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ${CUDA_TOOLKIT_ROOT_DIR}/lib
                ${NVIDIA_HPC_SDK_PATHS}
            PATH_SUFFIXES
                lib64
                lib
                target/x86_64-linux/lib
                targets/x86_64-linux/lib
                Linux_x86_64/*/math_libs/*/targets/x86_64-linux/lib
                */math_libs/*/targets/x86_64-linux/lib
                */math_libs/lib64
                Linux_x86_64/*/cuda/lib64
                */cuda/lib64
            DOC "NVIDIA cuBLAS library"
        )
    endif()

    if(CUBLAS_LIBRARY)
        message(STATUS "Found cuBLAS: ${CUBLAS_LIBRARY}")
        # NOTE (WP0 S1): this line is the family/library conflation itself --
        # "we found cuBLAS" is being used to answer "is there a CUDA backend".
        # It stays for now so S1 is bit-identical; S2 replaces it with a
        # derivation from the hardware. The library axis is recorded alongside.
        set(BATCHLAS_HAS_CUDA_BACKEND TRUE PARENT_SCOPE)
    else()
        message(WARNING "NVIDIA GPU detected but cuBLAS library not found. Add its path to CMAKE_PREFIX_PATH if needed.")
    endif()

    # ---- axis 3: which NVIDIA math libraries are actually present ----------
    #
    # cuSOLVER and cuSPARSE were never probed. They are pulled in blind via
    # CUDA::cusolver / CUDA::cusparse on a flag they did not influence, so a
    # toolkit missing either one fails at link time rather than at configure
    # time, and nothing can ask "is cuSOLVER available?" in order to route
    # around it. Probe them separately now.
    if(BATCHLAS_ENABLE_CUBLAS AND CUBLAS_LIBRARY)
        set(BATCHLAS_HAS_CUBLAS TRUE PARENT_SCOPE)
    endif()

    find_library(CUSOLVER_LIBRARY
        NAMES cusolver
        PATHS ${BATCHLAS_CUDA_PATH}/lib64 ${BATCHLAS_CUDA_PATH}/lib
              ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
        PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib
        DOC "NVIDIA cuSOLVER library")
    if(BATCHLAS_ENABLE_CUSOLVER AND CUSOLVER_LIBRARY)
        message(STATUS "Found cuSOLVER: ${CUSOLVER_LIBRARY}")
        set(BATCHLAS_HAS_CUSOLVER TRUE PARENT_SCOPE)
    endif()

    find_library(CUSPARSE_LIBRARY
        NAMES cusparse
        PATHS ${BATCHLAS_CUDA_PATH}/lib64 ${BATCHLAS_CUDA_PATH}/lib
              ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
        PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib
        DOC "NVIDIA cuSPARSE library")
    if(BATCHLAS_ENABLE_CUSPARSE AND CUSPARSE_LIBRARY)
        message(STATUS "Found cuSPARSE: ${CUSPARSE_LIBRARY}")
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

function(find_onemkl_libs)
    set(MKL_ROOT)
    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT "$ENV{MKLROOT}")
    elseif(EXISTS "/opt/intel/oneapi/mkl")
        set(MKL_ROOT "/opt/intel/oneapi/mkl")
    endif()

    if(NOT MKL_ROOT)
        message(STATUS "Intel oneAPI MKL not found, skipping Intel MKL backend detection")
        return()
    endif()

    message(STATUS "Searching for Intel oneAPI MKL in: ${MKL_ROOT}")

    find_library(MKL_CORE_LIBRARY
        NAMES mkl_core
        PATHS "${MKL_ROOT}"
        PATH_SUFFIXES lib lib/intel64
        NO_DEFAULT_PATH
        DOC "Intel oneAPI MKL core library"
    )

    if(MKL_CORE_LIBRARY)
        message(STATUS "Found MKL core: ${MKL_CORE_LIBRARY}")
        set(BATCHLAS_HAS_MKL_BACKEND TRUE PARENT_SCOPE)
        set(BATCHLAS_MKL_MANUAL_INSTALL TRUE PARENT_SCOPE)
        set(BATCHLAS_MKL_INCLUDE_DIR "${MKL_ROOT}/include" PARENT_SCOPE)
        message(STATUS "Intel MKL backend will be enabled")
    else()
        message(STATUS "Intel MKL library not found")
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
    find_library(CBLAS_LIBRARY NAMES cblas blas
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
        NO_DEFAULT_PATH)

    if(NOT LAPACKE_LIBRARY)
        find_library(LAPACKE_LIBRARY NAMES lapacke)
    endif()
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

if(BATCHLAS_ENABLE_MKL AND NOT MKL_FOUND)
    find_onemkl_libs()
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

    set(_BATCHLAS_MATHDX_HINTS)
    foreach(_hint ${BATCHLAS_MATHDX_ROOT} $ENV{BATCHLAS_MATHDX_ROOT} $ENV{mathdx_ROOT})
        if(_hint AND EXISTS "${_hint}")
            list(APPEND _BATCHLAS_MATHDX_HINTS "${_hint}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _BATCHLAS_MATHDX_HINTS)

    if(_BATCHLAS_MATHDX_HINTS)
        find_package(mathdx CONFIG QUIET PATHS ${_BATCHLAS_MATHDX_HINTS})
    else()
        find_package(mathdx CONFIG QUIET)
    endif()

    set(BATCHLAS_MATHDX_TARGETS "")
    if(TARGET mathdx::cublasdx)
        list(APPEND BATCHLAS_MATHDX_TARGETS mathdx::cublasdx)
        if(BATCHLAS_ENABLE_CUBLASDX)
            set(BATCHLAS_HAS_CUBLASDX TRUE PARENT_SCOPE)
        endif()
    endif()
    if(TARGET mathdx::cusolverdx)
        list(APPEND BATCHLAS_MATHDX_TARGETS mathdx::cusolverdx)
        if(BATCHLAS_ENABLE_CUSOLVERDX)
            set(BATCHLAS_HAS_CUSOLVERDX TRUE PARENT_SCOPE)
        endif()
    endif()

    set(BATCHLAS_ENABLE_CUBLASDX_WRAPPER OFF)
    set(BATCHLAS_MATHDX_INCLUDE_DIRS "")
    if(BATCHLAS_MATHDX_TARGETS)
        message(STATUS "Found MathDx package with targets: ${BATCHLAS_MATHDX_TARGETS}")
        if(TARGET mathdx::cublasdx)
            set(BATCHLAS_ENABLE_CUBLASDX_WRAPPER ON)
        endif()
    else()
        set(_BATCHLAS_MATHDX_ROOT_CANDIDATES ${_BATCHLAS_MATHDX_HINTS})
        if(NOT _BATCHLAS_MATHDX_ROOT_CANDIDATES)
            file(GLOB _BATCHLAS_MATHDX_DISCOVERED_ROOTS LIST_DIRECTORIES TRUE
                "/opt/nvidia/mathdx/*"
                "/usr/local/nvidia/mathdx/*")
            list(APPEND _BATCHLAS_MATHDX_ROOT_CANDIDATES ${_BATCHLAS_MATHDX_DISCOVERED_ROOTS})
        endif()
        list(REMOVE_DUPLICATES _BATCHLAS_MATHDX_ROOT_CANDIDATES)

        set(_BATCHLAS_MATHDX_CUBLASDX_HEADER_FOUND OFF)
        foreach(_root ${_BATCHLAS_MATHDX_ROOT_CANDIDATES})
            if(EXISTS "${_root}/include/cublasdx/include/cublasdx.hpp")
                list(APPEND BATCHLAS_MATHDX_INCLUDE_DIRS
                    "${_root}/include"
                    "${_root}/include/cublasdx/include")
                set(_BATCHLAS_MATHDX_CUBLASDX_HEADER_FOUND ON)
            elseif(EXISTS "${_root}/include/cublasdx.hpp")
                list(APPEND BATCHLAS_MATHDX_INCLUDE_DIRS "${_root}/include")
                set(_BATCHLAS_MATHDX_CUBLASDX_HEADER_FOUND ON)
            endif()
            if(EXISTS "${_root}/include/cusolverdx/include/cusolverdx.hpp")
                list(APPEND BATCHLAS_MATHDX_INCLUDE_DIRS
                    "${_root}/include"
                    "${_root}/include/cusolverdx/include")
            elseif(EXISTS "${_root}/include/cusolverdx.hpp")
                list(APPEND BATCHLAS_MATHDX_INCLUDE_DIRS "${_root}/include")
            endif()
            if(EXISTS "${_root}/external/cutlass/include")
                list(APPEND BATCHLAS_MATHDX_INCLUDE_DIRS "${_root}/external/cutlass/include")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES BATCHLAS_MATHDX_INCLUDE_DIRS)

        if(BATCHLAS_MATHDX_INCLUDE_DIRS)
            message(STATUS "Using MathDx headers from: ${BATCHLAS_MATHDX_INCLUDE_DIRS}")
            if(_BATCHLAS_MATHDX_CUBLASDX_HEADER_FOUND)
                set(BATCHLAS_ENABLE_CUBLASDX_WRAPPER ON)
            endif()
        elseif(BATCHLAS_MATHDX_ROOT)
            message(WARNING "BATCHLAS_MATHDX_ROOT was set to '${BATCHLAS_MATHDX_ROOT}', but no MathDx package config or headers were found there")
        else()
            message(STATUS "MathDx package not found; cuBLASDx/cuSolverDx wrappers will remain disabled unless headers are otherwise visible")
        endif()
    endif()
endif()

if(BATCHLAS_MKL_INCLUDE_DIR)
    target_include_directories(batchlas_dep_options INTERFACE
        $<BUILD_INTERFACE:${BATCHLAS_MKL_INCLUDE_DIR}>
    )
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
if(BATCHLAS_HAS_MKL_BACKEND)
    target_compile_definitions(batchlas_dep_options INTERFACE BATCHLAS_HAS_MKL_BACKEND=1)
    if(BATCHLAS_MKL_MANUAL_INSTALL)
        target_compile_definitions(batchlas_dep_options INTERFACE MKL_ILP64)
    endif()
endif()

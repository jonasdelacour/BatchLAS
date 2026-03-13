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
        endif()
    else()
        message(STATUS "MKL not found via CMake package, falling back to manual search")
    endif()
else()
    message(STATUS "MKL backend disabled")
endif()

function(find_nvidia_libs)
    if(NOT BATCHLAS_ENABLE_CUDA)
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
        set(BATCHLAS_HAS_CUDA_BACKEND TRUE PARENT_SCOPE)
    else()
        message(WARNING "NVIDIA GPU detected but cuBLAS library not found. Add its path to CMAKE_PREFIX_PATH if needed.")
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

        set(BATCHLAS_ROCM_LINK_LIBRARIES ${ROCBLAS_LIBRARY} ${HIPSPARSE_LIBRARY} ${ROCSOLVER_LIBRARY} PARENT_SCOPE)
        set(BATCHLAS_HAS_ROCM_BACKEND TRUE PARENT_SCOPE)
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

if(BATCHLAS_ENABLE_CUDA)
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
    endif()
    if(TARGET mathdx::cusolverdx)
        list(APPEND BATCHLAS_MATHDX_TARGETS mathdx::cusolverdx)
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
endif()
if(BATCHLAS_HAS_MKL_BACKEND)
    target_compile_definitions(batchlas_dep_options INTERFACE BATCHLAS_HAS_MKL_BACKEND=1)
    if(BATCHLAS_MKL_MANUAL_INSTALL)
        target_compile_definitions(batchlas_dep_options INTERFACE MKL_ILP64)
    endif()
endif()

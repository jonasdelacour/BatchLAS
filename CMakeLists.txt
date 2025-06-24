cmake_minimum_required(VERSION 3.14)
project(BatchLAS VERSION 0.1.0 LANGUAGES CXX)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Print the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Define compile flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# Include CMakeDependentOption module
include(CMakeDependentOption)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BATCHLAS_BUILD_TESTS "Build BatchLAS tests" ON)
option(BATCHLAS_BUILD_EXAMPLES "Build BatchLAS examples" OFF)
option(BATCHLAS_BUILD_DOCS "Build BatchLAS documentation" OFF)
option(BATCHLAS_ENABLE_CUDA "Enable CUDA support" OFF)
option(BATCHLAS_ENABLE_OPENMP "Enable OpenMP support" OFF)
option(BATCHLAS_ENABLE_ROCM "Enable ROCm support even when no AMD GPU is detected" OFF)
option(BATCHLAS_BUILD_PYTHON "Build Python bindings" OFF)
option(BATCHLAS_ENABLE_NETLIB "Enable Netlib BLAS/LAPACK backend" ON)
option(BATCHLAS_ENABLE_MKL "Enable oneMKL backend" OFF)
set(BATCHLAS_HAS_HOST_BACKEND FALSE)
set(BATCHLAS_HAS_MKL_BACKEND FALSE)
set(BATCHLAS_HAS_CUDA_BACKEND FALSE)
set(BATCHLAS_HAS_ROCM_BACKEND FALSE)

# Default GPU architectures (override with -D or ccmake)
set(BATCHLAS_AMD_ARCH "amd_gpu_gfx942" CACHE STRING "AMD GPU architecture for ROCm")
set(BATCHLAS_NVIDIA_ARCH "sm_50" CACHE STRING "NVIDIA GPU architecture for CUDA")

# SYCL is mandatory for this project
message(STATUS "SYCL support is mandatory for BatchLAS")

# Find dependencies
find_package(OpenMP QUIET)

if(BATCHLAS_ENABLE_MKL)
  # Try to find oneMKL via its CMake package configuration.  This is available
  # when Intel oneAPI is installed and `setvars.sh` has been sourced.
  find_package(MKL CONFIG QUIET)
  if(MKL_FOUND)
    # Determine which imported target exposes the SYCL interface.
    # The name changed from MKL::MKL_DPCPP (<= 2024.x) to MKL::MKL_SYCL (2025.x).
    if(TARGET MKL::MKL_DPCPP)
        set(_MKL_SYCL_TARGET MKL::MKL_DPCPP)
    elseif(TARGET MKL::MKL_SYCL)
        set(_MKL_SYCL_TARGET MKL::MKL_SYCL)
    else()
        message(WARNING
            "oneMKL was found but provides no SYCL interface target "
            "(MKL::MKL_DPCPP or MKL::MKL_SYCL). The MKL backend will be disabled.")
    endif()

    if(DEFINED _MKL_SYCL_TARGET)
        message(STATUS "Found oneMKL SYCL target: ${_MKL_SYCL_TARGET}")
        # Interface target that pulls in the canonical MKL SYCL linkage
        add_library(batchlas_mkl INTERFACE)
        add_library(batchlas::mkl ALIAS batchlas_mkl)
        target_link_libraries(batchlas_mkl INTERFACE ${_MKL_SYCL_TARGET})
        # All BatchLAS code calling MKL uses the ILP64 interfaces
        target_compile_definitions(batchlas_mkl INTERFACE MKL_ILP64)
        set(BATCHLAS_HAS_MKL_BACKEND TRUE)
    endif()
  else()
    message(STATUS "MKL not found via CMake package, falling back to manual search")
  endif()
else()
  message(STATUS "MKL backend disabled")
endif()

find_program(SYCL_LS sycl-ls)
if(SYCL_LS)
  get_filename_component(SYCL_BIN_DIR ${SYCL_LS} DIRECTORY)
  set(SYCL_DIR ${SYCL_BIN_DIR}/../)
  find_program(DPCPP_COMPILER NAMES ${SYCL_BIN_DIR}/icpx ${SYCL_BIN_DIR}/clang++)

  cmake_dependent_option(ENABLE_SYCL "Build SYCL GPU code" ON "DPCPP_COMPILER" OFF)

  if(DPCPP_COMPILER)
    message("Found dpc++ compiler: ${DPCPP_COMPILER}")
    # Cache the compiler choice to stop CMake from thinking the value keeps changing
    set(CMAKE_CXX_COMPILER "${DPCPP_COMPILER}" CACHE FILEPATH "DPC++ compiler" FORCE)
    message(STATUS "C++ compiler locked to: ${CMAKE_CXX_COMPILER}")
  else()
    message("Could not find dpc++ compiler")
  endif()

endif()

# List of SYCL targets detected by the helper functions
set(BATCHLAS_SYCL_TARGETS "")



# Function to detect SYCL GPU architectures
function(detect_sycl_gpu_architectures)
    if(NOT SYCL_LS)
        message(STATUS "sycl-ls not found, skipping automatic GPU target detection")
        return()
    endif()
    
    message(STATUS "Detecting GPU architectures using sycl-ls...")
    
    # Get basic device information
    execute_process(
        COMMAND ${SYCL_LS}
        OUTPUT_VARIABLE SYCL_LS_BASIC_OUTPUT
        ERROR_VARIABLE SYCL_LS_BASIC_ERROR
        RESULT_VARIABLE SYCL_LS_BASIC_RESULT
    )
    
    if(NOT SYCL_LS_BASIC_RESULT EQUAL 0)
        message(WARNING "Failed to execute sycl-ls: ${SYCL_LS_BASIC_ERROR}")
        return()
    endif()
    
    # Get detailed device information including architecture
    execute_process(
        COMMAND ${SYCL_LS} --verbose
        OUTPUT_VARIABLE SYCL_LS_OUTPUT
        ERROR_VARIABLE SYCL_LS_ERROR
        RESULT_VARIABLE SYCL_LS_RESULT
    )
    
    if(NOT SYCL_LS_RESULT EQUAL 0)
        message(WARNING "Failed to execute sycl-ls --verbose: ${SYCL_LS_ERROR}")
        return()
    endif()
    
    # Setup lists to store unique architecture flags and vendor types
    set(GPU_ARCH_FLAGS)
    set(HAS_NVIDIA_GPU FALSE)
    set(HAS_AMD_GPU FALSE)
    set(HAS_INTEL_GPU FALSE)
    
    # First, detect vendors from basic output
    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[cuda:gpu\\]")
        set(HAS_NVIDIA_GPU TRUE)
        message(STATUS "Detected NVIDIA GPU")
    endif()
    
    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[amd:gpu\\].*AMD")
        set(HAS_AMD_GPU TRUE)
        message(STATUS "Detected AMD GPU")
    endif()
    
    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[intel:gpu\\].*Intel")
        set(HAS_INTEL_GPU TRUE)
        message(STATUS "Detected Intel GPU")
    endif()
    
    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[level_zero:gpu\\]")
        set(HAS_INTEL_GPU TRUE)
        message(STATUS "Detected Intel Level Zero GPU")
    endif()
    
    # Look for architecture lines in the verbose output
    string(REGEX MATCHALL "Architecture: ([^\n]+)" ARCH_MATCHES "${SYCL_LS_OUTPUT}")
    foreach(ARCH_MATCH ${ARCH_MATCHES})
        string(REGEX REPLACE "Architecture: ([^\n]+)" "\\1" ARCH_FLAG "${ARCH_MATCH}")
        string(STRIP "${ARCH_FLAG}" ARCH_FLAG)
        # If the detected AMD architecture lacks the vendor prefix, add it.
        if(ARCH_FLAG MATCHES "^gfx[0-9]+$")
            set(ARCH_FLAG "amd_gpu_${ARCH_FLAG}")
        endif()
        # Only keep GPU architecture names (avoid CPU architectures like intel_cpu_spr)
        if(ARCH_FLAG MATCHES "_gpu_")
            list(FIND GPU_ARCH_FLAGS "${ARCH_FLAG}" ARCH_INDEX)
            if(ARCH_INDEX EQUAL -1)
                list(APPEND GPU_ARCH_FLAGS "${ARCH_FLAG}")
                message(STATUS "Detected GPU architecture: ${ARCH_FLAG}")
            endif()
        endif()
    endforeach()
    
    # Store detected architectures for later
    if(GPU_ARCH_FLAGS)
        list(APPEND BATCHLAS_SYCL_TARGETS ${GPU_ARCH_FLAGS})
        message(STATUS "Added targets to SYCL compilation: ${GPU_ARCH_FLAGS}")
    else()
        message(STATUS "No specific GPU architectures detected, using default JIT compilation")
    endif()

    # Propagate list of targets to parent scope
    set(BATCHLAS_SYCL_TARGETS "${BATCHLAS_SYCL_TARGETS}" PARENT_SCOPE)
    
    # Add vendor-specific flags
    if(HAS_NVIDIA_GPU)
        message(STATUS "Adding NVIDIA-specific flags")
        
        # Set BATCHLAS_ENABLE_CUDA to ON when NVIDIA GPU is detected
        set(BATCHLAS_ENABLE_CUDA ON PARENT_SCOPE)
        
        # Attempt to find CUDA installation path
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "Found CUDA at: ${CUDA_TOOLKIT_ROOT_DIR}")
            add_compile_options(-fsycl-unnamed-lambda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
            add_link_options(-fsycl-unnamed-lambda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
            # Store the CUDA path for further use
            set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR} PARENT_SCOPE)
        else()
            # Try to find CUDA using environment variable
            if(DEFINED ENV{CUDA_PATH})
                message(STATUS "Using CUDA from environment variable: $ENV{CUDA_PATH}")
                add_compile_options(-fsycl-unnamed-lambda --cuda-path=$ENV{CUDA_PATH})
                add_link_options(-fsycl-unnamed-lambda --cuda-path=$ENV{CUDA_PATH})
                set(CUDA_PATH $ENV{CUDA_PATH} PARENT_SCOPE)
            else()
                # Try common CUDA installation paths
                foreach(CUDA_DIR "/usr/local/cuda" "/opt/cuda")
                    if(EXISTS "${CUDA_DIR}")
                        message(STATUS "Found CUDA at: ${CUDA_DIR}")
                        add_compile_options(-fsycl-unnamed-lambda --cuda-path=${CUDA_DIR})
                        add_link_options(-fsycl-unnamed-lambda --cuda-path=${CUDA_DIR})
                        set(CUDA_PATH ${CUDA_DIR} PARENT_SCOPE)
                        break()
                    endif()
                endforeach()
                
                if(NOT EXISTS "/usr/local/cuda" AND NOT EXISTS "/opt/cuda")
                    message(WARNING "NVIDIA GPU detected but could not find CUDA installation path. Specify with --cuda-path manually if needed.")
                    add_compile_options(-fsycl-unnamed-lambda)
                    add_link_options(-fsycl-unnamed-lambda)
                endif()
            endif()
        endif()
    endif()
    
    if(HAS_AMD_GPU)
        message(STATUS "Adding AMD-specific flags")
        # Add AMD-specific flags (for GPU only)
        add_compile_options(-fno-sycl-id-queries-fit-in-int)
    endif()
    
    if(HAS_INTEL_GPU)
        message(STATUS "Adding Intel GPU-specific flags")
        # Add Intel GPU-specific flags
        add_compile_options(-fintelfpga)
    endif()
    
    # Set properties for the entire project
    if(HAS_NVIDIA_GPU OR HAS_AMD_GPU OR HAS_INTEL_GPU)
        # Common flags for all GPU types if needed
        add_compile_options(-fsycl-dead-args-optimization)
    endif()
endfunction()

# Detect whether a SYCL CPU device exists and, if so, add a CPU target
function(detect_sycl_cpu_target)
    if(NOT SYCL_LS)
        message(STATUS "sycl-ls not found, skipping CPU target detection")
        return()
    endif()

    execute_process(
        COMMAND ${SYCL_LS}
        OUTPUT_VARIABLE SYCL_LS_CPU_OUTPUT
        ERROR_VARIABLE SYCL_LS_CPU_ERROR
        RESULT_VARIABLE SYCL_LS_CPU_RESULT
    )

    if(NOT SYCL_LS_CPU_RESULT EQUAL 0)
        message(WARNING "Failed to execute sycl-ls: ${SYCL_LS_CPU_ERROR}")
        return()
    endif()

    if(SYCL_LS_CPU_OUTPUT MATCHES "\[.+:cpu\]")
        message(STATUS "Detected SYCL CPU device, adding spir64_x86_64 target")
        list(APPEND BATCHLAS_SYCL_TARGETS spir64_x86_64)
        set(BATCHLAS_SYCL_TARGETS "${BATCHLAS_SYCL_TARGETS}" PARENT_SCOPE)
    else()
        message(STATUS "No SYCL CPU device detected")
    endif()
endfunction()

# Add SYCL specific compiler flags
add_compile_options(-fsycl -Wno-c++20-extensions)
add_link_options(-fsycl)

# Detect and add GPU architecture targets
detect_sycl_gpu_architectures()

# Detect CPU target if available
detect_sycl_cpu_target()

# Function to find NVIDIA libraries like cuBLAS
function(find_nvidia_libs)
    if(NOT BATCHLAS_ENABLE_CUDA)
        return()
    endif()

    message(STATUS "Searching for NVIDIA CUDA libraries...")
    
    # Identify potential HPC SDK base path if using NVIDIA HPC SDK
    set(NVIDIA_HPC_SDK_BASE "")
    
    if(CUDA_PATH)
        # Check if CUDA_PATH is from NVIDIA HPC SDK
        if(CUDA_PATH MATCHES ".*/nvidia/hpc_sdk/.*" OR 
           CUDA_PATH MATCHES ".*/nvhpc/.*")
           
            # Extract HPC SDK base path
            string(REGEX REPLACE "(.*nvidia/hpc_sdk)/.*" "\\1" POTENTIAL_HPC_SDK_BASE "${CUDA_PATH}")
            if(EXISTS "${POTENTIAL_HPC_SDK_BASE}")
                set(NVIDIA_HPC_SDK_BASE "${POTENTIAL_HPC_SDK_BASE}")
                message(STATUS "Detected NVIDIA HPC SDK installation at: ${NVIDIA_HPC_SDK_BASE}")
            endif()
            
            # Alternative pattern for NVHPC directory structure
            if(NOT NVIDIA_HPC_SDK_BASE)
                string(REGEX REPLACE "(.*nvhpc)/.*" "\\1" POTENTIAL_HPC_SDK_BASE "${CUDA_PATH}")
                if(EXISTS "${POTENTIAL_HPC_SDK_BASE}")
                    set(NVIDIA_HPC_SDK_BASE "${POTENTIAL_HPC_SDK_BASE}")
                    message(STATUS "Detected NVIDIA HPC SDK installation at: ${POTENTIAL_HPC_SDK_BASE}")
                endif()
            endif()
            
            # Extract platform and version information if possible
            if(NVIDIA_HPC_SDK_BASE)
                string(REGEX REPLACE "${NVIDIA_HPC_SDK_BASE}/(.*)/cuda.*" "\\1" HPC_SDK_PLATFORM_VERSION "${CUDA_PATH}")
                set(NVIDIA_HPC_SDK_PLATFORM_VERSION "${HPC_SDK_PLATFORM_VERSION}")
                message(STATUS "HPC SDK platform path component: ${NVIDIA_HPC_SDK_PLATFORM_VERSION}")
                
                # Try to extract the actual version number
                if(CUDA_PATH MATCHES ".*/([0-9]+\\.[0-9]+)/cuda.*")
                    string(REGEX REPLACE ".*/([0-9]+\\.[0-9]+)/cuda.*" "\\1" HPC_SDK_VERSION "${CUDA_PATH}")
                    message(STATUS "HPC SDK version: ${HPC_SDK_VERSION}")
                    
                    # Construct potential math_libs path based on detected version
                    set(POTENTIAL_MATH_LIBS_DIR "${NVIDIA_HPC_SDK_BASE}/${NVIDIA_HPC_SDK_PLATFORM_VERSION}/math_libs")
                    if(EXISTS "${POTENTIAL_MATH_LIBS_DIR}")
                        message(STATUS "Found HPC SDK math_libs directory: ${POTENTIAL_MATH_LIBS_DIR}")
                        
                        # Find the latest version in math_libs directory if version not specified
                        file(GLOB MATH_LIBS_VERSIONS "${POTENTIAL_MATH_LIBS_DIR}/*")
                        list(SORT MATH_LIBS_VERSIONS)
                        list(REVERSE MATH_LIBS_VERSIONS) # Latest version first
                        
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
    
    # List of potential NVIDIA HPC SDK and CUDA Toolkit installations paths
    set(NVIDIA_HPC_SDK_PATHS
        "${NVIDIA_HPC_SDK_BASE}"
        "/opt/nvidia/hpc_sdk"
        "/usr/local/nvidia/hpc_sdk"
        "$ENV{NVHPC_ROOT}"
    )
    
    # First try the specific math_libs path if we found it
    if(DEFINED MATH_LIBS_DIR)
        find_library(CUBLAS_LIBRARY 
            NAMES cublas
            PATHS "${MATH_LIBS_DIR}"
            PATH_SUFFIXES 
                "targets/x86_64-linux/lib"
                "lib64"
                "lib"
            NO_DEFAULT_PATH
            DOC "NVIDIA cuBLAS library"
        )
    endif()
    
    # If not found, try broader search
    if(NOT CUBLAS_LIBRARY)
        find_library(CUBLAS_LIBRARY 
            NAMES cublas
            PATHS
                # Check standard CUDA paths first
                ${CUDA_PATH}/lib64
                ${CUDA_PATH}/lib
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ${CUDA_TOOLKIT_ROOT_DIR}/lib
                # Check NVIDIA HPC SDK paths with version structure
                ${NVIDIA_HPC_SDK_PATHS}
            PATH_SUFFIXES 
                # Common CUDA paths
                lib64 lib
                target/x86_64-linux/lib
                targets/x86_64-linux/lib
                # Try to match HPC SDK folder structure with wildcards
                Linux_x86_64/*/math_libs/*/targets/x86_64-linux/lib
                */math_libs/*/targets/x86_64-linux/lib
                */math_libs/lib64
                # Other potential paths if standard structure changes
                Linux_x86_64/*/cuda/lib64
                */cuda/lib64
            DOC "NVIDIA cuBLAS library"
        )
    endif()

    if(CUBLAS_LIBRARY)
        message(STATUS "Found cuBLAS: ${CUBLAS_LIBRARY}")
        get_filename_component(CUBLAS_LIBRARY_DIR ${CUBLAS_LIBRARY} DIRECTORY)
        set(CUBLAS_LIBRARY_DIR ${CUBLAS_LIBRARY_DIR} PARENT_SCOPE)
        
        # Try to find include directory (usually one level up from lib)
        get_filename_component(CUBLAS_PARENT_DIR "${CUBLAS_LIBRARY_DIR}" DIRECTORY)
        if(EXISTS "${CUBLAS_PARENT_DIR}/include")
            set(CUBLAS_INCLUDE_DIR "${CUBLAS_PARENT_DIR}/include")
        else()
            # Try to find at same level as targets
            get_filename_component(TARGETS_PARENT_DIR "${CUBLAS_PARENT_DIR}" DIRECTORY)
            if(EXISTS "${TARGETS_PARENT_DIR}/include")
                set(CUBLAS_INCLUDE_DIR "${TARGETS_PARENT_DIR}/include")
            endif()
        endif()
        
        # Add library directories to paths
        if(DEFINED CUBLAS_INCLUDE_DIR)
            message(STATUS "Found cuBLAS include: ${CUBLAS_INCLUDE_DIR}")
            include_directories(${CUBLAS_INCLUDE_DIR})
        else()
            message(WARNING "cuBLAS include directory not found. You may need to specify it manually.")
        endif()
        link_directories(${CUBLAS_LIBRARY_DIR})
        
        # Look for other libraries in the same directory
        find_library(CUBLASLT_LIBRARY cublasLt PATHS ${CUBLAS_LIBRARY_DIR} NO_DEFAULT_PATH)
        find_library(CUSOLVER_LIBRARY cusolver PATHS ${CUBLAS_LIBRARY_DIR} NO_DEFAULT_PATH)
        find_library(CUSPARSE_LIBRARY cusparse PATHS ${CUBLAS_LIBRARY_DIR} NO_DEFAULT_PATH)
        
        if(CUBLASLT_LIBRARY)
            message(STATUS "Found cuBLASLt: ${CUBLASLT_LIBRARY}")
        endif()
        if(CUSOLVER_LIBRARY)
            message(STATUS "Found cuSOLVER: ${CUSOLVER_LIBRARY}")
        endif()
        if(CUSPARSE_LIBRARY)
            message(STATUS "Found cuSPARSE: ${CUSPARSE_LIBRARY}")
        endif()
        
        # Export found libraries to parent scope
        set(CUBLAS_LIBRARY ${CUBLAS_LIBRARY} PARENT_SCOPE)
        set(CUBLAS_INCLUDE_DIR ${CUBLAS_INCLUDE_DIR} PARENT_SCOPE)
        set(CUBLASLT_LIBRARY ${CUBLASLT_LIBRARY} PARENT_SCOPE)
        set(CUSOLVER_LIBRARY ${CUSOLVER_LIBRARY} PARENT_SCOPE)
        set(CUSPARSE_LIBRARY ${CUSPARSE_LIBRARY} PARENT_SCOPE)
        
        # Set flag that we've found NVIDIA libraries
        set(NVIDIA_LIBS_FOUND TRUE PARENT_SCOPE)
        
        # Set CUDA backend macro
        set(BATCHLAS_HAS_CUDA_BACKEND TRUE PARENT_SCOPE)
        message(STATUS "CUDA backend will be enabled")
    else()
        message(WARNING "NVIDIA GPU detected but cuBLAS library not found. Add its path to CMAKE_PREFIX_PATH if needed.")
    endif()
endfunction()

# Find CUDA libraries if CUDA is enabled (either manually or by GPU detection)
if(BATCHLAS_ENABLE_CUDA)
    enable_language(CUDA)
    find_nvidia_libs()
endif()

# Function to find AMD ROCm libraries
function(find_rocm_libs)
    # First check if we can find the ROCm path
    set(ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    elseif(EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    endif()
    
    if(NOT ROCM_PATH)
        message(STATUS "ROCm path not found, skipping ROCm backend detection")
        return()
    endif()
    
    message(STATUS "Searching for ROCm libraries in: ${ROCM_PATH}")
    
    # Look for hipBLAS
    find_library(HIPBLAS_LIBRARY 
        NAMES hipblas
        PATHS ${ROCM_PATH}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
        DOC "AMD hipBLAS library"
    )
    
    if(HIPBLAS_LIBRARY)
        message(STATUS "Found hipBLAS: ${HIPBLAS_LIBRARY}")
        get_filename_component(HIPBLAS_LIBRARY_DIR ${HIPBLAS_LIBRARY} DIRECTORY)
        
        # Check for include directory
        if(EXISTS "${ROCM_PATH}/include")
            set(HIPBLAS_INCLUDE_DIR "${ROCM_PATH}/include")
        endif()
        
        # Look for other libraries in the same directory
        find_library(ROCBLAS_LIBRARY rocblas PATHS ${HIPBLAS_LIBRARY_DIR} NO_DEFAULT_PATH)
        find_library(HIPSPARSE_LIBRARY hipsparse PATHS ${HIPBLAS_LIBRARY_DIR} NO_DEFAULT_PATH)
        find_library(ROCSOLVER_LIBRARY rocsolver PATHS ${HIPBLAS_LIBRARY_DIR} NO_DEFAULT_PATH)
        
        if(ROCBLAS_LIBRARY)
            message(STATUS "Found rocBLAS: ${ROCBLAS_LIBRARY}")
        endif()
        if(HIPSPARSE_LIBRARY)
            message(STATUS "Found hipSPARSE: ${HIPSPARSE_LIBRARY}")
        endif()
        if(ROCSOLVER_LIBRARY)
            message(STATUS "Found rocSOLVER: ${ROCSOLVER_LIBRARY}")
        endif()
        
        # Export found libraries to parent scope
        set(HIPBLAS_LIBRARY ${HIPBLAS_LIBRARY} PARENT_SCOPE)
        set(HIPBLAS_INCLUDE_DIR ${HIPBLAS_INCLUDE_DIR} PARENT_SCOPE)
        set(ROCBLAS_LIBRARY ${ROCBLAS_LIBRARY} PARENT_SCOPE)
        set(HIPSPARSE_LIBRARY ${HIPSPARSE_LIBRARY} PARENT_SCOPE)
        set(ROCSOLVER_LIBRARY ${ROCSOLVER_LIBRARY} PARENT_SCOPE)
        
        # Add to include directories if found
        if(DEFINED HIPBLAS_INCLUDE_DIR)
            message(STATUS "Found hipBLAS include: ${HIPBLAS_INCLUDE_DIR}")
            include_directories(${HIPBLAS_INCLUDE_DIR})
        endif()
        link_directories(${HIPBLAS_LIBRARY_DIR})
        
        # Set flag that we've found ROCm libraries
        set(ROCM_LIBS_FOUND TRUE PARENT_SCOPE)
        
        # Set ROCm backend macro
        set(BATCHLAS_HAS_ROCM_BACKEND TRUE PARENT_SCOPE)
        message(STATUS "ROCm backend will be enabled")
    else()
        message(STATUS "hipBLAS library not found in ROCm installation")
    endif()
endfunction()

# Check for ROCm libraries if an AMD GPU was detected **or** the user explicitly enabled ROCm support
if(HAS_AMD_GPU OR BATCHLAS_ENABLE_ROCM)
    find_rocm_libs()
endif()

# Finalise SYCL targets based on detected backends
set(DETECTED_AMD_ARCH "")
set(DETECTED_NVIDIA_ARCH "")
foreach(_arch ${BATCHLAS_SYCL_TARGETS})
    if(_arch MATCHES "amd_gpu_gfx[0-9]+")
        set(DETECTED_AMD_ARCH ${_arch})
    elseif(_arch MATCHES "nvidia_gpu_sm_[0-9]+")
        set(DETECTED_NVIDIA_ARCH ${_arch})
    endif()
endforeach()

if(BATCHLAS_HAS_ROCM_BACKEND)
    set(_AMD_ARCH "${BATCHLAS_AMD_ARCH}")
    # Guarantee the canonical prefix for AMD targets provided by users or defaults
    if(_AMD_ARCH MATCHES "^gfx[0-9]+$")
        set(_AMD_ARCH "amd_gpu_${_AMD_ARCH}")
    endif()
    if(DETECTED_AMD_ARCH AND "${BATCHLAS_AMD_ARCH}" STREQUAL "amd_gpu_gfx942")
        set(_AMD_ARCH ${DETECTED_AMD_ARCH})
    endif()
    list(FIND BATCHLAS_SYCL_TARGETS ${_AMD_ARCH} _amd_idx)
    if(_amd_idx EQUAL -1)
        list(APPEND BATCHLAS_SYCL_TARGETS ${_AMD_ARCH})
    endif()
    message(STATUS "Compiling for AMD architecture: ${_AMD_ARCH}")
endif()

if(BATCHLAS_ENABLE_CUDA)
    # Format the default NVIDIA architecture string
    set(_NVIDIA_ARCH "nvidia_gpu_${BATCHLAS_NVIDIA_ARCH}")
    
    # Debug the detected architecture
    message(STATUS "Detected NVIDIA architecture: ${DETECTED_NVIDIA_ARCH}")
    
    # Use detected architecture if available and default is still the original sm_50
    if(DETECTED_NVIDIA_ARCH AND "${BATCHLAS_NVIDIA_ARCH}" STREQUAL "sm_50")
        message(STATUS "Using detected NVIDIA architecture instead of default")
        set(_NVIDIA_ARCH ${DETECTED_NVIDIA_ARCH})
    endif()
    
    # Make sure the architecture is in the targets list
    list(FIND BATCHLAS_SYCL_TARGETS ${_NVIDIA_ARCH} _nv_idx)
    if(_nv_idx EQUAL -1)
        list(APPEND BATCHLAS_SYCL_TARGETS ${_NVIDIA_ARCH})
    endif()
    
    message(STATUS "Compiling for NVIDIA architecture: ${_NVIDIA_ARCH}")
endif()

# Check for oneMKL for Intel GPUs
function(find_onemkl_libs)
    # Check for Intel oneAPI MKL
    set(MKL_ROOT)
    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT $ENV{MKLROOT})
    elseif(EXISTS "/opt/intel/oneapi/mkl")
        set(MKL_ROOT "/opt/intel/oneapi/mkl")
    endif()
    
    if(NOT MKL_ROOT)
        message(STATUS "Intel oneAPI MKL not found, skipping Intel MKL backend detection")
        return()
    endif()
    
    message(STATUS "Searching for Intel oneAPI MKL in: ${MKL_ROOT}")
    
    # Look for oneAPI MKL libraries
    find_library(MKL_CORE_LIBRARY 
        NAMES mkl_core
        PATHS ${MKL_ROOT}
        PATH_SUFFIXES lib lib/intel64
        NO_DEFAULT_PATH
        DOC "Intel oneAPI MKL core library"
    )
    
    if(MKL_CORE_LIBRARY)
        message(STATUS "Found MKL core: ${MKL_CORE_LIBRARY}")
        get_filename_component(MKL_LIBRARY_DIR ${MKL_CORE_LIBRARY} DIRECTORY)
        
        # Check for include directory
        if(EXISTS "${MKL_ROOT}/include")
            set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
        endif()
        
        # Look for other MKL libraries
        find_library(MKL_SYCL_LIBRARY mkl_sycl PATHS ${MKL_LIBRARY_DIR} NO_DEFAULT_PATH)
        find_library(MKL_INTEL_THREAD_LIBRARY mkl_intel_thread PATHS ${MKL_LIBRARY_DIR} NO_DEFAULT_PATH)
        
        if(MKL_SYCL_LIBRARY)
            message(STATUS "Found MKL SYCL: ${MKL_SYCL_LIBRARY}")
        endif()
        
        # Export found libraries to parent scope
        set(MKL_CORE_LIBRARY ${MKL_CORE_LIBRARY} PARENT_SCOPE)
        set(MKL_INCLUDE_DIR ${MKL_INCLUDE_DIR} PARENT_SCOPE)
        set(MKL_SYCL_LIBRARY ${MKL_SYCL_LIBRARY} PARENT_SCOPE)
        
        # Add to include directories if found
        if(DEFINED MKL_INCLUDE_DIR)
            message(STATUS "Found MKL include: ${MKL_INCLUDE_DIR}")
            include_directories(${MKL_INCLUDE_DIR})
        endif()
        link_directories(${MKL_LIBRARY_DIR})
        
        # Set flag that we've found MKL libraries
        set(MKL_LIBS_FOUND TRUE PARENT_SCOPE)
        
        # Set MKL backend macro
        set(BATCHLAS_HAS_MKL_BACKEND TRUE PARENT_SCOPE)
        message(STATUS "Intel MKL backend will be enabled")
    else()
        message(STATUS "Intel MKL library not found")
    endif()
endfunction()

# Check for MKL libraries on CPU or Intel GPU if the MKL package wasn't found
if(BATCHLAS_ENABLE_MKL AND NOT MKL_FOUND)
    find_onemkl_libs()
endif()

# Function to locate Netlib BLAS/LAPACK libraries
function(find_netlib_libs)
    if(NOT BATCHLAS_ENABLE_NETLIB)
        return()
    endif()

    message(STATUS "Searching for Netlib BLAS/LAPACK libraries")

    find_library(LAPACKE_LIBRARY NAMES lapacke)
    find_library(CBLAS_LIBRARY NAMES cblas blas)

    if(LAPACKE_LIBRARY AND CBLAS_LIBRARY)
        message(STATUS "Found LAPACKE: ${LAPACKE_LIBRARY}")
        message(STATUS "Found CBLAS: ${CBLAS_LIBRARY}")
        set(BATCHLAS_LAPACKE_LIBRARY ${LAPACKE_LIBRARY} PARENT_SCOPE)
        set(BATCHLAS_CBLAS_LIBRARY ${CBLAS_LIBRARY} PARENT_SCOPE)
        set(BATCHLAS_HAS_HOST_BACKEND TRUE PARENT_SCOPE)
    else()
        message(WARNING "LAPACKE/CBLAS libraries not found - disabling host backend")
        set(BATCHLAS_HAS_HOST_BACKEND FALSE PARENT_SCOPE)
    endif()
endfunction()

# Detect host libraries
find_netlib_libs()

# Apply final SYCL target flags
list(REMOVE_DUPLICATES BATCHLAS_SYCL_TARGETS)
if(BATCHLAS_SYCL_TARGETS)
    string(REPLACE ";" "," SYCL_TARGETS_STRING "${BATCHLAS_SYCL_TARGETS}")
    add_compile_options(-fsycl-targets=${SYCL_TARGETS_STRING})
    add_link_options(-fsycl-targets=${SYCL_TARGETS_STRING})
    message(STATUS "Using SYCL targets: ${SYCL_TARGETS_STRING}")
endif()

# This ensures proper integration with Intel's SYCL implementation
message(STATUS "Using Intel oneAPI DPC++ compiler for SYCL: ${CMAKE_CXX_COMPILER}")

# Define compile-time backend macros
if(BATCHLAS_HAS_HOST_BACKEND)
    add_compile_definitions(BATCHLAS_HAS_HOST_BACKEND=1)
endif()

if(BATCHLAS_HAS_CUDA_BACKEND)
    add_compile_definitions(BATCHLAS_HAS_CUDA_BACKEND=1)
endif()

if(BATCHLAS_HAS_ROCM_BACKEND)
    add_compile_definitions(BATCHLAS_HAS_ROCM_BACKEND=1)
endif()

if(BATCHLAS_HAS_MKL_BACKEND)
    add_compile_definitions(BATCHLAS_HAS_MKL_BACKEND=1)
    if(NOT MKL_FOUND)
        # When using a manually discovered MKL installation, ensure ILP64
        # interfaces are enabled globally.
        add_compile_definitions(MKL_ILP64)
    endif()
endif()

# Generate backend configuration header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/backend_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/batchlas/backend_config.h
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/batchlas/backend_config.h
        DESTINATION include/batchlas)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /opt/include
)

# Add library subdirectories
add_subdirectory(src)

# Tests
if(BATCHLAS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(benchmarks)
endif()

# Python bindings
if(BATCHLAS_BUILD_PYTHON)
    add_subdirectory(python)
endif()
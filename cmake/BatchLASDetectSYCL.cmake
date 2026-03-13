if(NOT SYCL_LS)
    find_program(SYCL_LS sycl-ls)
endif()

if(SYCL_LS)
    get_filename_component(SYCL_BIN_DIR "${SYCL_LS}" DIRECTORY)
    set(SYCL_DIR "${SYCL_BIN_DIR}/../")
    message(STATUS "Using SYCL tools from: ${SYCL_DIR}")
else()
    message(WARNING "sycl-ls not found - GPU architecture auto-detection will be skipped")
endif()

set(BATCHLAS_SYCL_TARGETS "")
set(BATCHLAS_DETECTED_NVIDIA_GPU FALSE)
set(BATCHLAS_DETECTED_AMD_GPU FALSE)
set(BATCHLAS_DETECTED_INTEL_GPU FALSE)
set(BATCHLAS_SYCL_EXTRA_CXX_OPTIONS "")
set(BATCHLAS_SYCL_EXTRA_LINK_OPTIONS "")
set(BATCHLAS_SYCL_NO_CPU_CXX_OPTIONS "")
set(BATCHLAS_SYCL_BASE_CXX_OPTIONS
    -fsycl
    -Wno-c++20-extensions
    -Wno-option-ignored
)
set(BATCHLAS_SYCL_BASE_LINK_OPTIONS -fsycl)

function(detect_sycl_gpu_architectures)
    if(NOT SYCL_LS)
        message(STATUS "sycl-ls not found, skipping automatic GPU target detection")
        return()
    endif()

    message(STATUS "Detecting GPU architectures using sycl-ls...")

    execute_process(
        COMMAND "${SYCL_LS}"
        OUTPUT_VARIABLE SYCL_LS_BASIC_OUTPUT
        ERROR_VARIABLE SYCL_LS_BASIC_ERROR
        RESULT_VARIABLE SYCL_LS_BASIC_RESULT
    )
    if(NOT SYCL_LS_BASIC_RESULT EQUAL 0)
        message(WARNING "Failed to execute sycl-ls: ${SYCL_LS_BASIC_ERROR}")
        return()
    endif()

    execute_process(
        COMMAND "${SYCL_LS}" --verbose
        OUTPUT_VARIABLE SYCL_LS_OUTPUT
        ERROR_VARIABLE SYCL_LS_ERROR
        RESULT_VARIABLE SYCL_LS_RESULT
    )
    if(NOT SYCL_LS_RESULT EQUAL 0)
        message(WARNING "Failed to execute sycl-ls --verbose: ${SYCL_LS_ERROR}")
        return()
    endif()

    set(_gpu_arch_flags "")
    set(_compile_options "")
    set(_link_options "")
    set(_has_nvidia_gpu FALSE)
    set(_has_amd_gpu FALSE)
    set(_has_intel_gpu FALSE)
    set(_cuda_enabled "${BATCHLAS_ENABLE_CUDA}")
    set(_cuda_path "")

    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[cuda:gpu\\]")
        set(_has_nvidia_gpu TRUE)
        message(STATUS "Detected NVIDIA GPU")
    endif()
    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[amd:gpu\\].*AMD")
        set(_has_amd_gpu TRUE)
        message(STATUS "Detected AMD GPU")
    endif()
    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[intel:gpu\\].*Intel")
        set(_has_intel_gpu TRUE)
        message(STATUS "Detected Intel GPU")
    endif()
    if(SYCL_LS_BASIC_OUTPUT MATCHES "\\[level_zero:gpu\\]")
        set(_has_intel_gpu TRUE)
        message(STATUS "Detected Intel Level Zero GPU")
    endif()

    string(REGEX MATCHALL "Architecture: ([^\n]+)" ARCH_MATCHES "${SYCL_LS_OUTPUT}")
    foreach(ARCH_MATCH ${ARCH_MATCHES})
        string(REGEX REPLACE "Architecture: ([^\n]+)" "\\1" ARCH_FLAG "${ARCH_MATCH}")
        string(STRIP "${ARCH_FLAG}" ARCH_FLAG)
        if(ARCH_FLAG MATCHES "^gfx[0-9]+$")
            set(ARCH_FLAG "amd_gpu_${ARCH_FLAG}")
        endif()
        if(ARCH_FLAG MATCHES "_gpu_")
            list(FIND _gpu_arch_flags "${ARCH_FLAG}" _arch_index)
            if(_arch_index EQUAL -1)
                list(APPEND _gpu_arch_flags "${ARCH_FLAG}")
                message(STATUS "Detected GPU architecture: ${ARCH_FLAG}")
            endif()
        endif()
    endforeach()

    if(_gpu_arch_flags)
        list(APPEND BATCHLAS_SYCL_TARGETS ${_gpu_arch_flags})
        message(STATUS "Added targets to SYCL compilation: ${_gpu_arch_flags}")
    else()
        message(STATUS "No specific GPU architectures detected, using default JIT compilation")
    endif()

    if(_has_nvidia_gpu)
        message(STATUS "Adding NVIDIA-specific flags")
        set(_cuda_enabled ON)

        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND AND CUDAToolkit_BIN_DIR)
            get_filename_component(_cuda_path "${CUDAToolkit_BIN_DIR}" DIRECTORY)
            message(STATUS "Found CUDA at: ${_cuda_path}")
        elseif(DEFINED ENV{CUDA_PATH})
            set(_cuda_path "$ENV{CUDA_PATH}")
            message(STATUS "Using CUDA from environment variable: ${_cuda_path}")
        else()
            foreach(CUDA_DIR "/usr/local/cuda" "/opt/cuda")
                if(EXISTS "${CUDA_DIR}")
                    set(_cuda_path "${CUDA_DIR}")
                    message(STATUS "Found CUDA at: ${_cuda_path}")
                    break()
                endif()
            endforeach()
        endif()

        list(APPEND _compile_options -fsycl-unnamed-lambda)
        list(APPEND _link_options -fsycl-unnamed-lambda)
        if(_cuda_path)
            list(APPEND _compile_options "--cuda-path=${_cuda_path}")
            list(APPEND _link_options "--cuda-path=${_cuda_path}")
        else()
            message(WARNING "NVIDIA GPU detected but could not find CUDA installation path. Specify with --cuda-path manually if needed.")
        endif()
    endif()

    if(_has_amd_gpu)
        message(STATUS "Adding AMD-specific flags")
        list(APPEND _compile_options -fno-sycl-id-queries-fit-in-int)
    endif()

    if(_has_intel_gpu)
        message(STATUS "Adding Intel GPU-specific flags")
        list(APPEND _compile_options -fintelfpga)
    endif()

    if(_has_nvidia_gpu OR _has_amd_gpu OR _has_intel_gpu)
        list(APPEND _compile_options -fsycl-dead-args-optimization)
    endif()

    set(BATCHLAS_ENABLE_CUDA "${_cuda_enabled}" PARENT_SCOPE)
    set(BATCHLAS_CUDA_PATH "${_cuda_path}" PARENT_SCOPE)
    set(BATCHLAS_DETECTED_NVIDIA_GPU "${_has_nvidia_gpu}" PARENT_SCOPE)
    set(BATCHLAS_DETECTED_AMD_GPU "${_has_amd_gpu}" PARENT_SCOPE)
    set(BATCHLAS_DETECTED_INTEL_GPU "${_has_intel_gpu}" PARENT_SCOPE)
    set(BATCHLAS_SYCL_TARGETS "${BATCHLAS_SYCL_TARGETS}" PARENT_SCOPE)
    set(BATCHLAS_SYCL_EXTRA_CXX_OPTIONS "${_compile_options}" PARENT_SCOPE)
    set(BATCHLAS_SYCL_EXTRA_LINK_OPTIONS "${_link_options}" PARENT_SCOPE)
endfunction()

function(detect_sycl_cpu_target)
    if(NOT BATCHLAS_CPU_TARGET STREQUAL "auto")
        if(BATCHLAS_CPU_TARGET STREQUAL "none")
            message(STATUS "CPU target override set to 'none' - disabling CPU target")
            set(BATCHLAS_HAS_CPU_TARGET OFF PARENT_SCOPE)
            return()
        elseif(BATCHLAS_CPU_TARGET STREQUAL "native_cpu")
            message(STATUS "CPU target override set to native_cpu")
            set(BATCHLAS_HAS_CPU_TARGET ON PARENT_SCOPE)
            list(FIND BATCHLAS_SYCL_TARGETS "native_cpu" _cpu_idx)
            if(_cpu_idx EQUAL -1)
                list(APPEND BATCHLAS_SYCL_TARGETS "native_cpu")
                message(STATUS "Added CPU target to SYCL compilation: native_cpu")
            endif()
            set(BATCHLAS_SYCL_TARGETS "${BATCHLAS_SYCL_TARGETS}" PARENT_SCOPE)
            return()
        elseif(BATCHLAS_CPU_TARGET STREQUAL "spir64_x86_64")
            message(STATUS "CPU target override set to spir64_x86_64")
            set(BATCHLAS_HAS_CPU_TARGET ON PARENT_SCOPE)
            list(FIND BATCHLAS_SYCL_TARGETS "spir64_x86_64" _cpu_idx)
            if(_cpu_idx EQUAL -1)
                list(APPEND BATCHLAS_SYCL_TARGETS "spir64_x86_64")
                message(STATUS "Added CPU target to SYCL compilation: spir64_x86_64")
            endif()
            set(BATCHLAS_SYCL_TARGETS "${BATCHLAS_SYCL_TARGETS}" PARENT_SCOPE)
            return()
        else()
            message(WARNING "Unknown BATCHLAS_CPU_TARGET value: ${BATCHLAS_CPU_TARGET} (expected auto|native_cpu|spir64_x86_64|none). Falling back to auto.")
        endif()
    endif()

    if(NOT SYCL_LS)
        message(STATUS "sycl-ls not found, skipping CPU target detection")
        set(BATCHLAS_HAS_CPU_TARGET OFF PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND "${SYCL_LS}"
        OUTPUT_VARIABLE SYCL_LS_CPU_OUTPUT
        ERROR_VARIABLE SYCL_LS_CPU_ERROR
        RESULT_VARIABLE SYCL_LS_CPU_RESULT
    )
    if(NOT SYCL_LS_CPU_RESULT EQUAL 0)
        message(WARNING "Failed to execute sycl-ls: ${SYCL_LS_CPU_ERROR}")
        set(BATCHLAS_HAS_CPU_TARGET OFF PARENT_SCOPE)
        return()
    endif()

    if(SYCL_LS_CPU_OUTPUT MATCHES "\\[opencl:cpu\\]" OR SYCL_LS_CPU_OUTPUT MATCHES "\\[host:cpu\\]")
        message(STATUS "Detected SYCL OpenCL CPU device")
        set(BATCHLAS_HAS_CPU_TARGET ON PARENT_SCOPE)
        list(FIND BATCHLAS_SYCL_TARGETS "spir64_x86_64" _cpu_idx)
        if(_cpu_idx EQUAL -1)
            list(APPEND BATCHLAS_SYCL_TARGETS "spir64_x86_64")
            message(STATUS "Added CPU target to SYCL compilation: spir64_x86_64")
        endif()
    elseif(SYCL_LS_CPU_OUTPUT MATCHES "\\[native_cpu:cpu\\]")
        message(STATUS "Detected SYCL native_cpu device")
        set(BATCHLAS_HAS_CPU_TARGET ON PARENT_SCOPE)
        list(FIND BATCHLAS_SYCL_TARGETS "native_cpu" _cpu_idx)
        if(_cpu_idx EQUAL -1)
            list(APPEND BATCHLAS_SYCL_TARGETS "native_cpu")
            message(STATUS "Added CPU target to SYCL compilation: native_cpu")
        endif()
    else()
        message(STATUS "No SYCL CPU device detected")
        set(BATCHLAS_HAS_CPU_TARGET OFF PARENT_SCOPE)
    endif()

    set(BATCHLAS_SYCL_TARGETS "${BATCHLAS_SYCL_TARGETS}" PARENT_SCOPE)
endfunction()

detect_sycl_gpu_architectures()
detect_sycl_cpu_target()

set(DETECTED_AMD_ARCH "")
set(DETECTED_NVIDIA_ARCH "")
foreach(_arch ${BATCHLAS_SYCL_TARGETS})
    if(_arch MATCHES "amd_gpu_gfx[0-9]+")
        set(DETECTED_AMD_ARCH "${_arch}")
    elseif(_arch MATCHES "nvidia_gpu_sm_[0-9]+")
        set(DETECTED_NVIDIA_ARCH "${_arch}")
    endif()
endforeach()

if(BATCHLAS_ENABLE_ROCM)
    set(_AMD_ARCH "${BATCHLAS_AMD_ARCH}")
    if(_AMD_ARCH MATCHES "^gfx[0-9]+$")
        set(_AMD_ARCH "amd_gpu_${_AMD_ARCH}")
    endif()
    if(DETECTED_AMD_ARCH AND "${BATCHLAS_AMD_ARCH}" STREQUAL "amd_gpu_gfx942")
        set(_AMD_ARCH "${DETECTED_AMD_ARCH}")
    endif()
    list(FIND BATCHLAS_SYCL_TARGETS "${_AMD_ARCH}" _amd_idx)
    if(_amd_idx EQUAL -1)
        list(APPEND BATCHLAS_SYCL_TARGETS "${_AMD_ARCH}")
    endif()
    message(STATUS "Compiling for AMD architecture: ${_AMD_ARCH}")
endif()

if(BATCHLAS_ENABLE_CUDA)
    set(_NVIDIA_ARCH "nvidia_gpu_${BATCHLAS_NVIDIA_ARCH}")
    message(STATUS "Detected NVIDIA architecture: ${DETECTED_NVIDIA_ARCH}")
    if(DETECTED_NVIDIA_ARCH AND "${BATCHLAS_NVIDIA_ARCH}" STREQUAL "sm_50")
        message(STATUS "Using detected NVIDIA architecture instead of default")
        set(_NVIDIA_ARCH "${DETECTED_NVIDIA_ARCH}")
    endif()
    list(FIND BATCHLAS_SYCL_TARGETS "${_NVIDIA_ARCH}" _nv_idx)
    if(_nv_idx EQUAL -1)
        list(APPEND BATCHLAS_SYCL_TARGETS "${_NVIDIA_ARCH}")
    endif()
    message(STATUS "Compiling for NVIDIA architecture: ${_NVIDIA_ARCH}")
endif()

list(REMOVE_DUPLICATES BATCHLAS_SYCL_TARGETS)
set(BATCHLAS_SYCL_TARGETS_STRING "")
set(BATCHLAS_SYCL_TARGETS_NO_CPU "${BATCHLAS_SYCL_TARGETS}")
list(FILTER BATCHLAS_SYCL_TARGETS_NO_CPU EXCLUDE REGEX "cpu|spir64")
set(BATCHLAS_SYCL_TARGETS_NO_CPU_STRING "")

if(BATCHLAS_SYCL_TARGETS)
    string(REPLACE ";" "," BATCHLAS_SYCL_TARGETS_STRING "${BATCHLAS_SYCL_TARGETS}")
    message(STATUS "Using SYCL targets: ${BATCHLAS_SYCL_TARGETS_STRING}")
endif()
if(BATCHLAS_SYCL_TARGETS_NO_CPU)
    string(REPLACE ";" "," BATCHLAS_SYCL_TARGETS_NO_CPU_STRING "${BATCHLAS_SYCL_TARGETS_NO_CPU}")
endif()

if(BATCHLAS_CPU_TARGET STREQUAL "native_cpu" OR BATCHLAS_CPU_TARGET STREQUAL "spir64_x86_64")
    set(BATCHLAS_HAS_CPU_TARGET ON)
endif()

set(HAS_CPU_SYCL_TARGET OFF)
foreach(_target ${BATCHLAS_SYCL_TARGETS})
    if(_target MATCHES "cpu" OR _target MATCHES "spir64" OR _target MATCHES "native_cpu")
        set(HAS_CPU_SYCL_TARGET ON)
        break()
    endif()
endforeach()

if(NOT HAS_CPU_SYCL_TARGET)
    if(BATCHLAS_HAS_CPU_TARGET)
        message(STATUS "CPU device detected by sycl-ls, but no CPU target in fsycl-targets")
        message(STATUS "CPU kernels will not be compiled - disabling CPU-dependent tests/benchmarks")
    endif()
    set(BATCHLAS_HAS_CPU_TARGET OFF)
endif()

if(NOT BATCHLAS_ENABLE_CPU_TESTS)
    message(STATUS "CPU-dependent tests and benchmarks manually disabled")
    set(BATCHLAS_HAS_CPU_TARGET OFF)
endif()

if(BATCHLAS_NATIVE_CPU_DISABLE_VECZ)
    list(FIND BATCHLAS_SYCL_TARGETS "native_cpu" _native_cpu_idx)
    if(NOT _native_cpu_idx EQUAL -1)
        message(STATUS "Disabling vecz for native_cpu backend (-mllvm -sycl-native-cpu-no-vecz)")
        list(APPEND BATCHLAS_SYCL_EXTRA_CXX_OPTIONS -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz)
        list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz)
    endif()
endif()

if(BATCHLAS_ENABLE_CUDA)
    if(BATCHLAS_DISABLE_CUDA_FTZ)
        message(STATUS "Disabling FTZ for CUDA device code (link-time device compilation)")
        list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS
            -Xsycl-target-backend=nvptx64-nvidia-cuda
            --ftz=false
            -Xcuda-ptxas
            --ftz=false
        )
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        message(STATUS "Enabling line info for CUDA device code")
        list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS
            -Xsycl-target-backend=nvptx64-nvidia-cuda
            --generate-line-info
            -Xcuda-ptxas
            -w
        )
    endif()
    if(BATCHLAS_KEEP_CUDA_INTERMEDIATES)
        message(STATUS "Preserving CUDA/SYCL CUDA intermediates for device-code inspection")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/cuda-intermediates")
        list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS
            -Xsycl-target-backend=nvptx64-nvidia-cuda
            --save-temps
            -Xcuda-ptxas
            -v
        )
    endif()
endif()

foreach(_opt IN LISTS BATCHLAS_SYCL_BASE_CXX_OPTIONS BATCHLAS_SYCL_EXTRA_CXX_OPTIONS)
    target_compile_options(batchlas_sycl_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${_opt}>)
    target_compile_options(batchlas_sycl_no_cpu_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${_opt}>)
endforeach()

foreach(_opt IN LISTS BATCHLAS_SYCL_BASE_LINK_OPTIONS BATCHLAS_SYCL_EXTRA_LINK_OPTIONS)
    target_link_options(batchlas_sycl_options INTERFACE "${_opt}")
endforeach()

if(BATCHLAS_SYCL_TARGETS_STRING)
    target_compile_options(batchlas_sycl_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING}>
    )
    target_link_options(batchlas_sycl_options INTERFACE
        -fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING}
    )
endif()

if(BATCHLAS_SYCL_TARGETS_NO_CPU_STRING)
    target_compile_options(batchlas_sycl_no_cpu_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fsycl-targets=${BATCHLAS_SYCL_TARGETS_NO_CPU_STRING}>
    )
endif()

message(STATUS "Using Intel oneAPI DPC++ compiler for SYCL: ${CMAKE_CXX_COMPILER}")

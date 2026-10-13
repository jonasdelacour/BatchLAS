# AdaptiveCpp (acpp) counterpart of BatchLASDetectSYCL.cmake (DPC++). Sets every variable
# the rest of the build reads from that module. Devices come from acpp-info (the acpp
# runtime's own view), falling back to nvidia-smi; sycl-ls belongs to DPC++.
# Design: docs/design/sycl-implementations.md section 5.1.
set(BATCHLAS_DETECT_SYCL_FUNCTIONS_ONLY TRUE)
include("${CMAKE_CURRENT_LIST_DIR}/BatchLASDetectSYCL.cmake")
unset(BATCHLAS_DETECT_SYCL_FUNCTIONS_ONLY)

# acpp-info output -> one "backend|type|name|arch|subgroup|vendor" record per device.
function(batchlas_parse_acpp_info text out_var)
    string(REPLACE "\r\n" "\n" _text "${text}")
    string(REPLACE ";" "," _text "${_text}")
    string(REPLACE "|" "/" _text "${_text}")
    string(REPLACE "\n" ";" _lines "${_text}")
    list(APPEND _lines "Device 999999:")
    set(_out "")
    set(_in_device FALSE)
    foreach(_line IN LISTS _lines)
        string(STRIP "${_line}" _l)
        if(_l MATCHES "^Device [0-9]+:$" OR _l MATCHES "^\\*+ Devices for backend")
            if(_in_device AND _backend)
                list(APPEND _out "${_backend}|${_type}|${_name}|${_arch}|${_sg}|${_vendor}")
            endif()
            set(_in_device FALSE)
            if(_l MATCHES "^Device")
                set(_in_device TRUE)
            endif()
            set(_backend "")
            set(_type "cpu")
            set(_name "")
            set(_arch "unknown")
            set(_sg 1)
            set(_vendor "")
        elseif(_l MATCHES "^Name: (.+)$")
            set(_name "${CMAKE_MATCH_1}")
        elseif(_l MATCHES "^Vendor: (.+)$")
            set(_vendor "${CMAKE_MATCH_1}")
        elseif(_l MATCHES "^Backend: (.+)$")
            set(_backend "${CMAKE_MATCH_1}")
        elseif(_l MATCHES "^Arch: (.+)$")
            set(_arch "${CMAKE_MATCH_1}")
        elseif(_l MATCHES "^Is GPU: 1$")
            set(_type "gpu")
        elseif(_l MATCHES "^sub_group_sizes: ([0-9]+)")
            set(_sg "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

# Script mode (tests/cmake/detect_sycl_arch_tests.cmake) wants the parser only.
if(CMAKE_SCRIPT_MODE_FILE)
    return()
endif()

set(BATCHLAS_ACPP_TARGETS "generic" CACHE STRING
    "--acpp-targets for every BatchLAS TU (generic = single-pass SSCP, JIT-compiled at first launch)")
if(BATCHLAS_ACPP_TARGETS MATCHES ";" OR BATCHLAS_ACPP_TARGETS STREQUAL "")
    message(FATAL_ERROR "BATCHLAS_ACPP_TARGETS must be one acpp target set without ';' "
        "(got '${BATCHLAS_ACPP_TARGETS}'); 'generic' already covers every backend")
endif()

get_filename_component(_acpp_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_acpp_real "${CMAKE_CXX_COMPILER}" REALPATH)
get_filename_component(_acpp_real_bin "${_acpp_real}" DIRECTORY)
get_filename_component(_acpp_root "${_acpp_real_bin}" DIRECTORY)
find_program(BATCHLAS_ACPP_INFO acpp-info HINTS "${_acpp_bin}" "${_acpp_real_bin}" NO_DEFAULT_PATH)

set(_acpp_devices "")
set(_acpp_info_ok FALSE)
if(BATCHLAS_ACPP_INFO)
    execute_process(COMMAND "${BATCHLAS_ACPP_INFO}" OUTPUT_VARIABLE _acpp_info_text
        ERROR_QUIET RESULT_VARIABLE _acpp_info_rc TIMEOUT 180)
    if(_acpp_info_rc EQUAL 0)
        set(_acpp_info_ok TRUE)
        batchlas_parse_acpp_info("${_acpp_info_text}" _acpp_devices)
        message(STATUS "Detecting devices using ${BATCHLAS_ACPP_INFO}")
    endif()
endif()
if(NOT _acpp_info_ok)
    # The CUDA runtime backend must exist in the install for an NVIDIA GPU to be usable.
    set(_rt "${_acpp_root}/lib/hipSYCL/librt-backend")
    if(EXISTS "${_rt}-omp${CMAKE_SHARED_LIBRARY_SUFFIX}")
        list(APPEND _acpp_devices "OpenMP|cpu|AdaptiveCpp OpenMP host device|native_cpu|1|")
    endif()
    if(APPLE AND EXISTS "${_rt}-metal${CMAKE_SHARED_LIBRARY_SUFFIX}")
        list(APPEND _acpp_devices "Metal|gpu|Apple GPU|applegpu_unknown|32|Apple")
    endif()
    if(NOT APPLE)
        find_program(BATCHLAS_NVIDIA_SMI nvidia-smi)
    endif()
    if(BATCHLAS_NVIDIA_SMI AND EXISTS "${_rt}-cuda${CMAKE_SHARED_LIBRARY_SUFFIX}")
        execute_process(COMMAND "${BATCHLAS_NVIDIA_SMI}" --query-gpu=compute_cap,name --format=csv,noheader
            OUTPUT_VARIABLE _smi ERROR_QUIET RESULT_VARIABLE _smi_rc TIMEOUT 30)
        if(_smi_rc EQUAL 0)
            string(REPLACE "\n" ";" _smi "${_smi}")
            foreach(_row IN LISTS _smi)
                if(_row MATCHES "^([0-9]+)\\.([0-9]+), (.+)$")
                    list(APPEND _acpp_devices "CUDA|gpu|${CMAKE_MATCH_3}|sm_${CMAKE_MATCH_1}${CMAKE_MATCH_2}|32|NVIDIA")
                endif()
            endforeach()
        endif()
    endif()
    message(STATUS "acpp-info unavailable; devices from the acpp install (and nvidia-smi off macOS)")
endif()

set(BATCHLAS_DETECTED_NVIDIA_GPU FALSE)
set(BATCHLAS_DETECTED_AMD_GPU FALSE)
set(BATCHLAS_DETECTED_INTEL_GPU FALSE)
set(BATCHLAS_DETECTED_APPLE_GPU FALSE)
set(DETECTED_AMD_ARCH "")
set(DETECTED_NVIDIA_ARCH "")
set(_has_omp FALSE)
set(_entries "")
set(_min_budget 0)
set(_min_safe 0)
foreach(_dev IN LISTS _acpp_devices)
    string(REPLACE "|" ";" _f "${_dev}")
    list(GET _f 0 _backend)
    list(GET _f 1 _type)
    list(GET _f 2 _name)
    list(GET _f 3 _arch)
    list(GET _f 4 _sg)
    list(GET _f 5 _vendor)
    if(_backend STREQUAL "OpenMP")
        set(_has_omp TRUE)
        set(_arch "native_cpu")
    elseif(_type STREQUAL "gpu" AND _backend STREQUAL "CUDA" AND _arch MATCHES "^sm_[0-9]+a?$")
        set(BATCHLAS_DETECTED_NVIDIA_GPU TRUE)
        set(_arch "nvidia_gpu_${_arch}")
        if(NOT DETECTED_NVIDIA_ARCH)
            set(DETECTED_NVIDIA_ARCH "${_arch}")
        endif()
    elseif(_type STREQUAL "gpu" AND _backend MATCHES "HIP" AND _arch MATCHES "^gfx[0-9a-f]+")
        set(BATCHLAS_DETECTED_AMD_GPU TRUE)
        set(_arch "amd_gpu_${_arch}")
        set(DETECTED_AMD_ARCH "${_arch}")
    elseif(_type STREQUAL "gpu" AND _backend STREQUAL "Metal")
        set(BATCHLAS_DETECTED_APPLE_GPU TRUE)
        string(REGEX REPLACE "^applegpu_" "" _arch "${_arch}")
        set(_arch "apple_gpu_${_arch}")
    elseif(_type STREQUAL "gpu" AND _backend MATCHES "Level Zero|OpenCL" AND _vendor MATCHES "Intel")
        set(BATCHLAS_DETECTED_INTEL_GPU TRUE)
        set(_arch "intel_gpu")
    endif()
    batchlas_architecture_table_local_mem_bytes("${_arch}" "${_type}" _local)
    batchlas_subgroup_workspace_budget_bytes("${_arch}" "${_local}" "${_type}" _budget)
    batchlas_safe_subgroups_per_workgroup("${_arch}" "${_type}" _safe)
    set(_entry "AdaptiveCpp ${_backend}|${_type}|${_name}|${_arch}|${_sg}|${_local}|${_budget}")
    list(FIND _entries "${_entry}" _idx)
    if(_idx EQUAL -1)
        list(APPEND _entries "${_entry}")
        message(STATUS "Detected ${_backend} ${_type}: ${_name} (${_arch})")
    endif()
    if(_type STREQUAL "gpu")
        if(_min_budget EQUAL 0 OR _budget LESS _min_budget)
            set(_min_budget "${_budget}")
        endif()
        if(_min_safe EQUAL 0 OR _safe LESS _min_safe)
            set(_min_safe "${_safe}")
        endif()
    endif()
endforeach()
if(_min_budget EQUAL 0)
    set(_min_budget 28672)
endif()
if(_min_safe EQUAL 0)
    set(_min_safe 2)
endif()
set(BATCHLAS_DETECTED_DEVICE_LIMIT_ENTRIES "${_entries}")
set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES "${_min_budget}")
set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SAFE_SUBGROUPS_PER_WORKGROUP "${_min_safe}")

# Same AUTO|ON|OFF contract as the DPC++ module: ON without a CUDA device is an error.
if(BATCHLAS_ENABLE_CUDA_MODE STREQUAL "ON" AND NOT BATCHLAS_DETECTED_NVIDIA_GPU)
    message(FATAL_ERROR
        "BATCHLAS_ENABLE_CUDA=ON but the AdaptiveCpp runtime exposes no CUDA device "
        "(acpp-info/nvidia-smi found none, or ${_acpp_root} has no CUDA backend). "
        "Configure with -DBATCHLAS_ENABLE_CUDA=AUTO or =OFF.")
endif()
if(BATCHLAS_ENABLE_CUDA_MODE STREQUAL "AUTO" AND BATCHLAS_DETECTED_NVIDIA_GPU)
    set(BATCHLAS_CUDA_ENABLED ON)
endif()
if(BATCHLAS_ENABLE_METAL_MODE STREQUAL "ON" AND NOT BATCHLAS_DETECTED_APPLE_GPU)
    message(FATAL_ERROR
        "BATCHLAS_ENABLE_METAL=ON but the AdaptiveCpp runtime exposes no Metal GPU "
        "(acpp-info lists none, or ${_acpp_root} was built without WITH_METAL_BACKEND). "
        "Configure with -DBATCHLAS_ENABLE_METAL=AUTO or =OFF.")
endif()
# OFF with a visible Metal GPU would compile the Metal guards out while default queues still land on
# it, so the NETLIB fallback's native double and vector kernels would reach the Metal JIT unguarded.
if(BATCHLAS_DETECTED_APPLE_GPU AND BATCHLAS_ENABLE_METAL_MODE STREQUAL "OFF"
   AND ("$ENV{ACPP_VISIBILITY_MASK}" STREQUAL "" OR "$ENV{ACPP_VISIBILITY_MASK}" MATCHES "metal"))
    message(FATAL_ERROR
        "BATCHLAS_ENABLE_METAL=OFF but the AdaptiveCpp runtime exposes a Metal GPU, which default "
        "queues would still use without the Metal backend's guards. For a host-only build hide it "
        "with ACPP_VISIBILITY_MASK=omp (configure, build and every run); otherwise use AUTO or ON.")
endif()
if(BATCHLAS_DETECTED_APPLE_GPU AND NOT BATCHLAS_ENABLE_METAL_MODE STREQUAL "OFF")
    set(BATCHLAS_HAS_SYCL_BACKEND TRUE)
    # Device headers swap their vector registers for plain structs under this, for the SSCP
    # Metal emitter; CUDA codegen keeps the vectors.
    set(BATCHLAS_DEVICE_NO_VECTOR_TYPES TRUE)
endif()
# Read by .github/ci/compare_failures.py to pick the failure ledger of a Metal tree.
if(BATCHLAS_HAS_SYCL_BACKEND)
    set(BATCHLAS_GPU_FAMILY_RESOLVED "METAL" CACHE INTERNAL "GPU family the tree was configured for")
else()
    set(BATCHLAS_GPU_FAMILY_RESOLVED "" CACHE INTERNAL "GPU family the tree was configured for")
endif()
set(BATCHLAS_CUDA_PATH "")

if(BATCHLAS_CPU_TARGET STREQUAL "none" OR NOT BATCHLAS_ENABLE_CPU_TESTS)
    set(BATCHLAS_HAS_CPU_TARGET OFF)
elseif(_has_omp AND BATCHLAS_ACPP_TARGETS MATCHES "^(generic|omp)")
    # Host kernels exist only when the target set compiles for the host (generic or omp*).
    set(BATCHLAS_HAS_CPU_TARGET ON)
else()
    set(BATCHLAS_HAS_CPU_TARGET OFF)
endif()
# Metal has no fp64: double runs only on the host (OpenMP device or LAPACKE).
if(BATCHLAS_HAS_SYCL_BACKEND)
    if(BATCHLAS_DEVICE_FP64_MODE STREQUAL "ON")
        message(FATAL_ERROR "BATCHLAS_DEVICE_FP64=ON but the Metal GPU has no double precision; "
            "use AUTO or OFF (double then runs on the host backend)")
    endif()
    set(BATCHLAS_HAS_DEVICE_FP64 OFF)
    if(NOT BATCHLAS_HAS_CPU_TARGET)
        message(FATAL_ERROR "The Metal backend needs the AdaptiveCpp OpenMP host device "
            "(BATCHLAS_CPU_TARGET=auto, BATCHLAS_ENABLE_CPU_TESTS=ON, ACPP_VISIBILITY_MASK including omp): "
            "double and the verification layer run there")
    endif()
endif()
message(STATUS "Native SYCL GPU backend (Metal): ${BATCHLAS_HAS_SYCL_BACKEND}; "
    "device fp64: ${BATCHLAS_HAS_DEVICE_FP64} (BATCHLAS_DEVICE_FP64=${BATCHLAS_DEVICE_FP64})")
if(NOT BATCHLAS_CPU_TARGET MATCHES "^(auto|none)$")
    message(WARNING "BATCHLAS_CPU_TARGET=${BATCHLAS_CPU_TARGET} is a DPC++ target; "
        "with AdaptiveCpp the CPU runs through the OpenMP backend (auto|none)")
endif()

# acpp compiles every backend from one target set, so there is no CPU-free variant.
set(BATCHLAS_SYCL_TARGETS "${BATCHLAS_ACPP_TARGETS}")
set(BATCHLAS_SYCL_TARGETS_STRING "${BATCHLAS_ACPP_TARGETS}")
set(BATCHLAS_SYCL_TARGETS_NO_CPU "")
set(BATCHLAS_SYCL_TARGETS_NO_CPU_STRING "")
set(BATCHLAS_SYCL_BACKEND_OPTION "")
set(BATCHLAS_SYCL_BACKEND_OPTION_STRING "")
set(BATCHLAS_SYCL_CONSUMER_OPTIONS "--acpp-targets=${BATCHLAS_ACPP_TARGETS}")
message(STATUS "Using AdaptiveCpp targets: ${BATCHLAS_ACPP_TARGETS}")

# Literal target names, not a foreach: check_exported_package.py lints only calls it can name.
target_compile_options(batchlas_sycl_options INTERFACE
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:--acpp-targets=${BATCHLAS_ACPP_TARGETS}>>")
target_link_options(batchlas_sycl_options INTERFACE "$<BUILD_INTERFACE:--acpp-targets=${BATCHLAS_ACPP_TARGETS}>")
target_compile_options(batchlas_sycl_no_cpu_options INTERFACE
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:--acpp-targets=${BATCHLAS_ACPP_TARGETS}>>")
target_link_options(batchlas_sycl_no_cpu_options INTERFACE "$<BUILD_INTERFACE:--acpp-targets=${BATCHLAS_ACPP_TARGETS}>")

# Device __mulsc3/__muldc3/__divsc3/__divdc3 (Annex G) in every TU; SSCP ships none, and a shared
# definitions TU cannot link (see the header). Joined -include<path>: one option, never de-duplicated.
set(BATCHLAS_ACPP_ANNEXG_HEADER "${PROJECT_SOURCE_DIR}/src/sycl/annexg_complex.hh")
target_compile_options(batchlas_sycl_options INTERFACE
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-include${BATCHLAS_ACPP_ANNEXG_HEADER}>>")
target_compile_options(batchlas_sycl_no_cpu_options INTERFACE
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-include${BATCHLAS_ACPP_ANNEXG_HEADER}>>")

# The acpp driver adds -ffp-contract=fast at -O2 and above (fusing across statements); DPC++ fuses
# only within one expression. Spelling it out keeps both rounding the same way (design page, R11).
target_compile_options(batchlas_sycl_options INTERFACE
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=on>>")
target_compile_options(batchlas_sycl_no_cpu_options INTERFACE
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=on>>")

message(STATUS "Using AdaptiveCpp for SYCL: ${CMAKE_CXX_COMPILER}")

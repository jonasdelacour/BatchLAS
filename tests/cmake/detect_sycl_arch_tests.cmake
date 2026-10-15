# cmake -DBATCHLAS_SOURCE_DIR=<src> -P detect_sycl_arch_tests.cmake
#
# Script-mode checks of the NVIDIA architecture recovery in BatchLASDetectSYCL.cmake:
# no compiler, GPU or sycl-ls is needed, so they run on any box.
if(DEFINED BATCHLAS_EXPECT_FATAL)
    include("${BATCHLAS_SOURCE_DIR}/cmake/BatchLASDetectSYCL.cmake")
    batchlas_rewrite_unaliased_nvidia_targets("nvidia_gpu_sm_100;nvidia_gpu_sm_120"
        "nvidia_gpu_sm_100;nvidia_gpu_sm_120" _t _o)
    return()
endif()

include("${BATCHLAS_SOURCE_DIR}/cmake/BatchLASDetectSYCL.cmake")

set(_failures 0)
function(expect what actual expected)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(SEND_ERROR "${what}: got '${actual}', expected '${expected}'")
    else()
        message(STATUS "ok  ${what}")
    endif()
endfunction()

# Trimmed from a real `sycl-ls --verbose` (intel/llvm sycl, 2026-09) on an RTX PRO 6000
# Blackwell (compute capability 12.0), plus an sm_89 card and an OpenCL CPU whose
# architecture is genuinely unknown to the same table.
set(_sycl_ls "Platforms: 2
Platform [#1]:
    Version  : CUDA 13.2
    Name     : NVIDIA CUDA BACKEND
    Devices  : 2
        Device [#0]:
        Type              : gpu
        Version           : 12.0
        Name              : NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
        Vendor            : NVIDIA Corporation
        Driver            : CUDA 13.2
        info::device::sub_group_sizes: 32
        Architecture: unknown
        Device [#1]:
        Type              : gpu
        Version           : 8.9
        Name              : NVIDIA GeForce RTX 4090
        Vendor            : NVIDIA Corporation
        Architecture: nvidia_gpu_sm_89
Platform [#2]:
    Version  : OpenCL 3.0 LINUX
    Name     : Intel(R) OpenCL
        Device [#0]:
        Type              : cpu
        Version           : 3.0
        Name              : AMD Ryzen Threadripper PRO 7975WX 32-Cores
        Vendor            : Intel(R) Corporation
        Architecture: unknown
")

batchlas_normalize_sycl_ls_architectures("${_sycl_ls}" _norm)
string(REGEX MATCHALL "Architecture: [^\n]+" _archs "${_norm}")
expect("unknown NVIDIA arch recovered from its compute capability; others untouched"
    "${_archs}"
    "Architecture: nvidia_gpu_sm_120;Architecture: nvidia_gpu_sm_89;Architecture: unknown")

# A compute capability must not leak into the next device: a non-NVIDIA device with no
# numeric Version keeps "unknown" even after an NVIDIA one.
batchlas_normalize_sycl_ls_architectures("Device [#0]:
Version : 12.0
Vendor : NVIDIA Corporation
Device [#1]:
Vendor : NVIDIA Corporation
Architecture: unknown
" _leak)
string(REGEX MATCH "Architecture: [^\n]+" _leak_arch "${_leak}")
expect("compute capability is reset per device" "${_leak_arch}" "Architecture: unknown")

# What the recovery is for: the Blackwell card gets NVIDIA's rows, not the
# unrecognised-GPU ones (budget 28672).
batchlas_architecture_table_local_mem_bytes("nvidia_gpu_sm_120" "gpu" _bytes)
batchlas_subgroup_workspace_budget_bytes("nvidia_gpu_sm_120" "${_bytes}" "gpu" _budget)
expect("sm_120 workspace budget" "${_budget}" "45056")

batchlas_rewrite_unaliased_nvidia_targets("nvidia_gpu_sm_120;native_cpu" "nvidia_gpu_sm_120"
    _targets _option)
expect("unaliased target becomes the generic triple" "${_targets}" "nvptx64-nvidia-cuda;native_cpu")
expect("backend option carries the architecture as one SHELL: item" "${_option}"
    "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_120")

batchlas_rewrite_unaliased_nvidia_targets("nvidia_gpu_sm_89;nvidia_gpu_sm_120" "nvidia_gpu_sm_120"
    _mixed _mixed_option)
expect("an aliased target is kept next to the generic one" "${_mixed}"
    "nvidia_gpu_sm_89;nvptx64-nvidia-cuda")

batchlas_rewrite_unaliased_nvidia_targets("nvidia_gpu_sm_89;native_cpu" "" _untouched _no_option)
expect("nothing unaliased leaves the targets alone" "${_untouched}" "nvidia_gpu_sm_89;native_cpu")
expect("nothing unaliased adds no option" "${_no_option}" "")

# AdaptiveCpp: trimmed acpp-info (develop 9ec394ef, macOS 27) listing the OpenMP host device and
# an Apple M5 Max, plus an OpenCL GPU whose vendor is not Intel and must not become intel_gpu.
include("${BATCHLAS_SOURCE_DIR}/cmake/BatchLASSyclAcpp.cmake")
batchlas_parse_acpp_info("***************** Devices for backend OpenMP *****************
Device 0:
 General device information:
  Name: AdaptiveCpp OpenMP host device
  Backend: OpenMP
  Vendor: the AdaptiveCpp project
  Arch: <native-cpu>
  Is CPU: 1
  Is GPU: 0
***************** Devices for backend Metal *****************
Device 0:
 General device information:
  Name: Apple M5 Max
  Backend: Metal
  Vendor: Apple
  Arch: applegpu_g17s
  Driver version: Version 27.0.1 (Build 26A434), gpufamilies: Apple1,Apple2
  Is CPU: 0
  Is GPU: 1
  sub_group_sizes: 32
***************** Devices for backend OpenCL *****************
Device 0:
  Name: Apple M5 Max
  Backend: OpenCL
  Vendor: Apple
  Is GPU: 1
" _acpp_devices)
expect("acpp-info records carry the vendor as a sixth field" "${_acpp_devices}"
    "OpenMP|cpu|AdaptiveCpp OpenMP host device|<native-cpu>|1|the AdaptiveCpp project;Metal|gpu|Apple M5 Max|applegpu_g17s|32|Apple;OpenCL|gpu|Apple M5 Max|unknown|1|Apple")
batchlas_architecture_table_local_mem_bytes("apple_gpu_g17s" "gpu" _apple_bytes)
batchlas_subgroup_workspace_budget_bytes("apple_gpu_g17s" "${_apple_bytes}" "gpu" _apple_budget)
expect("Apple GPU workspace budget" "${_apple_budget}" "28672")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -DBATCHLAS_SOURCE_DIR=${BATCHLAS_SOURCE_DIR} -DBATCHLAS_EXPECT_FATAL=1
            -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE _fatal_rc OUTPUT_QUIET ERROR_VARIABLE _fatal_err)
if(_fatal_rc EQUAL 0 OR NOT _fatal_err MATCHES "carries a single architecture")
    message(SEND_ERROR "two unaliased NVIDIA targets must be a configure error (rc=${_fatal_rc})")
else()
    message(STATUS "ok  two unaliased NVIDIA targets are a configure error")
endif()

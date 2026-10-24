# DPC++ (intel/llvm, icpx). The AdaptiveCpp counterpart is BatchLASSyclAcpp.cmake, which
# include()s this file with BATCHLAS_DETECT_SYCL_FUNCTIONS_ONLY for the shared derivations.
#
# Under `cmake -P` there is no project to configure and no toolchain to detect, so a
# script that include()s this file gets the derivation functions and nothing else. That
# is what lets the per-architecture caps be checked for AMD and Intel from a box that has
# neither.
if(CMAKE_SCRIPT_MODE_FILE)
    set(BATCHLAS_DETECT_SYCL_FUNCTIONS_ONLY TRUE)
endif()

set(BATCHLAS_SYCL_TARGETS "")
set(BATCHLAS_DETECTED_NVIDIA_GPU FALSE)
set(BATCHLAS_DETECTED_AMD_GPU FALSE)
set(BATCHLAS_SYCL_EXTRA_CXX_OPTIONS "")
set(BATCHLAS_SYCL_EXTRA_LINK_OPTIONS "")
set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES 28672)

set(BATCHLAS_SYCL_BASE_CXX_OPTIONS
    -fsycl
    -Wno-c++20-extensions
    -Wno-option-ignored
)
# The SYCL device link runs llvm-foreach over the device modules sycl-post-link splits
# out. Parallelising it is pure scheduling - the device images come out byte-identical
# (factorization 26.5 s -> 17.0 s at N=4). It cannot split one module, so a library
# dominated by one large image (latrd, posv, gesv) gains nothing, and executable links
# carry no device code. N=4 is conservative against oversubscription under `make -j`.
set(BATCHLAS_SYCL_BASE_LINK_OPTIONS -fsycl)
if(BATCHLAS_SYCL_LINK_JOBS GREATER 1)
    list(APPEND BATCHLAS_SYCL_BASE_LINK_OPTIONS
        -fsycl-max-parallel-link-jobs=${BATCHLAS_SYCL_LINK_JOBS})
endif()

# The per-architecture local-memory table. Its one caller is the tile cap below, which
# needs it as the HISTORICAL figure it must not move away from. It is deliberately NOT a device capacity -- no kernel may size anything from
# it, because a build-host table cannot bind the device the process later opens; every
# resident capacity asks DeviceProperty::LOCAL_MEM_SIZE instead
# (src/util/resident_capacity.hh).
function(batchlas_architecture_table_local_mem_bytes architecture device_type out_var)
    set(_table_bytes 32768)

    if("${device_type}" STREQUAL "gpu")
        if("${architecture}" MATCHES "^nvidia_gpu_sm_[0-9]+$")
            set(_table_bytes 49152)
        elseif("${architecture}" MATCHES "^amd_gpu_gfx[0-9]+$")
            set(_table_bytes 65536)
        elseif("${architecture}" MATCHES "^intel")
            set(_table_bytes 65536)
        elseif("${architecture}" MATCHES "^apple_gpu_")
            # Hypothesis (Metal maxThreadgroupMemoryLength on M5 Max), not a measured budget.
            set(_table_bytes 32768)
        endif()
    endif()

    set(${out_var} "${_table_bytes}" PARENT_SCOPE)
endfunction()

# The DEVICE-GEMM tiles' compile-time workspace cap, and the only value in the generated
# header anything reads (group_blas_subgroup_common.hh's kSubgroupWorkspaceBudgetBytes).
#
# It is derived from the ARCHITECTURE TABLE and never from a device capacity, even a true
# one. The budget sizes fixed arrays inside device structs and drives five `if constexpr`
# gates spanning gemm, symm, herk, syrk and syr2k, so it is a tuning constant wearing a
# capacity's clothes: raising it changes both which tile variants compile in and how many
# subgroups each stages. Feeding sm_89's real 101,376 into it would retune five ops as a
# side effect of correcting a capacity table, and no benchmark in this tree has ever
# timed any budget but the one each architecture already ships.
#
# The `- 4096` reserve and the 16384 floor are likewise historical, not derived: together
# with the table they reproduce, byte for byte, the budget every architecture had before
# any of this was touched (NVIDIA 45,056, AMD and Intel 61,440, unrecognised GPU 28,672).
# Changing any of the three is a measured change, not a cleanup.
# evidence: docs/perf/gemm.md#the-subgroup-workspace-budget
function(batchlas_subgroup_workspace_budget_bytes architecture local_mem_bytes device_type out_var)
    set(_workspace_budget_bytes "${local_mem_bytes}")

    if("${device_type}" STREQUAL "gpu")
        batchlas_architecture_table_local_mem_bytes("${architecture}" "${device_type}" _table_bytes)
        math(EXPR _workspace_budget_bytes "${_table_bytes} - 4096")

        if(_workspace_budget_bytes LESS 16384)
            set(_workspace_budget_bytes 16384)
        endif()
    endif()

    set(${out_var} "${_workspace_budget_bytes}" PARENT_SCOPE)
endfunction()

# DPC++ names an NVIDIA device's architecture only if its SYCL table lists it, and that
# table ends at sm_90a (intel/llvm sycl tip, 2026-09: no sm_100/sm_120). sycl-ls then
# prints "Architecture: unknown", and a Blackwell card fell into the unrecognised-GPU
# rows of the tables above and out of -fsycl-targets. The device's "Version" line still
# carries the compute capability, so recover nvidia_gpu_sm_<cc> from it.
function(batchlas_normalize_sycl_ls_architectures sycl_ls_text out_var)
    string(REPLACE "\r\n" "\n" _text "${sycl_ls_text}")
    string(REPLACE "\n" ";" _lines "${_text}")
    set(_out "")
    set(_cc "")
    set(_vendor "")
    foreach(_line IN LISTS _lines)
        string(STRIP "${_line}" _trimmed)
        if(_trimmed MATCHES "^(Platform|Device) \\[#")
            set(_cc "")
            set(_vendor "")
        elseif(_trimmed MATCHES "^Version[ ]*:[ ]*([0-9]+)\\.([0-9]+)$")
            set(_cc "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        elseif(_trimmed MATCHES "^Vendor[ ]*:[ ]*(.*)$")
            set(_vendor "${CMAKE_MATCH_1}")
        elseif(_trimmed STREQUAL "Architecture: unknown" AND _cc AND _vendor MATCHES "NVIDIA")
            string(REPLACE "unknown" "nvidia_gpu_sm_${_cc}" _line "${_line}")
        endif()
        string(APPEND _out "${_line}\n")
    endforeach()
    set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

# An nvidia_gpu_sm_<N> the compiler has no -fsycl-targets alias for is spelled as the
# generic triple plus a backend architecture, which gives the same codegen (-target-cpu,
# __SYCL_CUDA_ARCH__ and the PTX .target are all sm_<N>). The bare triple would silently
# build sm_75. The triple carries one architecture, so two unaliased ones cannot coexist.
# The option is one SHELL: item because CMake de-duplicates options, and the FTZ link
# option repeats -Xsycl-target-backend=nvptx64-nvidia-cuda.
function(batchlas_rewrite_unaliased_nvidia_targets targets unaliased out_targets out_option)
    list(LENGTH unaliased _count)
    if(_count GREATER 1)
        message(FATAL_ERROR
            "This compiler has no -fsycl-targets alias for ${unaliased}, and the generic "
            "nvptx64-nvidia-cuda target carries a single architecture. Build for one of "
            "them with -DBATCHLAS_NVIDIA_ARCH=<sm_N>.")
    endif()
    set(_targets "")
    set(_option "")
    foreach(_target IN LISTS targets)
        list(FIND unaliased "${_target}" _idx)
        if(_idx EQUAL -1)
            list(APPEND _targets "${_target}")
        else()
            string(REGEX REPLACE "^nvidia_gpu_" "" _sm "${_target}")
            list(APPEND _targets "nvptx64-nvidia-cuda")
            set(_option "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${_sm}")
        endif()
    endforeach()
    set(${out_targets} "${_targets}" PARENT_SCOPE)
    set(${out_option} "${_option}" PARENT_SCOPE)
endfunction()

# Asks the driver alone (-###, nothing is compiled) whether it knows the alias.
function(batchlas_nvidia_target_has_alias target out_var)
    set(_args -fsycl "-fsycl-targets=${target}" "-###" -x c++ -c /dev/null -o /dev/null)
    if(BATCHLAS_CUDA_PATH)
        list(APPEND _args "--cuda-path=${BATCHLAS_CUDA_PATH}")
    endif()
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" ${_args}
        OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
    if(_err MATCHES "SYCL target is invalid")
        set(${out_var} FALSE PARENT_SCOPE)
    else()
        set(${out_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

# FindCUDAToolkit searches CUDAToolkit_ROOT, $CUDA_PATH, nvcc on PATH and /usr/local/cuda. The
# NVIDIA HPC SDK installs the toolkit as <sdk>/<release>/cuda/<version> and puts nothing on a new
# login's PATH, so a box with only the SDK found no toolkit at all. ARGN (acpp's configured CUDA)
# is tried first, then the newest SDK toolkit. The pick is cached as CUDAToolkit_ROOT.
function(batchlas_locate_cuda_toolkit)
    if(CUDAToolkit_ROOT OR DEFINED ENV{CUDAToolkit_ROOT} OR DEFINED ENV{CUDA_PATH})
        return()
    endif()
    find_program(_batchlas_nvcc nvcc)
    set(_nvcc "${_batchlas_nvcc}")
    unset(_batchlas_nvcc CACHE)
    if(_nvcc OR EXISTS "/usr/local/cuda/bin/nvcc")
        return()
    endif()
    set(_best "")
    foreach(_dir IN LISTS ARGN)
        if(EXISTS "${_dir}/bin/nvcc")
            set(_best "${_dir}")
            break()
        endif()
    endforeach()
    if(NOT _best)
        file(GLOB _sdk_toolkits LIST_DIRECTORIES true "/opt/nvidia/hpc_sdk/*/*/cuda/[0-9]*")
        set(_best_cuda "0")
        set(_best_sdk "0")
        foreach(_dir IN LISTS _sdk_toolkits)
            if(EXISTS "${_dir}/bin/nvcc" AND _dir MATCHES "/([0-9.]+)/cuda/([0-9.]+)$")
                if(CMAKE_MATCH_2 VERSION_GREATER _best_cuda OR
                   (CMAKE_MATCH_2 VERSION_EQUAL _best_cuda AND CMAKE_MATCH_1 VERSION_GREATER _best_sdk))
                    set(_best "${_dir}")
                    set(_best_cuda "${CMAKE_MATCH_2}")
                    set(_best_sdk "${CMAKE_MATCH_1}")
                endif()
            endif()
        endforeach()
    endif()
    if(_best)
        get_filename_component(_best "${_best}" REALPATH)
        set(CUDAToolkit_ROOT "${_best}" CACHE PATH "CUDA toolkit root (no nvcc on PATH; located by BatchLAS)")
        message(STATUS "CUDA toolkit (no nvcc on PATH): ${_best}")
    endif()
endfunction()

# Runs the compiler's own sycl-ls against its own runtime, once. A sycl-ls on PATH can belong to
# another install (oneAPI's has no CUDA adapter), and neither runs without <prefix>/lib on the
# loader path, which a plain login shell does not have: detection used to find nothing and the
# build silently had no GPU target.
macro(batchlas_run_sycl_ls)
    get_filename_component(_batchlas_cxx_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_batchlas_cxx_real "${CMAKE_CXX_COMPILER}" REALPATH)
    get_filename_component(_batchlas_cxx_real_dir "${_batchlas_cxx_real}" DIRECTORY)
    get_filename_component(_batchlas_sycl_root "${_batchlas_cxx_real_dir}" DIRECTORY)
    find_program(SYCL_LS sycl-ls HINTS "${_batchlas_cxx_dir}" "${_batchlas_cxx_real_dir}" NO_DEFAULT_PATH)
    find_program(SYCL_LS sycl-ls)
    set(BATCHLAS_SYCL_RUNTIME_DIR "")
    if(EXISTS "${_batchlas_sycl_root}/lib/libsycl.so")
        set(BATCHLAS_SYCL_RUNTIME_DIR "${_batchlas_sycl_root}/lib")
    endif()
    set(BATCHLAS_SYCL_LS_OUTPUT "")
    set(BATCHLAS_SYCL_LS_VERBOSE_OUTPUT "")
    if(SYCL_LS)
        set(_batchlas_sycl_ls_env "")
        if(BATCHLAS_SYCL_RUNTIME_DIR AND "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
            set(_batchlas_sycl_ls_env "LD_LIBRARY_PATH=${BATCHLAS_SYCL_RUNTIME_DIR}")
        elseif(BATCHLAS_SYCL_RUNTIME_DIR)
            set(_batchlas_sycl_ls_env "LD_LIBRARY_PATH=${BATCHLAS_SYCL_RUNTIME_DIR}:$ENV{LD_LIBRARY_PATH}")
        endif()
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${_batchlas_sycl_ls_env} "${SYCL_LS}"
            OUTPUT_VARIABLE BATCHLAS_SYCL_LS_OUTPUT ERROR_VARIABLE _batchlas_sycl_ls_err
            RESULT_VARIABLE _batchlas_sycl_ls_rc TIMEOUT 180)
        if(_batchlas_sycl_ls_rc EQUAL 0)
            execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${_batchlas_sycl_ls_env} "${SYCL_LS}" --verbose
                OUTPUT_VARIABLE BATCHLAS_SYCL_LS_VERBOSE_OUTPUT ERROR_QUIET TIMEOUT 180)
            message(STATUS "Detecting devices using ${SYCL_LS}")
        else()
            message(WARNING "${SYCL_LS} failed (${_batchlas_sycl_ls_rc}): ${_batchlas_sycl_ls_err}")
            set(BATCHLAS_SYCL_LS_OUTPUT "")
        endif()
    else()
        message(WARNING "No sycl-ls next to ${CMAKE_CXX_COMPILER} or on PATH; devices cannot be detected")
    endif()
endmacro()

# Fills BATCHLAS_DETECTED_{NVIDIA,AMD}_GPU, BATCHLAS_DETECTED_NVIDIA_TARGETS and the non-NVIDIA GPU
# targets. NVIDIA codegen flags are added later, once BATCHLAS_NVIDIA_ARCH has had its say.
function(detect_sycl_gpu_architectures)
    batchlas_normalize_sycl_ls_architectures("${BATCHLAS_SYCL_LS_VERBOSE_OUTPUT}" _verbose)

    set(_targets "")
    set(_nvidia_targets "")
    set(_fallback_entries "")
    set(_compile_options "")
    set(_has_nvidia_gpu FALSE)
    set(_has_amd_gpu FALSE)

    if(BATCHLAS_SYCL_LS_OUTPUT MATCHES "\\[cuda:gpu\\]")
        set(_has_nvidia_gpu TRUE)
    endif()
    if(BATCHLAS_SYCL_LS_OUTPUT MATCHES "\\[amd:gpu\\].*AMD|\\[hip:gpu\\].*AMD")
        set(_has_amd_gpu TRUE)
    endif()

    string(REGEX MATCHALL "Architecture: ([^\n]+)" _arch_matches "${_verbose}")
    foreach(_match IN LISTS _arch_matches)
        string(REGEX REPLACE "Architecture: ([^\n]+)" "\\1" _arch "${_match}")
        string(STRIP "${_arch}" _arch)
        if(_arch MATCHES "^gfx[0-9]+$")
            set(_arch "amd_gpu_${_arch}")
        endif()
        if(_arch MATCHES "^nvidia_gpu_")
            list(APPEND _nvidia_targets "${_arch}")
        elseif(_arch MATCHES "_gpu_")
            list(APPEND _targets "${_arch}")
        endif()
    endforeach()

    # No CUDA device from sycl-ls although the toolchain has a CUDA adapter (sycl-ls missing or
    # broken, or a runtime that did not load): ask the driver. Without the adapter the runtime
    # could not run NVPTX kernels, so building them would only hide the problem.
    file(GLOB _cuda_adapter "${BATCHLAS_SYCL_RUNTIME_DIR}/libur_adapter_cuda.so*")
    if(NOT _has_nvidia_gpu AND _cuda_adapter)
        find_program(BATCHLAS_NVIDIA_SMI nvidia-smi)
        if(BATCHLAS_NVIDIA_SMI)
            execute_process(COMMAND "${BATCHLAS_NVIDIA_SMI}" --query-gpu=compute_cap,name --format=csv,noheader
                OUTPUT_VARIABLE _smi ERROR_QUIET RESULT_VARIABLE _smi_rc TIMEOUT 30)
            if(_smi_rc EQUAL 0)
                string(REPLACE "\n" ";" _smi "${_smi}")
                foreach(_row IN LISTS _smi)
                    if(_row MATCHES "^([0-9]+)\\.([0-9]+), (.+)$")
                        set(_has_nvidia_gpu TRUE)
                        list(APPEND _nvidia_targets "nvidia_gpu_sm_${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
                        list(APPEND _fallback_entries
                            "nvidia-smi|gpu|${CMAKE_MATCH_3}|nvidia_gpu_sm_${CMAKE_MATCH_1}${CMAKE_MATCH_2}|32")
                    endif()
                endforeach()
                if(_has_nvidia_gpu)
                    message(STATUS "sycl-ls listed no CUDA device; using nvidia-smi")
                endif()
            endif()
        endif()
    endif()

    list(REMOVE_DUPLICATES _nvidia_targets)
    list(REMOVE_DUPLICATES _targets)
    foreach(_t IN LISTS _nvidia_targets _targets)
        message(STATUS "Detected GPU architecture: ${_t}")
    endforeach()

    if(_has_amd_gpu)
        list(APPEND _compile_options -fno-sycl-id-queries-fit-in-int)
    endif()

    set(BATCHLAS_DETECTED_NVIDIA_GPU "${_has_nvidia_gpu}" PARENT_SCOPE)
    set(BATCHLAS_DETECTED_AMD_GPU "${_has_amd_gpu}" PARENT_SCOPE)
    set(BATCHLAS_DETECTED_NVIDIA_TARGETS "${_nvidia_targets}" PARENT_SCOPE)
    set(BATCHLAS_FALLBACK_DEVICE_ENTRIES "${_fallback_entries}" PARENT_SCOPE)
    set(BATCHLAS_SYCL_TARGETS "${_targets}" PARENT_SCOPE)
    set(BATCHLAS_SYCL_EXTRA_CXX_OPTIONS "${_compile_options}" PARENT_SCOPE)
endfunction()

function(collect_sycl_device_limit_info)
    batchlas_normalize_sycl_ls_architectures("${BATCHLAS_SYCL_LS_VERBOSE_OUTPUT}" SYCL_LS_DEVICE_LIMITS_OUTPUT)
    string(REPLACE "\r\n" "\n" SYCL_LS_DEVICE_LIMITS_OUTPUT "${SYCL_LS_DEVICE_LIMITS_OUTPUT}")
    string(REPLACE "\r" "\n" SYCL_LS_DEVICE_LIMITS_OUTPUT "${SYCL_LS_DEVICE_LIMITS_OUTPUT}")
    string(REPLACE "\n" ";" _sycl_ls_lines "${SYCL_LS_DEVICE_LIMITS_OUTPUT}")

    set(_min_gpu_workspace_budget_bytes 0)
    set(_current_device_type "")
    set(_in_device FALSE)

    foreach(_line IN LISTS _sycl_ls_lines)
        string(STRIP "${_line}" _trimmed_line)
        if(_trimmed_line STREQUAL "")
            continue()
        endif()

        if(_trimmed_line MATCHES "^Platform \\\[#")
            set(_in_device FALSE)
            continue()
        endif()

        if(_trimmed_line MATCHES "^Device \\\[#")
            set(_in_device TRUE)
            set(_current_device_type "")
            continue()
        endif()

        if(_in_device)
            if(_trimmed_line MATCHES "^Type[ ]*:[ ]*(.+)$")
                set(_current_device_type "${CMAKE_MATCH_1}")
            elseif(_trimmed_line MATCHES "^Architecture:[ ]*(.+)$")
                set(_current_architecture "${CMAKE_MATCH_1}")
                if("${_current_device_type}" STREQUAL "gpu")
                    batchlas_architecture_table_local_mem_bytes("${_current_architecture}" "gpu" _current_local_mem_bytes)
                    batchlas_subgroup_workspace_budget_bytes("${_current_architecture}" "${_current_local_mem_bytes}" "gpu" _current_workspace_budget_bytes)
                    if(_min_gpu_workspace_budget_bytes EQUAL 0 OR _current_workspace_budget_bytes LESS _min_gpu_workspace_budget_bytes)
                        set(_min_gpu_workspace_budget_bytes "${_current_workspace_budget_bytes}")
                    endif()
                endif()
            endif()
        endif()
    endforeach()

    foreach(_dev IN LISTS BATCHLAS_FALLBACK_DEVICE_ENTRIES)
        string(REPLACE "|" ";" _f "${_dev}")
        list(GET _f 3 _arch)
        batchlas_architecture_table_local_mem_bytes("${_arch}" gpu _local)
        batchlas_subgroup_workspace_budget_bytes("${_arch}" "${_local}" gpu _budget)
        if(_min_gpu_workspace_budget_bytes EQUAL 0 OR _budget LESS _min_gpu_workspace_budget_bytes)
            set(_min_gpu_workspace_budget_bytes "${_budget}")
        endif()
    endforeach()

    if(_min_gpu_workspace_budget_bytes EQUAL 0)
        set(_min_gpu_workspace_budget_bytes 28672)
    endif()

    set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES "${_min_gpu_workspace_budget_bytes}" PARENT_SCOPE)
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

    set(SYCL_LS_CPU_OUTPUT "${BATCHLAS_SYCL_LS_OUTPUT}")
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

if(NOT BATCHLAS_DETECT_SYCL_FUNCTIONS_ONLY)

batchlas_run_sycl_ls()
detect_sycl_gpu_architectures()
detect_sycl_cpu_target()
collect_sycl_device_limit_info()

set(DETECTED_AMD_ARCH "")
foreach(_arch ${BATCHLAS_SYCL_TARGETS})
    if(_arch MATCHES "amd_gpu_gfx[0-9]+")
        set(DETECTED_AMD_ARCH "${_arch}")
    endif()
endforeach()

# With ROCm on, BATCHLAS_AMD_ARCH replaces the detected AMD target, like BATCHLAS_NVIDIA_ARCH below.
if(BATCHLAS_ENABLE_ROCM AND BATCHLAS_AMD_ARCH)
    string(REGEX REPLACE "^amd_gpu_" "" _AMD_ARCH "${BATCHLAS_AMD_ARCH}")
    set(_AMD_ARCH "amd_gpu_${_AMD_ARCH}")
    list(FILTER BATCHLAS_SYCL_TARGETS EXCLUDE REGEX "^amd_gpu_")
    list(APPEND BATCHLAS_SYCL_TARGETS "${_AMD_ARCH}")
elseif(BATCHLAS_ENABLE_ROCM AND NOT DETECTED_AMD_ARCH)
    message(FATAL_ERROR "BATCHLAS_ENABLE_ROCM=ON but no AMD GPU was detected; name the target "
        "with -DBATCHLAS_AMD_ARCH=gfx<N>")
endif()
foreach(_arch IN LISTS BATCHLAS_SYCL_TARGETS)
    if(_arch MATCHES "^amd_gpu_")
        message(STATUS "Compiling for AMD architecture: ${_arch}")
    endif()
endforeach()

# BATCHLAS_NVIDIA_ARCH replaces the detected NVIDIA targets: a cross build, or one of several GPUs.
set(_nvidia_targets "${BATCHLAS_DETECTED_NVIDIA_TARGETS}")
if(BATCHLAS_NVIDIA_ARCH)
    string(REGEX REPLACE "^nvidia_gpu_" "" _sm "${BATCHLAS_NVIDIA_ARCH}")
    string(REGEX REPLACE "^sm_" "" _sm "${_sm}")
    if(NOT _sm MATCHES "^[0-9]+a?$")
        message(FATAL_ERROR "BATCHLAS_NVIDIA_ARCH='${BATCHLAS_NVIDIA_ARCH}': expected sm_<N>, e.g. sm_89")
    endif()
    set(_nvidia_targets "nvidia_gpu_sm_${_sm}")
endif()
if(BATCHLAS_ENABLE_CUDA_MODE STREQUAL "ON" AND NOT _nvidia_targets)
    message(FATAL_ERROR
        "BATCHLAS_ENABLE_CUDA=ON but no NVIDIA target: sycl-ls (${SYCL_LS}) printed no '[cuda:gpu]' "
        "and nvidia-smi found no GPU usable by this compiler's runtime. Use a DPC++ with a CUDA "
        "backend (oneAPI + the Codeplay oneAPI-for-NVIDIA plugin, or an intel/llvm built with "
        "--cuda), name the target with -DBATCHLAS_NVIDIA_ARCH=sm_<N>, or configure with "
        "-DBATCHLAS_ENABLE_CUDA=AUTO or =OFF.")
endif()

if(_nvidia_targets)
    # BATCHLAS_ENABLE_CUDA governs only the cuBLAS/cuSOLVER backend; NVPTX device code needs the
    # toolkit's libdevice whatever it says.
    if(BATCHLAS_ENABLE_CUDA_MODE STREQUAL "AUTO")
        set(BATCHLAS_CUDA_ENABLED ON)
    endif()
    list(APPEND BATCHLAS_SYCL_TARGETS ${_nvidia_targets})
    batchlas_locate_cuda_toolkit()
    find_package(CUDAToolkit QUIET)
    set(BATCHLAS_CUDA_PATH "")
    if(CUDAToolkit_FOUND AND CUDAToolkit_BIN_DIR)
        get_filename_component(BATCHLAS_CUDA_PATH "${CUDAToolkit_BIN_DIR}" DIRECTORY)
        message(STATUS "Found CUDA at: ${BATCHLAS_CUDA_PATH}")
        list(APPEND BATCHLAS_SYCL_EXTRA_CXX_OPTIONS "--cuda-path=${BATCHLAS_CUDA_PATH}")
        list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS "--cuda-path=${BATCHLAS_CUDA_PATH}")
    else()
        message(WARNING "NVIDIA target ${_nvidia_targets} but no CUDA toolkit found; "
            "pass -DCUDAToolkit_ROOT=<toolkit>")
    endif()
    list(APPEND BATCHLAS_SYCL_EXTRA_CXX_OPTIONS -fsycl-unnamed-lambda)
    list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS -fsycl-unnamed-lambda)
endif()
if(BATCHLAS_SYCL_TARGETS MATCHES "_gpu_")
    list(APPEND BATCHLAS_SYCL_EXTRA_CXX_OPTIONS -fsycl-dead-args-optimization)
endif()

list(REMOVE_DUPLICATES BATCHLAS_SYCL_TARGETS)
set(_unaliased_nvidia_targets "")
foreach(_target IN LISTS BATCHLAS_SYCL_TARGETS)
    if(_target MATCHES "^nvidia_gpu_sm_[0-9]+a?$")
        batchlas_nvidia_target_has_alias("${_target}" _has_alias)
        if(NOT _has_alias)
            list(APPEND _unaliased_nvidia_targets "${_target}")
        endif()
    endif()
endforeach()
set(BATCHLAS_SYCL_BACKEND_OPTION "")
set(BATCHLAS_SYCL_BACKEND_OPTION_STRING "")
if(_unaliased_nvidia_targets)
    batchlas_rewrite_unaliased_nvidia_targets("${BATCHLAS_SYCL_TARGETS}" "${_unaliased_nvidia_targets}"
        BATCHLAS_SYCL_TARGETS BATCHLAS_SYCL_BACKEND_OPTION)
    message(STATUS "No -fsycl-targets alias for ${_unaliased_nvidia_targets} in this compiler; "
        "using nvptx64-nvidia-cuda with ${BATCHLAS_SYCL_BACKEND_OPTION}")
    list(APPEND BATCHLAS_SYCL_EXTRA_CXX_OPTIONS "${BATCHLAS_SYCL_BACKEND_OPTION}")
    list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS "${BATCHLAS_SYCL_BACKEND_OPTION}")
    string(REGEX REPLACE "^SHELL:" "" BATCHLAS_SYCL_BACKEND_OPTION_STRING "${BATCHLAS_SYCL_BACKEND_OPTION}")
endif()
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

# vecz crashes clang on the native_cpu backend.
list(FIND BATCHLAS_SYCL_TARGETS "native_cpu" _native_cpu_idx)
if(NOT _native_cpu_idx EQUAL -1)
    message(STATUS "Disabling vecz for native_cpu backend (-mllvm -sycl-native-cpu-no-vecz)")
    list(APPEND BATCHLAS_SYCL_EXTRA_CXX_OPTIONS -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz)
    list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS -Xclang=-mllvm -Xclang=-sycl-native-cpu-no-vecz)
endif()

# Every flag-plus-argument pair below is one SHELL: item: CMake de-duplicates options, so a
# repeated -Xsycl-target-backend / -Xcuda-ptxas would otherwise leave its argument bare.
if(BATCHLAS_CUDA_ENABLED)
    if(BATCHLAS_DISABLE_CUDA_FTZ)
        # Only the -Xsycl-target-backend half. The '-Xcuda-ptxas --ftz=false'
        # pair that used to follow is a hard build failure on CUDA >= 12.8
        # ("ptxas fatal : Unknown option '-ftz'"), which is why this whole block
        # used to be skipped for one hardcoded compiler path. Like --save-temps below, this
        # reaches only the image's runtime JIT -compile-opts (driver -###), never the AOT step.
        message(STATUS "Passing --ftz=false to the CUDA image's runtime JIT options (not the AOT ptxas)")
        list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS
            "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --ftz=false")
    endif()
    if(BATCHLAS_KEEP_CUDA_INTERMEDIATES)
        message(STATUS "Preserving CUDA/SYCL CUDA intermediates for device-code inspection")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/cuda-intermediates")
        list(APPEND BATCHLAS_SYCL_EXTRA_LINK_OPTIONS
            "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --save-temps"
            "SHELL:-Xcuda-ptxas -v")
    endif()
endif()

# Every option below is wrapped in $<BUILD_INTERFACE:...>.
#
# batchlas_sycl_options is part of the installed export set (BatchLAS::batchlas
# links it), so without the wrapper the exported package forced this machine's
# entire SYCL device-codegen line onto every consumer translation unit:
# -fsycl -fsycl-unnamed-lambda -fsycl-dead-args-optimization -Xclang=-mllvm
# --cuda-path=/usr/local/cuda-13.2 -fsycl-targets=nvidia_gpu_sm_89,native_cpu.
# A consumer using GCC died on "unrecognized command-line option '-fsycl'"; a
# consumer picking its own -fsycl-targets had it silently overridden. None of it
# is needed - the public header surface is deliberately SYCL-free.
#
# In-tree targets (src/, tests/, benchmarks/, python/) consume the *build*
# interface and are completely unaffected.
#
# Commas are escaped as $<COMMA> because a bare comma inside a generator
# expression separates parameters (-fsycl-targets=a,b is the one that matters).
function(batchlas_build_interface_option out_var opt)
    string(REPLACE "," "$<COMMA>" _escaped "${opt}")
    set(${out_var} "${_escaped}" PARENT_SCOPE)
endfunction()

foreach(_opt IN LISTS BATCHLAS_SYCL_BASE_CXX_OPTIONS BATCHLAS_SYCL_EXTRA_CXX_OPTIONS)
    batchlas_build_interface_option(_opt_esc "${_opt}")
    target_compile_options(batchlas_sycl_options INTERFACE
        "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${_opt_esc}>>")
    target_compile_options(batchlas_sycl_no_cpu_options INTERFACE
        "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${_opt_esc}>>")
endforeach()

# BATCHLAS_SYCL_GLOBAL_OFFSET (cmake/BatchLASOptions.cmake). The flag reaches the device
# codegen through the link driver; probed first so a compiler without the GlobalOffset pass
# keeps building (with the twins) instead of failing every link on an unknown -mllvm option.
if(NOT BATCHLAS_SYCL_GLOBAL_OFFSET)
    set(_go_dir "${PROJECT_BINARY_DIR}/CMakeFiles/batchlas_global_offset_probe")
    file(MAKE_DIRECTORY "${_go_dir}")
    file(WRITE "${_go_dir}/probe.cc" "int main() { return 0; }\n")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -mllvm -enable-global-offset=false -fsyntax-only "${_go_dir}/probe.cc"
        RESULT_VARIABLE _go_rc OUTPUT_QUIET ERROR_VARIABLE _go_err)
    if(_go_rc EQUAL 0)
        list(APPEND BATCHLAS_SYCL_BASE_LINK_OPTIONS "SHELL:-mllvm -enable-global-offset=false")
        message(STATUS "SYCL global-offset kernel twins: off (-mllvm -enable-global-offset=false)")
    else()
        message(WARNING "${CMAKE_CXX_COMPILER} rejects -mllvm -enable-global-offset=false; "
            "keeping the _with_offset kernel twins (slower device links): ${_go_err}")
    endif()
else()
    message(STATUS "SYCL global-offset kernel twins: kept (BATCHLAS_SYCL_GLOBAL_OFFSET=ON)")
endif()

foreach(_opt IN LISTS BATCHLAS_SYCL_BASE_LINK_OPTIONS BATCHLAS_SYCL_EXTRA_LINK_OPTIONS)
    batchlas_build_interface_option(_opt_esc "${_opt}")
    target_link_options(batchlas_sycl_options INTERFACE "$<BUILD_INTERFACE:${_opt_esc}>")
    target_link_options(batchlas_sycl_no_cpu_options INTERFACE "$<BUILD_INTERFACE:${_opt_esc}>")
endforeach()

# The native_cpu device frontend inherits the host's -fmath-errno, so std::sqrt
# in a kernel stays an errno-setting libm call instead of llvm.sqrt, and every
# device link warns "Undefined function sqrt ... may result in runtime errors".
# Device code has no errno; the host compile is left alone.
list(FIND BATCHLAS_SYCL_TARGETS "native_cpu" _native_cpu_idx)
if(NOT _native_cpu_idx EQUAL -1)
    target_compile_options(batchlas_sycl_options INTERFACE
        "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xsycl-target-frontend=native_cpu -fno-math-errno>>")
endif()

if(BATCHLAS_SYCL_TARGETS_STRING)
    batchlas_build_interface_option(_targets_esc "-fsycl-targets=${BATCHLAS_SYCL_TARGETS_STRING}")
    target_compile_options(batchlas_sycl_options INTERFACE
        "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${_targets_esc}>>"
    )
    target_link_options(batchlas_sycl_options INTERFACE
        "$<BUILD_INTERFACE:${_targets_esc}>"
    )
endif()

if(BATCHLAS_SYCL_TARGETS_NO_CPU_STRING)
    batchlas_build_interface_option(_targets_no_cpu_esc "-fsycl-targets=${BATCHLAS_SYCL_TARGETS_NO_CPU_STRING}")
    target_compile_options(batchlas_sycl_no_cpu_options INTERFACE
        "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${_targets_no_cpu_esc}>>"
    )
    target_link_options(batchlas_sycl_no_cpu_options INTERFACE
        "$<BUILD_INTERFACE:${_targets_no_cpu_esc}>"
    )
endif()

# The driver links libsycl by name (its own -L makes the directory "implicit") and records no
# RUNPATH, so every library and test only loaded with the toolchain's lib/ on LD_LIBRARY_PATH.
# Built and installed binaries carry it instead.
if(BATCHLAS_SYCL_RUNTIME_DIR AND NOT BATCHLAS_SYCL_RUNTIME_DIR IN_LIST CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES)
    list(APPEND CMAKE_BUILD_RPATH "${BATCHLAS_SYCL_RUNTIME_DIR}")
    list(APPEND CMAKE_INSTALL_RPATH "${BATCHLAS_SYCL_RUNTIME_DIR}")
endif()

# Some intel/llvm builds (e79d489d, Jan 2026) lower reqd_work_group_size(1, 1, N) on NVPTX to
# reqntidx=N then reqntidx=1 and no reqntidz: ptxas sees `.reqntid 1, 1` and every launch fails
# with CUDA_ERROR_INVALID_VALUE. The same builds emit .maxntid for max_work_group_size, which
# ptxas rejects next to .reqntid. Newer builds have the opposite behaviour (known defect #20),
# so the gemm register functors pick their spelling from this probe of the device IR.
set(BATCHLAS_NVPTX_REQD_WG_OK OFF)
set(_reqd_probe_target "")
foreach(_t IN LISTS BATCHLAS_SYCL_TARGETS)
    if(_t MATCHES "^nvidia_gpu_sm_" AND NOT _reqd_probe_target)
        set(_reqd_probe_target "${_t}")
    endif()
endforeach()
if(_reqd_probe_target)
    set(_reqd_dir "${PROJECT_BINARY_DIR}/CMakeFiles/batchlas_reqd_wg_probe")
    file(MAKE_DIRECTORY "${_reqd_dir}")
    file(WRITE "${_reqd_dir}/probe.cc"
        "#include <sycl/sycl.hpp>\n"
        "struct K { int* p; [[sycl::reqd_work_group_size(1, 1, 256)]] void operator()(sycl::nd_item<3> it) const"
        " { p[it.get_global_linear_id()] = 1; } };\n"
        "void f(sycl::queue& q, int* p) { q.parallel_for(sycl::nd_range<3>({1, 1, 256}, {1, 1, 256}), K{p}); }\n")
    set(_reqd_args -fsycl -fsycl-targets=${_reqd_probe_target} -fsycl-device-only -S)
    if(BATCHLAS_CUDA_PATH)
        list(APPEND _reqd_args "--cuda-path=${BATCHLAS_CUDA_PATH}")
    endif()
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" ${_reqd_args} -o "${_reqd_dir}/probe.ll" "${_reqd_dir}/probe.cc"
        RESULT_VARIABLE _reqd_rc OUTPUT_QUIET ERROR_QUIET)
    if(_reqd_rc EQUAL 0 AND EXISTS "${_reqd_dir}/probe.ll")
        file(READ "${_reqd_dir}/probe.ll" _reqd_ir)
        if(_reqd_ir MATCHES "\"reqntidx\", i32 256" AND _reqd_ir MATCHES "\"reqntidz\", i32 1")
            set(BATCHLAS_NVPTX_REQD_WG_OK ON)
        endif()
    endif()
    message(STATUS "NVPTX reqd_work_group_size lowering: ${BATCHLAS_NVPTX_REQD_WG_OK} (${_reqd_probe_target})")
endif()

# What a consumer's own SYCL translation units need (BatchLAS_SYCL_COMPILE_OPTIONS).
set(BATCHLAS_SYCL_CONSUMER_OPTIONS -fsycl)

message(STATUS "Using Intel oneAPI DPC++ compiler for SYCL: ${CMAKE_CXX_COMPILER}")

endif() # NOT BATCHLAS_DETECT_SYCL_FUNCTIONS_ONLY

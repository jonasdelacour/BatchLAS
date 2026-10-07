# Parallel GPU tests through CTest resource allocation.
#
# Writes <build>/ctest_resources.json (one "gpus" entry per device) and gives every
# GPU test RESOURCE_GROUPS "gpus:1". scripts/ctest_gpus.sh passes the spec file and
# -j<total slots>; tests/ctest_gpu_env.sh maps the allocation to CUDA_VISIBLE_DEVICES.
# Plain ctest (no --resource-spec-file) ignores RESOURCE_GROUPS, and the launcher is
# a pass-through without an allocation, so it runs exactly as before.
# Included from the top-level CMakeLists.txt so tests/ and python/ both see it.
# evidence: AGENTS.md section 8 (Testing Policy)
include_guard(GLOBAL)

set(BATCHLAS_TEST_GPUS "auto" CACHE STRING
    "GPUs in <build>/ctest_resources.json: auto (nvidia-smi --list-gpus), <n>, or 0 to disable")
set(BATCHLAS_TEST_GPU_SLOTS "2" CACHE STRING
    "Concurrent GPU tests per device under scripts/ctest_gpus.sh")

if(NOT BATCHLAS_TEST_GPU_SLOTS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "BATCHLAS_TEST_GPU_SLOTS='${BATCHLAS_TEST_GPU_SLOTS}': expected a positive integer")
endif()

# Isolation is CUDA_VISIBLE_DEVICES only, so the SYCL runtime must expose CUDA devices
# ([cuda:gpu] in sycl-ls; BATCHLAS_ENABLE_CUDA governs only cuBLAS). A Level Zero or
# HIP tree would run count x slots tests on one default device: no spec at all.
set(BATCHLAS_TEST_GPU_COUNT 0)
set(_why "")
find_program(BATCHLAS_TEST_SH sh)
if(BATCHLAS_TEST_GPUS STREQUAL "0")
    set(_why "BATCHLAS_TEST_GPUS=0")
elseif(NOT BATCHLAS_TEST_GPUS STREQUAL "auto" AND NOT BATCHLAS_TEST_GPUS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "BATCHLAS_TEST_GPUS='${BATCHLAS_TEST_GPUS}': expected auto, 0 or a GPU count")
elseif(NOT BATCHLAS_DETECTED_NVIDIA_GPU OR WIN32 OR NOT BATCHLAS_TEST_SH)
    set(_why "needs a [cuda:gpu] in sycl-ls and a POSIX sh")
    if(NOT BATCHLAS_TEST_GPUS STREQUAL "auto")
        message(FATAL_ERROR "BATCHLAS_TEST_GPUS=${BATCHLAS_TEST_GPUS}: per-test GPU isolation "
            "(tests/ctest_gpu_env.sh) ${_why}; set it to 0 or auto")
    endif()
elseif(BATCHLAS_TEST_GPUS STREQUAL "auto")
    find_program(BATCHLAS_NVIDIA_SMI nvidia-smi)
    if(BATCHLAS_NVIDIA_SMI)
        execute_process(COMMAND "${BATCHLAS_NVIDIA_SMI}" --list-gpus
            OUTPUT_VARIABLE _gpu_list RESULT_VARIABLE _gpu_rc ERROR_QUIET TIMEOUT 30)
        if(_gpu_rc EQUAL 0)
            string(REGEX MATCHALL "GPU [0-9]+:" _gpu_lines "${_gpu_list}")
            list(LENGTH _gpu_lines BATCHLAS_TEST_GPU_COUNT)
        endif()
    endif()
    set(_why "nvidia-smi --list-gpus found none")
else()
    set(BATCHLAS_TEST_GPU_COUNT ${BATCHLAS_TEST_GPUS})
endif()

set(_spec "${CMAKE_BINARY_DIR}/ctest_resources.json")
set(BATCHLAS_TEST_GPU_LAUNCHER "")
if(BATCHLAS_TEST_GPU_COUNT GREATER 0)
    set(_entries "")
    math(EXPR _last "${BATCHLAS_TEST_GPU_COUNT} - 1")
    foreach(_g RANGE ${_last})
        list(APPEND _entries "{ \"id\": \"${_g}\", \"slots\": ${BATCHLAS_TEST_GPU_SLOTS} }")
    endforeach()
    string(REPLACE ";" ",\n        " _entries "${_entries}")
    file(WRITE "${_spec}"
        "{\n  \"version\": { \"major\": 1, \"minor\": 0 },\n  \"local\": [\n    {\n"
        "      \"gpus\": [\n        ${_entries}\n      ]\n    }\n  ]\n}\n")
    set(BATCHLAS_TEST_GPU_LAUNCHER "${BATCHLAS_TEST_SH}" "${PROJECT_SOURCE_DIR}/tests/ctest_gpu_env.sh")
    math(EXPR _slots_total "${BATCHLAS_TEST_GPU_COUNT} * ${BATCHLAS_TEST_GPU_SLOTS}")
    message(STATUS "GPU test resources: ${BATCHLAS_TEST_GPU_COUNT} GPU(s) x ${BATCHLAS_TEST_GPU_SLOTS} slot(s) "
                   "-> scripts/ctest_gpus.sh runs ctest -j${_slots_total}")
else()
    file(REMOVE "${_spec}")
    message(STATUS "GPU test resources: none (${_why}); use plain serial ctest")
endif()

# add_test COMMAND for a GPU test binary: the launcher (if any) then the target.
function(batchlas_gpu_test_command out_var target)
    set(${out_var} ${BATCHLAS_TEST_GPU_LAUNCHER} "$<TARGET_FILE:${target}>" PARENT_SCOPE)
endfunction()

function(batchlas_test_needs_gpu test_name)
    if(BATCHLAS_TEST_GPU_COUNT GREATER 0)
        set_tests_properties(${test_name} PROPERTIES RESOURCE_GROUPS "gpus:1")
    endif()
endfunction()

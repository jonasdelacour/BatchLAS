file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/batchlas")

set(BATCHLAS_DEVICE_LIMIT_ENTRY_LINES "")
foreach(_device_limit_entry IN LISTS BATCHLAS_DETECTED_DEVICE_LIMIT_ENTRIES)
    string(REPLACE "|" ";" _device_limit_fields "${_device_limit_entry}")
    list(LENGTH _device_limit_fields _device_limit_field_count)
    if(_device_limit_field_count EQUAL 7)
        list(GET _device_limit_fields 0 _device_limit_platform_name)
        list(GET _device_limit_fields 1 _device_limit_type)
        list(GET _device_limit_fields 2 _device_limit_name)
        list(GET _device_limit_fields 3 _device_limit_architecture)
        list(GET _device_limit_fields 4 _device_limit_subgroup_size)
        list(GET _device_limit_fields 5 _device_limit_local_mem_bytes)
        list(GET _device_limit_fields 6 _device_limit_workspace_budget_bytes)
        string(APPEND BATCHLAS_DEVICE_LIMIT_ENTRY_LINES
            "    DetectedDeviceLimit{\"${_device_limit_platform_name}\", \"${_device_limit_type}\", \"${_device_limit_name}\", \"${_device_limit_architecture}\", ${_device_limit_subgroup_size}u, ${_device_limit_local_mem_bytes}ull, ${_device_limit_workspace_budget_bytes}ull},\n")
    endif()
endforeach()

list(LENGTH BATCHLAS_DETECTED_DEVICE_LIMIT_ENTRIES BATCHLAS_DEVICE_LIMIT_ENTRY_COUNT)
if(NOT DEFINED BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES)
    set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SUBGROUP_WORKSPACE_BUDGET_BYTES 28672)
endif()
if(NOT DEFINED BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SAFE_SUBGROUPS_PER_WORKGROUP)
    set(BATCHLAS_DETECTED_DEVICE_LIMIT_MIN_GPU_SAFE_SUBGROUPS_PER_WORKGROUP 2)
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/backend_config.h.in"
    "${PROJECT_BINARY_DIR}/include/batchlas/backend_config.h"
)

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/device_limits.h.in"
    "${PROJECT_BINARY_DIR}/include/batchlas/device_limits.hh"
)

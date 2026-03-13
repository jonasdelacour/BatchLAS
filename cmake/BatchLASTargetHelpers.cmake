function(batchlas_apply_object_options target)
    set(options NO_CPU_TARGETS)
    cmake_parse_arguments(BATCHLAS_OBJ "${options}" "" "" ${ARGN})

    target_link_libraries(${target} PRIVATE
        batchlas_build_options
        batchlas_dep_options
    )

    if(BATCHLAS_ENABLE_SYCL)
        if(BATCHLAS_OBJ_NO_CPU_TARGETS AND BATCHLAS_SYCL_TARGETS_NO_CPU_STRING)
            target_link_libraries(${target} PRIVATE batchlas_sycl_no_cpu_options)
        else()
            target_link_libraries(${target} PRIVATE batchlas_sycl_options)
        endif()
    endif()
endfunction()

function(batchlas_configure_binary_target target)
    set(options NO_FACADE)
    set(multiValueArgs LIBRARIES)
    cmake_parse_arguments(BATCHLAS_BIN "${options}" "" "${multiValueArgs}" ${ARGN})

    set(_batchlas_binary_links ${BATCHLAS_BIN_LIBRARIES})
    if(NOT BATCHLAS_BIN_NO_FACADE)
        list(APPEND _batchlas_binary_links batchlas)
    else()
        list(APPEND _batchlas_binary_links
            batchlas_build_options
            batchlas_dep_options
            batchlas_sycl_options
        )
    endif()

    if(_batchlas_binary_links)
        target_link_libraries(${target} PRIVATE ${_batchlas_binary_links})
    endif()
endfunction()

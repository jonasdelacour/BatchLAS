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
    target_link_libraries(${target} PRIVATE batchlas ${ARGN})
endfunction()

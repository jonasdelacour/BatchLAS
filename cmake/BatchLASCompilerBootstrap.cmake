if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(SYCL_LS sycl-ls)
    if(SYCL_LS)
        get_filename_component(_BATCHLAS_SYCL_BIN_DIR "${SYCL_LS}" DIRECTORY)
        find_program(DPCPP_COMPILER
            NAMES icpx clang++
            HINTS "${_BATCHLAS_SYCL_BIN_DIR}"
            NO_DEFAULT_PATH
        )
        if(NOT DPCPP_COMPILER)
            find_program(DPCPP_COMPILER NAMES icpx clang++)
        endif()

        if(DPCPP_COMPILER)
            message(STATUS "Auto-detected SYCL compiler: ${DPCPP_COMPILER}")
            set(CMAKE_CXX_COMPILER "${DPCPP_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
        else()
            message(WARNING "sycl-ls found but no SYCL compiler detected. Please set CXX environment variable or -DCMAKE_CXX_COMPILER.")
        endif()
    else()
        message(WARNING "sycl-ls not found. SYCL support is mandatory. Please set CXX environment variable or -DCMAKE_CXX_COMPILER to icpx or a SYCL-capable compiler.")
    endif()
else()
    message(STATUS "Using user-specified compiler: ${CMAKE_CXX_COMPILER}")
endif()

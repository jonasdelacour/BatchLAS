# This file runs before project(), so BATCHLAS_IS_TOP_LEVEL does not exist yet;
# compare the directories directly. Picking a compiler is a whole-project
# decision, and the set() below is a CACHE ... FORCE write - doing that from
# inside an add_subdirectory()/FetchContent subproject would silently replace
# the consuming project's compiler. If BatchLAS is a subproject the parent has
# already run project() and CMAKE_CXX_COMPILER is defined anyway; the guard only
# matters for a parent that declared LANGUAGES NONE.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        message(STATUS "BatchLAS is a subproject: leaving CMAKE_CXX_COMPILER to the parent project. "
                       "BatchLAS needs a SYCL compiler (e.g. DPC++/icpx); set CMAKE_CXX_COMPILER accordingly.")
    endif()
elseif(NOT DEFINED CMAKE_CXX_COMPILER)
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

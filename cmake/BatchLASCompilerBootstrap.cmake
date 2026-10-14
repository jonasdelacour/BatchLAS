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
elseif((NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" MATCHES "(^|/)(acpp|syclcc|syclcc-clang)$") OR
       (NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "" AND CMAKE_HOST_APPLE) OR
       (BATCHLAS_SYCL_IMPL MATCHES "^[Aa][Cc][Pp][Pp]$" AND
        NOT CMAKE_CXX_COMPILER MATCHES "(^|/)(acpp|syclcc|syclcc-clang)$"))
    # AdaptiveCpp: never fall through to the sycl-ls search below, which picks a DPC++ clang++.
    # On macOS it is the only SYCL compiler, so a bare configure there looks for it too.
    # ccmake configures AUTO (DPC++) first, so ACPP also replaces a cached non-acpp compiler;
    # CMake then drops the cache and reconfigures, and AUTO resolves to ACPP from the driver.
    set(_batchlas_cached_is_acpp FALSE)
    if(DEFINED CMAKE_CXX_COMPILER)
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --acpp-version
            OUTPUT_VARIABLE _batchlas_acpp_version ERROR_QUIET RESULT_VARIABLE _batchlas_acpp_rc TIMEOUT 60)
        if(_batchlas_acpp_rc EQUAL 0 AND _batchlas_acpp_version MATCHES "AdaptiveCpp")
            set(_batchlas_cached_is_acpp TRUE)
        endif()
    endif()
    if(_batchlas_cached_is_acpp)
        message(STATUS "Using user-specified compiler: ${CMAKE_CXX_COMPILER}")
    elseif("$ENV{CXX}" STREQUAL "" OR DEFINED CMAKE_CXX_COMPILER)
        # /opt/adaptivecpp is the release symlink on the Linux boxes, ~/opt/adaptivecpp the
        # macOS install (scripts/macos/build_adaptivecpp_metal.sh); PATH wins over both.
        find_program(BATCHLAS_ACPP_COMPILER acpp PATHS /opt/adaptivecpp/bin "$ENV{HOME}/opt/adaptivecpp/bin")
        if(BATCHLAS_ACPP_COMPILER)
            message(STATUS "Auto-detected AdaptiveCpp driver: ${BATCHLAS_ACPP_COMPILER}")
            if(DEFINED CMAKE_CXX_COMPILER)
                # project() keeps the old compiler for this pass; BatchLASOptions.cmake reads this.
                set(BATCHLAS_ACPP_SWAP_PENDING TRUE)
            endif()
            set(CMAKE_CXX_COMPILER "${BATCHLAS_ACPP_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
        else()
            message(WARNING "BATCHLAS_SYCL_IMPL=ACPP but no acpp on PATH, in /opt/adaptivecpp/bin or ~/opt/adaptivecpp/bin; "
                            "set -DCMAKE_CXX_COMPILER=<prefix>/bin/acpp.")
        endif()
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

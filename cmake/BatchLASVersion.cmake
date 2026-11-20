# include/batchlas/version.hh is checked in, not generated: header-only use (-I <checkout>/include)
# never runs CMake, so a generated header would need a checked-in fallback that can go stale
# silently. One checked-in copy plus this check keeps a single number per release: configure stops
# when the header and project(BatchLAS VERSION ...) disagree.
set(_batchlas_version_header "${PROJECT_SOURCE_DIR}/include/batchlas/version.hh")
file(STRINGS "${_batchlas_version_header}" _batchlas_version_defines
     REGEX "^#define BATCHLAS_VERSION_(MAJOR|MINOR|PATCH|STRING) ")
set(_batchlas_version_mismatch "")
foreach(_part MAJOR MINOR PATCH STRING)
    if(_part STREQUAL "STRING")
        set(_want "\"${PROJECT_VERSION}\"")
    else()
        set(_want "${PROJECT_VERSION_${_part}}")
    endif()
    set(_got "")
    foreach(_line IN LISTS _batchlas_version_defines)
        if(_line MATCHES "^#define BATCHLAS_VERSION_${_part} +(.+)$")
            string(STRIP "${CMAKE_MATCH_1}" _got)
        endif()
    endforeach()
    if(NOT _got STREQUAL _want)
        string(APPEND _batchlas_version_mismatch
            "\n  BATCHLAS_VERSION_${_part} is '${_got}', project() says '${_want}'")
    endif()
endforeach()
if(_batchlas_version_mismatch)
    message(FATAL_ERROR
        "version: ${_batchlas_version_header} disagrees with project(BatchLAS VERSION "
        "${PROJECT_VERSION}) in CMakeLists.txt:${_batchlas_version_mismatch}\n"
        "Edit both together when bumping the version.")
endif()
unset(_batchlas_version_header)
unset(_batchlas_version_defines)
unset(_batchlas_version_mismatch)

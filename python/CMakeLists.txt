cmake_minimum_required(VERSION 3.14)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_CMAKE_DIR)
        find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}")
    else()
        message(FATAL_ERROR "Could not find pybind11. Install it with pip install pybind11")
    endif()
endif()

set(BATCHLAS_PY_PACKAGE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/batchlas")
set(BATCHLAS_PY_PACKAGE_BUILD "${CMAKE_BINARY_DIR}/python/batchlas")
file(MAKE_DIRECTORY "${BATCHLAS_PY_PACKAGE_BUILD}")

set(BATCHLAS_PY_EXTENSION_SOURCES
    batchlas/module.cc
    batchlas/bindings/types.cc
    batchlas/bindings/ops_blas.cc
    batchlas/bindings/ops_factorization.cc
    batchlas/bindings/ops_spectral.cc
    batchlas/bindings/ops_misc.cc
)

pybind11_add_module(_batchlas ${BATCHLAS_PY_EXTENSION_SOURCES})

target_compile_features(_batchlas PRIVATE cxx_std_17)
target_include_directories(_batchlas PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include"
)
target_link_libraries(_batchlas PRIVATE BatchLAS::batchlas)

set_target_properties(_batchlas PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_batchlas"
    LIBRARY_OUTPUT_DIRECTORY "${BATCHLAS_PY_PACKAGE_BUILD}"
    BUILD_RPATH "${CMAKE_BINARY_DIR}/src"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
)

set(BATCHLAS_PY_PACKAGE_FILES
    batchlas/__init__.py
    batchlas/_api.py
    batchlas/_options.py
)

foreach(_py_file IN LISTS BATCHLAS_PY_PACKAGE_FILES)
    get_filename_component(_py_name "${_py_file}" NAME)
    add_custom_command(
        TARGET _batchlas POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${_py_file}"
            "${BATCHLAS_PY_PACKAGE_BUILD}/${_py_name}"
        VERBATIM
    )
endforeach()

install(TARGETS _batchlas DESTINATION "${Python_SITEARCH}/batchlas")
install(FILES ${BATCHLAS_PY_PACKAGE_FILES} DESTINATION "${Python_SITEARCH}/batchlas")

if(BATCHLAS_BUILD_TESTS)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE BATCHLAS_PYTEST_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(BATCHLAS_PYTEST_RESULT EQUAL 0)
        add_test(
            NAME batchlas_python_build_import
            COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                "${Python_EXECUTABLE}" -m pytest
                "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )
    endif()
endif()

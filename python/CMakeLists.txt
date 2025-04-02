cmake_minimum_required(VERSION 3.14)

# Find Python and PyBind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

# If pybind11 was not found via config, try to find it via PythonLibs
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_CMAKE_DIR)
        find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}")
    else()
        message(FATAL_ERROR "Could not find pybind11. Please install it with pip install pybind11")
    endif()
endif()

# Set modified PyBind11 flags
set(PYBIND11_CPP_STANDARD -std=c++17)

# Create the Python extension module with a unique name to avoid conflicts with the main library
pybind11_add_module(batchlas_python batchlas/batchlas_bindings.cpp)

# Ensure we use position-independent code
set_property(TARGET batchlas_python PROPERTY POSITION_INDEPENDENT_CODE ON)

# Link against BatchLAS library, ensure WHOLE_ARCHIVE to include all symbols
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(batchlas_python PRIVATE 
        -Wl,--whole-archive batchlas -Wl,--no-whole-archive
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_link_libraries(batchlas_python PRIVATE 
        -Wl,-force_load batchlas
    )
else()
    target_link_libraries(batchlas_python PRIVATE batchlas)
endif()

# If CUDA is enabled, explicitly link against CUDA libraries
if(DEFINED CUBLAS_LIBRARY)
    target_link_libraries(batchlas_python PRIVATE ${CUBLAS_LIBRARY})
endif()
if(DEFINED CUBLASLT_LIBRARY)
    target_link_libraries(batchlas_python PRIVATE ${CUBLASLT_LIBRARY})
endif()
if(DEFINED CUSOLVER_LIBRARY)
    target_link_libraries(batchlas_python PRIVATE ${CUSOLVER_LIBRARY})
endif()
if(DEFINED CUSPARSE_LIBRARY)
    target_link_libraries(batchlas_python PRIVATE ${CUSPARSE_LIBRARY})
endif()

# Include directories
target_include_directories(batchlas_python PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_BINARY_DIR}/include
)

# Set the output directory and name for the Python module
set_target_properties(batchlas_python PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/batchlas
    OUTPUT_NAME "batchlas${Python_EXTENSION_SUFFIX}"
    PREFIX ""
)

# Install the Python module
install(TARGETS batchlas_python DESTINATION ${Python_SITEARCH}/batchlas)
install(FILES batchlas/__init__.py DESTINATION ${Python_SITEARCH}/batchlas)
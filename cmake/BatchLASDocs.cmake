# Documentation site target. The build itself lives in scripts/build_docs.sh so
# that CI can build the site without configuring a SYCL tree; this only wraps it.
# evidence: docs/developer/documentation.md#building-the-site

function(batchlas_add_docs_target)
    find_program(BATCHLAS_DOXYGEN_EXECUTABLE NAMES doxygen
        DOC "Doxygen 1.18+ used by the batchlas_docs target")
    if(NOT BATCHLAS_DOXYGEN_EXECUTABLE)
        message(FATAL_ERROR
            "BATCHLAS_BUILD_DOCS=ON but no doxygen was found. Install Doxygen 1.18+ "
            "or pass -DBATCHLAS_DOXYGEN_EXECUTABLE=/path/to/doxygen.")
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(docs_out "${PROJECT_BINARY_DIR}/docs")
    add_custom_target(batchlas_docs
        COMMAND ${CMAKE_COMMAND} -E env
                "DOXYGEN=${BATCHLAS_DOXYGEN_EXECUTABLE}"
                sh "${PROJECT_SOURCE_DIR}/scripts/build_docs.sh" "${docs_out}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Building the BatchLAS documentation site in ${docs_out}/html"
        VERBATIM)

    install(DIRECTORY "${docs_out}/html/"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}/html"
        OPTIONAL)
    message(STATUS "Documentation: target batchlas_docs -> ${docs_out}/html")
endfunction()

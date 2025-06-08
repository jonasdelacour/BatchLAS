set(EXTENSIONS_SOURCES
    ortho_matrixview.cc
    syevx_matrixview.cc
    inv_matrixview.cc
    lanczos_matrixview.cc
)

target_sources(batchlas PRIVATE ${EXTENSIONS_SOURCES})


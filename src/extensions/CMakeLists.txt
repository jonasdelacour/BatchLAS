set(EXTENSIONS_SOURCES
    ortho.cc
    ortho_matrixview.cc
    syevx_matrixview.cc
    syevx.cc
    lanczos.cc
)

target_sources(batchlas PRIVATE ${EXTENSIONS_SOURCES})
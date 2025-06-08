set(EXTENSIONS_SOURCES
    ortho.cc
    syevx.cc
    inv.cc
    lanczos.cc
)

target_sources(batchlas PRIVATE ${EXTENSIONS_SOURCES})


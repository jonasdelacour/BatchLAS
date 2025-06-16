set(EXTENSIONS_SOURCES
    ortho.cc
    syevx.cc
    inv.cc
    tridiag_solver.cc
    lanczos.cc
)

target_sources(batchlas PRIVATE ${EXTENSIONS_SOURCES})


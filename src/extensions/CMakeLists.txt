set(EXTENSIONS_SOURCES
    ortho.cc
    syevx.cc
    inv.cc
    tridiag_solver.cc
    lanczos.cc
    trmm.cc
    steqr.cc
    steqr_cta.cc
    stedc.cc
    stedc_flat.cc
    ritz_values.cc
    lascl.cc
)

target_sources(batchlas PRIVATE ${EXTENSIONS_SOURCES})


set(EXTENSIONS_CTA_SOURCES
    ormqr_cta.cc
    steqr_cta.cc
    sytrd_cta.cc
    sytrd_sb2st_cta.cc
)

set(EXTENSIONS_SOURCES
    ortho.cc
    syevx.cc
    syev_cta.cc
    syev_blocked.cc
    inv.cc
    tridiag_solver.cc
    lanczos.cc
    trmm.cc
    ormqr_blocked.cc
    steqr.cc
    latrd_lower_panel.cc
    sytrd_blocked.cc
    sytrd_sy2sb.cc
    sytrd_sb2st.cc
    band_reduction.cc
    stedc.cc
    stedc_flat.cc
    ritz_values.cc
    lascl.cc
)

target_sources(batchlas_extensions_obj PRIVATE ${EXTENSIONS_SOURCES})
target_sources(batchlas_extensions_cta_obj PRIVATE ${EXTENSIONS_CTA_SOURCES})


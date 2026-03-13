set(EXTENSIONS_CTA_SOURCES
    ormqr_cta.cc
    steqr_cta.cc
    stedc_merge_cta.cc
    sytrd_cta.cc
    sytrd_sb2st_cta.cc
)

set(EXTENSIONS_EIGEN_SOURCES
    ortho.cc
    iluk.cc
    syevx.cc
    syev_cta.cc
    syev_blocked.cc
    syev_two_stage.cc
    lanczos.cc
    ritz_values.cc
)

set(EXTENSIONS_FACTORIZATION_SOURCES
    gebrd.cc
    bdsqr.cc
    ormbr.cc
    gesvd_blocked.cc
    inv.cc
    ormqr_blocked.cc
    lascl.cc
)

set(EXTENSIONS_SYMMETRIC_SOURCES
    symm.cc
    syrk.cc
    syr2k.cc
    trmm.cc
)

set(EXTENSIONS_TRIDIAG_SOURCES
    tridiag_solver.cc
    steqr.cc
    steqr_wg.cc
    steqr_legacy.cc
    latrd_lower_panel.cc
    sytrd_blocked.cc
    sytrd_sy2sb.cc
    sytrd_sb2st.cc
    band_reduction.cc
    stedc.cc
    stedc_secular.cc
    stedc_merge_kernels.cc
    stedc_flat.cc
)

target_sources(batchlas_extensions_eigen_obj PRIVATE ${EXTENSIONS_EIGEN_SOURCES})
target_sources(batchlas_extensions_factorization_obj PRIVATE ${EXTENSIONS_FACTORIZATION_SOURCES})
target_sources(batchlas_extensions_symmetric_obj PRIVATE ${EXTENSIONS_SYMMETRIC_SOURCES})
target_sources(batchlas_extensions_tridiag_obj PRIVATE ${EXTENSIONS_TRIDIAG_SOURCES})
target_sources(batchlas_extensions_cta_obj PRIVATE ${EXTENSIONS_CTA_SOURCES})


set(EXTENSIONS_CTA_SOURCES
    gebrd_cta.cc
    ormqr_cta.cc
    steqr_cta.cc
    stedc_merge_cta.cc
    sytrd_cta.cc
    sytrd_sb2st_cta.cc
    syev_jacobi_cta.cc
    syev_cta_fused.cc
)

set(EXTENSIONS_EIGEN_SOURCES
    ortho.cc
    iluk.cc
    syevx.cc
    syevx_direct.cc
    syevx_direct_subset.cc
    syevx_lobpcg.cc
    syevx_filtered.cc
    syev_cta.cc
    uplo_mirror.cc
    syev_blocked.cc
    syev_two_stage.cc
    lanczos.cc
    ritz_values.cc
)

set(EXTENSIONS_FACTORIZATION_SOURCES
    gebrd.cc
    gebrd_blocked.cc
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

# The unit of SYCL *device* linking is the shared library, not the object file:
# one changed object redoes ptxas and the native_cpu AOT pass for every object in
# the library. These three groups used to be one 15-object library whose device
# link cost ~200 s, which every edit to any tridiagonal source paid in full.
#
# The grouping is NOT by topic, it is by device-code cluster: SYCL device
# functions are resolved across translation units within a library, so a source
# must sit with the sources whose device symbols it calls. Splitting through a
# cluster is a hard link error (`ptxas fatal: Unresolved extern function ...`),
# never a silent miscompile — so a bad regrouping fails the build immediately.
# Measured per-group device link, single-threaded: stedc ~21 s, tridiag ~15 s,
# sytrd ~190 s (sytrd is dominated by latrd_lower_panel.cc alone at ~110 s).
set(EXTENSIONS_TRIDIAG_SOURCES
    tridiag_solver.cc
    steqr.cc
    stebz.cc
    stein.cc
    steqr_wg.cc
    steqr_legacy.cc
)

# latrd_lower_panel.cc gets a library to itself: it is ~112 s of the ~170 s
# sytrd device link on its own. Measured, single-threaded: the 6-object library
# 170 s; latrd alone 112 s; the other five alone 69 s. Both halves link rc=0,
# which is what proves the clusters are disjoint. This trades ~11 s on a clean
# build (off the critical path at -j20) for ~101 s on every edit to the other
# five sources.
set(EXTENSIONS_LATRD_SOURCES
    latrd_lower_panel.cc
)

set(EXTENSIONS_SYTRD_SOURCES
    sytrd_blocked.cc
    sytrd_sy2sb.cc
    sytrd_sb2st.cc
    sytrd_sb2st_hh.cc
    band_reduction.cc
)

set(EXTENSIONS_STEDC_SOURCES
    stedc.cc
    stedc_secular.cc
    stedc_merge_kernels.cc
)

target_sources(batchlas_extensions_eigen_obj PRIVATE ${EXTENSIONS_EIGEN_SOURCES})
target_sources(batchlas_extensions_factorization_obj PRIVATE ${EXTENSIONS_FACTORIZATION_SOURCES})
target_sources(batchlas_extensions_symmetric_obj PRIVATE ${EXTENSIONS_SYMMETRIC_SOURCES})
target_sources(batchlas_extensions_tridiag_obj PRIVATE ${EXTENSIONS_TRIDIAG_SOURCES})
target_sources(batchlas_extensions_sytrd_obj PRIVATE ${EXTENSIONS_SYTRD_SOURCES})
target_sources(batchlas_extensions_latrd_obj PRIVATE ${EXTENSIONS_LATRD_SOURCES})
target_sources(batchlas_extensions_stedc_obj PRIVATE ${EXTENSIONS_STEDC_SOURCES})
target_sources(batchlas_extensions_cta_obj PRIVATE ${EXTENSIONS_CTA_SOURCES})

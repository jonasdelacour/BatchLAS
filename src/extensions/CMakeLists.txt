set(EXTENSIONS_CTA_SOURCES
    gebrd_cta.cc
    ormqr_cta.cc
    # WP4 Phases 1 and 2. THIS list and no other (W12): the blocked driver's
    # diagonal leaf IS potrf_cta_body, shared through potrf_cta_device.hh, so
    # potrf_blocked.cc lands here too. While every helper stays an inline
    # template a split would link by duplicating the body into both TUs and break
    # only when someone marks one SYCL_EXTERNAL -- ":53-57 below: a source must
    # sit with the sources whose device symbols it calls". This library is also
    # the only one configured NO_CPU_TARGETS (src/CMakeLists.txt:67-71), and
    # potrf's native route is GPU-only, so a native_cpu image of these templates
    # would be dead weight.
    potrf_cta.cc
    potrf_blocked.cc
    # WP5 scaffolding, and THIS list for the same W12 reason as potrf's pair: the
    # blocked geqrf driver's panel leaf will BE the CTA device function, shared
    # through a geqrf_cta_device.hh, so the two sources must sit in ONE
    # device-code cluster (":53-57 below: a source must sit with the sources
    # whose device symbols it calls"). Neither file contains device code YET --
    # both currently hold only the capability answers, which report the kernels
    # absent -- but the grouping is decided here so landing a kernel is not also
    # a CMake decision. NOTE the counter-example NOT to copy: ormqr_cta.cc is in
    # this list while ormqr_blocked.cc is in EXTENSIONS_FACTORIZATION_SOURCES,
    # which works only because those two share no device symbol. orgqr_blocked.cc
    # follows ormqr_blocked.cc for exactly that reason.
    geqrf_cta.cc
    geqrf_blocked.cc
    # WP6 scaffolding, and THIS list for the same W12 reason as the potrf and
    # geqrf pairs: the blocked getrf driver's diagonal-panel leaf will BE the CTA
    # device function, shared through a getrf_cta_device.hh, so the two sources
    # must sit in ONE device-code cluster (":77-86 below: a source must sit with
    # the sources whose device symbols it calls"). Neither file contains device
    # code YET -- both currently hold only the capability answers, which report
    # the kernels absent -- but the grouping is decided here so landing a kernel
    # is not also a CMake decision. NOTE that getrs_native.cc and
    # getri_blocked.cc are NOT here: both are a permutation kernel plus ROUTED
    # trsm calls and share no device symbol with this pair, so they follow
    # orgqr_blocked.cc into EXTENSIONS_FACTORIZATION_SOURCES. If either ever
    # calls the getrf CTA device body it moves here, and the failure mode of
    # getting that wrong is a hard link error, never a silent miscompile (:81-83).
    getrf_cta.cc
    getrf_blocked.cc
    steqr_cta.cc
    stedc_merge_cta.cc
    sytrd_cta.cc
    sytrd_sb2st_cta.cc
    syev_jacobi_cta.cc
    gesvdj_cta.cc
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
    bdsdc.cc
    ormbr.cc
    gesvd_blocked.cc
    inv.cc
    ormqr_blocked.cc
    # WP5 scaffolding. HERE and not in EXTENSIONS_CTA_SOURCES: orgqr ships as
    # ormqr applied to an identity, so it is built on ormqr_blocked.cc directly
    # above and shares device symbols with neither geqrf TU. The grouping rule is
    # the cluster, not the topic (see :53-57 below).
    orgqr_blocked.cc
    # WP6 scaffolding. HERE and not in EXTENSIONS_CTA_SOURCES: getrs is a row
    # permutation plus two ROUTED trsm calls and getri is "write P into C" plus
    # two ROUTED trsm calls, so neither shares a device symbol with the getrf
    # pair. The grouping rule is the cluster, not the topic (see :77-86 below),
    # and this is the same reason orgqr_blocked.cc sits here rather than beside
    # geqrf_cta.cc. If either ever calls the getrf CTA device body it MOVES.
    getrs_native.cc
    # The FUSED narrow-RHS getrs tier. HERE for the same reason getrs_native.cc is:
    # it calls no getrf CTA device function, so it shares no device symbol with
    # that cluster. Its kernel is entirely its own -- one work-group per matrix,
    # the interchange walk and both substitutions in one launch, no GEMM and no
    # laswp -- so it shares no device symbol with getrs_native.cc either.
    getrs_fused.cc
    getri_blocked.cc
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

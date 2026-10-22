# src/ops/<op>/: <op>.cc is the public entry point (listed in src/CMakeLists.txt with select/),
# kernels/ holds the op's native kernels, nested folders hold internal sub-ops (syev/stedc) and
# src/kernels/common/ holds device code shared by several ops. Folders are by topic; the
# OBJECT-library lists below are by device-code cluster, and a move between folders never
# changes which list a source is in.

set(EXTENSIONS_CTA_SOURCES
    gesvd/gebrd/gebrd_cta.cc
    ormqr/kernels/ormqr_cta.cc
    # WP4 Phases 1 and 2. THIS list and no other (W12): the blocked driver's
    # diagonal leaf IS potrf_cta_body, shared through potrf_cta_device.hh, so
    # potrf_blocked.cc lands here too. While every helper stays an inline
    # template a split would link by duplicating the body into both TUs and break
    # only when someone marks one SYCL_EXTERNAL -- "the device-link note below: a source must
    # sit with the sources whose device symbols it calls". This library is also
    # the only one configured NO_CPU_TARGETS (src/CMakeLists.txt, NO_CPU_TARGETS foreach), and
    # potrf's native route is GPU-only, so a native_cpu image of these templates
    # would be dead weight.
    potrf/kernels/potrf_cta.cc
    potrf/kernels/potrf_blocked.cc
    # WP6/P1: the register-resident TINY tier. Same device-code cluster as the pair
    # above -- the tests compare the two kernels bit-for-bit, and tiny_device.hh's
    # helpers are inline templates shared with the getrf/geqrf tiny kernels, so all
    # three must sit with the sources whose device symbols they call.
    potrf/kernels/potrf_tiny.cc
    # WP6/P2: the FUSED factor-and-solve tier. Same cluster as potrf_tiny.cc -- it
    # reuses tiny_device.hh's inline helpers and is GPU-only, so a native_cpu image
    # of it would be dead weight (this library is the only NO_CPU_TARGETS one).
    posv/kernels/posv_tiny.cc
    # WP6/P3: the LEFT-LOOKING panel leaf. Same device-code cluster again -- the tests
    # compare it against potrf_cta_body on the same matrices, and it is the blocked
    # driver's intended replacement leaf.
    potrf/kernels/potrf_lpanel.cc
    # WP5 scaffolding, and THIS list for the same W12 reason as potrf's pair: the
    # blocked geqrf driver's panel leaf will BE the CTA device function, shared
    # through a geqrf_cta_device.hh, so the two sources must sit in ONE
    # device-code cluster ("the device-link note below: a source must sit with the
    # sources whose device symbols it calls"). Both files now DO contain device code --
    # getrf_cta.cc defines GetrfPanelResidentKernel and GetrfPanelGlobalKernel,
    # and getrf_blocked_available<T>() is true for all four types -- so this
    # grouping is load-bearing, not speculative: getting it wrong is the
    # `ptxas fatal: Unresolved extern function` this comment warns about.
    # NOTE the counter-example NOT to copy: ormqr_cta.cc is in
    # this list while ormqr_blocked.cc is in EXTENSIONS_FACTORIZATION_SOURCES,
    # which works only because those two share no device symbol. orgqr_blocked.cc
    # follows ormqr_blocked.cc for exactly that reason.
    geqrf/kernels/geqrf_cta.cc
    geqrf/kernels/geqrf_blocked.cc
    # WP6/P1, the register-resident TINY tier for square n <= 32. HERE because the rule
    # "a source must sit with the sources whose device symbols it calls" places
    # it: geqrf_tiny.cc reuses geqrf_cta_device.hh's LarfgScalars/geqrf_larfg_scalars
    # verbatim rather than re-deriving the LAPACK real-beta tau convention, so it shares
    # a device-code cluster with the pair above. It is also GPU-only, which is what this
    # object library's NO_CPU_TARGETS configuration (src/CMakeLists.txt, NO_CPU_TARGETS foreach) wants.
    geqrf/kernels/geqrf_tiny.cc
    # WP6 scaffolding, and THIS list for the same W12 reason as the potrf and
    # geqrf pairs: the blocked getrf driver's diagonal-panel leaf will BE the CTA
    # device function, shared through a getrf_cta_device.hh, so the two sources
    # must sit in ONE device-code cluster ("the device-link note below: a source must sit with
    # the sources whose device symbols it calls"). Neither file contains device
    # code YET -- both currently hold only the capability answers, which report
    # the kernels absent -- but the grouping is decided here so landing a kernel
    # is not also a CMake decision. NOTE that getrs_native.cc and
    # getri_blocked.cc are NOT here: both are a permutation kernel plus ROUTED
    # trsm calls and share no device symbol with this pair, so they follow
    # orgqr_blocked.cc into EXTENSIONS_FACTORIZATION_SOURCES. If either ever
    # calls the getrf CTA device body it moves here, and the failure mode of
    # getting that wrong is a hard link error, never a silent miscompile (device-link note below).
    getrf/kernels/getrf_cta.cc
    getrf/kernels/getrf_blocked.cc
    # WP6/P1, the register-resident tier. HERE for a different reason than the pair
    # above -- it shares NO device symbol with getrf_cta.cc (its whole body is inline
    # in the lambda, and it borrows only lu_cabs1, an inline template) -- so the rule
    # "a source must sit with the sources whose device symbols it calls" does
    # not place it. What places it is that this object library is the only one
    # configured NO_CPU_TARGETS (src/CMakeLists.txt, NO_CPU_TARGETS foreach) and the tier is GPU-only:
    # a native_cpu image of these templates would be dead weight. The file's own rule
    # is "the cluster, not the topic", so the reason is stated rather than assumed.
    getrf/kernels/getrf_tiny.cc
    # WP6/P2: the FUSED factor-and-solve tier. HERE for getrf_tiny.cc's reason, not
    # for the shared-device-symbol one: it borrows only lu_cabs1, an inline template,
    # and the tier is GPU-only.
    gesv/kernels/gesv_tiny.cc
    # WP6/P4, the register-resident PANEL leaf for the blocked driver. HERE by the
    # cluster rule and not by the GPU-only one: getrf_panel_reg_device.hh includes
    # getrf_cta_device.hh for lu_cabs1 and lu_zero, and getrf_blocked.cc calls this
    # file's getrf_panel_reg_factorize, so all three share a device-code cluster.
    getrf/kernels/getrf_panel_reg.cc
    syev/steqr/steqr_cta.cc
    syev/stedc/stedc_merge_cta.cc
    syev/sytrd/sytrd_cta.cc
    syev/sytrd/sytrd_sb2st_cta.cc
    syev/kernels/syev_jacobi_cta.cc
    gesvd/kernels/gesvdj_cta.cc
    syev/kernels/syev_cta_fused.cc
)

set(EXTENSIONS_EIGEN_SOURCES
    ortho/ortho.cc
    iluk/iluk.cc
    syevx/syevx.cc
    syevx/kernels/syevx_direct.cc
    syevx/kernels/syevx_direct_subset.cc
    syevx/kernels/syevx_lobpcg.cc
    syevx/kernels/syevx_filtered.cc
    syev/kernels/syev_cta.cc
    syev/kernels/uplo_mirror.cc
    syev/kernels/syev_blocked.cc
    syev/kernels/syev_two_stage.cc
    lanczos/lanczos.cc
    ritz_values/ritz_values.cc
    syev_align/syev_align.cc
)

set(EXTENSIONS_FACTORIZATION_SOURCES
    gesvd/gebrd/gebrd.cc
    gesvd/gebrd/gebrd_blocked.cc
    gesvd/bdsqr/bdsqr.cc
    gesvd/bdsdc/bdsdc.cc
    gesvd/ormbr/ormbr.cc
    gesvd/kernels/gesvd_blocked.cc
    inv/inv.cc
    ormqr/kernels/ormqr_blocked.cc
    # WP5 scaffolding. HERE and not in EXTENSIONS_CTA_SOURCES: orgqr ships as
    # ormqr applied to an identity, so it is built on ormqr_blocked.cc directly
    # above and shares device symbols with neither geqrf TU. The grouping rule is
    # the cluster, not the topic (see the device-link note below).
    orgqr/kernels/orgqr_blocked.cc
    # WP6 scaffolding. HERE and not in EXTENSIONS_CTA_SOURCES: getrs is a row
    # permutation plus two ROUTED trsm calls and getri is "write P into C" plus
    # two ROUTED trsm calls, so neither shares a device symbol with the getrf
    # pair. The grouping rule is the cluster, not the topic (see the device-link note below),
    # and this is the same reason orgqr_blocked.cc sits here rather than beside
    # geqrf_cta.cc. If either ever calls the getrf CTA device body it MOVES.
    getrs/kernels/getrs_native.cc
    # The FUSED narrow-RHS getrs tier. HERE for the same reason getrs_native.cc is:
    # it calls no getrf CTA device function, so it shares no device symbol with
    # that cluster. Its kernel is entirely its own -- one work-group per matrix,
    # the interchange walk and both substitutions in one launch, no GEMM and no
    # laswp -- so it shares no device symbol with getrs_native.cc either.
    getrs/kernels/getrs_fused.cc
    getri/kernels/getri_blocked.cc
    lascl/lascl.cc
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
    syev/tridiag/tridiag_solver.cc
    syev/steqr/steqr.cc
    syev/tridiag/stebz.cc
    syev/tridiag/stein.cc
    syev/steqr/steqr_wg.cc
    syev/steqr/steqr_legacy.cc
)

# latrd_lower_panel.cc gets a library to itself: it is ~112 s of the ~170 s
# sytrd device link on its own. Measured, single-threaded: the 6-object library
# 170 s; latrd alone 112 s; the other five alone 69 s. Both halves link rc=0,
# which is what proves the clusters are disjoint. This trades ~11 s on a clean
# build (off the critical path at -j20) for ~101 s on every edit to the other
# five sources.
set(EXTENSIONS_LATRD_SOURCES
    syev/sytrd/latrd_lower_panel.cc
)

set(EXTENSIONS_SYTRD_SOURCES
    syev/sytrd/sytrd_blocked.cc
    syev/sytrd/sytrd_sy2sb.cc
    syev/sytrd/sytrd_sb2st.cc
    syev/sytrd/sytrd_sb2st_hh.cc
    syev/sytrd/band_reduction.cc
)

set(EXTENSIONS_STEDC_SOURCES
    syev/stedc/stedc.cc
    syev/stedc/stedc_secular.cc
    syev/stedc/stedc_merge_kernels.cc
)

target_sources(batchlas_extensions_eigen_obj PRIVATE ${EXTENSIONS_EIGEN_SOURCES})
target_sources(batchlas_extensions_factorization_obj PRIVATE ${EXTENSIONS_FACTORIZATION_SOURCES})
target_sources(batchlas_extensions_tridiag_obj PRIVATE ${EXTENSIONS_TRIDIAG_SOURCES})
target_sources(batchlas_extensions_sytrd_obj PRIVATE ${EXTENSIONS_SYTRD_SOURCES})
target_sources(batchlas_extensions_latrd_obj PRIVATE ${EXTENSIONS_LATRD_SOURCES})
target_sources(batchlas_extensions_stedc_obj PRIVATE ${EXTENSIONS_STEDC_SOURCES})
target_sources(batchlas_extensions_cta_obj PRIVATE ${EXTENSIONS_CTA_SOURCES})

# One device-link unit. Keeping it small matters: this is the .so whose device
# link ptxas runs on, and it already measures ~43.4 s with gemm_kernels.cc alone
# (scripts/register_probe.sh). Measure the delta when adding to it.
#
# gemv_native.cc goes HERE, in the ordinary list, and not into a
# NO_CPU_TARGETS object library, however tempting the link time makes that.
# batchlas_sycl_obj is built with batchlas_apply_object_options() with no
# NO_CPU_TARGETS flag (src/CMakeLists.txt, NO_CPU_TARGETS foreach), i.e. for
# -fsycl-targets=nvidia_gpu_sm_89,native_cpu. The native_cpu image is the whole
# WP7 deliverable: 20 of the 40 vendor-free gemv_tests failures are
# GemvMatrixViewTest/0..3 on a native_cpu Device("cpu") queue, and a GPU-only
# native gemv closes the other 20 while leaving the suite red and the burn-down
# unmoved. Measure the device-link delta and report it; do not pay for it by
# dropping the CPU target.
set(SYCL_SOURCES
    gemm/kernels/gemm_kernels.cc
    trsm/kernels/trsm_native.cc
    trsm/kernels/trsm_sg_left.cc
    gemv/kernels/gemv_native.cc
    spmm/kernels/spmm_native.cc
)

target_sources(batchlas_sycl_obj PRIVATE ${SYCL_SOURCES})

potri: ctest 3/3 pass (device_blas_v2_potri_tests, potri_tests 340s, potri_tests_heavy); bitid 3464/3464 identical (factorization .so md5 differs base b4107 vs cand 9c923)
gesvd-ormbr: ctest 9/10 pass; gesvd_tests fails 2 cases GesvdTest/{2,3}.CtaProviderCoversAllJobCombinations (float,double CUDA) = same on base, listed in known-failures.txt; bitid 3464/3464 identical
BASE union ctest (83 tests, GPU1): 81/83; fails gesvd_tests (CtaProviderCoversAll x2, known-failures.txt) and syev_blocked_tests_heavy (EigenvaluesOnlyLowerMatchesNetlib x4, known-failures-host-jrhnbr.txt)
potrf-cta-lpanel: ctest 23/23 pass (incl potrf/posv/potrf_candidates/potrf_mask heavy, potrf_tests_native, sygvd heavy); bitid 3464/3464 identical incl complex CTA (algo) and blocked fold n=256
bdsqr-bdsdc: ctest 8/10; gesvd_tests 2 known (same as base); device_blas_v2_bdsqr-bdsdc_tests SegFault: heap corruption (malloc unsorted list) at V2BidiagRealTest<float>.SelectSortCarriesColumns submit -> engineer fixing. bitid 3464/3464 identical (gesvd bdsdc/bdsqr all shapes)
  -> FIX dd429378 pushed to inv/bdsqr-bdsdc: test stride bug (per=n+rows*ld, C spans (n-1)*ld+rows) overran host vector by 7 elems; test-only, 22/22 + bdsdc/bdsqr pass
geqrf: ctest 42/44; syev_blocked_tests_heavy NETLIB x4 known (same as base); device_blas_v2_geqrf_tests fails ButterflyOrdersAreBitExact x4 types (period_bitdiff 84096/28032) + Geqr2ComposedFromOps cdouble exception -> engineer. bitid 3464/3464 identical
  -> FIX 37ccf05f pushed to inv/geqrf (test-only): butterfly input scaled by global idx%7 broke 4-item period (now lane%7; deliberate break caught all 4 types); Geqr2ComposedFromOps cd launch lacked WG bound -> 66 regs x1024 launch error; new test-local TEST_WG_BOUND macro (max_work_group_size; reqd only when BATCHLAS_NVPTX_REQD_WG_OK). 24/24, geqrf ctest 8/8
syev-drivers-fused: ctest 27/28; syev_blocked_tests_heavy NETLIB x4 known (same as base); device_blas_v2_syev-drivers-fused_tests pass; bitid 3464/3464 identical
larft-orgqr-ormqr: ctest 20/21 (gesvd_tests known x2 as base; incl ormqr_cta, ormqr(+_native), ormqr_blocked, orgqr, geqrf, syev_cta(+fused), device_blas_v2 pass). bitid: 3288/3464 identical; all real ormqx_cta (L+R, QR+QL, N/T) identical; differs (algo, expected): complex ormqx_cta 80 lines (L and R), ormqr blocked IMPL=device all types 48 (v2 larft), downstream complex syev:cta A 16, complex gesvd U/Vh 32. Unset-IMPL ormqr blocked identical.
tridiag-stebz-stein: ctest 37/38 (syev_blocked_tests_heavy NETLIB x4 known = base; stebz/stein/steqr/tridiag_mask/syevx*/syev*/device_blas_v2 pass); bitid 3464/3464 identical incl tridiagonal_solver, stebz, stein, syevx subset (algo ports bit-identical too)
gebrd: ctest 9/10 (gesvd_tests known x2 = base; device_blas_v2_gebrd, gesvd(+heavy), gesvd_mask(+heavy), gesvd_candidates(+heavy), gesvdj_cta, bdsqr, bdsdc pass). bitid 3040/3464: ALL real outputs identical (gebrd blocked/cta/unblocked f,d; real gesvd); complex differ (algo/fail ports): gebrd unblocked+blocked cf/cd all shapes (200 lines) and complex gesvd S/U/Vh/A (224 lines)

## potrf timing (13:16, both GPUs idle at gate, 14 reps)
cta float n16 1.084 (0.1221->0.1323, tight ranges: regression); n48 1.005
cta cfloat n16 0.895 (win); n48 1.060 (2.467->2.615, cand range tight 2.609-2.618: regression)
lpanel float n16 1.018, n32 0.970, n64 1.002 (neutral)
blocked cfloat n256 1.023, n512 1.087 (ranges overlap; noisy)

## bdsqr-bdsdc timing (13:16, gated idle, 14 reps): all 0.999-1.001, tight ranges. Neutral.
## gebrd first pass (13:17-15:27) DISCARDED: overlapped tdm-cuda bisect on GPU1 (max ranges 2-4x min). Rerun under flock from 15:28.
## Orchestrator note 2026-11-02: GPU0 gate withdrawn; flock /tmp/batchlas-measure.lock per block; GPU0 util recorded per block.

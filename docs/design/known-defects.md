# Known defects {#design_known_defects}

> **Status:** current · checked 2026-10-29

Open defects located to a line, with symptom, cause and what a fix needs. A fixed defect keeps a
short entry only while code, tests or other pages cite its heading or number. Performance debt is
listed per op under [../perf/](../perf/README.md).

## Defect 1: `ortho`'s transposed arm builds a view that does not describe the memory

Fixed (WP8). Under `transA = Trans`/`ConjTrans` the CGS arm built a view with `ld = m` and took
column `i` where the arm's vectors are rows; gemv's `can_run` refused the mismatched vector length,
so the call went to the vendor. Fix: `src/ops/ortho/ortho.cc:228-231`. Guard: `tests/ortho_tests.cc`
runs every algorithm under `NoTrans`, `Trans` and `ConjTrans`.

## Defect 3: `lanczos` issues a two-column multiply and consumes one column

Open: right answer, twice the level-3 work. `padded_vector` and `padded_output` are two columns
wide (`src/ops/lanczos/lanczos.cc:65-66`) so the sparse arm does not fall back to SpMV; the spmm/gemm
at `:122-124` fills both, and the kernel reads only column 0 (`local_v_next`, `:139`). The red
`lanczos_tests` cases are a different defect:
[the spectrum](#lanczos-returns-a-spectrum-about-three-times-too-wide).

## Defect 4: rocSPARSE conjugates a real transpose

Open, inferred (no AMD device here). `src/backends/rocsparse.cc:40-41` and `:73-74` pass
`enum_convert` straight through, and `src/linalg-impl.hh:274` maps `ConjTrans` to
`rocsparse_operation_conjugate_transpose` for every scalar type, which gave wrong results on real
scalars in cuSPARSE. **Fix needs:** a `rocsparse_op<T>` like `cusparse_op<T>`
(`src/backends/cusparse.cc`) on both operands, then `tests/spmm_tests.cc` on AMD hardware.

## Defect 5: netlib `trsm` reads `B` when `alpha == 0`

Fixed 2026-10-30 for NETLIB. The host solve computed `alpha * Bb.at(i, j, 0) - sum` with no
`alpha == 0` quick return, so unzeroed `BumpAllocator` memory gave `0 * NaN = NaN`; the batch loop in
`trsm_vendor` (`src/backends/netlib_lapack.cc`) now stores `B := 0` without reading `A` or `B`, as
BLAS does. Guard: `TrsmNetlib.AlphaZeroIgnoresNanB*` (every side, uplo, trans and diag, real and
complex, NaN `A` and `B`, padded `ld`).

Still open on the device routes, which read `B` at `alpha == 0`: the CTA kernel
(`src/ops/trsm/kernels/trsm_native.cc:264-269`, `dev_mul(alpha_d, B)`), the sub-group left kernel
(`src/ops/trsm/kernels/trsm_sg_left.cc:157`), the blocked route through its first diagonal block
(`trsm_native.cc:458-459` calls the CTA kernel with `alpha`; its trailing gemms take `beta = alpha`
and skip `C`, see Defect 11), and the CUDA complex substitute kernel (`src/backends/cublas.cc`, the
`alpha * Bb[j * ldb + i] - sum` arms of `trsm_vendor`). The real cuBLAS and rocBLAS calls follow
their libraries.

## Defect 11: native gemm reads C at beta zero

Fixed 2026-10-11. `LinearEpilogue` and `launch_direct` computed `alpha * acc + beta * prior` even at
`beta == 0`, so a dirty workspace gave NaN; `LinearEpilogue::store`
(`src/ops/gemm/kernels/epilogue_linear.hh`) now skips the load, and the geqrf/potrf zero fills that
hid it are deleted. Guard: `GemmCandidates.BetaZeroDoesNotReadC` (every pinned family, NaN `C`).

## Defect 12: vendor families accepted a heterogeneous batch

Fixed (WP3). Vendor arms pass the top-level extents to the library, so a heterogeneous batch
silently ran on the padded storage. Every family of `potrf`, `trsm`, `getrf`, `getrs`, `getri`,
`geqrf`, `orgqr`, `ormqr`, `syev`, `gesvd`, `gemv` and `spmm`'s vendor now refuses
`is_heterogeneous()` operands; `hemm`, `herk`, `her2k` throw `invalid_argument`. Only `gemm`
serves a heterogeneous batch (its per-item loop). Guard:
`BatchPolicy.HeterogeneousBatchNeverRunsAFamilyThatIgnoresIt` (`tests/select_tests.cc`).

## Defect 13: complex<double> cuBLAS calls segfault inside cuBLASLt

Fixed in 9f551ab0: cuBLASLt vector-loads `alpha`/`beta` at 16 bytes, so every cuBLAS call copies
them into `alignas(16)` locals (`src/backends/cublas.cc`). The gemv and spmm complex<double>
refusals are deleted (guard `GemvVendorRefusalTest.ComplexDoubleVendorRunsOnCuda`,
`tests/gemv_tests.cc`). **Open debt:** the now-redundant `m == 1 || n == 1` branch to
`cublasZgemmStridedBatched` (`cublas.cc:65-68`) can go, with its own route diff.

## Defect 14: the Hermitian drivers read the unreferenced triangle

Fixed 2026-10-26 in the drivers: `sytrd_cta`'s tile load reads only the named triangle (serving
`syev_cta` Upper, `syev_blocked` Lower at n <= 32 and `gesvd_cta` Upper), and `sytrd_sy2sb` uses
LAPACK's lower-only two-sided update. The remaining Upper-on-`blocked`/`two_stage` and
Lower-on-`syev_cta` paths go through an O(n^2) mirror that reads only the named triangle. Guards:
`SyevTest.AutoIgnoresUnreferencedTriangle` (+ heavy `AutoReadsOnlyTheNamedTriangleAtLargeN`),
`SyevCandidates.OtherTriangleIsNeverRead`,
`GesvdCandidates.HermitianFamiliesIgnoreTheUnreferencedTriangle`. Gap: `OtherTriangleIsNeverRead`
stops at n = 40, one short sy2sb panel. The double two-stage cost (1.34x at n = 480, untuned FP64
symm rows) was accepted.

## Defect 15: cuSOLVER `gesvdjBatched` faults on values-only non-square input

Fixed 2026-10-22: on CUDA `can_run(vendor)` is the binding's envelope (`max(m, n) <= 32`, packed A
and U, no thin factor, not values-only non-square); false on ROCm. Guard:
`GesvdCandidates.VendorPinRunsInsideItsEnvelope`.

## Defect 16: `ormqr` blocked throws at batch > 65535 {#defect-16-blocked-qr-past-the-grid}

Fixed. Blocked geqrf/ormqr past 65535 items returned silent wrong answers for items past the grid
(2x4090, sm_89). The blocked geqrf, orgqr, ormqr, gesvd and syev part loops cap every part at
`max_groups_yz / 2` (`select::grid_capped_chunk`, `src/select/grid_chunk.hh`; half because of
[defect 21](#defect-21-range-rounded-past-the-grid)); `can_run(Blocked)` has no batch term. Guards:
`*BlockedPastTheGridLimitIsBitIdentical` in `geqrf_tests` and `ormqr_blocked_tests`.

## Defect 17: cuSPARSE spmm: shapes refused in `can_run`, and the alignment waiver

Open (vendor behaviour). Complex `transB == ConjTrans` with one row of B returns an unchecked error
status and leaves `C` unwritten, so the CUDA vendor `can_run` refuses it (`src/ops/spmm/spmm.cc`).
cuSPARSE also mishandles operands off their natural alignment; alignment is a pointer property, not
a selection key, so `spmm_tests` skips misaligned cases unless the route is pinned `native`.

## Defect 18: `gesvd_cta` left vectors lose orthogonality as eps times kappa squared

Open. CTA mode always takes the normal-equations branch (`gesvd_direct_bidiag` is true only for
`Blocked`, `src/ops/gesvd/kernels/gesvd_blocked.cc:72-74`), and `build_bidiag_vectors` forms
\f$u_i = B v_i / \sigma_i\f$, scaling the error of \f$v_i\f$ by \f$\sigma_{\max}/\sigma_i\f$:
\f$\max|U^T U - I|\f$ reaches \f$\varepsilon\kappa^2\f$ (V stays at 1.5e-6 in float). Same root as
[gesvd defect A](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa). Reached under
Auto on sm_120 (14 double `vec=all` non-square rows), pin-only on sm_89. Red:
`GesvdTest.CtaProviderCoversAllJobCombinations`, float and double, in `tests/known-failures.txt`.
**Fix needs:** the direct bidiagonal solver in CTA mode, or no `cta` family for real general input.

## Defect 19: Upper potrf beyond the CTA ceiling has no vendor-free route

Fixed: the blocked driver has an Upper schedule (`potrf_blocked.cc`) and `can_run(blocked)` no longer
reads `uplo`; `lpanel` stays Lower-only. Tests: `PotrfBlockedTest.BlockedUpper*`. The Metal ledger
entry `PosvTest/5.PublicPosvSolvesOnBothTriangles` (`tests/known-failures-acpp-metal.txt`) needs a
re-run to confirm.

## Defect 20: `BATCHLAS_LAUNCH_BOUNDS` emits no `.maxntid`, so ptxas ignores the bound

Open, performance only. `intel::max_work_group_size(1, 1, N)` in the macro
(`src/sycl/kernel_attrs.hh:18-19`) lowers to no `.maxntid` on NVPTX with the `/opt/dpcpp-cuda`
DPC++, and `.minnctapersm` alone is dropped, so register counts are uncapped. Still affected:
`getrf_tiny.cc:90`, `gesv_tiny.cc:101`, `posv_tiny.cc:96`. Every gemm functor that uses the macro
also declares `reqd_work_group_size` (`.reqntid`;
[gemm.md](../perf/gemm.md#gemm-wp-q-the-launch-bound-was-ignored)). The intel/llvm build on the
2x4090 box behaves the other way round; the configure probe `BATCHLAS_NVPTX_REQD_WG_OK`
(`cmake/BatchLASDetectSYCL.cmake`) keeps `reqd_work_group_size` only where it lowers correctly.
**Fix needs:** the macro fixed or `reqd_work_group_size` per functor, then an A/B and register probe
per kernel (the tiny-solver tables were tuned unbounded) and re-pinned kernel hashes.

## Defect 21: a plain range launch is range-rounded past the grid {#defect-21-range-rounded-past-the-grid}

Open outside the capped part loops. A `parallel_for` over a plain `sycl::range<3>` with the batch in
dimension 0 is rounded up by DPC++ (`__pf_kernel_wrapper`), here to 65536, past CUDA's 65535 grid:
pinned blocked gesvd (float, n = 40) aborts from batch 65505. Sites: `src/ops/gesvd/ormbr/ormbr.cc:170`,
`src/kernels/common/larft_wy.hh:262`, `src/ops/orgqr/kernels/orgqr_blocked.cc:225,256`,
`src/ops/syev/kernels/uplo_mirror.cc:29`, `src/ops/syev/sytrd/band_reduction.cc` (nine launches),
`src/ops/syev/sytrd/sytrd_sy2sb.cc:126`. **Workaround:** the blocked geqrf, orgqr, ormqr, gesvd and
syev (cta, blocked) part loops keep every part at `max_groups_yz / 2`; test
`Gesvd*.BlockedPastTheGridIsBitIdentical`. syev two_stage (band_reduction, sy2sb) has no part loop.
**Fix needs:** `nd_range` with an explicit local size, or the batch out of dimension 0.

## Defect 22: acpp multipass drops an over-ceiling launch silently {#getrf-register-panel-unlaunched-on-acpp-multipass}

Worked around; the silent drop is open. The register-panel leaf's row cap comes from DPC++ ptxas
register counts, but acpp multipass cubins allocate more (110/150/144/222 vs 64/104/96/176 for
float, double, cfloat, cdouble), so a 512-thread double or cfloat panel exceeds the register ceiling
and the acpp CUDA queue drops the launch without an error; garbage pivots then fault getrs and the
lost context fails the rest of the process. **Workaround:** on acpp the cap is 256 rows
(`kPanelRegImplMaxRows`, `src/ops/getrf/kernels/getrf_panel_reg.cc:82`), and
`RegPanelCapacityIsOneSpelling` checks the pivots at the advertised cap. **Fix needs:** a launch error
from acpp's CUDA backend or a per-kernel register query; any other compile-time register gate is
exposed the same way.

## Fixed, cited by number

| # | defect | fix |
|---|---|---|
| 2 | `cond` and `norm` called `syev_vendor_or_throw` directly and threw in a vendor-free build | both call the public `syev` and `syev_buffer_size` |
| 23 | `gebrd_cta`, `ormqx_cta` and `syev_jacobi_cta` returned per partition before work-group barriers on a batch tail | fixed 2026-10-28: dead partitions run a zero tile with every store guarded by `live`; `*BatchTailStoresNothingAndKeepsItemsBitwise` |

## LOBPCG picked a spurious zero Ritz value from a null P column

Fixed 2026-10-09; kept because the symptom points at the wrong component. `syevx` Smallest returned
0 for some eigenvalues when the projected solve ran `jacobi` or `vendor`, which suggested a vendor
`ld > n` bug; the real cause was null P columns (exact-zero coefficients) adding a 0 eigenvalue to
\f$S^T A S\f$. The deflation kernel (`src/ops/syevx/kernels/syevx_lobpcg.cc`) now puts the sentinel on
every column at or after `block_vectors` with \f$\lVert s_c\rVert^2 \le (n\varepsilon)^2\f$. Guards:
`Pinned/SyevxProjectedSyevRouteTest.SmallestDenseMatchesReferenceSyev/*` (`tests/syevx_tests.cc`).

## Known defects: the CTA SYTRD Lower path

Worked around, not reproduced. `src/ops/syev/kernels/syev_cta.cc:131-133` says the CTA SYTRD/SYEV
pipeline is wrong for `Uplo::Lower`, so Lower input is mirrored into Upper first; the fused kernel
(`syev_cta_fused.cc`) also always runs Upper. No test or failure is recorded. To settle it, pin the
Lower reduction on graded complex input with a poisoned upper triangle and compare against Upper;
then fix the kernel and drop the mirror, or record the failure here.

## linalg::qr returns a wrong QR after an earlier call in the process

Observed; cause not located; the wrapper is withheld (`include/batchlas/blas/linalg-ops.hh:836`).
Composing `geqrf`, `triangular_mask_into` and `orgqr` returned \f$QR \ne A\f$ after an earlier
`linalg::qr` call in the same process, and passed alone. No repro is checked in: compose the three
calls twice in one process. Look first at state that outlives a call: the `Queue`'s arena
(@ref design_runtime_internals), a static, or a stale event.

## lanczos returns a spectrum about three times too wide

Open, cause not located. Toeplitz tridiag(-5, 2, -5), n = 64, float, CSR returns eigenvalues over
about [-28, 32] against the exact [-8, 12], at batch 1 too. Red in `tests/known-failures.txt`:
`LanczosTestBase.LanczosTest`, `LanczosTestBase.ToeplitzEigenpairs`, `LanczosStride.ClosedFormSpectrum`.
**Lead (unconfirmed):** the SVQB2 re-orthogonalization (the `ortho` call in `lanczos_iteration`,
`src/ops/lanczos/lanczos.cc`) rotates \f$[v_{it-1}, v_{it}]\f$ after their alphas and betas were
recorded, breaking the three-term recurrence; raising steqr's `max_sweeps` to 50 changed nothing.
**To fix:** compare the recorded alphas/betas with \f$V^T A V\f$ for one item.

## Unverified candidates

Located to a line, not confirmed by a test.

- **`gesvdj_cta` global rescale ignores columns 32..63 on the C=64 rung**
  (`src/ops/gesvd/kernels/gesvdj_cta.cc:355`): overflow headroom is narrower than designed; see
  [global power-of-two scaling](gesvd.md#gesvdj_cta-global-power-of-two-scaling).
- **Recursive `stedc` under `NoEigenVectors`** may merge from unset leaf vectors
  (`src/ops/syev/stedc/stedc.cc:537`); see
  [stedc](../perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors).
- **`compute_optimal_wg_size` assumes a 32-bit `long`:** `31 - __builtin_clzl`
  (`include/batchlas/util/kernel-heuristics.hh:104,109`) is a negative shift on LP64. Fix with
  `std::bit_floor`.
- **`gesv` and `posv` throw `internal_error` for a caller error** (heterogeneous or bad-shape batch,
  `src/ops/gesv/gesv.cc:89,92`, `src/ops/posv/posv.cc:96,99`); see
  [error model](error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy).
- Matrix containers, the environment parser and the 3-D tile-group race are tracked on their pages:
  [matrix model](matrix-model.md#matrix-model-open-debts),
  [environment](environment.md#environment-parser-defects-recorded-not-fixed),
  [device-group BLAS](device-group-blas.md#device-group-blas-the-3-d-launch-generic-fallback).

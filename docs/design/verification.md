# Verification library {#design_verification}

> **Status:** current · checked 2026-10-21 · tolerances calibrated 2026-10-09 on threadripper02
> (RTX PRO 6000, sm_120)

BatchLAS checks its own results in three places: the tests, the benchmark harnesses
(`benchmarks/factor_bench.cc`, the `*_acc` and accuracy harnesses) and the tuner (all 19
`tools/tune/*_spec.cc`), which refuses to time a wrong answer. All three use one facility: residual
checks as `MatrixView` statics, and `batchlas::verify` for the tolerance policy and the LAPACKE
references. In a test the entry point is `EXPECT_VERIFY` (`tests/test_utils.hh`).

## Verification: the copies it replaced

Survey of 2026-10-09 (branch `tuning-deep-sm120`), before the library:

| Concept | Copies | Differences between copies |
| --- | --- | --- |
| promotion and complex helpers (`up`, `ab`, `cj`, `mk`, `Prom`) | ~30 | some promote to double, some keep working precision |
| NaN-propagating maximum | 2 | tests used `std::max`, which drops NaN: `syev_cta_tests.cc:69`, `syev_blocked_tests.cc:128,142`, `getrf_tests.cc:323,1689` passed a NaN residual |
| seeded generator | 8, plus ~25 `std::mt19937` sites | |
| SPD and permuted-LU inputs | 2 each, plus per-test variants | identical code, different buffer types |
| factorization residuals (potrf, getrf) | 4 and 4+ | pivot layout, Upper/Lower support |
| solve residual | 7 | only one handled getri and large-n probe columns |
| BLAS componentwise backward error | 8 inline | tests used an absolute `k·eps·bound` instead |
| orthogonality | ~12 | Frobenius in the tuner, elementwise max in the tests |
| eigen residual | 6 | two denominators |
| LAPACKE references (`?syevd`, `?gesvd`, `?sterf`, `?getrf`) | ~10 | |
| tolerances | 3 schemes, 20+ constants | fixed per type vs `c·n·eps` with c from 4 to 200 |

`tools/tune/residuals.hh` (a port of `benchmarks/factor_bench.cc`) and the fixed per-type
`test_utils::tolerance` are deleted. "Is LAPACKE available" is one switch,
`BATCHLAS_VERIFY_HAVE_LAPACKE`, set by `batchlas::verify`.

## Verification: decisions

| Question | Decision |
| --- | --- |
| Where the checks live | static functions of `MatrixView<T, MatrixFormat::Dense>` (`include/batchlas/blas/matrix.hh`, implementation `src/util/matrix_checks.cc`); every utility a check needs is a `Matrix`/`MatrixView` member or static, or an existing library function. `batchlas::verify` keeps only the tolerance policy and the LAPACKE references |
| Data | the library's own `MatrixView` and `VectorView`; no other matrix, batch, operand or item type. A site holding raw buffers builds a view over them |
| Memory | host-readable views (unified or host allocations); the caller waits on its queue first. A check promotes its items with `MatrixView::astype<promoted_type>(items)` |
| Queue | the caller's: every check that launches kernels takes `Queue& ctx` first and waits before it returns (at least once per item group); on an out-of-order Queue it runs on an in-order sibling that first waits on ctx's pending work (`select::on_in_order_queue`), which costs one Queue construction per call, so pass item spans rather than loop per item. `product` takes `const Queue&` and returns its Event without waiting; `values_error`, `pivot_ratio` and `pivots_valid` are host loops and take no Queue. On a device without fp64 the check runs on a CPU queue ([no-fp64 devices](#verification-no-fp64-device)) |
| Arithmetic | every check promotes to `double` / `std::complex<double>` |
| Maxima | NaN-propagating everywhere: a NaN anywhere in a check fails it |
| Items checked | `default_items()` (`{0, batch/2, batch−1}`) by default; an explicit list otherwise, `all_items()` for every item. Check a batch with one call over its items, not one call per item: each call waits on the queue |
| Empty item set | a batch-0 operand with defaulted items checks nothing: every check returns 0 (passes), `form_q` returns a batch-0 `Matrix`, `pivots_valid` returns true. One policy for `MatrixView` and `VectorView` operands (`VectorView::default_items()` is `{}` for batch 0) |
| Operand structure | operands are read as stored. A check against a triangle or a mirrored matrix applies `triangularize`, `hermitize` or `symmetrize` to a copy first; `trsm_residual` (`Uplo`, `Diag`), `potrf_residual` (`Uplo`) and `eigen_residual` (lower triangle) take the structure their factor or input has |
| Tolerance | `c · f(n) · eps(T)` per check kind, one named entry per kind (below); no fixed per-type bounds |
| Inputs | `MatrixView` fills (device kernels, deterministic in the seed): `fill_random`, `fill_random_spd`, `fill_random_pivoted` (dominant, rows a random n-cycle per item), `fill_graded_hermitian` (condition \f$10^c\f$); verify has no generators |
| Independence | promotion (`astype`), transforms, `fill_identity`, `fill_random`, `linalg::axpby_into`, `linalg::triangular_mask_into` and `norm` (Frobenius) are library utilities; no routed operation (gemm, potrf, ...) is called. Products (`MatrixView::product`, op() by index map), the componentwise backward error, reflector application and row interchanges are `matrix_checks.cc`'s own SYCL kernels, so no check runs through the op it judges |

## Verification: layout

| Where | Contents |
| --- | --- |
| `MatrixView` statics | `product`; `potrf_residual`, `getrf_residual` (packed 1-based int32 pivots), `qr_residual`, `form_q` (Q's first columns from geqrf storage, a `Matrix<promoted_type>`; one item, or a list of items packed in order), `qr_reconstruction` (`‖A − Q·triu(R)‖_F / ‖A‖_F`), `similarity_residual` (`‖A − Q·B·Qᴴ‖_F / ‖A‖_F`, judged as `factorization`), `svd_residual` (`‖A − U·diag(s)·Vh‖_F / ‖A‖_F` over the leading min(m, n) singular triplets, judged as `factorization` at n = max(m, n)), `solve_residual` (an empty B for getri; 8 seeded probe columns above \f$2^{28}\f$ multiply-adds per item), `lu_solve_residual` (`op(P·L·U)` formed per item), `gemm_backward_error` / `gemv_backward_error` / `spmm_backward_error` (componentwise, `|r| / (|α||A||B| + |β||C|)`, C's triangle optional), `gemm_max_denominator` (over an item list, like the others), `rank2k_backward_error` (her2k and syr2k over one triangle), `trsm_residual`, `orthogonality`, `eigen_residual`, `values_error`, `pivots_valid`, `pivot_ratio` (`max cabs1(L(i,k)U(k,k)) / cabs1(U(k,k))`, the partial-pivoting property no residual sees); scalar helpers `promoted_type`, `up`, `make`, `conj`, `cabs1`, `finite` |
| `MatrixView` / `VectorView` members | `all_items()`, `default_items()` (both views); `MatrixView::astype<U>(items)` (the listed items, promoted and packed) |
| `include/batchlas/verify/tolerance.hh` | `enum class Check` (seven kinds, below), `bound<T>(Check, n[, Slack])`, `pass<T>(Check, n, value[, Slack])` (recorded), `within<T>` (the same comparison, never recorded), `Slack`, `pivot_ratio_bound<T>()`, `eps<T>()`, `nanmax` |
| `include/batchlas/verify/reference.hh` | LAPACKE: eigenvalues (`?syevd`/`?heevd`) and singular values (`?gesdd`) of `double` or `std::complex<double>` data only (any other type does not compile), `?getrf` pivots in the data's own precision (`s/d/c/zgetrf`: float data never reaches the host dgetrf, see @ref perf_lu), `?geqrf` tau, `?sterf`; each on a packed vector or on `(view, item)`. Each returns `false` without LAPACKE, empty input included, and the caller skips, so a check without a reference never passes. With LAPACKE (`have_reference`) `false` is a failed reference and the caller fails. A double reference other than `?sterf` throws `batchlas::unsupported` when the host dgemm probe (shared with the NETLIB backend, `src/backends/host_blas_health.hh`) fails (OpenBLAS 0.3.20 on Cooperlake without `OPENBLAS_CORETYPE`): its blocked code runs on that dgemm |

CMake target `batchlas::verify` (`cmake/BatchLASVerify.cmake`): a small static host library
(`src/verify/reference.cc`, the dgemm probe) carrying LAPACKE and `BATCHLAS_VERIFY_HAVE_LAPACKE`; not
installed. `tests/test_utils.hh` adds the one gtest wrapper, `EXPECT_VERIFY(T, kind, n, value)` and
`EXPECT_VERIFY_SLACK(T, kind, n, value, Slack{factor, "reason"})`; a failure prints value, bound,
kind, n and the slack.

Every check returns the worst value over the checked items as a `double`; the caller compares it with
`bound`. Items are processed in groups that keep each promoted operand under \f$2^{25}\f$ elements.

## Checks on a device without fp64 {#verification-no-fp64-device}

The checks promote to `double`, and a Metal build has no double kernels for the GPU
(`BATCHLAS_HAS_DEVICE_FP64=0`). `check_queue()` in `src/util/matrix_checks.cc` therefore waits on the
caller's queue and returns a leaked, per-thread CPU `Queue` (the AdaptiveCpp OpenMP device) when
`ctx.device().supports_fp64()` is false. Every promoted allocation, kernel and Frobenius norm of the
check runs there. The operands stay where they are: library matrices are USM shared, which the host
device reads in place. A `malloc_device` operand is not host-readable and must be copied first. The
checks are not demoted to `float`, which would make them as inexact as the code they judge. The fp64
policy this answers: [AdaptiveCpp Metal](sycl-implementations.md#sycl-impl-metal-fp64-policy).

## Verification: tolerance kinds

`bound<T>(kind, n) = c · f(n) · eps(T)`, with `eps(T)` the unit roundoff of T's real type:

| `Check` | Quantity | f(n) | c |
| --- | --- | --- | --- |
| `factorization` | `‖A − LLᴴ‖_F / ‖A‖_F`, `‖PA − LU‖_F / ‖A‖_F`, `‖A − QR‖_F / ‖A‖_F`, `‖A − UΣVᴴ‖_F / ‖A‖_F` | n (`svd_residual`: max(m, n)) | 16 |
| `solve` | `‖AX − B‖_F / (‖A‖_F ‖X‖_F)`; `trsm_residual`'s normwise `‖op(A)X − αB‖_F / (‖A‖_F‖X‖_F + |α|‖B‖_F)` | n | 16 |
| `blas` | componentwise backward error (gemm, gemv, spmm, rank-k, trmm; the tuner's componentwise trsm check) | k + 2 (k the inner dimension; trmm and trsm: the order of A) | 8 |
| `orthogonality` | `‖QᴴQ − I‖_F`, Q built from Householder reflectors or by a direct orthogonalization of given columns (geqrf, orgqr, `form_q`; every `ortho` algorithm) | n = rows of Q | 32 |
| `orthogonality_rotations` | `‖QᴴQ − I‖_F`, eigen- or singular vectors from an iterative solver (Jacobi, steqr/stedc-based syev, syevx, stein, bdsqr, bdsdc, gesvd's U and V) | n = columns of Q (vectors checked) | 256 |
| `eigen_residual` | `‖AV − VΛ‖_F / ‖A‖_F` | n | 32 |
| `values` | `max |λ − λ_ref| / ‖A‖_2` (eigenvalues and singular values) | n | 32 |

Each c is the smallest power of two at least 4× the largest `value / (f(n)·eps)` recorded over the
tests' passing cases (all dtypes, the ladder of sizes they use); see
[calibration](#verification-calibration). Only rows recorded with factor 1 enter a kind's maximum;
slacked rows (factor ≠ 1) are listed separately, so a slacked site cannot inflate its kind's c. A
calibrated c more than 4× away from the specified one is reported, not adopted.

- **Two orthogonality kinds.** A product of Householder reflectors is orthogonal to a few ulps per
  column; vectors accumulated over many rotation sweeps (or rebuilt by divide and conquer) lose
  orthogonality with the sweep count. One c for both is either 8× too loose for QR or red for the
  Jacobi solvers (measured below).
- **Their n.** `orthogonality` takes the rows of Q: Householder error grows with the reflector length.
  `orthogonality_rotations` takes the vectors checked, the columns of Q (for a row check, the columns
  of Qᴴ), as it was calibrated; counting rows would loosen thin and partial sets (stein's kb of n,
  gesvd's thin U) by up to 32×.
- **`blas` grows as k + 2**, not k: the α and β scalings and the complex multiply add roundings that a
  k = 1 cell hits (cdouble rows reached 0.88 of `4·k·eps` at k = 1).

**Slack.** A site whose quantity legitimately sits off its kind's bound passes `Slack{factor, reason}`:
`bound<T>(kind, n, slack) = factor · bound<T>(kind, n)`. The factor may be above or below 1; it must
be finite and positive, and the reason is mandatory (`batchlas::invalid_argument` otherwise). With
`BATCHLAS_VERIFY_RECORD=<path>` (a raw `getenv` on every check, not the settings snapshot) every `pass`
appends `kind dtype n value bound factor reason`: `value` raw (never divided by the factor), `bound`
the kind's unslacked bound, `factor` 1 and `reason` `-` for the plain overloads, whitespace in a reason
written as `_`. Every factor above 1 goes to the user for a decision. `within<T>` makes the same
comparison without recording, for predicates evaluated on results expected to be wrong.

**Pivot ratio.** `pivot_ratio` is not a `c · f(n) · eps` quantity: cabs1 partial pivoting keeps it at
most 1 in exact arithmetic, so `pivot_ratio_bound<T>() = 1 + 64 eps(T)` (32 machine epsilons). The
slack covers the factor's own rounding: `L(i,k) = A′(i,k) / U(k,k)` is computed in T (LAPACK and most
kernels form the reciprocal of the pivot, then scale), so `L(i,k)·U(k,k)` recovers `A′(i,k)` only to
a few ulps of T. The product itself is formed in double and adds nothing.

## Verification: calibration

Recorded 2026-10-09 with `BATCHLAS_VERIFY_RECORD` over the test binaries on threadripper02 (RTX PRO
6000, sm_120), one file per area: LU (`getrf`, `getrs`, `gesv`, `getri` and their candidate tests;
119,879 rows), QR and Cholesky (`geqrf`, `orgqr`, `ormqr`, `potrf`, `posv` and candidates; 18,125),
eigen and SVD (`syev*`, `sytrd*`, `gesvd*`, `gesvdj_cta`, `bdsdc`, `bdsqr`; 70,736), BLAS
(`device_blas`, `gemm`, `gemv`, `spmm`, `syrk`, `syr2k`, `symm`, `trmm`, `trsm` and candidates;
30,074). "Need" is the largest `value / (f(n)·eps)` over the factor-1 rows of the kind; "rule" is the
smallest power of two ≥ 4 × the larger need of the two classes.

| Kind | Need, float/cfloat | Need, double/cdouble | Rule | c | Change |
| --- | --- | --- | --- | --- | --- |
| `factorization` | 3.15 (cfloat, n = 1) | 3.75 (cdouble, n = 1) | 16 | 16 | none |
| `solve` | 2.69 (cfloat, n = 1) | 3.04 (cdouble, n = 1) | 16 | 16 | none |
| `blas` | 1.11 (float, k = 2) | 1.28 (double, k = 2) | 8 | 8 | 4 → 8 |
| `orthogonality` | 4.65 (cfloat, n = 1) | 6.00 (cdouble, n = 1) | 32 | 32 | 16 → 32 |
| `orthogonality_rotations` | 59.4 (float, n = 32, `BdsdcTest.Graded`) | 23.3 (double, n = 64) | 256 | 256 | new kind (was 16) |
| `eigen_residual` | 5.09 (float, n = 31) | 4.14 (cdouble, n = 32) | 32 | 32 | none |
| `values` | 3.87 (float, n = 12) | 4.66 (cdouble, n = 32) | 32 | 32 | none |

The `orthogonality` maximum comes from 1 × 1 Q (a single reflector, a few ulps of `|q|² − 1`); above
n = 1 its need is 2.8 (cdouble), which alone would round to 16. The eigen and SVD file's
orthogonality rows are re-labelled `orthogonality_rotations` (every orthogonality check in those
binaries is that kind). At a single kind with c = 16, 26 cases (Jacobi syev, `gesvdj_cta`, the Jacobi
and CTA families of `syev_candidates`, `BdsdcTest.Graded`,
`GesvdGeneralComplexTest.GeneralComplexAboveThirtyTwo`) were red, with needs from 16.3 to 59.4.

**Re-check with the per-kind n convention** (`ortho`, `trmm`, `syevx`, the syev CTA and Jacobi
eigenvalues, `syev`'s and `gesvd`'s known spectra on the library): 213,235 rows over 22 test binaries
(20 of them record), 2026-10-09, same machine. Every need stays at or below c/4, so no c changes:

| Kind | Need, float/cfloat | Need, double/cdouble |
| --- | --- | --- |
| `factorization` | 1.44 (cfloat, n = 1) | 2.24 (cdouble, n = 1) |
| `solve` | 1.02 (float, n = 1) | 0.82 (cdouble, n = 1) |
| `blas` | 0.72 (cfloat, k = 1) | 1.17 (cdouble, k = 1) |
| `orthogonality` | 5.95 (cfloat, n = 12, `ortho`) | 4.65 (cdouble, n = 10, `ortho`) |
| `orthogonality_rotations` | 42.5 (float, n = 63, `gesvdj_cta`) | 23.3 (double, n = 64, `gesvdj_cta`) |
| `eigen_residual` | 5.09 (float, n = 31) | 4.14 (cdouble, n = 32) |
| `values` | 4.62 (float, n = 31, Jacobi) | 4.66 (cdouble, n = 32) |

The only `orthogonality` site that counts rows rather than columns is `orgqr_tests` (4 × 4, factor 1).
No bound moved by more than 4× through the n convention.

**Accepted slack.** Every site whose factor is not 1, with the user's decision of 2026-10-09:

| Site | Kind | Factor | Bound | Reason |
| --- | --- | --- | --- | --- |
| `tests/bdsdc_tests.cc` `GradedWithVectorsHighCondition` | `orthogonality_rotations` | 8 float, 0.5 double | 2048 n eps, 128 n eps | graded κ ~ 1e6 divide and conquer at n = 64: the repair's Gram-Schmidt accumulates error; measured need 1246 float (4.75e-3), 71.5 double |
| `tests/gesvd_tests.cc` `gesvd_cta_slack` (real `cta` pins) | `orthogonality_rotations` | 0.25 float, 0.5 double | 64 n eps, 128 n eps | the cta provider always takes the normal-equations bidiagonal path, which squares the condition number; measured need 52 float, 113 double |
| `tests/gesvd_tests.cc` `expect_orthonormal_columns` under `BATCHLAS_GESVD_BIDIAG=normal`, float | `orthogonality_rotations` | 0.5 | 128 n eps | normal equations square the condition number; measured on one site only (`ThinTallUnderNormalEquationsBidiag`, n = 48): need 7.18 (2.05e-5); a whole-binary sweep under `BATCHLAS_GESVD_BIDIAG=normal` is not measured |
| `tests/geqrf_tests.cc` `orth_slack`, `tests/orgqr_tests.cc` `orth_slack` | `orthogonality` | ≤ 1 | the file's old `0.5 (m+k) eps / √k`, at most the kind's bound | tightening: keeps each file's pre-library tolerance |
| `tests/geqrf_tests.cc` `residual_slack`, `tests/geqrf_candidates_tests.cc` `tol`, `tests/orgqr_tests.cc` `recon_slack` | `factorization` | ≤ 1 | the file's old `0.5 (m+n) eps` with its tiny-order floor | tightening (docs/perf/qr.md) |
| `tests/potrf_tests.cc` `kLeafSlack` | `factorization` | 0.5 | 8 n eps | tightening: the leaf tolerance 4 n eps of docs/perf/potrf.md |
| `tests/device_blas_tests.cc` `tighten` | `blas` | ≤ 1 | the old absolute tolerance | tightening: exact small-integer data |

The bdsdc and gesvd factors were accepted as 128×/8× and 4×/8× of c = 16 under the single kind; on
`orthogonality_rotations` they keep each accepted absolute bound (2048, 128, 64, 128 n eps), not 16×
looser.

**Tightening slacks** (factor ≤ 1, the older stricter bound as the reason; `git grep 'Slack{'` lists
them, 26 sites in 10 test files on 2026-10-21): gemm and gemv float against the old vendor-comparison
tolerance (`vendor_comparison_tolerance`, `vendor_comparison_slack`), the sytrd eig(T)-vs-eig(A) float
bound of 1e-6 relative, device_blas's exact data, the sb2st float spectra (1/16 per chase step, at
most 1; double stays at factor 1), trsm's vendor-substitute `‖AX − B‖ / ‖B‖ < 1e-5`, the float
spectrum bounds of lanczos, ritz_values, steqr, syev_blocked, syevx and sytrd_blocked, and
ormqr_blocked's float reconstruction (`recon_slack`, 1/64). Where the old check was entrywise and the
library's is normwise, the slack keeps the old power only on average: ormqr_blocked's 1/64 catches a
single `1e-5` error in a Q entry in column j only where ‖R(j,:)‖ is at least 0.76 of the average row
norm (j ≤ 45 of 64 on random input; a deliberate one-entry break was caught in every float case); a
column with a small row of R, near the end, is not caught. ormqr's identity applications are judged by
`qr_reconstruction` against the input and `orthogonality`, not componentwise (Q's near-zero entries
make an elementwise relative error meaningless).

**Tuner specs.** `svd_residual` keeps the `factorization` constant 16: its measured need is 3.73
float / 1.29 double over gesvd_tests and 5.17 / 5.68 over the tuner's native arms, but 25.3 for
cuSOLVER's 2x2 Hermitian double cells, which the tuner therefore records `bad` (the rule would give
128, an 8x loosening that needs the maintainer's acceptance; not adopted). The `cta` arm is `bad` on
square vectors cells too: U's orthogonality goes like eps·κ² on the normal-equations path
([gesvd defect A](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa)), up to 2.3e4
n eps; no slack.

**Kept outside the library.** Local references that are deliberately not `batchlas::verify`:

| Site | Why it stays |
| --- | --- |
| `tests/device_blas_tests.cc` `reference_rankk` (herk with complex α, 2 sites) | the device herk applies a complex α and then zeroes the imaginary part of the diagonal; no library check expresses that |
| `tests/geqrf_tests.cc` `host_geqr2` (`ConventionMatchesReferenceLapackWithoutAVendor`) | an independent host xGEQR2 is the oracle for LAPACK's reflector sign convention, which every residual (and `form_q`, which shares the convention) passes by construction |
| `tests/syev_blocked_tests.cc` `syev<Backend::NETLIB>` reference | a second implementation of the op under test, compared result to result |
| `tests/sytrd_cta_tests.cc` `ref_sytd2_upper` (host unblocked sytd2) | an independent oracle for d, e, tau and the stored reflectors, a convention no residual sees; its deviations are judged at `factorization` (the similarity itself goes through `similarity_residual`) |
| `tests/sytrd_sb2st_hh_tests.cc` similarity check | measures with `verify::orthogonality` but keeps its own tolerance |
| `tests/syev_jacobi_cta_tests.cc` `reference_jacobi_eigenvalues` | an independent cyclic two-sided Jacobi in double on exact float input, with the file's own 2n eps ‖A‖_F bound: it isolates the float kernel's rounding from a LAPACKE reference's |
| `tests/steqr_tests.cc` `GradedTridiagonalRelativeAccuracy`, `TridiagSolverGradedAndSaturatingBatch` | a long-double Sturm-bisection oracle and per-eigenvalue relative median/max thresholds; no relative-accuracy kind exists |
| `tests/sytrd_blocked_tests.cc` `GridMatchesLegacyTridiagonal` `elem_tol` | elementwise d and \|e\| agreement (1e-11 n relative) of two LATRD implementations: intermediate outputs, no kind fits |
| `tests/sytrd_blocked_tests.cc` `TrailingUpdateRoutesAgree` relation | `syr2k ≤ 4 · gemm + 16 u` compares two measured errors (each route is also checked at its kind) |
| `tests/syevx_tests.cc` LOBPCG instrumentation tests | staged vs host-read history (rates, residual norms, Ritz values): non-eigen quantities from two paths of the code under test |
| `tests/cond_tests.cc` log10 and `ztol` checks | the random log-cond tolerances against the generator's target, the log10 tolerances and the band-structure zero checks are properties of the inputs and the estimator, not a residual |
| `tests/norm_tests.cc` `expect_norm_close` | a four-line scalar relative error (judged at `blas`): the library has no scalar-vs-scalar error |
| `tests/linalg_layer_tests.cc` `NormMatchesTheHostFrobeniusNorm`, `CondMatchesTheDiagonalReference` | one scalar per item against a closed form on exact small-integer data: no scalar-vs-scalar kind |
| `tests/device_blas_tests.cc` `expect_vector_near` / `expect_matrix_near` and the gemv/dot/conj spot checks | hand-computed results on small-integer data, exact in float: any error is a bug, and a componentwise kind would loosen them |
| `tests/iluk_tests.cc` MILU diagonal value, generator Hermitian symmetry, host-vs-device factor entries, the 2D-Laplacian smallest eigenvalue | one exactly computable factor entry; a property of the input generator; two factorization implementations compared entry by entry (intermediate output); an iterative LOBPCG run judged at its own convergence tolerance, not a residual |
| `tests/geqrf_tests.cc` tau against `host_geqr2` and `verify::geqrf_tau`, the native-vs-vendor drop-in and Register-vs-Resident panel comparisons | LAPACK's reflector convention (a residual passes any sign) and two implementations of the op compared result to result, normwise |
| `tests/getrf_tests.cc` tiny-vs-cta elementwise factor agreement | two kernels compared element by element, the pivots exactly: the identity pad must not change the selection or the arithmetic |
| `tests/gesvd_tests.cc` `DefaultProviderKeepsSingularValuesAtHighCondition` `sv_rel_tol` | per-singular-value relative error on graded input (kappa 1e5) against the closed-form spectrum: the quantity the normal-equations path destroys; a normwise `values_error` scaled by sigma_max cannot see the small values, and no relative-accuracy kind exists |
| `tests/device_blas_tests.cc` Auto-vs-Generic and group-vs-item policy comparisons (gemv, ger, syr2k `EXPECT_NEAR` at 1e-4/1e-5) | two execution policies of the same op compared result to result: each policy's arithmetic must agree elementwise, which a backward error against the inputs would loosen |

## Verification: acceptance

- **Inputs are library fills.** Test inputs come from `MatrixView::fill_random`, `fill_random_spd`,
  `fill_random_pivoted` and `fill_graded_hermitian`. Inputs of sweeps archived before 2026-10-11 do
  not reproduce.
- **Every check proven.** Each residual has a test with a correct input (passes) and one deliberate
  corruption per axis it reads: a perturbed element, a wrong `ld`, a wrong batch `stride`, the last
  item (not item 0) corrupted, a NaN. Each must fail.
- **Results unchanged for migrated sites.** A migrated tuner spec gives the same verdict on every cell
  of a preview run of its op; factor_bench's `residual` column moves by no more than rounding; a
  migrated test keeps its case list, and the deliberate-break rule (@ref dev_agent_guide §8) is re-run
  for it.
- **One switch.** No `HAVE_LAPACKE` macro other than `BATCHLAS_VERIFY_HAVE_LAPACKE`, and a vendor-free,
  LAPACKE-free configure still builds with the references returning `false`.
- **No new copies.** Grep finds one definition of each helper: a `MatrixView`/`VectorView` member or
  static, or `include/batchlas/verify/` (`nanmax`, the tolerances, the references).

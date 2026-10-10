# Verification library {#design_verification}

> **Status:** implemented 2026-10-09 (tests, benchmarks and the tuner verify through it; tolerances
> calibrated on threadripper02, RTX PRO 6000 sm_120) · scope: one host-side verification facility ·
> machine-independent

BatchLAS checks its own results in three places: the tests, the benchmark harnesses and the
tuner, which refuses to time a wrong answer. Each grew its own input generators, references,
residuals and tolerances. This page specifies one facility, `batchlas::verify`, that all three use.

## Why one facility

A survey of the tree on 2026-10-09 (branch `tuning-deep-sm120`) found:

| Concept | Copies | Differences between copies |
| --- | --- | --- |
| promotion and complex helpers (`up`, `ab`, `cj`, `mk`, `Prom`) | ~30 | some promote to double, some keep working precision |
| NaN-propagating maximum | 2 | tests use `std::max`, which drops NaN: `syev_cta_tests.cc:69`, `syev_blocked_tests.cc:128,142`, `getrf_tests.cc:323,1689` pass a NaN residual |
| seeded generator | 8, plus ~25 `std::mt19937` sites | |
| SPD and permuted-LU inputs | 2 each, plus per-test variants | identical code, different buffer types |
| factorization residuals (potrf, getrf) | 4 and 4+ | pivot layout, Upper/Lower support |
| solve residual | 7 | only one handles getri and large-n probe columns |
| BLAS componentwise backward error | 8 inline | tests use an absolute `k·eps·bound` instead |
| orthogonality | ~12 | Frobenius in the tuner, elementwise max in the tests |
| eigen residual | 6 | two denominators |
| LAPACKE references (`?syevd`, `?gesvd`, `?sterf`, `?getrf`) | ~10 | |
| tolerances | 3 schemes, 20+ constants | fixed per type vs `c·n·eps` with c from 4 to 200 |

`tools/tune/residuals.hh` says it is a port of `benchmarks/factor_bench.cc`. "Is LAPACKE
available" is decided under three macro names (`BATCHLAS_TUNE_HAVE_LAPACKE`,
`BATCHLAS_GEQRF_TESTS_HAVE_LAPACKE`, `BATCHLAS_GETRF_TESTS_HAVE_LAPACKE`).

## Verification: decisions

| Question | Decision |
| --- | --- |
| Data | the library's own `MatrixView<T, MatrixFormat::Dense>` and `VectorView<T>`; no new matrix or batch type. A site holding raw buffers builds a view over them |
| Memory | host-readable views (unified or host allocations); the caller waits on its queue first |
| Arithmetic | every check promotes to `double` / `std::complex<double>` |
| Maxima | NaN-propagating everywhere: a NaN anywhere in a check fails it |
| Items checked | `{0, batch/2, batch−1}` by default; an overload takes an explicit item list, `all_items(batch)` lists every item (saturating-batch tests) |
| Tolerance | `c · f(n) · eps(T)` per check kind, one named entry per kind (below); no fixed per-type bounds |
| Device-side checks | out of scope: miniacc's accuracy benchmarks keep `gemm`/`norm` on the device and only take their LAPACKE references from here |
| Public API | no: internal, not installed |

## Layout

Header-only, `include/batchlas/verify/`, namespace `batchlas::verify`, CMake INTERFACE target
`batchlas::verify` defined at the top level (tests are under `BATCHLAS_BUILD_TESTS`, the tuner
and benchmarks under `BATCHLAS_BUILD_BENCHMARKS`, so neither directory can own it).

| Header | Contents |
| --- | --- |
| `scalar.hh` | `promoted_t<T>`, `up`, `conj` (in T for every dtype), `make<T>`, `abs`, `cabs1` (`|re| + |im|`, LAPACK's pivot metric), `finite` (every component), `eps<T>()`, `nanmax`, `Rng` (factor_bench's historical LCG, constants unchanged: `next()` in [−1, 1), `uniform01()`, `normal()`) |
| `inputs.hh` | `fill_spd`, `fill_lu` (dominant, then row-permuted per item), `fill_random`, `fill_graded_hermitian` (condition ~1e±6), `fill_reflectors` (larfg on random columns, LAPACK storage) |
| `norms.hh` | `view(data, rows, cols, ld, stride = 0, batch = 1)` (a `KernelMatrixView` over raw host memory, `const T*` too; stride 0 is `ld·cols`); Frobenius, max-abs and one-norm of one item, promoted |
| `residuals.hh` | `potrf_residual`, `getrf_residual` (packed 1-based int32 pivots), `qr_residual`, `form_q` (Q's first columns from geqrf storage, the same reflector loop as `qr_residual`), `qr_reconstruction` (`‖A − Q·triu(R)‖_F / ‖A‖_F`, Q explicit), `similarity_residual` (`‖A − Q·B·Qᴴ‖_F / ‖A‖_F`, B a tridiagonal or banded reduction stored densely; judged as `factorization`), `solve_residual` (null B for getri; probe columns above 2^28 elements; an overload reads A through a `Shape` and a `Transpose`), `lu_solve_residual` (the solve quantity for X from getrf factors and pivots: `op(P·L·U)` formed per checked item), `gemm_backward_error` / `gemv_backward_error` / `spmm_backward_error` (componentwise, `|r| / (|α||A||B| + |β||C|)`), `gemm_max_denominator` (the largest of those denominators, to turn an absolute tolerance into a componentwise one), `rank2k_backward_error` (her2k and syr2k over one triangle), `trsm_residual`, `orthogonality`, `eigen_residual`, `values_error`, `pivots_valid`, `pivot_ratio` (`max cabs1(L(i,k)U(k,k)) / cabs1(U(k,k))`, the partial-pivoting property no residual sees) |
| `reference.hh` | LAPACKE: eigenvalues (`?syevd`/`?heevd`), singular values (`?gesdd`), `?getrf` pivots in the data's own precision (`s/d/c/zgetrf`: float data never reaches the host dgetrf, see @ref perf_lu), `?geqrf` factor and tau, `?sterf`. Each returns `false` without LAPACKE, empty input included, and the caller reports NaN or skips, so a check without a reference never passes |
| `tolerance.hh` | `enum class Check` (seven kinds, below), `bound<T>(Check, n[, Slack])`, `pass<T>(Check, n, value[, Slack])` (recorded), `within<T>` (the same comparison, never recorded), `Slack`, `pivot_ratio_bound<T>()` |

`tests/test_utils.hh` adds the one gtest wrapper, `EXPECT_VERIFY(T, kind, n, value)` and
`EXPECT_VERIFY_SLACK(T, kind, n, value, Slack{factor, "reason"})`; a failure prints value, bound,
kind, n and the slack. The library itself stays gtest-free.

Every residual returns the worst value over the checked items as a `double`; the caller compares
it with `bound`. Nothing in the library calls a BatchLAS operation, so a reference can never go
through the tables or routes under test (agent guide §8 rule 7).

## Tolerance kinds

`bound<T>(kind, n) = c · f(n) · eps(T)`, with `eps(T)` the unit roundoff of T's real type:

| `Check` | Quantity | f(n) | c |
| --- | --- | --- | --- |
| `factorization` | `‖A − LLᴴ‖_F / ‖A‖_F`, `‖PA − LU‖_F / ‖A‖_F`, `‖A − QR‖_F / ‖A‖_F` | n | 16 |
| `solve` | `‖AX − B‖_F / (‖A‖_F ‖X‖_F)`; `trsm_residual`'s normwise `‖op(A)X − αB‖_F / (‖A‖_F‖X‖_F + |α|‖B‖_F)` | n | 16 |
| `blas` | componentwise backward error (gemm, gemv, spmm, rank-k, trmm; the tuner's componentwise trsm check) | k + 2 (k the inner dimension; trmm and trsm: the order of A) | 8 |
| `orthogonality` | `‖QᴴQ − I‖_F`, Q built from Householder reflectors or by a direct orthogonalization of given columns (geqrf, orgqr, `form_q`; every `ortho` algorithm) | n = rows of Q | 32 |
| `orthogonality_rotations` | `‖QᴴQ − I‖_F`, eigen- or singular vectors from an iterative solver (Jacobi, steqr/stedc-based syev, syevx, stein, bdsqr, bdsdc, gesvd's U and V) | n = columns of Q (vectors checked) | 256 |
| `eigen_residual` | `‖AV − VΛ‖_F / ‖A‖_F` | n | 32 |
| `values` | `max |λ − λ_ref| / ‖A‖_2` (eigenvalues and singular values) | n | 32 |

Each c is the smallest power of two at least 4× the largest `value / (f(n)·eps)` recorded over
the migrated tests' passing cases (all dtypes, the ladder of sizes they use); see
[calibration](#verification-calibration). Only rows recorded with factor 1 enter a kind's maximum;
slacked rows (factor ≠ 1, below) are listed separately, so a slacked site cannot inflate its
kind's c. A calibrated c more than 4× away from the specified one is reported, not adopted.

Orthogonality is two kinds because the two sources differ by an order of magnitude: a product of
Householder reflectors is orthogonal to a few ulps per column, while vectors accumulated over many
sweeps of rotations (or rebuilt by divide and conquer) lose orthogonality with the sweep count.
One c for both is either 8× too loose for QR or red for the Jacobi solvers (measured below).

The two orthogonality kinds count n as each was calibrated. `orthogonality` takes the number of rows
of Q: Householder error grows with the reflector length (the QR sites and factor_bench always passed
m). `orthogonality_rotations` takes the number of vectors checked, the columns of Q (for a row check,
the columns of Qᴴ): its c was calibrated that way, and counting rows would loosen thin and partial
sets of vectors (stein's kb of n, gesvd's thin U) by up to 32×.

The `blas` growth is k + 2, not k: the α and β scalings and the complex multiply add roundings
that a k = 1 cell hits (cdouble rows reached 0.88 of `4·k·eps` at k = 1).

**Slack.** A site whose quantity legitimately sits off its kind's bound passes
`Slack{factor, reason}`: `bound<T>(kind, n, slack) = factor · bound<T>(kind, n)`. The factor may be
above or below 1 (a site may also tighten); it must be finite and positive, and the reason is
mandatory (`std::invalid_argument` otherwise). With `BATCHLAS_VERIFY_RECORD=<path>` every `pass`
appends `kind dtype n value bound factor reason`: `value` is the raw value (never divided by the
factor), `bound` the kind's unslacked bound, `factor` 1 and `reason` `-` for the plain overloads,
and whitespace in a reason is written as `_`. The calibration excludes every row with factor ≠ 1
from its kind's maximum and lists it separately (site reason, factor, raw `value / (f(n)·eps)`);
every factor above 1 goes to the user for a decision. `within<T>` makes the same comparison
without recording, for predicates evaluated on results that are expected to be wrong.

**Pivot ratio.** `pivot_ratio` is not a `c · f(n) · eps` quantity: cabs1 partial pivoting keeps it
at most 1 in exact arithmetic, so `pivot_ratio_bound<T>() = 1 + 64 eps(T)` (32 machine epsilons).
The slack covers the factor's own rounding: `L(i,k) = A′(i,k) / U(k,k)` is computed in T (LAPACK
and most kernels form the reciprocal of the pivot, then scale), so `L(i,k)·U(k,k)` recovers
`A′(i,k)` only to a few ulps of T. The product itself is formed in double and adds nothing.

## Verification: calibration

Recorded 2026-10-09 with `BATCHLAS_VERIFY_RECORD` over the migrated test binaries on threadripper02
(RTX PRO 6000, sm_120), one file per migration stage: LU (`getrf`, `getrs`, `gesv`, `getri` and
their candidate tests; 119,879 rows), QR and Cholesky (`geqrf`, `orgqr`, `ormqr`, `potrf`, `posv`
and candidates; 18,125), eigen and SVD (`syev*`, `sytrd*`, `gesvd*`, `gesvdj_cta`, `bdsdc`,
`bdsqr`; 70,736), BLAS (`device_blas`, `gemm`, `gemv`, `spmm`, `syrk`, `syr2k`, `symm`, `trmm`,
`trsm` and candidates; 30,074). "Need" is the largest `value / (f(n)·eps)` over the factor-1 rows of
the kind; "rule" is the smallest power of two ≥ 4 × the larger need of the two classes.

| Kind | Need, float/cfloat | Need, double/cdouble | Rule | c | Change |
| --- | --- | --- | --- | --- | --- |
| `factorization` | 3.15 (cfloat, n = 1) | 3.75 (cdouble, n = 1) | 16 | 16 | none |
| `solve` | 2.69 (cfloat, n = 1) | 3.04 (cdouble, n = 1) | 16 | 16 | none |
| `blas` | 1.11 (float, k = 2) | 1.28 (double, k = 2) | 8 | 8 | 4 → 8 |
| `orthogonality` | 4.65 (cfloat, n = 1) | 6.00 (cdouble, n = 1) | 32 | 32 | 16 → 32 |
| `orthogonality_rotations` | 59.4 (float, n = 32, `BdsdcTest.Graded`) | 23.3 (double, n = 64) | 256 | 256 | new kind (was 16) |
| `eigen_residual` | 5.09 (float, n = 31) | 4.14 (cdouble, n = 32) | 32 | 32 | none |
| `values` | 3.87 (float, n = 12) | 4.66 (cdouble, n = 32) | 32 | 32 | none |

The `orthogonality` maximum comes from 1 × 1 Q (a single reflector, a few ulps of
`|q|² − 1`); above n = 1 its need is 2.8 (cdouble), which alone would round to 16. The
orthogonality rows of the eigen and SVD file were recorded under the single kind before the split
and are re-labelled `orthogonality_rotations` (every orthogonality check in those binaries is now
that kind). Before the split, 26 cases (Jacobi syev, `gesvdj_cta`, the Jacobi and CTA families of
`syev_candidates`, `BdsdcTest.Graded`, `GesvdGeneralComplexTest.GeneralComplexAboveThirtyTwo`) were
red at c = 16 with needs from 16.3 to 59.4.

**Re-check after the final pass** (the per-kind n convention above; `ortho`, `trmm`, `syevx`, the
syev CTA and Jacobi eigenvalues, `syev`'s and `gesvd`'s known spectra newly on the library): 213,235
rows over 22 changed test binaries (20 of them record), 2026-10-09, same machine. Every kind's need
stays at or below c/4, so no c changes:

| Kind | Need, float/cfloat | Need, double/cdouble |
| --- | --- | --- |
| `factorization` | 1.44 (cfloat, n = 1) | 2.24 (cdouble, n = 1) |
| `solve` | 1.02 (float, n = 1) | 0.82 (cdouble, n = 1) |
| `blas` | 0.72 (cfloat, k = 1) | 1.17 (cdouble, k = 1) |
| `orthogonality` | 5.95 (cfloat, n = 12, `ortho`) | 4.65 (cdouble, n = 10, `ortho`) |
| `orthogonality_rotations` | 42.5 (float, n = 63, `gesvdj_cta`) | 23.3 (double, n = 64, `gesvdj_cta`) |
| `eigen_residual` | 5.09 (float, n = 31) | 4.14 (cdouble, n = 32) |
| `values` | 4.62 (float, n = 31, Jacobi) | 4.66 (cdouble, n = 32) |

The only `orthogonality` site that changed from columns to rows is `orgqr_tests` (4 × 4, factor 1);
`ortho_tests` is new on the library. No bound moved by more than 4× through the n convention.

**Accepted slack.** Every site whose factor is not 1, with the user's decision of 2026-10-09:

| Site | Kind | Factor | Bound | Reason |
| --- | --- | --- | --- | --- |
| `tests/bdsdc_tests.cc` `GradedWithVectorsHighCondition` | `orthogonality_rotations` | 8 float, 0.5 double | 2048 n eps, 128 n eps | graded κ ~ 1e6 divide and conquer at n = 64: the repair's Gram-Schmidt accumulates error; measured need 1246 float (4.75e-3), 71.5 double |
| `tests/gesvd_tests.cc` `gesvd_cta_slack` (real `cta` pins) | `orthogonality_rotations` | 0.25 float, 0.5 double | 64 n eps, 128 n eps | the cta provider always takes the normal-equations bidiagonal path, which squares the condition number; measured need 52 float, 113 double |
| `tests/gesvd_tests.cc` `expect_orthonormal_columns` under `BATCHLAS_GESVD_BIDIAG=normal`, float | `orthogonality_rotations` | 0.5 | 128 n eps | normal equations square the condition number; measured on one site only (`ThinTallUnderNormalEquationsBidiag`, n = 48): need 7.18 (2.05e-5); a whole-binary sweep under `BATCHLAS_GESVD_BIDIAG=normal` is not measured |
| `tests/geqrf_tests.cc` `orth_slack`, `tests/orgqr_tests.cc` `orth_slack` | `orthogonality` | ≤ 1 | the file's old `0.5 (m+k) eps / √k`, at most the kind's bound | tightening: keeps each file's pre-migration tolerance |
| `tests/geqrf_tests.cc` `residual_slack`, `tests/geqrf_candidates_tests.cc` `tol`, `tests/orgqr_tests.cc` `recon_slack` | `factorization` | ≤ 1 | the file's old `0.5 (m+n) eps` with its tiny-order floor | tightening (docs/perf/qr.md) |
| `tests/potrf_tests.cc` `kLeafSlack` | `factorization` | 0.5 | 8 n eps | tightening: the leaf tolerance 4 n eps of docs/perf/potrf.md |
| `tests/device_blas_tests.cc` `tighten` | `blas` | ≤ 1 | the old absolute tolerance | tightening: exact small-integer data |

The tail migration (the last tests on the library) kept every older bound that was stricter than the
kind's as a `Slack` with factor ≤ 1 and the old bound as its reason, so no test lost power: gemm and
gemv float against the old vendor-comparison tolerance (`old_vendor_tolerance`,
`old_relative_tolerance`), the sytrd eig(T)-vs-eig(A) float bound of 1e-6 relative, device_blas's
exact data, the sb2st float spectra (1/16 per chase step, at most 1; double stays at factor 1), trsm's vendor-substitute `‖AX − B‖ / ‖B‖ < 1e-5`, and the
float spectrum bounds of lanczos, ritz_values, steqr, syev_blocked, syevx and sytrd_blocked.
`git grep 'Slack{'` lists them with their reasons.

The bdsdc and gesvd factors were accepted as 128×/8× and 4×/8× of c = 16 before the split; they
are re-expressed on `orthogonality_rotations` so that each absolute bound is the one accepted
(2048, 128, 64, 128 n eps), not 16× looser.

**Kept outside the library.** Local references that are deliberately not `batchlas::verify`:

| Site | Why it stays |
| --- | --- |
| `tests/device_blas_tests.cc` `reference_rankk` (herk with complex α, 2 sites) | the device herk applies a complex α and then zeroes the imaginary part of the diagonal; no library check expresses that |
| `tests/geqrf_tests.cc` `host_geqr2` (`ConventionMatchesReferenceLapackWithoutAVendor`) | an independent host xGEQR2 is the oracle for LAPACK's reflector sign convention, which every residual (and `form_q`, which shares the convention) passes by construction |
| `tests/syev_blocked_tests.cc` `syev<Backend::NETLIB>` reference | a second implementation of the op under test, compared result to result |
| `tests/gesvd_tests.cc` `expect_reconstruction` (with its per-type `gesvd_recon_tol`), `tests/gesvdj_cta_tests.cc`, `tests/bdsdc_tests.cc` and `tests/bdsqr_tests.cc` reconstruction | `‖A − UΣVᴴ‖_F / ‖A‖_F` has no library residual: `qr_reconstruction` reads R as upper triangular and ΣVᴴ is not, and a componentwise `gemm_backward_error` of U·(ΣVᴴ) against A judges the SVD's normwise backward error elementwise. The library covers U and V orthogonality and the values |
| `tests/sytrd_cta_tests.cc` `ref_sytd2_upper` (host unblocked sytd2) | an independent oracle for d, e, tau and the stored reflectors, a convention no residual sees; its deviations are judged at `factorization` (the similarity itself goes through `similarity_residual`) |
| `tests/sytrd_sb2st_hh_tests.cc` similarity check | measures with `verify::orthogonality` but keeps its own tolerance |
| `tests/syev_jacobi_cta_tests.cc` `reference_jacobi_eigenvalues` | an independent cyclic two-sided Jacobi in double on exact float input, with the file's own 2n eps ‖A‖_F bound: it isolates the float kernel's rounding from a LAPACKE reference's |
| `tests/steqr_tests.cc` `GradedTridiagonalRelativeAccuracy` | a long-double Sturm-bisection oracle and per-eigenvalue relative median/max thresholds; no relative-accuracy kind exists |
| `tests/sytrd_blocked_tests.cc` `GridMatchesLegacyTridiagonal` `elem_tol` | elementwise d and \|e\| agreement (1e-11 n relative) of two LATRD implementations: intermediate outputs, no kind fits |
| `tests/sytrd_blocked_tests.cc` `TrailingUpdateRoutesAgree` relation | `syr2k ≤ 4 · gemm + 16 u` compares two measured errors (each route is also checked at its kind) |
| `tests/syevx_tests.cc` LOBPCG instrumentation tests | staged vs host-read history (rates, residual norms, Ritz values): non-eigen quantities from two paths of the code under test |
| `tests/cond_tests.cc` log10 and `ztol` checks | the random log-cond tolerances against the generator's target, the log10 tolerances and the band-structure zero checks are properties of the inputs and the estimator, not a residual |
| `tests/norm_tests.cc` `expect_norm_close`, `tests/potrf_tests.cc` `host_real` / `host_imag` | a four-line scalar relative error (judged at `blas`) and scalar part accessors: the library has no scalar-vs-scalar error or part accessor |

## Rollout

One PR to main (`verify-library`). The library and its first users (the tuner's potrf, posv, trsm and
gemm specs, `benchmarks/factor_bench.cc`; `tools/tune/residuals.hh` and factor_bench's inline copies
deleted; the three LAPACKE macros became `BATCHLAS_VERIFY_HAVE_LAPACKE`) landed first. The tests were
then migrated per area (LU; QR and Cholesky; eigen, SVD and tridiagonal; BLAS) on task branches merged
into it, and the library was extended mid-way where a migration needed it (`view`, `form_q`,
`qr_reconstruction`, the `Shape` and `Transpose` solve overload, `rank2k_backward_error`, `geqrf_tau`,
`Slack`; later `lu_solve_residual`, `pivot_ratio`, working-precision `getrf_pivots` and the split
orthogonality kinds). A final pass moved the remaining local checks onto it
(syevx, ortho, trmm, the syev CTA and Jacobi eigenvalue references, gesvd's singular-value bounds,
syev's closed-form spectra, the candidates tests' scalar helpers, getrf's LAPACKE pivots). A tail pass moved the last users of the fixed per-type `test_utils::tolerance` (deleted) onto it, ormqr's identity applications among them: `op(Q)` judged by `qr_reconstruction` against the input and `orthogonality`, not componentwise (Q's near-zero entries make an elementwise relative error meaningless).

The tiered-tuning PR (19 specs) follows it and writes its specs against this library, so
`lu_residuals.hh`, `qr_common.hh` and `host_reference.hh` from the `tuning-deep-sm120` branch
never land.

## Verification: acceptance

- **Inputs unchanged.** `fill_spd` and `fill_lu` produce byte-identical buffers to
  `factor_bench.cc` and `tools/tune/residuals.hh` for every dtype at a fixed seed (golden hashes
  in the library's tests), so archived sweeps and ledger records stay comparable.
- **Every check proven.** Each residual has a test with a correct input (passes) and one
  deliberate corruption per axis it reads: a perturbed element, a wrong `ld`, a wrong batch
  `stride`, the last item (not item 0) corrupted, a NaN. Each must fail.
- **Results unchanged for migrated sites.** A migrated tuner spec gives the same verdict on every
  cell of a preview run of its op; factor_bench's `residual` column moves by no more than
  rounding; a migrated test keeps its case list, and the repo's deliberate-break rule (agent
  guide §8) is re-run for it.
- **One switch.** No `HAVE_LAPACKE` macro other than `BATCHLAS_VERIFY_HAVE_LAPACKE` remains, and
  a vendor-free, LAPACKE-free configure still builds with the references reporting NaN.
- **No new copies.** After each step, grep finds no definition of the migrated helpers outside
  `include/batchlas/verify/`.

# Verification library {#design_verification}

> **Status:** current · checked 2026-10-29 · tolerances calibrated 2026-10-09 on threadripper02
> (RTX PRO 6000, sm_120)

Tests, benchmark harnesses (`benchmarks/factor_bench.cc`, the `*_acc` harnesses) and the tuner
(`tools/tune/*_spec.cc`, which refuses to time a wrong answer) check results with one facility:
residual checks as `MatrixView` statics, and `batchlas::verify` for the tolerance policy and the
LAPACKE references. In a test the entry point is `EXPECT_VERIFY` (`tests/test_utils.hh`).

## Verification: decisions

| Question | Decision |
| --- | --- |
| Where checks live | statics of `MatrixView<T, MatrixFormat::Dense>` (`include/batchlas/blas/matrix.hh`, `src/util/matrix_checks.cc`); `batchlas::verify` keeps only tolerances and LAPACKE references |
| Data | the library's `MatrixView` / `VectorView`; raw buffers get a view |
| Memory | host-readable views (USM shared or host); the caller waits on its queue first |
| Queue | checks that launch kernels take `Queue& ctx` first and wait before returning; an out-of-order queue runs on an in-order sibling (`select::on_in_order_queue`, one Queue per call), so pass item spans, not one call per item. `product` returns its Event unwaited; `values_error`, `pivot_ratio`, `pivots_valid` are host loops. No fp64: [CPU queue](#verification-no-fp64-device) |
| Arithmetic | promoted to `double` / `std::complex<double>` (`astype<promoted_type>(items)`) |
| Maxima | NaN-propagating: a NaN anywhere fails the check |
| Items | `default_items()` = `{0, batch/2, batch−1}`, an explicit list, or `all_items()`; batch 0 checks nothing and passes. Items are grouped to keep each promoted operand under \f$2^{25}\f$ elements |
| Structure | operands are read as stored; apply `triangularize` / `hermitize` / `symmetrize` to a copy first. `trsm_residual`, `potrf_residual`, `eigen_residual` take their own `Uplo` |
| Inputs | `fill_random`, `fill_random_spd`, `fill_random_pivoted`, `fill_graded_hermitian` (condition \f$10^c\f$), device-side and seeded |
| Independence | no routed operation (gemm, potrf, ...) is called; products, componentwise backward error, reflector application and row interchanges are `matrix_checks.cc`'s own kernels |

## Verification: layout

| Where | Contents |
| --- | --- |
| `MatrixView` statics | `product`; residuals `potrf_residual`, `getrf_residual` (1-based int32 pivots), `qr_residual`, `qr_reconstruction`, `similarity_residual`, `svd_residual` (leading \f$\min(m, n)\f$ triplets, \f$n = \max(m, n)\f$), `solve_residual` (8 seeded probe columns above \f$2^{28}\f$ multiply-adds per item), `lu_solve_residual`, `trsm_residual`, `eigen_residual`, `generalized_eigen_residual` (itype 1-3); componentwise `gemm_/gemv_/spmm_/rank2k_backward_error`, `gemm_max_denominator`; `orthogonality`, `form_q`; `values_error`, `pivots_valid`, `pivot_ratio`; scalar helpers `up`, `make`, `conj`, `cabs1`, `finite` |
| `include/batchlas/verify/tolerance.hh` | `enum class Check`, `bound<T>`, `pass<T>` (recorded), `within<T>` (not recorded), `Slack`, `pivot_ratio_bound<T>()`, `eps<T>()`, `nanmax` |
| `include/batchlas/verify/reference.hh` | LAPACKE eigenvalues and singular values (double data only), `?getrf` pivots in the data's precision, `?geqrf` tau, `?sterf`. Without LAPACKE each returns `false` and the caller skips; with it `false` fails. A double reference other than `?sterf` throws `batchlas::unsupported` when the host dgemm probe (`src/backends/host_blas_health.hh`) fails |

CMake target `batchlas::verify` (`cmake/BatchLASVerify.cmake`, not installed) carries LAPACKE and
the one switch `BATCHLAS_VERIFY_HAVE_LAPACKE`. `EXPECT_VERIFY(T, kind, n, value)` and
`EXPECT_VERIFY_SLACK(T, kind, n, value, Slack{factor, "reason"})` print value, bound, kind, n and
slack on failure. Every check returns the worst value over its items as a `double`.

## Checks on a device without fp64 {#verification-no-fp64-device}

The checks need double kernels, which a Metal build lacks (`BATCHLAS_HAS_DEVICE_FP64=0`). When
`ctx.device().supports_fp64()` is false, `check_queue()` (`src/util/matrix_checks.cc`) waits on the
caller's queue and returns a leaked per-thread CPU `Queue`, where every promoted allocation, kernel
and norm runs. Operands stay put (USM shared is host-readable); a `malloc_device` operand must be
copied first. Demoting the checks to `float` would make them as inexact as the code they judge.
Policy: [AdaptiveCpp Metal](sycl-implementations.md#sycl-impl-metal-fp64-policy).

## Verification: tolerance kinds

`bound<T>(kind, n)` \f$= c \cdot f(n) \cdot \varepsilon(T)\f$, where \f$\varepsilon\f$ (`eps`) is the unit roundoff of T's real type:

| `Check` | Quantity | \f$f(n)\f$ | \f$c\f$ |
| --- | --- | --- | --- |
| `factorization` | \f$\lVert A - LL^{H}\rVert\f$, \f$\lVert PA - LU\rVert\f$, \f$\lVert A - QR\rVert\f$, \f$\lVert A - U\Sigma V^{H}\rVert\f$, over \f$\lVert A\rVert_F\f$ | \f$n\f$ (svd: \f$\max(m, n)\f$) | 16 |
| `solve` | \f$\lVert AX - B\rVert_F / (\lVert A\rVert_F \lVert X\rVert_F)\f$; trsm's normwise residual | \f$n\f$ | 16 |
| `blas` | componentwise backward error (gemm, gemv, spmm, rank-k, trmm, the tuner's trsm) | \f$k + 2\f$ | 8 |
| `orthogonality` | \f$\lVert Q^{H}Q - I\rVert_F\f$, Householder or direct orthogonalization (geqrf, orgqr, `form_q`, `ortho`) | rows of \f$Q\f$ | 32 |
| `orthogonality_rotations` | \f$\lVert Q^{H}Q - I\rVert_F\f$, iterative eigen/singular vectors (Jacobi, steqr/stedc syev, syevx, stein, bdsqr, bdsdc, gesvd) | columns checked | 256 |
| `eigen_residual` | \f$\lVert AV - V\Lambda\rVert_F / \lVert A\rVert_F\f$ | \f$n\f$ | 32 |
| `values` | \f$\max \lvert \lambda - \lambda_{\mathrm{ref}} \rvert / \lVert A\rVert_2\f$ | \f$n\f$ | 32 |

- **Two orthogonality kinds:** Householder products are orthogonal to a few ulps per column; vectors
  from many rotation sweeps lose orthogonality with the sweep count. One \f$c\f$ is 8x too loose for QR or
  red for Jacobi. Counting rows for `orthogonality_rotations` would loosen thin sets up to 32x.
- **`blas` grows as \f$k + 2\f$:** the \f$\alpha\f$/\f$\beta\f$ scalings and complex multiply add roundings a \f$k = 1\f$ cell hits.
- **Slack:** `Slack{factor, reason}` scales a site's bound either way; the factor must be finite and
  positive and the reason non-empty (`batchlas::invalid_argument`). Every factor above 1 needs the
  maintainer's decision. `BATCHLAS_VERIFY_RECORD=<path>`
  ([read raw](environment.md#environment-verify-record-is-read-raw)) makes every `pass` append
  `kind dtype n value bound factor reason` (raw value, unslacked bound).
- **Pivot ratio:** \f$\max_i \operatorname{cabs1}(L_{ik}U_{kk}) / \operatorname{cabs1}(U_{kk})\f$, where \f$\operatorname{cabs1}(z) = \lvert \operatorname{Re} z\rvert + \lvert \operatorname{Im} z\rvert\f$, is at most 1 under partial pivoting, so
  `pivot_ratio_bound<T>()` \f$= 1 + 64\,\varepsilon(T)\f$, covering \f$L\f$'s rounding in T.

## Verification: calibration

Each \f$c\f$ is the smallest power of two at least 4x the largest \f$\mathrm{value} / (f(n)\,\varepsilon)\f$ ("need") over the
tests' passing factor-1 rows; slacked rows cannot inflate a kind. A \f$c\f$ more than 4x from the
specified one is reported, not adopted. Recorded 2026-10-09 with `BATCHLAS_VERIFY_RECORD` over the
LU, QR/Cholesky, eigen/SVD and BLAS test binaries on threadripper02 (RTX PRO 6000, sm_120), about
239k rows:

| Kind | Need, float/cfloat | Need, double/cdouble | \f$c\f$ |
| --- | --- | --- | --- |
| `factorization` | 3.15 (cfloat, \f$n = 1\f$) | 3.75 (cdouble, \f$n = 1\f$) | 16 |
| `solve` | 2.69 (cfloat, \f$n = 1\f$) | 3.04 (cdouble, \f$n = 1\f$) | 16 |
| `blas` | 1.11 (float, \f$k = 2\f$) | 1.28 (double, \f$k = 2\f$) | 8 |
| `orthogonality` | 4.65 (cfloat, \f$n = 1\f$) | 6.00 (cdouble, \f$n = 1\f$) | 32 |
| `orthogonality_rotations` | 59.4 (float, \f$n = 32\f$, `BdsdcTest.Graded`) | 23.3 (double, \f$n = 64\f$) | 256 |
| `eigen_residual` | 5.09 (float, \f$n = 31\f$) | 4.14 (cdouble, \f$n = 32\f$) | 32 |
| `values` | 3.87 (float, \f$n = 12\f$) | 4.66 (cdouble, \f$n = 32\f$) | 32 |

A same-day re-check under the per-kind \f$n\f$ convention (213,235 rows, 22 binaries) kept every need at or
below \f$c/4\f$. With a single orthogonality kind at \f$c = 16\f$, 26 Jacobi, `gesvdj_cta`, bdsdc and complex
gesvd cases were red (needs 16.3-59.4), hence `orthogonality_rotations`.

**Factors above 1** (`git grep 'Slack'` in `tests/` lists every site; the rest are tightenings,
\f$\mathrm{factor} < 1\f$, keeping a file's older stricter bound):

| Site | Kind | Factor | Reason |
| --- | --- | --- | --- |
| `tests/bdsdc_tests.cc` graded orthogonality | `orthogonality_rotations` | 8 float, 0.5 double | graded \f$\kappa \sim 10^6\f$ D&C at \f$n = 64\f$: the repair's Gram-Schmidt; need 1246 float, 71.5 double |
| `tests/bdsdc_tests.cc` graded `svd_residual`, float | `factorization` | 2 | same input; need 24 |
| `tests/syfn_tests.cc` `inv_sqrt` (2 sites) | `factorization` | 100 | forward error of \f$A^{-1/2}\f$ scales with \f$\operatorname{cond}(A) = 100\f$ |

The real `cta` gesvd pins (`gesvd_cta_slack`, 0.25 float / 0.5 double) and `BATCHLAS_GESVD_BIDIAG=normal`
(0.5 float, one site measured) are below 1 on `orthogonality_rotations` but square \f$\kappa\f$. In the tuner,
`svd_residual` keeps \f$c = 16\f$ (native arms need at most 5.68); cuSOLVER's \f$2 \times 2\f$ Hermitian double cells
(need 25.3) and `cta` square vector cells (\f$\varepsilon\kappa^2\f$,
[defect A](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa)) are recorded `bad`.

**Kept outside the library**, deliberately: independent oracles for a convention no residual sees
(`host_geqr2`, `ref_sytd2_upper`, `reference_jacobi_eigenvalues`, the long-double Sturm oracle and
relative-accuracy thresholds in `steqr_tests`, `sv_rel_tol` in `gesvd_tests`); two implementations or
policies of one op compared result to result (NETLIB syev, LATRD grid vs legacy, `TrailingUpdateRoutesAgree`,
tiny-vs-cta getrf, device_blas Auto-vs-Generic, geqrf native-vs-vendor); exact small-integer data
where any error is a bug (device_blas, `linalg_layer_tests`, `norm_tests`); herk's zeroed diagonal
imaginary part (`reference_rankk`); and properties of generators or estimators (`cond_tests`,
`fill_graded_hermitian`'s spectrum ratio, `iluk_tests`, syevx instrumentation).

## Verification: acceptance

- **Inputs are library fills** (`fill_random`, `fill_random_spd`, `fill_random_pivoted`,
  `fill_graded_hermitian`). Sweeps archived before 2026-10-11 do not reproduce.
- **Every check proven:** each residual has a passing correct input and one deliberate corruption
  per axis it reads (an element, `ld`, batch `stride`, the last item, a NaN), each of which fails
  (`tests/verify_residuals_tests.cc`).
- **One switch:** no `HAVE_LAPACKE` macro other than `BATCHLAS_VERIFY_HAVE_LAPACKE`; a vendor-free,
  LAPACKE-free configure builds with references returning `false`.
- **No new copies:** one definition of each helper, a `MatrixView`/`VectorView` member or static or
  `include/batchlas/verify/`.

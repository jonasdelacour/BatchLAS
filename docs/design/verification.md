# Verification library {#design_verification}

> **Status:** specified 2026-10-09, not implemented · scope: tests, benchmarks and the tuner on one
> host-side verification facility · machine-independent

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
| Items checked | `{0, batch/2, batch−1}` by default; an overload takes an explicit item list, `all_items(view)` checks every item (saturating-batch tests) |
| Tolerance | `c · f(n) · eps(T)` per check kind, one named entry per kind (below); no fixed per-type bounds |
| Device-side checks | out of scope: miniacc's accuracy benchmarks keep `gemm`/`norm` on the device and only take their LAPACKE references from here |
| Public API | no: internal, not installed |

## Layout

Header-only, `include/batchlas/verify/`, namespace `batchlas::verify`, CMake INTERFACE target
`batchlas::verify` defined at the top level (tests are under `BATCHLAS_BUILD_TESTS`, the tuner
and benchmarks under `BATCHLAS_BUILD_BENCHMARKS`, so neither directory can own it).

| Header | Contents |
| --- | --- |
| `scalar.hh` | `promoted_t<T>`, `up`, `conj`, `abs`, `cabs1` (`|re| + |im|`, LAPACK's pivot metric), `finite` (every component), `eps<T>()`, `nanmax`, `Rng` (the LCG of `factor_bench.cc`, `uniform()`, `normal()`) |
| `inputs.hh` | `fill_spd`, `fill_lu` (dominant, then row-permuted per item), `fill_random`, `fill_graded_hermitian` (condition ~1e±6), `fill_reflectors` (larfg on random columns, LAPACK storage) |
| `norms.hh` | `view(data, rows, cols, ld, stride = 0, batch = 1)` (a `KernelMatrixView` over raw host memory, `const T*` too; stride 0 is `ld·cols`); Frobenius, max-abs and one-norm of one item, promoted |
| `residuals.hh` | `potrf_residual`, `getrf_residual` (packed 1-based int32 pivots), `qr_residual`, `form_q` (Q's first columns from geqrf storage, the same reflector loop as `qr_residual`), `qr_reconstruction` (`‖A − Q·triu(R)‖_F / ‖A‖_F`, Q explicit), `solve_residual` (null B for getri; probe columns above 2^28 elements; an overload reads A through a `Shape` and a `Transpose`), `lu_solve_residual` (the solve quantity for X from getrf factors and pivots: `op(P·L·U)` formed per checked item), `gemm_backward_error` / `gemv_backward_error` / `spmm_backward_error` (componentwise, `|r| / (|α||A||B| + |β||C|)`), `rank2k_backward_error` (her2k and syr2k over one triangle), `trsm_residual`, `orthogonality`, `eigen_residual`, `values_error`, `pivots_valid`, `pivot_ratio` (`max cabs1(L(i,k)U(k,k)) / cabs1(U(k,k))`, the partial-pivoting property no residual sees) |
| `reference.hh` | LAPACKE: eigenvalues (`?syevd`/`?heevd`), singular values (`?gesdd`), `?getrf` pivots in the data's own precision (`s/d/c/zgetrf`: float data never reaches the host dgetrf, see @ref perf_lu), `?geqrf` factor and tau, `?sterf`. Each returns `false` without LAPACKE, and the caller reports NaN, so a check without a reference fails rather than passes |
| `tolerance.hh` | `enum class Check`, `bound<T>(Check, n[, Slack])`, `pass<T>(Check, n, value[, Slack])` (recorded), `within<T>` (the same comparison, never recorded), `Slack`, `pivot_ratio_bound<T>()` |

`tests/test_utils.hh` adds the one gtest wrapper, `EXPECT_VERIFY(T, kind, n, value)` and
`EXPECT_VERIFY_SLACK(T, kind, n, value, Slack{factor, "reason"})`; a failure prints value, bound,
kind, n and the slack. The library itself stays gtest-free.

Every residual returns the worst value over the checked items as a `double`; the caller compares
it with `bound`. Nothing in the library calls a BatchLAS operation, so a reference can never go
through the tables or routes under test (agent guide §8 rule 7).

## Tolerance kinds

`bound<T>(kind, n) = c · f(n) · eps(T)`, with `eps(T)` the unit roundoff of T's real type:

| `Check` | Quantity | f(n) | c (initial) |
| --- | --- | --- | --- |
| `factorization` | `‖A − LLᴴ‖_F / ‖A‖_F`, `‖PA − LU‖_F / ‖A‖_F`, `‖A − QR‖_F / ‖A‖_F` | n | 16 |
| `solve` | `‖AX − B‖_F / (‖A‖_F ‖X‖_F)` | n | 16 |
| `blas` | componentwise backward error | k + 2 (k the inner dimension) | 4 |
| `orthogonality` | `‖QᴴQ − I‖_F` | n | 16 |
| `eigen_residual` | `‖AV − VΛ‖_F / ‖A‖_F` | n | 32 |
| `values` | `max |λ − λ_ref| / ‖A‖_2` (eigenvalues and singular values) | n | 32 |

The initial c values are calibrated by the implementation: each kind is set to the smallest
power of two at least 4× the largest residual observed over the migrated tests' passing cases
(all dtypes, the ladder of sizes they use), and the calibration table is recorded on this page.
A kind whose calibrated c differs from the table above by more than 4× is reported, not silently
adopted.

The `blas` growth is k + 2, not k: the α and β scalings and the complex multiply add roundings
that a k = 1 cell hits (cdouble rows reached 0.88 of `4·k·eps` at k = 1).

**Slack.** A site whose quantity legitimately sits off its kind's bound passes
`Slack{factor, reason}`: `bound<T>(kind, n, slack) = factor · bound<T>(kind, n)`. The factor may be
above or below 1 (a site may also tighten); it must be finite and positive, and the reason is
mandatory (`std::invalid_argument` otherwise). With `BATCHLAS_VERIFY_RECORD=<path>` every `pass`
appends `kind dtype n value bound factor reason`: `value` is the raw value (never divided by the
factor), `bound` the kind's unslacked bound, `factor` 1 and `reason` `-` for the plain overloads,
and whitespace in a reason is written as `_`. The calibration lists every factor above 1 for a
user decision. `within<T>` makes the same comparison without recording, for predicates evaluated
on results that are expected to be wrong.

**Pivot ratio.** `pivot_ratio` is not a `c · f(n) · eps` quantity: cabs1 partial pivoting keeps it
at most 1 exactly, so `pivot_ratio_bound<T>() = 1 + 64 eps(T)` (32 machine epsilons of slack for
the rounding of `L(i,k)·U(k,k)` in T).

## Rollout

One self-contained PR to main, in three reviewable stages (commits), each building and passing:

1. **Library and its first users.** The headers and `batchlas::verify`; the tuner's specs on
   main (potrf, posv, trsm, gemm) and `benchmarks/factor_bench.cc` on it;
   `tools/tune/residuals.hh` and factor_bench's inline copies deleted. The three LAPACKE macros
   become `BATCHLAS_VERIFY_HAVE_LAPACKE`, owned by the target.
2. **LU, QR and Cholesky tests**, with their `*_candidates_tests`.
3. **Eigen, SVD, tridiagonal and BLAS tests**, plus the LAPACKE references of the accuracy
   benchmarks.

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

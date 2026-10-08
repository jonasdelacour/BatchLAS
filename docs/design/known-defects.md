# Known defects, located and not fixed

> **Status:** current · checked 2026-10-06

Each entry is a defect located to a line and left in the tree. Entries give the symptom, the
cause with its location, any workaround, and the status. Performance debt is listed per op under
[../perf/](../perf/README.md), not here.

Entries keep their numbers so that other pages can cite them (for example, "known-defects #13").
Entries 1, 3, 11, 12 and 14 are cited from code and carry a `Defect N:` heading. Entries 7 to 9 are
closed and kept only for numbering.

## At a glance

| # | where | status |
|---|---|---|
| 1 | `src/extensions/ortho.cc:189-191` | open, latent (routed to the vendor) |
| 2 | `src/extra/cond.cc:48,54,131` | open, throws in a vendor-free build |
| 3 | `src/extensions/lanczos.cc:112-117` | open, 2x work, right answer |
| 4 | `src/backends/rocsparse.cc:30-31,62-63` | open, inferred, unobservable here |
| 5 | `src/backends/netlib_lapack.cc:484,496,513,525` | open, NaN from unwritten workspace |
| 6 | `src/backends/netlib_lapack.cc:1250` | open, wrong answer at padded `ld` |
| 7 | `src/backends/trsm_route.hh` (deleted) | closed: not a defect |
| 8 | `src/backends/syrk_custom_dispatch.cc` (deleted) | closed when the file was removed |
| 9 | `src/backends/syr2k_custom_dispatch.cc` (deleted) | closed when the file was removed |
| 10 | `src/extensions/latrd_lower_panel.cc` | fixed, barrier armed; residual rate not bounded |
| 11 | `src/sycl/gemm/epilogue_linear.hh`, `src/sycl/gemm_kernels.cc` | open, worked around in `geqrf_blocked` |
| 12 | `src/ops/potrf/potrf.cc`, `src/ops/trsm/trsm.cc`, `src/backends/cusolver.cc:72-77` | open, silent wrong answer on a direct call |
| 13 | `src/backends/cublas.cc`, cuBLASLt | gemm worked around; gemv open |
| 14 | Hermitian drivers in gesvd and syev | open, wrong answer under Auto |
| 15 | cuSOLVER `gesvdjBatched` | open, pinned vendor only |
| 16 | `ormqr_blocked` sub-kernels | open, throws at batch > 65535 |
| 17 | cuSPARSE spmm | open, plus an alignment waiver |
| — | CTA SYTRD Lower path | worked around, not reproduced |
| — | `linalg::qr` | wrapper withheld, cause not located |

## Defect 1: `ortho`'s transposed arm builds a view that does not describe the memory

- **Symptom:** under `transA = Trans` or `ConjTrans`, the CGS arm builds a view whose extents and
  leading dimension do not describe the memory, and passes a vector of the wrong length.
- **Cause:** `src/extensions/ortho.cc:189-191` (was `:218-224` when filed):

  ```cpp
  auto A_i = transA == Transpose::NoTrans
        ? MatrixView<T, fmt>(A.data_ptr(), m, i, m, A.stride(), batch_size)
        : MatrixView<T, fmt>(A.data_ptr(), i, m, m, A.stride(), batch_size);
  auto C      = VectorView(Ymem.data(), i, batch_size);
  auto A_next = A(Slice(), i);
  ```

  `A_i` has `ld = m`, but the leading dimension of the first `i` rows of `A` is `A.ld()`. `A_next`
  is column `i`, but on this arm the vectors are rows of `A`.
- **Why it is not live:** gemv's `can_run` (`src/ops/gemv/gemv.cc`) refuses every native family
  when `X.size() != red_len` or `Y.size() != out_len`, so the call always goes to the vendor.
- **Test gap:** `tests/ortho_tests.cc:249` and `:293` use only `{Transpose::NoTrans}`. The
  transposed arm has never run in the suite.
- **Fix needs:** `A_i` with `ld = A.ld()` and stored-orientation extents, `A_next` as the i-th row
  vector, and a transposed test.
- **Status:** open, latent.

## 2. `cond` calls the vendor `syev` directly

- **Symptom:** throws in a vendor-free build.
- **Cause:** `src/extra/cond.cc:48`, `:54` and `:131` call
  `blas::dispatch::detail::syev_vendor_or_throw`, which bypasses the public `syev` selection.
- **Fix needs:** call the public `syev` and `syev_buffer_size`, and move the workspace query with
  the call. The shims are in `src/ops/syev/vendor.hh`.
- **Status:** open.

## Defect 3: `lanczos` issues a two-column multiply and consumes one column

- **Symptom:** the result is correct, but both the sparse and dense arms do twice the level-3 work.
- **Cause:** `src/extensions/lanczos.cc:112-117`. `padded_vector` and `padded_output` are two
  columns wide (`:56`). The consumer reads only column 0, `local_v_next = Span(v_next_ptr + bid*2*n, n)`
  at `:132`.
- **Note:** `lanczos_tests` fails the same way under `BATCHLAS_SPMM_ROUTE=vendor`
  (`LanczosTestBase.LanczosTest`, `LanczosTestBase.ToeplitzEigenpairs`). A route flip here says
  nothing about routing.
- **Status:** open.

## 4. rocSPARSE conjugates a real transpose

- **Symptom (inferred):** on real scalars, `ConjTrans` is passed as the conjugating enum, which
  should give wrong results as it did in cuSPARSE.
- **Cause:** `src/backends/rocsparse.cc:30-31` and `:62-63` pass `enum_convert` results straight
  through. That conversion (`src/linalg-impl.hh:327-329`) maps `ConjTrans` to
  `rocsparse_operation_conjugate_transpose` for every scalar type.
- **Reference fix:** `cusparse_op<T>` in `src/backends/cusparse.cc:29-39`.
- **Status:** inferred, not observed. There is no AMD device here.
  `scripts/rocm_syntax_check.sh` compiles the ROCm TUs but cannot run a kernel.
- **Fix needs:** a `rocsparse_op<T>` applied to both operands, then `tests/spmm_tests.cc` on AMD
  hardware.

## 5. netlib `trsm` reads `B` when `alpha == 0`

- **Symptom:** `NaN` from an unwritten workspace.
- **Cause:** `src/backends/netlib_lapack.cc:484`, `:496`, `:513` and `:525`, in all four arms of
  the host solve:

  ```cpp
  T x = alpha * Bb.at(i, j, 0) - sum;
  ```

  There is no `alpha == 0` quick return. `BumpAllocator` memory is not zeroed, and `0 * NaN` is
  `NaN`.
- **Fix needs:** substitute `T(0)` for the `alpha` term, as the native bodies do, and add an
  `alpha == 0` case to `tests/trsm_tests.cc` with `B` poisoned by `NaN`. A merely wrong `B` cannot
  reveal the bug. The same defect in `spmm` (`netlib_lapack.cc:228,252`) is fixed; see
  [../perf/spmm.md](../perf/spmm.md).
- **Status:** open.

## 6. netlib `getri` ignores the leading dimension

- **Symptom:** a wrong answer when `Ab` or `Cb` has a padded `ld`. If `C` is the tighter view,
  it also writes past its last column.
- **Cause:** `src/backends/netlib_lapack.cc:1250` copies `n * n` contiguous elements and reads
  neither `ld`.
- **Fix needs:** the per-column `std::copy_n` already used at `:841`.
- **Status:** open. Recorded in [../perf/lu.md](../perf/lu.md).

## 7. CLOSED: `trsm`'s heterogeneous-batch rejection can fire

The field is written: `trsm_op_shape` sets `s.heterogeneous_batch` from both `A` and `B`. The
gate is therefore reachable. The old `trsm_route.hh` is deleted; the term now sits in the native
families' `can_run` (`src/ops/trsm/trsm.cc`).

The gate is armed. Its test is `TrsmCandidates.HeterogeneousBatchHasNoNativeRoute`
(`tests/trsm_candidates_tests.cc`), which asserts that every native pin refuses a heterogeneous `A`
or `B`. Dropping the term turns that test red for all four CUDA dtypes. The vendor arm still lacks
the term (defect 12).

## 8, 9. CLOSED: forced-route defects in the level-3 dispatchers

Both were removed with the old level-3 dispatchers. `BATCHLAS_SYRK_ROUTE=native` and
`BATCHLAS_SYR2K_ROUTE=native` now run the tile kernel, which writes only the named triangle
(`SyrkCudaCustomTest.AutoAndNativeRoutesLeaveTheOtherHalfUntouched`). The old `DiagFullGemm` route
wrote both triangles and was reachable only through those pins.

## Defect 10: the grid `latrd` path had an unsynchronised cross-sub-group read-after-write

- **Symptom:** `SytrdBlockedLatrdGridCudaTest.GridMatchesLegacyTridiagonal` gave a wrong answer in
  about one run in seven, with eigenvalue differences of \f$5\times10^{-5}\f$ against a tolerance
  of \f$10^{-8}\f$.
- **Cause:** in `src/extensions/latrd_lower_panel.cc`, the column-update loop writes
  `Ab(r, i)` for `r = rlo + lid`. The sumsq loop immediately after reads the same column for
  `r = slo + lid`. No barrier separated them. For group 0, `slo == rlo + 1`, so each work-item
  reads the row written by `lid + 1`. The race shows only at a sub-group boundary. It needs
  `wg > 32` and `chunk > 32`. The grid rewrite (`87f6887`, 2026-08-03) dropped the barrier that
  the legacy kernel has.
- **Fix (applied):** the barrier is restored between the write and sumsq loops.
- **Arming (2026-09-15, shared library relinked for each leg):**

  | barrier | geometry | expected | observed |
  |---|---|---|---|
  | present | amplified (`BATCHLAS_LATRD_GRID_GROUPS=2 BATCHLAS_LATRD_GRID_WG=256`, `n=1024`), x20 | green | green |
  | **deleted** | **amplified, x20** | **red** | **red, 20 of 20** |
  | deleted | default, x40 | red | green, 40 of 40 |
  | restored | amplified, x20 | green | green |

  The red leg fails at `tests/sytrd_blocked_tests.cc:545`.
- **Not established:** the residual rate at the default geometry. The default window is about 16
  times narrower than the amplified one, so 40 runs cannot see it. The barrier removes the hazard by
  the memory model. Confirming that needs many more default runs or
  `compute-sanitizer --tool racecheck`.
- **Warning:** forcing a larger work-group does not reproduce the defect. The window depends on the
  group count. Forcing `G` down raises `chunk`. `tests/sytrd_blocked_tests.cc:597-612` is an
  unreliable guard, not a blind one.
- **Status:** fixed, armed on the amplified geometry.

## Defect 11: native gemm reads C at beta zero

- **Symptom:** `GeqrfTest.BlockedIgnoresAGarbageWorkspace` (`tests/geqrf_tests.cc`) returns `NaN`
  for all four scalar types when the trailing update runs as a native gemm. The caller's workspace
  is `0xff`-filled, and the test is 64 x 64.
- **Cause:** `LinearEpilogue::apply` in `src/sycl/gemm/epilogue_linear.hh` computes
  `alpha * acc + beta * prior` unconditionally. `launch_direct` in `src/sycl/gemm_kernels.cc` reads
  `c_ptr` the same way. `0 * NaN` is `NaN`. Reference BLAS does not read `C` at `beta == 0`.
- **Workaround:** `geqrf_blocked_dispatch` zero-fills `W1` and `W2` on each call
  (`src/extensions/geqrf_blocked.cc`).
- **Fix needs:** the two-line change to skip `C` at `beta == 0` in both places, plus a gemm timing
  A/B. When it lands, delete the memset. The test stays as the guard.
- **Status:** open. Found 2026-09-30 on threadripper02 (RTX PRO 6000, sm_120).

## Defect 12: vendor potrf and trsm accept a heterogeneous batch

- **Symptom:** a direct heterogeneous call factors or solves the padded matrix with no error. The
  backward `L^H` solve couples the active rows to the padding. `posv` refuses the input upstream.
- **Cause:** `potrf`'s and `trsm`'s `can_run(Vendor)` have no heterogeneity term, whereas the native
  families do. `potrf_vendor` (`src/backends/cusolver.cc:72-77`) passes `descrA.rows()`, the full
  storage order, to `cusolverDn?potrf[Batched]`. The cuBLAS `trsm` path has no active-dims handling.
- **Fix needs:** a `!A.is_heterogeneous()` term on each vendor `can_run`, or a per-item loop.
- **Test gap:** no test constructs a heterogeneous `potrf` or `trsm`.
- **Level-3 four: closed.** `symm`, `syrk`, `syr2k` and `trmm` refuse a heterogeneous operand in
  `can_run(Vendor)` as well as natively. Auto throws, or `NoRouteError` when vendor-free. Tests:
  `{Symm,Syrk,Syr2k,Trmm}Candidates{,Cpu}.HeterogeneousBatchHasNoRoute`.
- **Status:** open for `potrf` and `trsm`.

## Defect 13: complex<double> cuBLAS calls with a unit dimension segfault inside cuBLASLt

- **Symptom:** every complex<double> `cublasGemmEx`, `cublasGemmStridedBatchedEx` with `m == 1` or
  `n == 1`, and `cublasZgemvStridedBatched`, segfaults inside `cublasLtZZZMatmul`. This happens only
  in a BatchLAS process. A standalone program with the same libraries does not crash. Seen on
  threadripper02 (cuBLAS 13.4.1, HPC SDK 26.5, 2026-10-04).
- **Root cause:** unknown.
- **Reached by:** `trsm` with `blocked` and one right-hand side. The parent build crashes on
  `trsm cdouble L/R order 64-384 q 1 batch 128` under Auto, and also in `getrf_tests`
  (`LuTest/7`) and `ortho_tests` (`OrthoMatrixTest/7`).
- **Workaround (gemm):** `gemm_vendor_impl` in `src/backends/cublas.cc` calls
  `cublasZgemmStridedBatched` when `m` or `n` is 1. Guard:
  `TrsmNativeBlocked.ComplexDoubleSingleRhsTrailingGemm`.
- **Status:** gemm worked around. gemv open: `gemv_vendor` still crashes `ortho_tests`.

## Defect 14: the Hermitian drivers read the unreferenced triangle

With large finite poison in the triangle the caller did not name:

- `gesvd_cta` with Upper, and `gesvd_blocked` with Lower at n <= 32 (tested at 8 and 32; 48 is
  fine), return wrong singular values for all four dtypes.
- `syev_cta` with Upper (n = 5, 17, 32) reads the lower triangle. `syev_blocked` with Lower at
  n <= 32 reads the upper. `syev_two_stage` with Lower at n = 40 reads the upper.

Auto reaches some of these. gesvd sends Hermitian Upper n <= 32 to `cta`. syev sends cfloat
n = 9..32 and cdouble n <= 32 to `syev_cta`. syev `two_stage` with Lower is the Auto choice for
jobz=N above n = 320, and for float V with n = 449..1024. A caller that stores only the upper
triangle gets a wrong answer today.

- **Tests:** `GesvdCandidates.DISABLED_HermitianFamiliesIgnoreTheUnreferencedTriangle`
  (`tests/gesvd_candidates_tests.cc`) and `SyevCandidates.OtherTriangleIsNeverRead`
  (`tests/syev_candidates_tests.cc`). Both are skipped for exactly these drivers.
- **Fix needs:** mirror or read only the named triangle in each driver, then remove the skip list.
  gesvd's `can_run` also refuses blocked Hermitian Upper, which the driver could run by mirroring.
- **Status:** open.

## 15. cuSOLVER `gesvdjBatched` faults on values-only non-square input

- **Symptom:** `BATCHLAS_GESVD_ROUTE=vendor`, float 8x4, jobs N/N, batch 3 gives
  `CUDA_ERROR_ILLEGAL_ADDRESS`.
- **Cause:** gesvd's vendor `can_run` is `has_vendor` alone. Other cuSOLVER refusals
  (`max(m, n) > 32`, non-packed batches, thin factors) arrive as launch-time throws.
- **Workaround:** Auto never sends this shape to the vendor, because a native family runs first.
- **Status:** open, pinned vendor only.

## 16. `ormqr` blocked throws at batch > 65535

- **Symptom:** `ormqr` float L T m=2 k=1 q=1 batch=65536 under Auto gives "Number of work-groups
  exceed limit for dimension 2".
- **Cause:** `ormqr_blocked`'s sub-kernels put the batch on grid dimension 2. The driver does not
  check the limit, and `can_run` does not model it. Auto therefore does not fall to the vendor.
  Other ops that put the batch on dimension 2 may share the limit; see
  [flat-kernel-selection.md](flat-kernel-selection.md) §11.
- **Fix needs:** fold the batch into dimension 0, or loop over batch chunks. As a stopgap, add a
  `batch <= 65535` term to the driver and `can_run`.
- **Status:** open.

## 17. cuSPARSE spmm: refused shapes and the alignment waiver

Measured by calling `backend::spmm_vendor` directly, so these are vendor behaviours.

- **Complex, `transB == ConjTrans`, nrhs 1:** the vendor returns an error status the caller never
  checks, and `C` is left unwritten. The vendor `can_run` now refuses these on CUDA, so Auto runs
  `direct`.
- **complex<double> N/N with one column:** segfaults inside cuSPARSE, as in defect 13. A `vendor`
  pin now falls back to Auto.
- **R3 waiver:** cuSPARSE silently mishandles operands that are off their natural alignment.
  Alignment is a property of the pointers, not of the selection key, so `can_run` does not model
  it. `spmm_tests` skips misaligned cases unless the route is pinned to `native`.

## Known defects: the CTA SYTRD Lower path

**Status:** worked around on both CTA paths; not reproduced.

`src/extensions/syev_cta.cc` (the `uplo_eff` note) says the CTA SYTRD/SYEV pipeline is wrong for
`Uplo::Lower`. It runs the Upper pipeline and first symmetrizes `A_upper := conj(A_lower)`. No test
or failure was recorded. The fused kernel (`syev_cta_fused.cc`) also always runs Upper, so neither
path reaches the Lower `sytrd_cta` kernel.

To settle it, pin the Lower reduction on graded complex input with a poisoned upper triangle.
Compare it against the Upper pipeline. Then fix the kernel and drop the mirror, or record the
failure here. The mirror interacts with [defect 14](#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle),
because `syev_cta` with Upper reads the lower triangle.

## linalg::qr returns a wrong QR after an earlier call in the process

**Status:** observed; cause not located. The wrapper is withheld.

`linalg::qr` is absent from `include/batchlas/blas/linalg-ops.hh`. Composing `geqrf`,
`triangular_mask_into` and `orgqr` returns \f$QR \ne A\f$ after an earlier `linalg::qr` call in the
same process. Run alone, it passes. The repro is `tests/linalg_layer_tests.cc`, 4x.

The `linalg::` value-returning wrappers free their scratch while kernels may still be enqueued.
The workspace arena belongs to the `Queue` (see @ref design_runtime_internals). Look at state that
outlives one call first: arena reuse, a static, or a stale event.

**To fix:** locate the cross-call state with the two-call repro, then add the wrapper with that
repro as its guard.

## Unverified candidates

Located to a line but not confirmed by a test. Each moves to a numbered entry when confirmed or
refuted.

- **`gesvdj_cta` global rescale ignores columns 32..63 on the C=64 rung.** `src/extensions/gesvdj_cta.cc:354`
  reads `Nrm_local[base_n + lane]` only for `lane < CC`. Correctness holds, but the overflow headroom
  for 33 to 64 columns is narrower than the design claims. See
  [global power-of-two scaling](gesvd.md#gesvdj_cta-global-power-of-two-scaling). To settle it, use
  graded input with the largest column norm in columns 32..63, near overflow.
- **Recursive `stedc` may merge from unset eigenvectors under `NoEigenVectors`.**
  `src/extensions/stedc.cc:535` forwards `jobz` to the leaf, but merges always consume the leaf
  vectors. The level-synchronous driver ignores `jobz` (`:640`). To settle it, call
  `stedc(..., NoEigenVectors, ...)` with `StedcAlgorithm::Recursive` on a poisoned eigenvector
  buffer. See [stedc: eigenvalues-only still builds eigenvectors](../perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors).
- **`compute_optimal_wg_size` assumes a 32-bit `long`.** `include/batchlas/util/kernel-heuristics.hh:103,108`
  uses `__builtin_clzl` with a 31 shift. On LP64 this shifts by a negative count, which is undefined
  behaviour. Fix with `63 - __builtin_clzl` or `std::bit_floor`. Callers: `src/matrix.cc:387,487`.
- **The generic (MKL-instantiated) `trmm` recursion reads the whole square of `A`.** Its base case
  at n <= 256 is a `gemm` with `beta = 1` on the diagonal block. It reads the unreferenced triangle
  and, under `Diag::Unit`, the stored diagonal. It accumulates into `C` instead of overwriting.
  See [the trmm generic recursion](../perf/level3.md#trmm-the-generic-recursion-reads-the-whole-square-of-a)
  (`src/extensions/trmm.cc:22-26`).
- **Matrix and vector containers.** Five defects in `matrix.hh` and `src/matrix.cc` are listed in
  [Matrix model: open debts](matrix-model.md#matrix-model-open-debts). They are an undefined
  `MatrixView::transpose`, packed-only addressing in `fill_triangular_random` and
  `fill_tridiag_toeplitz`, padding written by `fill_random`, a no-op slice assert in
  `KernelMatrixView`, and a `fill_diagonal(ctx, Span, k)` length assert that fires for `k != 0`.
- **Environment parser.** `BATCHLAS_SYEVX_SOFT_LOCK` is inverted: `=off` and an empty export read
  as on. `BATCHLAS_GESVD_BLOCKED_GEBRD_MIN` reads an unparseable value as 0, which widens the blocked
  path to every n. `BATCHLAS_TUNE_STEDC_MERGE_VARIANT` is cast to the enum with no range check. See
  [environment parser defects](environment.md#environment-parser-defects-recorded-not-fixed).
- **3-D tile-group race in `gemm`, `symm` and `trmm`.** With a `sycl::nd_item<3>` executor, every
  tile-group writes all of `C` through the non-register sub-group path. `symm` and `trmm` have no
  guard. The result is benign at `beta == 0` and wrong at `beta != 0` or for an in-place `trmm`.
  Read from the code; not reproduced. See
  [the 3-D launch generic fallback](device-group-blas.md#device-group-blas-the-3-d-launch-generic-fallback).
- **`francis_sweep` is declared and never defined.** `include/batchlas/blas/extensions.hh`
  declares it `BATCHLAS_API`. A call compiles and fails at link time. Delete the declaration or
  implement it.
- **`tridiagonal_solver` assumes `Q.ld() == n`.** `src/extensions/tridiag_solver.cc:29` addresses
  `Q` with stride `n`, but the identity fill uses `Q.ld()`. A padded `Q` gives a wrong answer. It
  also caps QR steps at six per eigenvalue with no convergence report. Nothing in `src/` calls it.
- **`UnifiedVector` move assignment leaks.** `operator=(UnifiedVector&&)` in
  `include/batchlas/util/sycl-vector.hh` overwrites `data_` without freeing it. Moving into a
  non-empty `UnifiedVector`, `Matrix` or `Vector` leaks USM shared memory. See
  [Matrix model: open debts](matrix-model.md#matrix-model-open-debts).
- **Kernel selection throws outside the error hierarchy.** `src/select/` throws `std::invalid_argument`
  and `std::runtime_error`. `batchlas::NoRouteError` derives from `std::runtime_error` only, so
  `catch (const batchlas::exception&)` misses all of them. `gesv` and `posv` throw
  `internal_error` for an empty or heterogeneous batch, where `invalid_argument` or `unsupported`
  would fit. Reclassifying either changes behaviour for callers that catch the current type. See
  [kernel selection throws outside the hierarchy](error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy).
- **`internal/sytrd_blocked.hh` declares a second, undefined `sytrd_blocked`.** It takes
  `Span<std::byte> ws` by value with no default `block_size`
  (`include/batchlas/internal/sytrd_blocked.hh:46-53`). The definition at
  `src/extensions/sytrd_blocked.cc:915` is the `const Span<std::byte>&` overload. A translation unit
  that sees only the internal header fails at link time. The fix is to make the internal
  declaration match, or drop it.

## Fixed while documenting

- **`miniacc --help` terminated its host.** `miniacc::ParseCommandLine` called `std::exit(0)`,
  which skips destructors in the caller's frames. It now sets `CliOptions::help_requested`, and
  `MiniAccMain` returns 0 (`include/batchlas/util/miniacc.hh`).

## Retracted claim

[../perf/lu.md](../perf/lu.md) reported a vendor gate defect: `getrs` sits in a `BATCHLAS_HAS_CUBLAS`
TU but calls cuSOLVER. Re-checked, the claim does not hold. `getrs_vendor` calls only
`cublas?getrsBatched` (`src/backends/cublas.cc:1491`), and the gate matches the definition. The
stated mismatch did not reproduce.

## Guards that cannot fail

Several test suites looked thorough but could not fail for the property they named. Each was
shown to be blind by applying a deliberate break, rebuilding, and running.

| guard | why it was blind | fix |
|---|---|---|
| `trsm` barrier regression | drove V1 at n=16, which never has more than one sub-group | drive V2 at order 48, q=976, batch=128 (`tests/trsm_tests.cc:538`). Orders 48, 77, 80 and 109 fail 90-128 of 128 items. |
| all 232 `gemv` cases | every case used the natural batch stride | four `stride_pad` cases, one per kernel body |
| `spmm` transposed `nnz` bound | a NaN at an out-of-range column is discarded by the kernel's own guard | poison with an in-range column and a large finite value |
| `gemv` tail sub-group | out-of-range writes went unseen | a guard band, poisoned before the call and checked after |
| `geqrf` tau/beta convention | the only check skipped in a vendor-free build | an independent host `xGEQR2` reference (`ConventionMatchesReferenceLapackWithoutAVendor`) |
| `potrf` no-fold residual | computed over the lower triangle only | poison the upper triangle and assert it is unchanged |

Checklist before trusting a guard:

1. Apply the break, rebuild the device-linked library, and watch the test go red. A stale `.so`
   passes everything.
2. Check what the kernel does with the poison. A defensive predicate between the poison and the
   assertion makes the case vacuous.
3. Vary each axis the test claims to cover: strides, square and non-square shapes, `ld != rows`,
   real data in complex tests, and batches with unequal `nnz`.
4. Check the elements past the last one asserted on.
5. Check that the suite runs in the build you care about. A `GTEST_SKIP` on vendor absence leaves
   a null suite.
6. A coverage row proves only that some shape reached a route. Only a break that turns one body red
   proves that body ran.
7. Make the break as narrow as the contract it denies.
8. `git diff` cannot verify the revert of an untracked file. Compare `md5sum` against a pristine copy.
9. A residual bound is not a convention test. Assert the convention.
10. For real scalars, `ConjTrans` is `Trans`, and conjugation is the identity. Complex test data
    needs an imaginary part that differs from the real part as a function of the indices.

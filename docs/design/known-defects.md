# Known defects, located and not fixed {#design_known_defects}

> **Status:** current · checked 2026-10-11

Each entry is a defect located to a line and left in the tree. Entries give the symptom, the
cause with its location, any workaround, and the status. Performance debt is listed per op under
[../perf/](../perf/README.md), not here.

Entries keep their numbers so that other pages can cite them (for example, "known-defects #13").
Entries 1, 3 and 11 to 17 are cited from code and carry a `Defect N:` heading. Entries 7 to 9 are
closed and kept only for numbering.

## At a glance

| # | where | status |
|---|---|---|
| 1 | `src/extensions/ortho.cc:190-192` | open, latent (routed to the vendor) |
| 2 | `src/extra/cond.cc:48,54,131` | open, throws in a vendor-free build |
| 3 | `src/extensions/lanczos.cc:111-116` | open, 2x work, right answer |
| 4 | `src/backends/rocsparse.cc:37-38,70-71` | open, inferred, unobservable here |
| 5 | `src/backends/netlib_lapack.cc:399,411,428,440` | open, NaN from unwritten workspace |
| 6 | `src/backends/netlib_lapack.cc:1165` | open, wrong answer at padded `ld` |
| 7 | `src/backends/trsm_route.hh` (deleted) | closed: not a defect |
| 8 | `src/backends/syrk_custom_dispatch.cc` (deleted) | closed when the file was removed |
| 9 | `src/backends/syr2k_custom_dispatch.cc` (deleted) | closed when the file was removed |
| 10 | `src/extensions/latrd_lower_panel.cc` | fixed, barrier armed; residual rate not bounded |
| 11 | `src/sycl/gemm/epilogue_linear.hh`, `src/sycl/gemm_kernels.cc` | fixed 2026-10-11, workarounds deleted |
| 12 | every `can_run` in `src/ops/`, `src/ops/level3/level3.cc` | fixed (WP3): a heterogeneous batch has no route |
| 13 | `src/backends/cublas.cc`, cuBLASLt | fixed (16-byte aligned scalars); the gemv/spmm refusals deleted (WP3) |
| 14 | Hermitian drivers in gesvd and syev | open, wrong answer under Auto |
| 15 | cuSOLVER `gesvdjBatched` | open, pinned vendor only |
| 16 | `ormqr_blocked` sub-kernels | open, not reproduced on threadripper02 (sm_120) |
| 17 | cuSPARSE spmm | open, plus an alignment waiver |
| 18 | `src/extensions/gesvd_blocked.cc:68-71,596` | open; reached under Auto on sm_120 (14 double `vec=all` non-square rows), pin-only on sm_89 |
| — | CTA SYTRD Lower path | worked around, not reproduced |
| — | `linalg::qr` | wrapper withheld, cause not located |
| — | `lanczos` spectrum ~3x too wide | open, wrong answer, cause not located |
| — | LOBPCG null P column | fixed 2026-10-09, kept for the misleading symptom |

## Defect 1: `ortho`'s transposed arm builds a view that does not describe the memory

- **Symptom:** under `transA = Trans` or `ConjTrans`, the CGS arm builds a view whose extents and
  leading dimension do not describe the memory, and passes a vector of the wrong length.
- **Cause:** `src/extensions/ortho.cc:190-192` (was `:218-224` when filed):

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
- **Test gap:** `tests/ortho_tests.cc:111` and `:154` use only `{Transpose::NoTrans}`. The
  transposed arm has never run in the suite.
- **Fix:** `A_i` has `ld = A.ld()` and stored-orientation extents, `A_next` is row `i`
  (conjugated around the gemv pair for complex), the normalisation steps by `ld`. SVQB/SVQB2 copy
  the conjugate transpose back on this arm and the projection against `M` uses `ConjTrans`.
  Householder (also what every algorithm becomes on a host device) QRs a transposed m x k copy and
  transposes the factor back. `tests/ortho_tests.cc` runs both orientations for every algorithm.
- **Status:** fixed (WP8).

## Defect 2: `cond` calls the vendor `syev` directly

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
- **Note:** the red `lanczos_tests` cases are not this defect. One cause, steqr reading item
  b's off-diagonals at stride n-1, is fixed; the remaining one (whole spectrum ~3x too wide, at
  batch 1 too) is open: see [lanczos returns a spectrum about three times too wide](#lanczos-returns-a-spectrum-about-three-times-too-wide).
- **Status:** open.

## Defect 4: rocSPARSE conjugates a real transpose

- **Symptom (inferred):** on real scalars, `ConjTrans` is passed as the conjugating enum, which
  should give wrong results as it did in cuSPARSE.
- **Cause:** `src/backends/rocsparse.cc:37-38` and `:70-71` pass `enum_convert` results straight
  through. That conversion (`src/linalg-impl.hh:275-281`) maps `ConjTrans` to
  `rocsparse_operation_conjugate_transpose` for every scalar type.
- **Reference fix:** `cusparse_op<T>` in `src/backends/cusparse.cc:29-40`.
- **Status:** inferred, not observed. There is no AMD device here.
  `scripts/rocm_syntax_check.sh` compiles the ROCm TUs but cannot run a kernel.
- **Fix needs:** a `rocsparse_op<T>` applied to both operands, then `tests/spmm_tests.cc` on AMD
  hardware.

## Defect 5: netlib `trsm` reads `B` when `alpha == 0`

- **Symptom:** `NaN` from an unwritten workspace.
- **Cause:** `src/backends/netlib_lapack.cc:399`, `:411`, `:428` and `:440`, in all four arms of
  the host solve:

  ```cpp
  T x = alpha * Bb.at(i, j, 0) - sum;
  ```

  There is no `alpha == 0` quick return. `BumpAllocator` memory is not zeroed, and `0 * NaN` is
  `NaN`.
- **Fix needs:** substitute `T(0)` for the `alpha` term, as the native bodies do, and add an
  `alpha == 0` case to `tests/trsm_tests.cc` with `B` poisoned by `NaN`. A merely wrong `B` cannot
  reveal the bug. The same defect in `spmm` (`netlib_lapack.cc:166`) is fixed; see
  [../perf/spmm.md](../perf/spmm.md).
- **Status:** open.

## Defect 6: netlib `getri` ignores the leading dimension

- **Symptom:** a wrong answer when `Ab` or `Cb` has a padded `ld`. If `C` is the tighter view,
  it also writes past its last column.
- **Cause:** `src/backends/netlib_lapack.cc:1165` copies `n * n` contiguous elements and reads
  neither `ld`.
- **Fix needs:** the per-column `std::copy_n` already used at `:756`.
- **Status:** open. Recorded in [../perf/lu.md](../perf/lu.md).

## Defect 7 (closed): `trsm`'s heterogeneous-batch rejection can fire

The field is written: `trsm_op_shape` sets `s.heterogeneous_batch` from both `A` and `B`. The
gate is therefore reachable. The old `trsm_route.hh` is deleted; the term now sits in the native
families' `can_run` (`src/ops/trsm/trsm.cc`).

The gate is armed. Its test is `TrsmCandidates.HeterogeneousBatchHasNoNativeRoute`
(`tests/trsm_candidates_tests.cc`), which asserts that every native pin refuses a heterogeneous `A`
or `B`. Dropping the term turns that test red for all four CUDA dtypes. The vendor arm still lacks
the term (defect 12).

## Defects 8 and 9 (closed): forced-route defects in the level-3 dispatchers

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

  The red leg fails at `tests/sytrd_blocked_tests.cc:390`.
- **Not established:** the residual rate at the default geometry. The default window is about 16
  times narrower than the amplified one, so 40 runs cannot see it. The barrier removes the hazard by
  the memory model. Confirming that needs many more default runs or
  `compute-sanitizer --tool racecheck`.
- **Warning:** forcing a larger work-group does not reproduce the defect. The window depends on the
  group count. Forcing `G` down raises `chunk`. `tests/sytrd_blocked_tests.cc:440-455` is an
  unreliable guard, not a blind one.
- **Status:** fixed, armed on the amplified geometry.

## Defect 11: native gemm reads C at beta zero

- **Symptom:** a beta = 0 gemm into a dirty workspace returned `NaN` when it ran as the native
  `direct`, `tiled` or float `reg` family. `geqrf_blocked` (W1/W2) and `potrf_blocked` (the W x W
  product) zero-filled their scratch to hide it. `orgqr_blocked` had no fill: the blocked `ormqr`
  apply's `W1 = V^H C` reads the caller's arena. Under the deep sm_120 gemm tables, complex
  `orgqr` at 1x1 and 33x20 routes that gemm to `direct` and `tiled`, so
  `OrgqrCandidates.ExactWorkspaceInAPoisonedArena` (0xff arena) went NaN for cfloat and cdouble.
  The sm_89/old tables sent those shapes to families that skip `C`, which hid it.
- **Cause:** `LinearEpilogue::apply` (`src/sycl/gemm/epilogue_linear.hh`), used by `tiled_general.hh`
  and `register_tiled_common.hh`, computed `alpha * acc + beta * prior` unconditionally.
  `launch_direct` in `src/sycl/gemm_kernels.cc` read `c_ptr` the same way. `0 * NaN` is `NaN`.
  Reference BLAS does not read `C` at `beta == 0`. `small`, `wide` and the 128x128 tile already
  skipped it.
- **Fix:** `LinearEpilogue::store` skips the load at `beta == 0`, and all three families call it.
  The geqrf and potrf zero fills are deleted. Timing A/B on sm_120 (batch 256-512, 6 alternating
  reps, medians): beta = 0 is 1-15% faster (the C load is gone), and beta = 1 is within noise
  (0.998-1.017).
- **Guards**, each pinning `direct` and `tiled` through `BATCHLAS_GEMM_ROUTE`'s `ScopedPin` so the
  gemm table cannot hide the family: `GemmCandidates.BetaZeroDoesNotReadC` (every family, NaN `C`,
  including 1x1x1), `OrgqrCandidates.PoisonedArenaUnderEveryPinnedNativeGemm`,
  `GeqrfTest.BlockedIgnoresAGarbageWorkspace` and `PotrfBlockedTest.BlockedDoesNotReadUninitialisedWorkspace`.
  With the epilogues reverted, those four go red for every GPU type, plus
  `OrgqrCandidates.ExactWorkspaceInAPoisonedArena` and `PotrfCandidates.ExactWorkspaceInAPoisonedArena`
  for cfloat and cdouble. Nothing else in those suites goes red.
- **Status:** fixed 2026-10-11 on threadripper02 (RTX PRO 6000, sm_120). Found 2026-09-30.

## Defect 12: vendor families accepted a heterogeneous batch

- **Symptom:** a heterogeneous call (per-item active dims) factored, solved or decomposed the padded
  storage with no error. `BatchPolicy.HeterogeneousBatchNeverRunsAFamilyThatIgnoresIt`
  (`tests/select_tests.cc`, batch 4, active orders {8, 5, 8, 3}) found it returning on CUDA for
  `trsm`, `potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `ormqr`, `syev`, `gesvd`, `spmm`
  (dense B), and `gemv` (float).
- **Cause:** each vendor arm passes the top-level extents (`descrA.rows()`) to a batched or
  per-item library call; `syev`'s and `gesvd`'s native families had no term either, and `ormqr`'s
  `can_run` read only `trans`. `hemm`, `herk` and `her2k` (no selection) reached gemm's per-item
  loop, which threw an unrelated dimension error.
- **Fix (WP3):** `!A.is_heterogeneous()` (and B, C where the op reads them) on every family of
  `potrf`, `trsm`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `ormqr`, `syev`, `gesvd`, `gemv` and
  `spmm`'s vendor. Under Auto such a call finds no runnable kernel (`std::runtime_error`), vendor-free
  `NoRouteError`; a pin throws "cannot run this shape". `hemm`, `herk` and `her2k` throw
  `batchlas::invalid_argument` after their shape check. `gesv` and `posv` already refused upstream.
  Only `gemm` serves a heterogeneous batch (its per-item loop).
- **Level-3 four:** `symm`, `syrk`, `syr2k` and `trmm` already refused one in every family. Tests:
  `{Symm,Syrk,Syr2k,Trmm}Candidates{,Cpu}.HeterogeneousBatchHasNoRoute`.
- **Status:** fixed.

## Defect 13: complex<double> cuBLAS calls segfault inside cuBLASLt

- **Symptom:** every complex<double> `cublasGemmEx`, `cublasGemmStridedBatchedEx` with `m == 1` or
  `n == 1`, and complex<double> gemv at any shape, segfaults inside `cublasLtZZZMatmul`. This happens only
  in a BatchLAS process. A standalone program with the same libraries does not crash. Seen on
  threadripper02 (cuBLAS 13.4.1, HPC SDK 26.5, 2026-10-04).
- **Root cause:** the host `alpha`/`beta` handed to cuBLAS were 8-byte aligned (a
  `std::complex<double>` parameter or closure member), and cuBLASLt loads them with a 16-byte
  vector load. Fixed in 9f551ab0: every cuBLAS call copies them into `alignas(16)` locals inside
  `impl::run_native` (`src/backends/cublas.cc`).
- **Reached by:** `trsm` with `blocked` and one right-hand side. The parent build crashes on
  `trsm cdouble L/R order 64-384 q 1 batch 128` under Auto, and also in `getrf_tests`
  (`LuTest/7`) and `ortho_tests` (`OrthoMatrixTest/7`).
- **Workaround (gemm), now redundant:** `gemm_vendor_impl` in `src/backends/cublas.cc` still calls
  `cublasZgemmStridedBatched` when `m` or `n` is 1. Guard:
  `TrsmNativeBlocked.ComplexDoubleSingleRhsTrailingGemm`. With the branch disabled (DPC++,
  threadripper02, sm_120, 2026-10-09), under `BATCHLAS_GEMM_ROUTE=vendor`:
  - `TrsmNativeBlocked.*` (13), `getrf_tests LuTest/7.*` (46), `ortho_tests` (16) and
    `gemm_tests */7.*` (28) pass;
  - coverage shows complex<double> vendor gemm reached at m or n = 1;
  - a direct call at m, n in {1, 64}, k in {64, 384}, batch 1 (`cublasGemmEx`) and 128
    (`cublasGemmStridedBatchedEx`) matches a host reference to 2.3e-15.

  The branch can be removed; it is kept until that change lands with its own route diff.
- **gemv refused (2026-10-09):** under a pinned vendor every non-trivial `GemvMatrixViewTest/7.*` and
  `GemvCoverageTest/7.*` case exits 139, batch 1 included (`cublasZgemv_v2 -> cublasLtZZZMatmul`).
  The deep sm_120 ledger (the cdouble cells of `benchmarks/results/tuning/ledger/gemv.sm_120.jsonl`) agrees: the vendor
  arm crashed in all 1458 batch > 1 cells and 191 of 196 batch-1 cells. The five surviving rows
  (trans=N, red=32768, out >= 64, batch 1) ranked vendor first, so nearest-row lookup sent other
  shapes into the crash. gemv's `can_run(Vendor)` (`src/ops/gemv/gemv.cc`) therefore refuses
  complex<double> on the CUDA backend: a strict pin throws `std::invalid_argument`, the `vendor` class
  word falls back to Auto, and Auto runs `direct`/`cta`. That term predated the alignment fix.
- **Re-measured without the refusals (WP3, 2026-10-12, threadripper02, sm_120, cuBLAS/cuSPARSE of
  HPC SDK 26.5):** with gemv's complex<double> CUDA term and spmm's complex<double> N/N one-column
  term removed, `BATCHLAS_GEMV_ROUTE=vendor` sent all 43 gemv calls of `GemvMatrixViewTest/3.*` and
  `GemvCoverageTest/3.*` to cuBLAS (46 tests pass), `BATCHLAS_SPMM_ROUTE=vendor` sent every
  `SpmmCoverageTest/3.*` call (48 pass, one N/N nrhs 1) to cuSPARSE, and `spmm_candidates_tests` ran
  the N/N nrhs 1 vendor at m 9/4/5 correctly. No process crashed, so both terms are deleted. Guard:
  `GemvVendorRefusalTest.ComplexDoubleVendorRunsOnCuda` (a strict vendor pin, verified answer).
- **Auto moved:** gemv rows that rank `vendor` first now run cuBLAS for complex<double>. All are
  untimed (`transcribed`): 188 of 1568 cdouble rows on sm_120 and 1360 of 1408 on sm_89. The Auto
  route diff of `gemv_tests`/`spmm_tests` on threadripper02 moved one decision (`m=64 n=32768
  batch=2` NoTrans, `direct` to `vendor`). spmm did not move. Details:
  [gemv.md](../perf/gemv.md#gemv-complex-double-vendor-route-change). The 2x4090 box was not
  re-checked.
- **Status:** fixed; no complex<double> refusal remains in `can_run`.

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

## Defect 15: cuSOLVER `gesvdjBatched` faults on values-only non-square input

- **Symptom:** `BATCHLAS_GESVD_ROUTE=vendor`, float 8x4, jobs N/N, batch 3 gives
  `CUDA_ERROR_ILLEGAL_ADDRESS`.
- **Cause:** gesvd's vendor `can_run` is `has_vendor` alone. Other cuSOLVER refusals
  (`max(m, n) > 32`, non-packed batches, thin factors) arrive as launch-time throws.
- **Workaround:** Auto never sends this shape to the vendor, because a native family runs first.
- **Status:** open, pinned vendor only.

## Defect 16: `ormqr` blocked throws at batch > 65535

- **Symptom:** `ormqr` float L T m=2 k=1 q=1 batch=65536 under Auto gives "Number of work-groups
  exceed limit for dimension 2".
- **Cause:** `ormqr_blocked`'s sub-kernels put the batch on grid dimension 2. The driver does not
  check the limit, and `can_run` does not model it. Auto therefore does not fall to the vendor.
  Other ops that put the batch on dimension 2 may share the limit; see
  [flat-kernel-selection.md](flat-kernel-selection.md) §11.
- **Not reproduced (WP3 review, threadripper02, sm_120, CUDA 13.2):** `ormqr_blocked` called
  directly at batch 65536 ran and applied Q correctly for every side and trans (complex: N, C) at
  m/k/q = 2/1/1, 17/17/3 and 40/33/5, all four types. The public gemm sub-calls refuse batch > 65535
  for every 3-D native launch (`ops::gemm::kMaxGridBatch`, `gemm.cc:101`), so in a vendor build
  they take the vendor. Vendor-free at L T batch 65536, double runs correctly (gemm `small`), while
  float (sub-trmm), complex<float> and complex<double> (sub-gemm) throw `NoRouteError` from the
  sub-op. Vendor-free there is no other ormqr family, so the outcome equals a `can_run` refusal; only
  the coverage miss row names the sub-op. The 2x4090 box was not re-checked. `can_run(Blocked)`
  therefore has no batch term: a term would make double vendor-free unservable past 65535. Test:
  `OrmqrCandidates.BlockedPastTheGridLimitRunsOrHasNoRoute` (L T, batch 65536, pinned `blocked`).
- **Fix needs:** a reproducer. If one is found: fold the batch into dimension 0 or loop over batch
  chunks; the driver itself still does not check.
- **Status:** open, not reproduced.

## Defect 17: cuSPARSE spmm: shapes refused in `can_run`, and the alignment waiver

Measured by calling `backend::spmm_vendor` directly, so these are vendor behaviours.

- **Complex, `transB == ConjTrans`, nrhs 1:** the vendor returns an error status the caller never
  checks, and `C` is left unwritten. The vendor `can_run` now refuses these on CUDA, so Auto runs
  `direct`.
- **complex<double> N/N with one column:** segfaulted inside cuSPARSE, as in defect 13. Re-run
  after defect 13's fix it runs correctly, and the `can_run` term is deleted (see defect 13).
- **R3 waiver:** cuSPARSE silently mishandles operands that are off their natural alignment.
  Alignment is a property of the pointers, not of the selection key, so `can_run` does not model
  it. `spmm_tests` skips misaligned cases unless the route is pinned to `native`.

## Defect 18: `gesvd_cta` left vectors lose orthogonality as eps times kappa squared

- **Symptom:** float, real general 16x16, `BATCHLAS_GESVD_ROUTE=cta`, 160 matrices drawn by
  `Matrix::Random`: \f$\max|U^T U - I|\f$ is 0.003 to 0.77 times \f$\varepsilon\kappa^2\f$. It is
  1.1e-3 at \f$\kappa = 174\f$, 0.20 at \f$2.9 \cdot 10^3\f$ and 0.996 at \f$10^4\f$. V stays at
  1.5e-6. `jacobi` and `blocked` stay at or below 6e-6 on the same matrices.
- **Cause:** CTA mode always takes the normal-equations branch (`gesvd_direct_bidiag`,
  `src/extensions/gesvd_blocked.cc:68-71`). V comes from the eigenvectors of \f$B^T B\f$, and
  `build_bidiag_vectors` forms \f$u_i = B v_i / \sigma_i\f$ (`:596`), which scales the error of
  \f$v_i\f$ by \f$\sigma_{\max}/\sigma_i\f$. Same root as
  [gesvd defect A](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa).
- **Reached by:** on sm_89 a `cta` pin only: no real general row of `tuned/gesvd.sm_89.txt` ranks `cta`
  first, and `jacobi`'s `can_run` admits every real shape that `cta` admits. On sm_120 Auto reaches it:
  the deep table ranks `cta` first in 14 double `vec=all` non-square rows (for example `m=8 n=32`,
  `m=32 n=8`); float `vec=none` rows also reach `cta`, but compute no vectors, so they cannot
  show the defect. Double's error is \f$\varepsilon_d\kappa^2\f$; no double
  failure has been reproduced.
- **Test:** `GesvdTest.CtaProviderCoversAllJobCombinations`, float/CUDA, listed in
  `tests/known-failures.txt`. Since the Philox generator, its `AN` matrix has \f$\kappa = 497\f$;
  8 of 80 seeds fail. The double instantiation still covers the four job combinations.
- **Fix needs:** the direct bidiagonal solver in CTA mode, as Blocked uses, or no `cta` family
  for real general input.
- **Status:** open; reached under Auto on sm_120 (14 double `vec=all` non-square rows), pin-only on sm_89.

## LOBPCG picked a spurious zero Ritz value from a null P column

**Fixed 2026-10-09; kept here because the symptom pointed at the wrong component.** With the deep
sm_120 tables, `syevx_tests` `FindLargestAndSmallest/SyevxJacobiTest.*/Smallest` returned 0 for
eigenvalues 1-3. The tables route LOBPCG's projected solves (float, jobz=V, n = 12 and 18,
batch 2) to the cuSOLVER `vendor` family, and pinning `vendor` under the old tables failed the
same way, so the vendor wrapper was the suspect, specifically its handling of the 12x12 corner
of an 18x18 slab (ld 18 > n).

That hypothesis is refuted. `SyevTest/*.VendorRouteHonoursLdSubViewsAndBatchStride`
(`tests/syev_tests.cc`) runs the pinned vendor on packed, ld > n, sub-view, padded-stride and
batch-1 layouts, all four dtypes, against a closed-form graded spectrum with poisoned padding,
and passes on the unchanged wrapper. Dumping the vendor's inputs and outputs inside LOBPCG showed
the eigenpairs it returned were correct. What it returned, though, were Ritz vectors whose P and R
coefficients were **exactly zero** for two converged columns. LOBPCG builds the next P from those
coefficients (`C_p`, `src/extensions/syevx_lobpcg.cc`), so two P columns were null, the next
\f$S^T A S\f$ had zero rows and columns at 6 and 7, and the projected problem gained an
eigenvalue 0. Taking the smallest end selects it whenever 0 lies below the wanted Ritz values (here
-0.337, -0.041, **0**, 0.74), which gives a zero column in X. The largest end never reaches it. The
native `jacobi` family hits it too: `Pinned/SyevxProjectedSyevRouteTest.SmallestDenseMatchesReferenceSyev/jacobi`
fails on the unfixed tree, as does `/vendor`, while `cta`, `cta_fused` and `blocked` leave
round-off in those coefficients and pass.

LOBPCG already planted a sentinel \f$\lVert S^T A S\rVert_F + 1\f$ on the diagonal for residual
columns that soft locking had zeroed. It did not do that for a null P column. It also could not
do it on a restart, where R sits in the P slot but the sentinel index assumed offset
`2 * block_vectors`. The deflation kernel now finds null directions from \f$S\f$ itself: every
column at or after `block_vectors` with \f$\lVert s_c\rVert^2 \le (n\,\varepsilon)^2\f$ gets the
sentinel, whatever made it null. Guards: the five `Pinned/SyevxProjectedSyevRouteTest.*` cases, which
pin each family so the result no longer depends on the table.

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

## lanczos returns a spectrum about three times too wide

**Status:** observed 2026-10-12 on threadripper02; cause not located.

Toeplitz tridiag(-5, 2, -5), n = 64, float, CSR: `lanczos` returns eigenvalues spanning about
[-28, 32] where the exact spectrum is [-8, 12], at batch 1 as well as batch 2. Each item of a
batch still matches that item solved in a uniform batch (`LanczosStride.MixedBatchItemsMatchUniformBatches`),
so this is not the steqr stride bug fixed 2026-10-12 (defect 3, note). Red tests, listed in `tests/known-failures.txt`:
`LanczosTestBase.LanczosTest`, `LanczosTestBase.ToeplitzEigenpairs`, `LanczosStride.ClosedFormSpectrum`.

**Lead (unconfirmed):** the SVQB2 re-orthogonalization in `lanczos_iteration`
(`src/extensions/lanczos.cc`) rotates the column pair \f$[v_{it-1}, v_{it}]\f$ after their
alphas and betas were recorded, which breaks the three-term recurrence the tridiagonal assumes.
Raising steqr's `max_sweeps` from 10 to 50 changed nothing.

**To fix:** compare the recorded alphas/betas against \f$V^T A V\f$ for one item; if they differ,
re-orthogonalize only against earlier vectors, or recompute alpha and beta after the rotation.

## Unverified candidates

Located to a line but not confirmed by a test. Each moves to a numbered entry when confirmed or
refuted.

- **`gesvdj_cta` global rescale ignores columns 32..63 on the C=64 rung.** `src/extensions/gesvdj_cta.cc:361`
  reads `Nrm_local[base_n + lane]` only for `lane < CC`. Correctness holds, but the overflow headroom
  for 33 to 64 columns is narrower than the design claims. See
  [global power-of-two scaling](gesvd.md#gesvdj_cta-global-power-of-two-scaling). To settle it, use
  graded input with the largest column norm in columns 32..63, near overflow.
- **Recursive `stedc` may merge from unset eigenvectors under `NoEigenVectors`.**
  `src/extensions/stedc.cc:536` forwards `jobz` to the leaf, but merges always consume the leaf
  vectors. The level-synchronous driver ignores `jobz` (`:640`). To settle it, call
  `stedc(..., NoEigenVectors, ...)` with `StedcAlgorithm::Recursive` on a poisoned eigenvector
  buffer. See [stedc: eigenvalues-only still builds eigenvectors](../perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors).
- **`compute_optimal_wg_size` assumes a 32-bit `long`.** `include/batchlas/util/kernel-heuristics.hh:103,108`
  uses `__builtin_clzl` with a 31 shift. On LP64 this shifts by a negative count, which is undefined
  behaviour. Fix with `63 - __builtin_clzl` or `std::bit_floor`. Callers: `src/matrix.cc:387,487`.
- **Matrix and vector containers.** Four defects in `matrix.hh` and `src/matrix.cc` are listed in
  [Matrix model: open debts](matrix-model.md#matrix-model-open-debts). They are an undefined
  `MatrixView::transpose`, packed-only addressing in `fill_triangular_random` and
  `fill_tridiag_toeplitz`, padding written by `fill_random`, and a `fill_diagonal(ctx, Span, k)`
  length assert that fires for `k != 0`.
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
- **`tridiagonal_solver` assumes `Q.ld() == n`.** `src/extensions/tridiag_solver.cc:30` addresses
  `Q` with stride `n`, but the identity fill uses `Q.ld()`. A padded `Q` gives a wrong answer. It
  also caps QR steps at six per eigenvalue with no convergence report. Nothing in `src/` calls it.
- **Kernel selection throws outside the error hierarchy.** `src/select/` throws `std::invalid_argument`
  and `std::runtime_error`, which `catch (const batchlas::exception&)` misses (`batchlas::NoRouteError`
  is a `batchlas::unsupported`). `gesv` and `posv` throw
  `internal_error` for an empty or heterogeneous batch, where `invalid_argument` or `unsupported`
  would fit. Reclassifying either changes behaviour for callers that catch the current type. See
  [kernel selection throws outside the hierarchy](error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy).
- **`internal/sytrd_blocked.hh` declares a second, undefined `sytrd_blocked`.** It takes
  `Span<std::byte> ws` by value with no default `block_size`
  (`include/batchlas/internal/sytrd_blocked.hh:46-53`). The definition at
  `src/extensions/sytrd_blocked.cc:916` is the `const Span<std::byte>&` overload. A translation unit
  that sees only the internal header fails at link time. The fix is to make the internal
  declaration match, or drop it.

## Fixed while documenting

- **`miniacc --help` terminated its host.** `miniacc::ParseCommandLine` called `std::exit(0)`,
  which skips destructors in the caller's frames. It now sets `CliOptions::help_requested`, and
  `MiniAccMain` returns 0 (`include/batchlas/util/miniacc.hh`).

## Retracted claim

[../perf/lu.md](../perf/lu.md) reported a vendor gate defect: `getrs` sits in a `BATCHLAS_HAS_CUBLAS`
TU but calls cuSOLVER. Re-checked, the claim does not hold. `getrs_vendor` calls only
`cublas?getrsBatched` (`src/backends/cublas.cc:1090`), and the gate matches the definition. The
stated mismatch did not reproduce.

## Guards that cannot fail

Several test suites looked thorough but could not fail for the property they named. Each was
shown to be blind by applying a deliberate break, rebuilding, and running.

| guard | why it was blind | fix |
|---|---|---|
| `trsm` barrier regression | drove V1 at n=16, which never has more than one sub-group | drive V2 at order 48, q=976, batch=128 (`tests/trsm_tests.cc:476-486`). Orders 48, 77, 80 and 109 fail 90-128 of 128 items. |
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

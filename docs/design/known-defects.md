# Known defects {#design_known_defects}

> **Status:** current · checked 2026-10-23

Each entry is a defect located to a line, with its symptom, cause, workaround and status. Open
entries are left in the tree; fixed and closed entries keep their numbered heading, because code,
tests and the failure ledgers cite the slug. Performance debt is listed per op under
[../perf/](../perf/README.md), not here.

## At a glance

| # | where | status |
|---|---|---|
| 1 | `src/ops/ortho/ortho.cc:228-231` | fixed (WP8) |
| 2 | `src/util/cond.cc`, `src/util/norm.cc` | fixed: both call the public `syev` |
| 3 | `src/ops/lanczos/lanczos.cc:117-122` | open, 2x work, right answer |
| 4 | `src/backends/rocsparse.cc:38-39,71-72` | open, inferred, unobservable here |
| 5 | `src/backends/netlib_lapack.cc:400,412,429,441` | open, NaN from unwritten workspace |
| 6 | `src/backends/netlib_lapack.cc:1160` | fixed 2026-10-12 (`db9112ed`) |
| 7 | `src/ops/trsm/trsm.cc:48,60` | closed: not a defect; the gate is armed |
| 8 | `src/backends/syrk_custom_dispatch.cc` (deleted) | closed when the file was removed |
| 9 | `src/backends/syr2k_custom_dispatch.cc` (deleted) | closed when the file was removed |
| 10 | `src/ops/syev/sytrd/latrd_lower_panel.cc` | fixed, barrier armed; residual rate not bounded |
| 11 | `src/ops/gemm/kernels/epilogue_linear.hh:14-15`, `src/ops/gemm/kernels/gemm_kernels.cc` | fixed 2026-10-11, workarounds deleted |
| 12 | every `can_run` in `src/ops/`, `src/ops/{hemm,herk,her2k}/` | fixed (WP3): a heterogeneous batch has no route |
| 13 | `src/backends/cublas.cc`, cuBLASLt | fixed (16-byte aligned scalars); the gemv/spmm refusals deleted (WP3) |
| 14 | Hermitian drivers in gesvd and syev | fixed in the drivers (`sytrd_cta` load, `sytrd_sy2sb` update) |
| 15 | cuSOLVER `gesvdjBatched` | open, pinned vendor only |
| 16 | blocked geqrf/orgqr/ormqr part loops | fixed: parts capped at half the grid (reproduced on sm_89) |
| 17 | cuSPARSE spmm | open, plus an alignment waiver |
| 18 | `src/ops/gesvd/kernels/gesvd_blocked.cc:71-74,647` | open; reached under Auto on sm_120 (14 double `vec=all` non-square rows), pin-only on sm_89; float and double red in `tests/known-failures.txt` |
| 19 | `src/ops/potrf/potrf.cc` `can_run` | fixed: `blocked` serves Upper |
| 20 | `src/sycl/kernel_attrs.hh:18-19` `BATCHLAS_LAUNCH_BOUNDS` | open, perf only: the bound is ignored on the tiny solvers; the 128x128 gemm worked around (WP-U) |
| 21 | plain `range<3>` launches with the batch in dimension 0 (`ormbr.cc:170` and others) | open; worked around in the QR, gesvd and syev part loops |
| 22 | `src/ops/getrf/kernels/getrf_panel_reg.cc` register cap; acpp multipass launches | worked around (256-row cap on acpp); the silent launch drop is open |
| — | CTA SYTRD Lower path | worked around, not reproduced |
| — | `linalg::qr` | wrapper withheld, cause not located |
| — | `lanczos` spectrum ~3x too wide | open, wrong answer, cause not located |
| — | LOBPCG null P column | fixed 2026-10-09, kept for the misleading symptom |

## Defect 1: `ortho`'s transposed arm builds a view that does not describe the memory

- **Symptom:** under `transA = Trans` or `ConjTrans`, the CGS arm built a view whose extents and
  leading dimension did not describe the memory, and passed a vector of the wrong length.
- **Cause:** the view was `MatrixView<T, fmt>(A.data_ptr(), i, m, m, A.stride(), batch_size)`, so
  `ld = m` where the first `i` rows of `A` have `ld = A.ld()`, and `A_next = A(Slice(), i)` was
  column `i` where this arm's vectors are rows of `A`.
- **Not live before the fix:** gemv's `can_run` (`src/ops/gemv/gemv.cc`) refuses every native
  family when `X.size() != red_len` or `Y.size() != out_len`, so the call went to the vendor.
- **Fix:** `src/ops/ortho/ortho.cc:228-231`. `A_i` has `ld = A.ld()` and stored-orientation
  extents, `A_next` is row `i` (conjugated around the gemv pair for complex), the normalisation
  steps by `ld`. SVQB/SVQB2 copy the conjugate transpose back on this arm and the projection
  against `M` uses `ConjTrans`. Householder (also what every algorithm becomes on a host device)
  QRs a transposed m x k copy and transposes the factor back.
- **Guard:** `tests/ortho_tests.cc:93` and `:136` run `NoTrans` and `Trans`/`ConjTrans` for every
  algorithm.
- **Status:** fixed (WP8).

## Defect 2: `cond` and `norm` called the vendor `syev` directly

- **Symptom:** the Spectral `cond` and `norm` threw in a vendor-free build on CUDA and ROCm.
- **Cause:** both called `ops::syev::detail::syev_vendor_or_throw` and its buffer-size query for every
  backend but SYCL, bypassing the public `syev` selection.
- **Fix:** `cond_spectral_impl` and `norm_spectral_vendor_impl` call the public `syev` and `syev_buffer_size` for
  every backend, so selection picks the vendor or a native route. Tests: `cond_tests` and `norm_tests` in a
  `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` tree.
- **Status:** fixed.

## Defect 3: `lanczos` issues a two-column multiply and consumes one column

- **Symptom:** the result is correct, but both the sparse and dense arms do twice the level-3 work.
- **Cause:** `src/ops/lanczos/lanczos.cc:117-122`. `padded_vector` and `padded_output` are two
  columns wide (`:63-64`). The consumer reads only column 0, `local_v_next = Span(v_next_ptr + bid*2*n, n)`
  at `:137`.
- **Note:** the red `lanczos_tests` cases are not this defect. One cause, steqr reading item
  b's off-diagonals at stride n-1, is fixed; the remaining one (whole spectrum ~3x too wide, at
  batch 1 too) is open: see [lanczos returns a spectrum about three times too wide](#lanczos-returns-a-spectrum-about-three-times-too-wide).
- **Status:** open.

## Defect 4: rocSPARSE conjugates a real transpose

- **Symptom (inferred):** on real scalars, `ConjTrans` is passed as the conjugating enum, which
  should give wrong results as it did in cuSPARSE.
- **Cause:** `src/backends/rocsparse.cc:38-39` and `:71-72` pass `enum_convert` results straight
  through. That conversion (`src/linalg-impl.hh:268-274`) maps `ConjTrans` to
  `rocsparse_operation_conjugate_transpose` for every scalar type.
- **Reference fix:** `cusparse_op<T>` in `src/backends/cusparse.cc:30-41`.
- **Status:** inferred, not observed. There is no AMD device here.
  `scripts/rocm_syntax_check.sh` compiles the ROCm TUs but cannot run a kernel.
- **Fix needs:** a `rocsparse_op<T>` applied to both operands, then `tests/spmm_tests.cc` on AMD
  hardware.

## Defect 5: netlib `trsm` reads `B` when `alpha == 0`

- **Symptom:** `NaN` from an unwritten workspace.
- **Cause:** `src/backends/netlib_lapack.cc:400`, `:412`, `:429` and `:441`, in all four arms of
  the host solve:

  ```cpp
  T x = alpha * Bb.at(i, j, 0) - sum;
  ```

  There is no `alpha == 0` quick return. `BumpAllocator` memory is not zeroed, and `0 * NaN` is
  `NaN`.
- **Fix needs:** substitute `T(0)` for the `alpha` term, as the native bodies do, and add an
  `alpha == 0` case to `tests/trsm_tests.cc` with `B` poisoned by `NaN`. A merely wrong `B` cannot
  reveal the bug. The same defect in the host `spmm` is fixed (`netlib_lapack.cc:143` and
  `:167-177` leave `B` and `C` unread); see [../perf/spmm.md](../perf/spmm.md).
- **Status:** open.

## Defect 6: netlib `getri` ignores the leading dimension

- **Symptom:** a wrong answer when `Ab` or `Cb` had a padded `ld`. If `C` was the tighter view, it
  also wrote past its last column.
- **Cause:** the host `getri` copied `n * n` contiguous elements from `A` to `C` and read neither
  `ld`.
- **Fix:** `src/backends/netlib_lapack.cc:1160` copies with the ld- and stride-aware
  `MatrixView::copy`, and the LAPACKE call passes `Cb.ld()` (`db9112ed`, 2026-10-12, B10).
- **Status:** fixed. Recorded in [../perf/lu.md](../perf/lu.md).

## Defect 7 (closed): `trsm`'s heterogeneous-batch rejection can fire

Not a defect. `trsm`'s `can_run` reads `A.is_heterogeneous()` and `B.is_heterogeneous()` directly,
in the native term (`src/ops/trsm/trsm.cc:48`) and in the vendor family (`:60`, since defect 12's
fix). Guard: `TrsmCandidates.HeterogeneousBatchHasNoNativeRoute`
(`tests/trsm_candidates_tests.cc:664`) asserts that every native pin refuses a heterogeneous `A`
or `B`; dropping the term turns it red for all four CUDA dtypes.

## Defects 8 and 9 (closed): forced-route defects in the level-3 dispatchers

The old level-3 dispatchers and their `DiagFullGemm` route, which wrote both triangles and was
reachable only through a `native` pin, are deleted. `BATCHLAS_SYRK_ROUTE=native` and
`BATCHLAS_SYR2K_ROUTE=native` run the tile kernel, which writes only the named triangle
(`SyrkCudaCustomTest.AutoAndNativeRoutesLeaveTheOtherHalfUntouched`, `tests/syrk_tests.cc:271`).

## Defect 10: the grid `latrd` path had an unsynchronised cross-sub-group read-after-write

- **Symptom:** `SytrdBlockedLatrdGridCudaTest.GridMatchesLegacyTridiagonal` gave a wrong answer in
  about one run in seven, with eigenvalue differences of \f$5\times10^{-5}\f$ against a tolerance
  of \f$10^{-8}\f$.
- **Cause:** in `src/ops/syev/sytrd/latrd_lower_panel.cc`, the column-update loop writes
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

  The red leg fails at `tests/sytrd_blocked_tests.cc:391`.
- **Not established:** the residual rate at the default geometry. The default window is about 16
  times narrower than the amplified one, so 40 runs cannot see it. The barrier removes the hazard by
  the memory model. Confirming that needs many more default runs or
  `compute-sanitizer --tool racecheck`.
- **Warning:** forcing a larger work-group does not reproduce the defect. The window depends on the
  group count. Forcing `G` down raises `chunk`. `SytrdBlockedLatrdGridCudaTest.ForcedGroupCountsAgree`
  (`tests/sytrd_blocked_tests.cc:441-457`) is an unreliable guard, not a blind one.
- **Status:** fixed, armed on the amplified geometry.

## Defect 11: native gemm reads C at beta zero

- **Symptom:** a beta = 0 gemm into a dirty workspace returned `NaN` when it ran as the native
  `direct`, `tiled` or float `reg` family. `geqrf_blocked` (W1/W2) and `potrf_blocked` (the W x W
  product) zero-filled their scratch to hide it. `orgqr_blocked` had no fill: the blocked `ormqr`
  apply's `W1 = V^H C` reads the caller's arena. Under the deep sm_120 gemm tables, complex
  `orgqr` at 1x1 and 33x20 routes that gemm to `direct` and `tiled`, so
  `OrgqrCandidates.ExactWorkspaceInAPoisonedArena` (0xff arena) went NaN for cfloat and cdouble.
  The sm_89/old tables sent those shapes to families that skip `C`, which hid it.
- **Cause:** `LinearEpilogue::apply` (`src/ops/gemm/kernels/epilogue_linear.hh`), used by `tiled_general.hh`
  and `register_tiled_common.hh`, computed `alpha * acc + beta * prior` unconditionally.
  `launch_direct` in `src/ops/gemm/kernels/gemm_kernels.cc` read `c_ptr` the same way. `0 * NaN` is `NaN`.
  Reference BLAS does not read `C` at `beta == 0`. `small`, `wide` and the 128x128 tile already
  skipped it.
- **Fix:** `LinearEpilogue::store` (`epilogue_linear.hh:14-15`) skips the load at `beta == 0`, and
  all three families call it.
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
  `spmm`'s vendor. Under Auto such a call finds no runnable kernel ("no runnable kernel", a
  `batchlas::unsupported`, `src/select/select.hh`), vendor-free `NoRouteError`; a pin throws "cannot
  run this shape" (a `batchlas::invalid_argument`). `hemm`, `herk` and `her2k` throw
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
  `impl::run_native` (`src/backends/cublas.cc:62-63`, `:151-152`, `:244-245`, `:417-418`,
  `:478-485`).
- **Reached by:** `trsm` with `blocked` and one right-hand side. The unfixed build crashed on
  `trsm cdouble L/R order 64-384 q 1 batch 128` under Auto, and also in `getrf_tests`
  (`LuTest/7`) and `ortho_tests` (`OrthoMatrixTest/7`).
- **Workaround (gemm), redundant:** `gemm_vendor_impl` (`src/backends/cublas.cc:43`) still calls
  `cublasZgemmStridedBatched` when `m` or `n` is 1 (`:69`). Guard:
  `TrsmNativeBlocked.ComplexDoubleSingleRhsTrailingGemm` (`tests/trsm_tests.cc:588`). With the
  branch disabled (DPC++, threadripper02, sm_120, 2026-10-09), under `BATCHLAS_GEMM_ROUTE=vendor`:
  - `TrsmNativeBlocked.*` (13), `getrf_tests LuTest/7.*` (46), `ortho_tests` (16) and
    `gemm_tests */7.*` (28) pass;
  - coverage shows complex<double> vendor gemm reached at m or n = 1;
  - a direct call at m, n in {1, 64}, k in {64, 384}, batch 1 (`cublasGemmEx`) and 128
    (`cublasGemmStridedBatchedEx`) matches a host reference to 2.3e-15.

  Open debt: remove the branch, with its own route diff.
- **The gemv and spmm refusals, deleted (WP3, 2026-10-12):** before the alignment fix, every
  non-trivial `GemvMatrixViewTest/7.*` and `GemvCoverageTest/7.*` case under a pinned vendor exited
  139, batch 1 included (`cublasZgemv_v2 -> cublasLtZZZMatmul`); in the deep sm_120 gemv ledger
  (the cdouble cells of `benchmarks/results/tuning/ledger/gemv.sm_120.jsonl`) the vendor arm crashed
  in all 1458 batch > 1 cells and 191 of 196 batch-1 cells, and its five surviving rows (trans=N,
  red=32768, out >= 64, batch 1) ranked vendor first, so nearest-row lookup sent other shapes into
  the crash. gemv's
  `can_run(Vendor)` therefore refused complex<double> on CUDA, as did spmm's for complex<double> N/N
  with one column. Re-measured without both terms (threadripper02, sm_120, cuBLAS/cuSPARSE of HPC
  SDK 26.5): `BATCHLAS_GEMV_ROUTE=vendor` sent all 43 gemv calls of `GemvMatrixViewTest/3.*` and
  `GemvCoverageTest/3.*` to cuBLAS (46 tests pass), `BATCHLAS_SPMM_ROUTE=vendor` sent every
  `SpmmCoverageTest/3.*` call (48 pass, one N/N nrhs 1) to cuSPARSE, and `spmm_candidates_tests` ran
  the N/N nrhs 1 vendor at m 9/4/5 correctly. No process crashed, so both terms are deleted. Guard:
  `GemvVendorRefusalTest.ComplexDoubleVendorRunsOnCuda` (`tests/gemv_tests.cc:1037`, a strict
  vendor pin, verified answer).
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

Auto reached some of these. gesvd sent Hermitian Upper n <= 32 to `cta`. Before the fix,
Auto sent cfloat n = 9..32 and cdouble n <= 32 to `syev_cta`, and `two_stage` with Lower was its
choice for jobz=N above n = 320 and for float V with n = 449..1024. `linalg::eigh`, `eigvalsh` and
the Hermitian `linalg::svd` mirror the named triangle on their copy first
(`MatrixView::hermitize`), so they were never affected.

- **Fix (2026-10-26), in the drivers:** the public `syev` mirrors nothing; each driver reads only
  the named triangle.
  - `sytrd_cta` (`src/ops/syev/sytrd/sytrd_cta.cc`, the tile load): each lane loads its row of the
    named triangle, coalesced, and writes the conjugate into the other half of the SLM tile; the
    diagonal's imaginary part is dropped, as `{c,z}hetrd` does. This serves `syev_cta` Upper,
    `syev_blocked` Lower at n <= 32 (`sytrd_blocked` hands n <= 32 to `sytrd_cta` Lower) and
    `gesvd_cta` Hermitian Upper. The reduction loops are unchanged.
  - `sytrd_sy2sb` (`src/ops/syev/sytrd/sytrd_sy2sb.cc`): the two full-block `ormqr` calls on the
    trailing block are replaced by LAPACK `dsytrd_sy2sb`'s two-sided update, which reads only the
    lower triangle: Y = A22 V T (`symm`/`hemm`, Lower), W = Y - V (T^H V^H Y) / 2, then
    A22 -= V W^H + W V^H (`syr2k`/`her2k`, or the gemm pair whose lower result depends only on
    lower entries). `syevx`'s direct subset path shares it.
  - Cost (2026-10-26, sm_89, `syev_two_stage_benchmark` jobz=V Lower kd = 32, base 024d64c1, alternating
    rounds): with symm/hemm's `mirror` family ([level-3](../perf/level3.md#level-3-the-mirror-family)) the
    lower-only update is faster than the old full-block one for float (n = 256..2048: 0.88-0.91x) and
    cfloat (n = 256..1024: 0.93-0.94x). double at n = 480, batch 512 is 1.34x slower (809 -> 1084 ms):
    the transcribed double symm rows rank the per-item cuBLAS loop first, and FP64 rows are not tuned on
    this part. Accepted by the owner; a later FP64 pass that measures those rows removes it.
  - Upper on `blocked` and `two_stage`, and Lower on `syev_cta`, still go through an in-place
    O(n^2) pass that reads the named triangle and writes the other (`uplo_mirror.hh`,
    `syev_cta.cc`): the Lower-only (resp. Upper-only) pipelines are served that way, which reads
    nothing unnamed. The documented contract stands: the other triangle may be overwritten.
- **Tests:** `SyevTest.AutoIgnoresUnreferencedTriangle` (n = 5, 32, 40) and its heavy-tier half
  `SyevTest.AutoReadsOnlyTheNamedTriangleAtLargeN` (n = 400, 1024, GPU backends only)
  (`tests/syev_tests.cc`: Auto, both `uplo`, both jobz, poison 9e3 against
  \f$\lVert A\rVert_2 = 1\f$, strided batch), `SyevCandidates.OtherTriangleIsNeverRead` (every
  pinned family, no skip list) and `GesvdCandidates.HermitianFamiliesIgnoreTheUnreferencedTriangle`
  (re-enabled; `PinnedCandidatesStraddleTheirLimits` poisons `cta` Upper too). NETLIB
  instantiations run only the small half: their route is reference LAPACK, which reads only
  `uplo`. In a build without LAPACKE (vendor-free) NETLIB syev has no route and runs neither.
- **Red set of the unfixed drivers** (2026-10-25, sm_89, full-target tree, `tuned/` tables of
  ab235065, no entry mirror): under Auto, float and double fail at n = 400 and 1024 Lower N
  (`two_stage`) and float also at n = 1024 Lower V; cfloat fails at n = 32 Upper V and N (`cta`)
  and at n = 400 and 1024 Lower N; cdouble at n = 5 Upper V and N, n = 32 Upper N (`cta`; its
  n = 32 V row is the vendor) and n = 400 and 1024 Lower N. Every other Auto shape (`jacobi`,
  `cta_fused`, `blocked` above n = 32, the vendor) stays green: those routes do not read the
  poison. `OtherTriangleIsNeverRead` fails for exactly `cta` Upper (n = 5, 17, 32), `blocked`
  Lower (n = 5, 17, 32) and `two_stage` Lower (n = 40), both jobz, all four dtypes; the gesvd test
  for `cta herm=U` at 8x8 and 32x32.
- **Known test gap:** `OtherTriangleIsNeverRead` stops at n = 40, a single short sy2sb panel; a
  full or second panel is reached only by the heavy Auto test (jobz=N, and float V at n = 1024).
- **gesvd blocked fixed 2026-10-22:** the Hermitian arm mirrors Lower into Upper before
  `syev_blocked` (Upper is mirrored the other way inside it), and `can_run` admits Upper.
  `PinnedCandidatesStraddleTheirLimits` now poisons blocked at every n, both triangles.
- **Status:** fixed in the drivers for syev and gesvd.

## Defect 15: cuSOLVER `gesvdjBatched` faults on values-only non-square input

- **Symptom:** `BATCHLAS_GESVD_ROUTE=vendor`, float 8x4, jobs N/N, batch 3 gives
  `CUDA_ERROR_ILLEGAL_ADDRESS`.
- **Cause:** gesvd's vendor `can_run` is `has_vendor` alone. Other cuSOLVER refusals
  (`max(m, n) > 32`, non-packed batches, thin factors) arrive as launch-time throws.
- **Fix (2026-10-22):** on CUDA `can_run(vendor)` is now the binding's envelope:
  \f$\max(m,n) \le 32\f$, packed A and U, no thin factor, and not values-only non-square. On ROCm
  (a "not implemented" stub) it is false. A refused pin throws `cannot run this shape`; Auto falls to
  the next entry or the last resort (`blocked`). `GesvdCandidates.VendorPinRunsInsideItsEnvelope`
  straddles every term.
- **Status:** fixed.

## Defect 16: `ormqr` blocked throws at batch > 65535 {#defect-16-blocked-qr-past-the-grid}

- **Symptom:** `ormqr` float L T m=2 k=1 q=1 batch=65536 under Auto gives "Number of work-groups
  exceed limit for dimension 2".
- **Cause:** `ormqr_blocked`'s sub-kernels put the batch on grid dimension 2. The driver does not
  check the limit, and `can_run` does not model it. Auto therefore does not fall to the vendor.
  Other ops that put the batch on dimension 2 may share the limit; see
  [flat-kernel-selection.md](flat-kernel-selection.md) §11.
- **Not reproduced (WP3 review, threadripper02, sm_120, CUDA 13.2):** `ormqr_blocked` called
  directly at batch 65536 ran and applied Q correctly for every side and trans (complex: N, C) at
  m/k/q = 2/1/1, 17/17/3 and 40/33/5, all four types. The public gemm and trmm sub-calls used to
  refuse batch > 65535 for every 3-D native launch, so vendor-free at L T batch 65536 float (sub-trmm),
  complex<float> and complex<double> (sub-gemm) threw `NoRouteError` from the sub-op. Since the
  level-3 and gemm batch chunking (`src/ops/level3/batched.hh`, gemm's `launch`) those natives run such a batch in chunks, so a
  sub-op no longer refuses it: the vendor-free tree (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`, threadripper02, 2026-10-21)
  runs L T batch 65536 for all four CUDA types and applies Q to every item. The 2x4090 box was not re-checked.
  `can_run(Blocked)` has no batch term. Test: `OrmqrCandidates.BlockedPastTheGridLimitRuns` (L T, batch 65536,
  pinned `blocked`; no `NoRouteError` tolerance).
- **Reproduced (2x4090, sm_89, 2026-10-23):** with the L2 chunk forced above the batch, pinned
  blocked geqrf (float, complex<float>, complex<double>) and ormqr (float) at batch 65537 returned
  items past the grid that differ from item 0: a silent wrong answer, not a throw. Whether it shows
  depends on the local size the runtime picks for the batch dimension, which is why the shapes above
  ran.
- **Fix:** the blocked geqrf, orgqr, ormqr, gesvd and syev part loops cap every part at
  `max_groups_yz / 2` (`select::grid_capped_chunk`, `src/select/grid_chunk.hh`; half because of
  [defect 21](#defect-21-range-rounded-past-the-grid)). Tests:
  `*BlockedPastTheGridLimitIsBitIdentical` in `geqrf_tests` and `ormqr_blocked_tests` (red under
  a deliberate no-cap break: geqrf float/cfloat/cdouble, ormqr float).
- **Status:** fixed in the blocked QR drivers.

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
- **Cause:** CTA mode always takes the normal-equations branch (`gesvd_direct_bidiag` is true only
  for `Blocked`, `src/ops/gesvd/kernels/gesvd_blocked.cc:71-74`). V comes from the eigenvectors of
  \f$B^T B\f$, and `build_bidiag_vectors` (`:517`) forms \f$u_i = B v_i / \sigma_i\f$ (`:582`), which scales the error of
  \f$v_i\f$ by \f$\sigma_{\max}/\sigma_i\f$. Same root as
  [gesvd defect A](../perf/gesvd.md#gesvd-defect-a-the-normal-equations-square-kappa).
- **Reached by:** on sm_89 a `cta` pin only: no real general row of `tuned/gesvd.sm_89.txt` ranks `cta`
  first, and `jacobi`'s `can_run` admits every real shape that `cta` admits. On sm_120 Auto reaches it:
  the deep table ranks `cta` first in 14 double `vec=all` non-square rows (for example `m=8 n=32`,
  `m=32 n=8`); float and double `vec=none` rows also reach `cta`, but compute no vectors, so they
  cannot show the defect. Double's error is \f$\varepsilon_d\kappa^2\f$.
- **Test:** `GesvdTest.CtaProviderCoversAllJobCombinations` (`tests/gesvd_tests.cc:547`), listed in
  `tests/known-failures.txt` for float/CUDA and double/CUDA. Float: since the Philox generator, its
  `AN` matrix has \f$\kappa = 497\f$; 8 of 80 seeds fail. Double (threadripper02, sm_120,
  2026-10-12): the same orthogonality check fails at 8.39e-12 against the \f$128\,n\varepsilon\f$
  `cta` slack; not seen on the 2x4090 box.
- **Fix needs:** the direct bidiagonal solver in CTA mode, as Blocked uses, or no `cta` family
  for real general input.
- **Status:** open; reached under Auto on sm_120 (14 double `vec=all` non-square rows), pin-only on sm_89.

## Defect 19: Upper potrf beyond the CTA ceiling has no vendor-free route

- **Symptom:** `potrf` with `Uplo::Upper` throws `NoRouteError` in a vendor-free build once n is
  above both the `tiny` range and the `cta` local-memory ceiling. On an Apple GPU (Metal, selector
  budget 20,480 B) the first case is complex<float> at n = 33:
  `posv_tests` `PosvTest/5.PublicPosvSolvesOnBothTriangles` (posv pinned `blocked`, whose potrf child
  selects for itself), listed in `tests/known-failures-acpp-metal.txt`. Lower, and n = 4 and 16,
  pass.
- **Cause:** in `can_run` (`src/ops/potrf/potrf.cc:48` and `:54`) `lpanel` and `blocked` require
  `Uplo::Lower`, so only `tiny` and `cta` serve Upper, and `cta` is capped by
  `potrf_cta_max_n_for_slm(budget)` (`:45`). With the vendor family absent nothing else is runnable. On CUDA
  the 97,280 B budget puts the ceiling higher, and a vendor build still has the `vendor` family.
- **Fix:** the blocked driver has an Upper schedule (`potrf_blocked.cc`): leaf `potrf` Upper on the diagonal
  block, `trsm` Left/Upper/ConjTrans on the row panel, and the trailing update `-U12^H U12` through the public
  gemm folded into the upper triangle only. `can_run` for `blocked` no longer reads `uplo`; `lpanel` stays
  Lower-only. `posv` Upper inherits it. Tests: `PotrfBlockedTest.BlockedUpper*`, `FacadeBlockedUpper`.
- **Status:** fixed. Metal's ledger entry in `tests/known-failures-acpp-metal.txt` needs a re-run to confirm.

## Defect 20: `BATCHLAS_LAUNCH_BOUNDS` emits no `.maxntid`, so ptxas ignores the bound

- **Symptom:** the device link prints `.minnctapersm is ignored when neither .maxntid nor .reqntid
  is specified` for every functor that uses the macro without a fixed launch size. Counted with
  `scripts/register_probe.sh` on sm_120 (2026-10-19): 149 in `batchlas_extensions_cta` (the only
  users there are `src/ops/getrf/kernels/getrf_tiny.cc:90`, `src/ops/gesv/kernels/gesv_tiny.cc:101`
  and `src/ops/posv/kernels/posv_tiny.cc:96`) and 24 in `batchlas_sycl` (the 128x128 gemm
  functors, then without `reqd_work_group_size`). Right answers; register counts are uncapped.
- **Cause:** `intel::max_work_group_size(1, 1, N)` in the macro (`src/sycl/kernel_attrs.hh:18-19`)
  lowers to no `.maxntid` on NVPTX with this DPC++, and `.minnctapersm` alone is dropped.
- **Workaround:** a functor with a fixed launch adds `[[sycl::reqd_work_group_size(1, 1, N)]]`
  (`.reqntid`), as wide v2 does
  ([gemm.md](../perf/gemm.md#gemm-wp-q-the-launch-bound-was-ignored)).
- **Fix needs:** the macro fixed, or `reqd_work_group_size` on each remaining functor, then an A/B
  and register probe per kernel: an applied bound moves registers and routing windows, and the
  tiny-solver tables were tuned with no bound in effect. Re-pins the potrf/posv/trsm kernel hashes.
- **Status:** open for the three tiny solvers. Every gemm functor that uses the macro also declares
  `reqd_work_group_size`: `BATCHLAS_GEMM_REQD_WG` in `register_128x128.hh:189,579,590` (u1 since
  WP-U, u4 since WP-N2) and `register_256x128.hh:256`, `BATCHLAS_WIDE_V2_REQD_WG` in
  `register_64x64_k16_wide.hh:883`. Measured after WP-U: 0 ignored bounds in `batchlas_sycl`, u1
  capped at 128 registers
  ([gemm.md](../perf/gemm.md#gemm-wp-u-the-u1-128x128-launch-bound-enforced)).
- **Compiler-dependent (2026-10-22, 2x4090 box):** the intel/llvm build there (`e79d489d`, Jan 2026)
  has the opposite behaviour. `max_work_group_size` emits `.maxntid 256, 1, 1` + `.minnctapersm`,
  while `reqd_work_group_size(1, 1, 256)` lowers to `reqntidx=256, reqntidx=1` and no `reqntidz`:
  ptxas then rejects `.maxntid` next to `.reqntid` ("Conflicting directives"), and with the bound
  dropped the `.reqntid 1, 1` kernels fail every launch with `CUDA_ERROR_INVALID_VALUE`. A configure
  probe of the device IR (`BATCHLAS_NVPTX_REQD_WG_OK`, `cmake/BatchLASDetectSYCL.cmake`) keeps the
  `reqd_work_group_size` spelling only where it lowers correctly; elsewhere the launch bound alone
  applies, which that compiler honours.

## Defect 21: a plain range launch is range-rounded past the grid {#defect-21-range-rounded-past-the-grid}

- **Symptom:** pinned blocked gesvd (float, n = 40, values and vectors, chunking off) aborts with
  `CUDA_ERROR_INVALID_VALUE` and then SIGFPE from batch 65505; batch 65504 runs. At n = 16 the first
  failing batch is 65535.
- **Cause:** `PackPPanelKernel` (`src/ops/gesvd/ormbr/ormbr.cc:170`) is a `parallel_for` over a
  plain `sycl::range<3>` with the batch in dimension 0. DPC++ range rounding
  (`__pf_kernel_wrapper`) rounds that dimension up, here to 65536, past CUDA's 65535 grid. A batch
  that fits the grid is therefore not enough. Other plain `range<3>` launches with the batch in
  dimension 0: `src/kernels/common/larft_wy.hh:261` (pack_v), `src/ops/orgqr/kernels/orgqr_blocked.cc:223,254`,
  `src/ops/syev/kernels/uplo_mirror.cc:28`, `src/ops/syev/sytrd/band_reduction.cc` (nine launches),
  `src/ops/syev/sytrd/sytrd_sy2sb.cc:121`, `src/util/transpose.cc:30`.
- **Workaround:** the part loops of blocked geqrf, orgqr, ormqr, gesvd (all types, complex general
  included) and syev (cta, blocked) keep every part at `max_groups_yz / 2` (`select::grid_capped_chunk`).
  Test: `Gesvd*.BlockedPastTheGridIsBitIdentical` (batch 65505, n = 40, float and complex<float>).
- **Fix needs:** `nd_range` launches with an explicit local size, or the batch out of dimension 0, at
  the sites above. syev two_stage (band_reduction, sy2sb) and `transpose` have no part loop and are not
  covered.
- **Status:** open outside the capped part loops.

## Defect 22: acpp multipass drops an over-ceiling launch silently {#getrf-register-panel-unlaunched-on-acpp-multipass}

- **Symptom:** on an AdaptiveCpp multipass `cuda:sm_89` tree (acpp `develop` 94ab0d58, CUDA 12.8)
  `LuTest/1.RegPanelAgreesWithTheLocalMemoryLeaf` (double) read the `-12345` pivot poison back at
  m = 512, and complex<float> failed the same way at the heavy Auto shapes. No error was raised at
  the launch or the next wait. getrs then used the garbage pivots, faulted (CUDA:700), and the lost
  context failed every later case of the process: about 300 ledger lines of cascade.
- **Cause:** the register-panel leaf's row cap is computed at compile time from register counts
  probed on DPC++'s ptxas (64/104/96/176 for float, double, cfloat, cdouble). acpp's multipass
  cubins allocate 110/150/144/222 (`cuobjdump --dump-resource-usage` of
  `libbatchlas_extensions_cta.so`), so a 512-thread double or cfloat panel is over the
  per-sub-partition register ceiling. The acpp CUDA queue drops that launch without reporting it,
  the same error-model gap as `QueueAsyncErrors.*` in `tests/known-failures-acpp-cuda.txt`.
- **Workaround:** on acpp the cap is at most 256 rows (`kPanelRegImplMaxRows`): 8 warps launch at
  any count up to 255 registers, the bound `kernel_max_wg_size` already uses
  (`src/sycl/impl.hh`). Taller panels take the local-memory leaf. `RegPanelCapacityIsOneSpelling`
  now checks the pivots written by its launch at the advertised cap, so a dropped launch is red.
- **Fix needs:** a launch error from the acpp CUDA backend for an out-of-resources launch, or a
  per-kernel register query; any other compile-time register gate is exposed the same way on acpp.
- **Status:** worked around for the register panel; the silent drop is open.

## LOBPCG picked a spurious zero Ritz value from a null P column

**Status:** fixed 2026-10-09; kept because the symptom points at the wrong component.

- **Symptom:** with the deep sm_120 tables, `syevx_tests`
  `FindLargestAndSmallest/SyevxJacobiTest.*/Smallest` returned 0 for eigenvalues 1-3. The tables
  route LOBPCG's projected solves (float, jobz=V, n = 12 and 18, batch 2) to the cuSOLVER `vendor`
  family, and a `vendor` pin under the old tables failed the same way.
- **Refuted suspect:** the vendor wrapper's handling of the 12x12 corner of an 18x18 slab
  (ld 18 > n). `SyevTest/*.VendorRouteHonoursLdSubViewsAndBatchStride` (`tests/syev_tests.cc:340`)
  runs the pinned vendor on packed, ld > n, sub-view, padded-stride and batch-1 layouts, all four
  dtypes, against a closed-form graded spectrum with poisoned padding, and passes on the unchanged
  wrapper. The eigenpairs the vendor returned inside LOBPCG were correct.
- **Cause:** for two converged columns the Ritz vectors' P and R coefficients were **exactly
  zero**. LOBPCG builds the next P from them (`C_p`, `src/ops/syevx/kernels/syevx_lobpcg.cc:309`),
  so two P columns were null, the next \f$S^T A S\f$ had zero rows and columns at 6 and 7, and the
  projected problem gained an eigenvalue 0. The smallest end selects it whenever 0 lies below the
  wanted Ritz values (here -0.337, -0.041, **0**, 0.74), giving a zero column in X; the largest end
  never reaches it. `jacobi` and `vendor` produce the exact zeros; `cta`, `cta_fused` and `blocked`
  leave round-off and pass.
- **Fix:** the sentinel \f$\lVert S^T A S\rVert_F + 1\f$ on the diagonal covered only residual
  columns zeroed by soft locking, and on a restart (R in the P slot) its index assumed offset
  `2 * block_vectors`. The deflation kernel now finds null directions from \f$S\f$ itself: every
  column at or after `block_vectors` with \f$\lVert s_c\rVert^2 \le (n\,\varepsilon)^2\f$ gets the
  sentinel.
- **Guards:** the five `Pinned/SyevxProjectedSyevRouteTest.SmallestDenseMatchesReferenceSyev/*`
  cases (`tests/syevx_tests.cc:1176`; `vendor`, `cta_fused`, `cta`, `jacobi`, `blocked`), which pin
  each family so the result does not depend on the table. `/jacobi` and `/vendor` fail on the
  unfixed tree.

## Known defects: the CTA SYTRD Lower path

**Status:** worked around on both CTA paths; not reproduced.

`src/ops/syev/kernels/syev_cta.cc:130-135` (the `uplo_eff` note) says the CTA SYTRD/SYEV pipeline
is wrong for `Uplo::Lower`. It runs the Upper pipeline and first symmetrizes
`A_upper := conj(A_lower)`. No test or failure is recorded. The fused kernel
(`syev_cta_fused.cc:52`) also always runs Upper, so neither path reaches the Lower `sytrd_cta`
kernel.

To settle it, pin the Lower reduction on graded complex input with a poisoned upper triangle.
Compare it against the Upper pipeline. Then fix the kernel and drop the mirror, or record the
failure here. The mirror interacts with [defect 14](#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle),
because `syev_cta` with Upper reads the lower triangle.

## linalg::qr returns a wrong QR after an earlier call in the process

**Status:** observed; cause not located. The wrapper is withheld.

`linalg::qr` is absent from `include/batchlas/blas/linalg-ops.hh`. Composing `geqrf`,
`triangular_mask_into` and `orgqr` returns \f$QR \ne A\f$ after an earlier `linalg::qr` call in the
same process. Run alone, it passes. No repro is checked in: the wrapper left the tree with its
test. Rebuild one by composing the three calls twice in one process.

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
(`src/ops/lanczos/lanczos.cc:105`, the `ortho` call at `:113`) rotates the column pair \f$[v_{it-1}, v_{it}]\f$ after their
alphas and betas were recorded, which breaks the three-term recurrence the tridiagonal assumes.
Raising steqr's `max_sweeps` from 10 to 50 changed nothing.

**To fix:** compare the recorded alphas/betas against \f$V^T A V\f$ for one item; if they differ,
re-orthogonalize only against earlier vectors, or recompute alpha and beta after the rotation.

## Unverified candidates

Located to a line but not confirmed by a test. Each moves to a numbered entry when confirmed or
refuted.

- **`gesvdj_cta` global rescale ignores columns 32..63 on the C=64 rung.** `src/ops/gesvd/kernels/gesvdj_cta.cc:354`
  reads `Nrm_local[base_n + lane]` only for `lane < CC`, one value per lane of 32. Correctness holds, but the overflow headroom
  for 33 to 64 columns is narrower than the design claims. See
  [global power-of-two scaling](gesvd.md#gesvdj_cta-global-power-of-two-scaling). To settle it, use
  graded input with the largest column norm in columns 32..63, near overflow.
- **Recursive `stedc` may merge from unset eigenvectors under `NoEigenVectors`.**
  `src/ops/syev/stedc/stedc.cc:537` forwards `jobz` to the leaf, but merges always consume the leaf
  vectors. The level-synchronous driver ignores `jobz` (`:641`). To settle it, call
  `stedc(..., NoEigenVectors, ...)` with `StedcAlgorithm::Recursive` on a poisoned eigenvector
  buffer. See [stedc: eigenvalues-only still builds eigenvectors](../perf/stedc.md#stedc-eigenvalues-only-still-builds-eigenvectors).
- **`compute_optimal_wg_size` assumes a 32-bit `long`.** `include/batchlas/util/kernel-heuristics.hh:104,109`
  uses `__builtin_clzl` with a 31 shift. On LP64 this shifts by a negative count, which is undefined
  behaviour. Fix with `63 - __builtin_clzl` or `std::bit_floor`. Callers on the affected
  REDUCTION and SCAN arms: `src/matrix.cc:406,506`.
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
- **`gesv` and `posv` throw `internal_error` for a caller error.** An empty or heterogeneous batch
  (`src/ops/gesv/gesv.cc:87,90`, `src/ops/posv/posv.cc:107,110`) is `internal_error` where
  `invalid_argument` or `unsupported` would fit. Reclassifying changes behaviour for callers that
  catch the current type. Kernel selection itself throws only `batchlas::` types (fixed in
  58dd7c0a). See
  [kernel selection throws](error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy).

## Fixed while documenting

- **`internal/sytrd_blocked.hh` declared a second, undefined `sytrd_blocked`.** It took
  `Span<std::byte> ws` by value with no default `block_size`, while the definition in
  `src/ops/syev/sytrd/sytrd_blocked.cc` is the `const Span<std::byte>&` overload, so a translation unit
  that saw only the internal header failed at link time. The header is deleted; the one
  declaration, with its contract, is in `include/batchlas/blas/extensions.hh:1055`.
- **`miniacc --help` terminated its host.** `miniacc::ParseCommandLine` called `std::exit(0)`,
  which skips destructors in the caller's frames. It sets `CliOptions::help_requested`, and
  `MiniAccMain` returns 0 (`include/batchlas/util/miniacc.hh:656,723`).

## Retracted claim

| Claim | Re-check | Verdict |
| --- | --- | --- |
| [../perf/lu.md](../perf/lu.md): `getrs` sits in a `BATCHLAS_HAS_CUBLAS` TU but calls cuSOLVER | `getrs_vendor` (`src/backends/cublas.cc:843`) calls only `cublas?getrsBatched`; the gate matches the definition | did not reproduce |
| `tridiagonal_solver` assumes `Q.ld() == n` and caps QR steps at six with no report | `src/ops/syev/tridiag/tridiag_solver.cc` addresses `Q` through `Q.ld()` (`:177`), allows `kTridiagMaxIterations = 30` steps per eigenvalue (`:30`) and throws `convergence_error` past that; `SteqrTest.TridiagSolverGradedAndSaturatingBatch` (`tests/steqr_tests.cc:1432`) runs it with `ld = n + 3` | does not hold |

## Guards that cannot fail

Test suites that looked thorough but could not fail for the property they named, each shown blind
by a deliberate break, rebuild and run.

| guard | why it was blind | fix |
|---|---|---|
| `trsm` barrier regression | drove V1 at n=16, which never has more than one sub-group | drive V2 at order 48, q=976, batch=128 (`TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly`, `tests/trsm_tests.cc:486`). Orders 48, 77, 80 and 109 fail 90-128 of 128 items. |
| all 232 `gemv` cases | every case used the natural batch stride | four `stride_pad` cases, one per kernel body |
| `spmm` transposed `nnz` bound | a NaN at an out-of-range column is discarded by the kernel's own guard | poison with an in-range column and a large finite value |
| `gemv` tail sub-group | out-of-range writes went unseen | a guard band, poisoned before the call and checked after |
| `geqrf` tau/beta convention | the only check skipped in a vendor-free build | an independent host `xGEQR2` reference (`ConventionMatchesReferenceLapackWithoutAVendor`) |
| `potrf` no-fold residual | computed over the lower triangle only | poison the upper triangle and assert it is unchanged |
| Python suite, any device fault | `_skip_if_unavailable` turned every `RuntimeError` into a skip (31 sites) | skip only on `NotImplementedError`, `NoRouteError` and the sub-group-32 message (`python/tests/test_batchlas.py`) |
| Python 1-D results (values, pivots, `m`, `info`, CSR arrays) | under pybind11 < 2.12 and NumPy 2 every one had stride 0; the count checks compared against a constant array, which a stride-0 array matches | build arrays from a shape container; `test_coverage.py` checks per-item values and strides |

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

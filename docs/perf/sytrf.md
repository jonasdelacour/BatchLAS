# LDL^T: sytrf, sytrs, sysv {#perf_sytrf}

> **Status:** current · routes and `can_run` windows only, nothing timed · threadripper02 (RTX PRO 6000, sm_120) and the RTX 4090 box (sm_89), CUDA 13.2 · 2026-10-25

Batched symmetric indefinite factorization by Bunch-Kaufman pivoting with LAPACK `?sytrf` semantics,
its solve `?sytrs`, and the composed `?sysv`. Consumers: KKT systems \f$[[H, A^T], [A, 0]]\f$ (optimize)
and `dsysv` (gfn2). Complex is symmetric, not Hermitian (`csytrf`).

## sytrf: contract

* **Storage.** Only the `uplo` triangle is read and written. It receives D and the multipliers of L
  (Lower) or U (Upper), exactly as LAPACK stores them; the other triangle is never touched.
* **Pivots.** LAPACK's `ipiv`, 1-based int32 packed into the first half of the int64 span, item b at
  `as_span<int>()[b n + k]` (getrf's packing). `ipiv[k] > 0`: 1 x 1 block, rows/columns k and
  `ipiv[k] - 1` interchanged. Lower `ipiv[k] = ipiv[k+1] = -p`: 2 x 2 block at (k, k+1), rows k+1 and
  p-1 interchanged. Upper `ipiv[k-1] = ipiv[k] = -p`: 2 x 2 block at (k-1, k), rows k-1 and p-1
  interchanged.
* **info.** Per item: 0, or i > 0 when D(i,i) (1-based) is exactly zero (`?sytf2`'s rule: the
  first column whose `max(|a_kk|, colmax)` is zero, in loop order, so the zero matrix gives 1 for
  Lower and n for Upper). The factorization still completes.
* **Pivot metric.** `cabs1 = |re| + |im|`, as `csytf2`.

## sysv and sytrs: contract

* `sytrs` reads A, `uplo` and `ipiv` exactly as `sytrf` left them and overwrites B with X. No
  singularity check: a singular item gives infinities or NaNs.
* `sysv` is `sytrf` then `sytrs`. An item with `info > 0` keeps its B (LAPACK `?sysv`); the other
  items are solved. Every route honours this, the fused `cta` kernel included.

## Selection

| op | family | driver | `can_run` |
|---|---|---|---|
| `sytrf` | `cta` | `sytrf_kernels.cc` `cta_launch`: one sub-group per matrix, logical lower triangle in local memory | native term, `n <= 32` (`sytrf.cc`, `can_run`) |
| `sytrf` | `wg` | `sytrf_kernels.cc` `sytrf_wg_dispatch`: one work-group (32..256) per matrix, unblocked `?sytf2` in place in global memory | native term |
| `sytrf` | `blocked` | `sytrf_kernels.cc` `sytrf_blocked_dispatch`: `?lasyf` panels of 31 columns (one work-group per matrix) and the trailing update through the public `gemm`; `?sytf2` on the last panel | native term (the gemm selects for itself) |
| `sytrf` | `vendor` | `cusolverDn<t>sytrf` per item (`cusolver_sytrf.cc`), LAPACKE `?sytrf` (`netlib_lapack.cc`) | CUDA with cuSOLVER or NETLIB with LAPACKE, `d.has_vendor`, homogeneous batch. rocSOLVER is not wired |
| `sysv` | `cta` | `sysv_cta_dispatch`: the `cta` factor and both solves in one launch | GPU, sub-group 32, homogeneous, `1 <= n <= 32` |
| `sysv` | `compose` | public `sytrf` (routed), then `sytrs` with the per-item status as a skip list | homogeneous batch |

* **Native term:** `B != NETLIB && d.is_gpu && d.has_sg32 && max_wg >= 32`, homogeneous batch,
  `n, batch >= 1`.
* **sytrs is not routed** (no `Op`, no route variable): the native kernel (one work-group per
  right-hand side, `sytrs_native_dispatch`) on every GPU backend, LAPACKE on NETLIB.
* **Last resort** (no table): sytrf `wg`, then `vendor`; sysv `compose`.
* **Pins:** `BATCHLAS_SYTRF_ROUTE` (`cta`, `wg`, `blocked`, `vendor`, `native`, `auto`),
  `BATCHLAS_SYSV_ROUTE` (`cta`, `compose`). A `cta` pin above n = 32 throws.

### sytrf: the transcribed tables

`tuned/{sytrf,sysv}.{sm_89,sm_120}.txt` are `transcribed`, **untimed**, generated from
`benchmarks/results/tuning/ledger/{sytrf,sysv}.<device>.jsonl` (sytrf: run `transcribed-sytrf-blocked`,
2026-10-25; sysv: `transcribed-30cc8e06`). Grid: n in {1, 2, 4, 8, 12, 16, 24, 32, 33, 48, ..., 2048},
batch in {128, ..., 32768}, sysv nrhs in {1, 4, 16, 64}.

| op | n <= 32 | 33 <= n <= 128 | n >= 256 |
|---|---|---|---|
| `sytrf` | `cta`, `wg`, `blocked`, `vendor` | `wg`, `blocked`, `vendor` | `blocked`, `wg`, `vendor` |
| `sysv` | `cta`, `compose` | `compose` | `compose` |

The order is a hypothesis, nothing was timed: native first because the vendor arm is a serial
per-item loop; above n = 128 `blocked` first because the unblocked `wg` kernel streams the trailing
matrix from global memory once per column, while `blocked` reads it once per 31-column panel and
hands most of the flops to the gemm. The 128 | 256 edge is a guess; the per-item cuSOLVER loop may
also win at large n and small batch. The specs `tools/tune/sytrf_spec.cc` and `sysv_spec.cc` exist
(Lower only, as potri); timing them is the next step:

```sh
build/tools/tune/batchlas_tune sytrf,sysv --tier preview --dtype float,cfloat --devices 1 --plan
build/tools/tune/batchlas_tune sytrf,sysv --tier preview --dtype float,cfloat --devices 1
python3 scripts/sweep_to_table.py --op sytrf --device sm_89
python3 scripts/sweep_to_table.py --op sysv --device sm_89
```

## sytrf: the reversal map

The kernels implement `?sytf2` and `?sytrs` once, in lower form (`sytrf_device.hh`). Upper runs the
same code on the index-reversed matrix, logical i = physical n-1-i. Under the reversal `?sytf2`'s
upper loop (k = n down to 1, 2 x 2 block at (k-1, k), interchange with k-1) is its lower loop step for
step (block at (k, k+1), interchange with k+1), the stored factor lands in the upper triangle, and
`ipiv` is written at the physical index. The one asymmetry is `I?AMAX`'s tie-break (first maximum in
storage order): the upper form takes the largest logical index on a tie.

The argmax reductions are a sub-group butterfly plus sub-groups in fixed order, so every lane gets
the same result and it is deterministic (a NaN magnitude is folded to +inf before any compare). The
pivot decisions taken from them are lane 0's, broadcast to the work-group ([termination](#sytrf-termination)).

## sytrf: the blocked route

`blocked` is LAPACK `?sytrf`'s structure, lower form (Upper through the reversal map above): `?lasyf`
on a panel, the trailing update \f$A_{22} \mathrel{-}= L_{21} W_{21}^T\f$, then the next panel;
`?sytf2` on the last one. Per panel, three device steps (`sytrf_kernels.cc`):

1. **Step kernel** (one work-group per matrix, `ldl_panel` in `sytrf_device.hh`): first the previous
   panel's `ldl_undo` (`?lasyf`'s closing loop, which restores the `?sytf2` storage form of its
   finished columns), then the panel: each column is formed in W from A and the panel's earlier
   columns (a gemv), the pivot is chosen exactly as in `?sytf2`, and only the non-updated column is
   interchanged in A.
2. **Trailing kernel**: the lower triangle of each 128-wide diagonal block of the trailing matrix.
3. **The public `gemm`** (`NoTrans`, `Trans`, \f$\alpha = -1, \beta = 1\f$; complex is not conjugated)
   for everything below the diagonal blocks (above them for Upper, in physical coordinates). It is
   table-routed and may be native or cuBLAS; the vendor-free build serves it natively.

**Uniform panels.** A batch shares one set of gemm sub-views, but a Bunch-Kaufman panel ends where
its last 2 x 2 block does, which differs per item. `?lasyf` stops one column early (`kb` = nb-1 or
nb). Here every panel is the uniform range [k0, k0 + 31) and always attempts its last column, so an
item ends at k0 + 31 or, when that step is a 2 x 2 block, at k0 + 32; W has 32 columns, the columns
an item did not compute are zero, and the gemm's k is a uniform 32. An item that ended late starts
the next panel one column late (`state` = {first, end} per item). Its column k0 + 31 is either
factored or still trailing; the trailing kernel updates it in the second case. The pivots are
`?sytf2`'s in exact arithmetic for any blocking, so ipiv matches `wg` and cuSOLVER entry for entry
on data without near ties (`BlockedMatchesWgAndVendor`). One consequence of the zero W columns: a
non-finite entry in a not-yet-factored column k0 + 31 meets a zero (Inf x 0 = NaN) where LAPACK would
not multiply; such an item's factor is already non-finite.

**Constants** (`sytrf_native.hh`, untimed): nb = 31, diagonal blocks of 128, the batch chunked by
getrf's L2 rule (`tuning::blocked_chunk_items`, `BLOCKED_L2_CHUNK_PERCENT`). Workspace: W
(n x 32 per chunk item), the state, the info fallback and three pointer arrays for the gemm views,
through `BumpAllocator::measuring()`.

**Tests** (`tests/sytrf_tests.cc`): `blocked` is in every route list (n straddling 31 | 32 and
62 | 63, both uplo, every generator, padded ld and stride, poisoned triangle). Also
`BlockedMatchesWgAndVendor` (ipiv vs `wg` and cuSOLVER; asserts that some item takes a 2 x 2 block
across the first boundary), `SingularColumnsReportTheFirstInLoopOrder` (n = 100, info the first
zero column in loop order; zero sets in panels only, in the `?sytf2` tail only, and in a panel and the
tail), the saturating batch at n = 170 (where the gemm runs), and below one panel
`CtaAndWgAgreeBitForBit` (blocked is the `wg` kernel there).

## sytrf: termination

Every device loop is bounded by n: `ldl_factor` and `ldl_panel` advance k by kstep in {1, 2},
`ldl_solve` advances by 1 or 2 per ipiv entry in both passes, and `ldl_undo` walks j down. The one
way these kernels can hang is a barrier that some lanes skip. The pivot decisions (`?sytf2`'s three
tests) were taken redundantly by every lane from the same values, which holds only if every lane
reads the same bits: a NaN compare, or items aliased by a stride shorter than the matrix (another
work-group writing the values being read), can split a work-group across the kstep branches, and the
next barrier deadlocks. The decisions are now lane 0's, broadcast (`Group::bcast`,
`pivot_stage`/`pivot_choice`), so the barrier count is uniform whatever the data. The argmax
reductions were already uniform: a butterfly over a strict total order (NaN folded to +inf, ties
broken by index).

`ldl_solve` reads ipiv as given. A list `sytrf` cannot have written (an entry 0 or outside
[-n, n], a 2 x 2 entry without its partner) used to index b out of bounds; now that item's X is
all NaN and the other items are solved. LAPACKE's `?sytrs` (NETLIB) does no such check.

Tests: `NonFiniteAndAliasedItemsTerminate` is a smoke test, not a guard for the broadcast: it runs a
NaN diagonal, Inf, an all-NaN item and an aliased batch on cta, wg and blocked and checks info > 0 and
the clean items' bit-identity, but it provokes no lane divergence on this box, so removing the
broadcast leaves it green (deliberate breaks below). `SytrsMalformedPivotsGiveNaN` guards the ipiv
check. No input was found that hangs the kernels as they were: the hang
recorded in PR #193 was seen with a deliberately broken kernel and its trigger was not kept, so the
broadcast removes the mechanism rather than a known trigger. The `ctest` entries carry
`TIMEOUT 1800`, and the suite is in the heavy tier (`SaturatingBatchIsBitIdenticalToItemZero`).

## sytrf: vendor

cuSOLVER 13.2 (`cusolverDn.h`) offers `cusolverDn<t>sytrf` (+ `_bufferSize`; legacy API, int32
`ipiv`, one matrix per call) and the 64-bit `cusolverDnXsytrs` (int64 `ipiv`). There is no
`cusolverDnXsytrf` and no batched sytrf or sytrs. The vendor family is therefore a per-item loop on one
stream with one shared work buffer; cuSOLVER writes LAPACK's `ipiv`, which the native `sytrs`
consumes directly (`VendorFactorSolvesWithNativeSytrs`). `Xsytrs` is not wired: it would need an int64
copy of the pivots and is still per item.

## sytrf: tests

`tests/sytrf_tests.cc`, every route pinned, padded `ld` and batch stride, both pads and the
unreferenced triangle poisoned with large finite values, complex with nonzero imaginary parts:

* `GeneratorsForceTheirPivotKinds`: random data has interchanges and 2 x 2 blocks, the dominant
  generator neither, every KKT item a 2 x 2 block.
* `FactorThenSolveEveryRoute`, `SysvEveryRoute`: residual through `EXPECT_VERIFY(Check::solve)`, both
  uplo, n straddling 32 | 33, 256 | 257 and the blocked panel (31 | 32, 62 | 63), `info == 0`,
  untouched triangle and pads.
* `SingularItemReportsInfoWithoutDisturbingTheOthers`: the zero matrix in slot 2 gives `info` 1
  (Lower) or n (Upper); every other item is bit-identical to a batch without it; sysv keeps its B.
* `WgPivotsMatchVendor`: on random data (no near ties) the `wg` kernel's ipiv equals cuSOLVER's entry
  for entry at n = 24, 100, 200; a residual cannot see a wrong pivot choice, and above n = 32 the
  argmax spans several sub-groups.
* `CtaAndWgAgreeBitForBit`, `SaturatingBatchIsBitIdenticalToItemZero` (batch 1024, identical KKT
  items), `VendorFactorSolvesWithNativeSytrs`, `CtaCeilingStraddled`, `ArgumentChecks`.
* The blocked and termination tests: see the two sections above.

### sytrf: deliberate breaks

Run on the RTX 4090 box, 2026-10-25, float and cfloat CUDA instantiations, every break reverted:

| break | red set |
|---|---|
| trailing kernel skips column k0 + 31 of an item whose panel ended there | `blocked` rows of `FactorThenSolveEveryRoute`, `BlockedMatchesWgAndVendor` (residual), the n = 170 saturating case |
| the step kernel skips `ldl_undo` | the same `blocked` rows, residuals only (ipiv still matches `wg` and cuSOLVER) |
| alpha = (1 + sqrt 17) / 7 (pivot choice; every factor stays valid) | `WgPivotsMatchVendor`, the "vs vendor" half of `BlockedMatchesWgAndVendor`; every residual green |
| `ldl_factor` reports the last zero column | `SingularItemReportsInfo...` (cta, compose), the `wg` rows of `SingularColumnsReportTheFirstInLoopOrder` with two zero columns (re-run 2026-10-26; blocked green: each set puts at most one zero column in its tail) |
| blocked tail info drops the Lower offset (`st = t`, 2026-10-26) | `SingularColumnsReportTheFirstInLoopOrder`, the `blocked` Lower {95} case only (info 3 for 96), all four CUDA types |
| `sytrs` drops the reversal of x for Upper | every Upper row through `sytrs`; all Lower and the fused `sysv` `cta` green |
| `sytrs` ignores the right-hand side's column offset | every `sytrs` row with nrhs >= 2; the fused `sysv` `cta` green |
| `sysv` `compose` passes no skip list | `SingularItemReportsInfo...`, `compose` only |
| `ldl_solve` pivot validation off | `SytrsMalformedPivotsGiveNaN`: an illegal-address abort |
| decisions not broadcast | nothing: the termination test provokes no divergence on this box |

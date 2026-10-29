# LDL^T: sytrf, sytrs, sysv {#perf_sytrf}

> **Status:** current · sytrf routes untimed; sytrs and sysv timed on the RTX 4090 box (sm_89, float and cfloat), sm_120 transcribed · CUDA 13.2 · 2026-10-26

Batched symmetric indefinite factorization by Bunch-Kaufman pivoting with LAPACK `?sytrf` semantics,
its solve `?sytrs`, and the composed `?sysv`. Consumers: KKT systems \f$[[H, A^T], [A, 0]]\f$ (optimize)
and `dsysv` (gfn2). Complex is symmetric, not Hermitian (`csytrf`).

## sytrf: contract

* **Storage.** Only the `uplo` triangle is read and written; it receives D and the multipliers of L
  (Lower) or U (Upper) as LAPACK stores them.
* **Pivots.** LAPACK's `ipiv`, 1-based int32 packed into the first half of the int64 span, item b at
  `as_span<int>()[b n + k]` (getrf's packing). `ipiv[k] > 0`: 1 x 1 block, k and `ipiv[k] - 1`
  interchanged. Lower `ipiv[k] = ipiv[k+1] = -p`: 2 x 2 block at (k, k+1), rows k+1 and p-1
  interchanged. Upper `ipiv[k-1] = ipiv[k] = -p`: 2 x 2 block at (k-1, k), rows k-1 and p-1.
* **info.** Per item: 0, or i > 0 when D(i,i) (1-based) is exactly zero, `?sytf2`'s rule (the first
  column in loop order whose `max(|a_kk|, colmax)` is zero: the zero matrix gives 1 for Lower, n for
  Upper). The factorization still completes.
* **Pivot metric.** `cabs1 = |re| + |im|`, as `csytf2`.

## sysv and sytrs: contract

* `sytrs` reads A, `uplo` and `ipiv` as `sytrf` left them and overwrites B with X. No singularity
  check (a singular item gives Inf or NaN). Workspace `sytrs_buffer_size()` bytes, nonzero for
  `blocked` only. A pivot list `sytrf` cannot have written (an entry 0 or outside [-n, n], a 2 x 2
  entry without its partner) gives that item an all-NaN X; LAPACKE's `?sytrs` (NETLIB) does no check.
* `sysv` is `sytrf` then `sytrs`. An item with `info > 0` keeps its B (LAPACK `?sysv`); every route,
  the fused `cta` included, honours this through sytrs's skip list.

## Selection

| op | family | driver | `can_run` |
|---|---|---|---|
| `sytrf` | `cta` | `cta_launch`: one sub-group per matrix, logical lower triangle in local memory | native, `n <= 32` |
| `sytrf` | `wg` | `sytrf_wg_dispatch`: one work-group per matrix, unblocked `?sytf2` in global memory | native |
| `sytrf` | `blocked` | `sytrf_blocked_dispatch`: `?lasyf` panels of 31 + the public `gemm` | native |
| `sytrf` | `vendor` | `cusolverDn<t>sytrf` per item (`cusolver_sytrf.cc`), LAPACKE `?sytrf` | vendor linked, `d.has_vendor`, homogeneous, unmasked; no rocSOLVER |
| `sytrs` | `cta` | `sytrs_cta_dispatch`: [converted solve](#sytrs-the-converted-solve) in local memory, a sub-group per rhs | native, `n <= 128`, `sytrs_cta_slm_bytes<T>(n) <= slm_budget` |
| `sytrs` | `blocked` | `sytrs_blocked_dispatch`: converted factor in workspace, two public `trsm` | native |
| `sytrs` | `wg` | `sytrs_native_dispatch`: one work-group per rhs, `?sytrs` step by step | native |
| `sytrs` | `vendor` | LAPACKE `?sytrs` per item | NETLIB with LAPACKE, homogeneous |
| `sysv` | `cta` | `sysv_cta_dispatch`: `cta` factor + sytrs `cta` solve, one launch | GPU, sub-group 32, homogeneous, `1 <= n <= 32` |
| `sysv` | `compose` | routed `sytrf`, then routed `sytrs` with the status as skip list | homogeneous |

* **Native:** `B != NETLIB && d.is_gpu && d.has_sg32 && max_wg >= 32`, homogeneous and unmasked,
  `n, batch >= 1` (sytrs: `nrhs >= 1`). `sytrf.cc` and `sytrs.hh` hold the predicates.
* **Last resort** (no table): sytrf `wg`, then `vendor`; sytrs `blocked`, then `vendor`; sysv `compose`.
* **Pins:** `BATCHLAS_SYTRF_ROUTE` (`cta`, `wg`, `blocked`, `vendor`), `BATCHLAS_SYTRS_ROUTE` (`cta`,
  `blocked`, `wg`, `vendor`), `BATCHLAS_SYSV_ROUTE` (`cta`, `compose`); each also takes `native`, `auto`.

### sytrf: the transcribed tables

`tuned/{sytrf,sysv}.{sm_89,sm_120}.txt` were transcribed, **untimed** (ledgers
`benchmarks/results/tuning/ledger/{sytrf,sysv}.<device>.jsonl`, 2026-10-25):

| op | n <= 32 | 33 <= n <= 128 | n >= 256 |
|---|---|---|---|
| `sytrf` | `cta`, `wg`, `blocked`, `vendor` | `wg`, `blocked`, `vendor` | `blocked`, `wg`, `vendor` |
| `sysv` | `cta`, `compose` | `compose` | `compose` |

The order is a hypothesis: native first because the vendor arm is a serial per-item loop; `blocked`
first above n = 128 because it reads the trailing matrix once per panel, not once per column. The
128 | 256 edge is a guess. **Open:** time sytrf (`tools/tune/sytrf_spec.cc`, Lower only) with
`batchlas_tune sytrf --tier preview --dtype float,cfloat`, then `scripts/sweep_to_table.py --op sytrf`.

sysv sm_89 float and cfloat are a `preview` sweep (run `20261026T152523-JRHNBR-1179869`, 691 rows):
`compose` wins much of n <= 32 from nrhs = 16 (8 x 16 x 8192: 0.133 vs 0.169 ms for `cta`). double,
cdouble and sm_120 keep the transcribed rows.

### sytrs: the tuned tables {#sytrs-selection}

`tuned/sytrs.sm_89.txt` float and cfloat: `preview` sweep of `tools/tune/sytrs_spec.cc` (A factored
once by the routed sytrf, untimed; Lower only; verified by `solve_residual`), run
`20261026T143047-JRHNBR-1000130`, batch 128..32768, 1256 cells. double, cdouble and all of sm_120
copy that order row for row (`transcribed-sytrs-d67ad754`; FP64 is not tuned on 1:64 parts), and
`can_run` drops `cta` where cdouble's factor does not fit local memory. Regions as in
[sytrs: measured](#sytrs-measured).

## sytrf: the reversal map

The kernels implement `?sytf2` and `?sytrs` once, in lower form (`sytrf_device.hh`). Upper runs the
same code on the index-reversed matrix, logical i = physical n-1-i: `?sytf2`'s upper loop becomes its
lower loop step for step, the factor lands in the upper triangle and `ipiv` is written at the physical
index. The one asymmetry is `I?AMAX`'s tie-break (first maximum in storage order): Upper takes the
largest logical index on a tie. The argmax reductions are a sub-group butterfly plus sub-groups in
fixed order, deterministic and uniform across lanes (NaN magnitude folded to +inf).

## sytrf: the blocked route

LAPACK `?sytrf`'s structure in lower form: `?lasyf` on a panel, the trailing update
\f$A_{22} \mathrel{-}= L_{21} W_{21}^T\f$, the next panel, `?sytf2` on the last. Per panel
(`sytrf_kernels.cc`):

1. **Step kernel** (one work-group per matrix, `ldl_panel`): the previous panel's `ldl_undo`
   (`?lasyf`'s closing loop), then the panel, each column formed in W and pivoted as in `?sytf2`.
2. **Trailing kernel**: the lower triangle of each 128-wide diagonal block.
3. **Public `gemm`** (`NoTrans`, `Trans`, \f$\alpha = -1, \beta = 1\f$, complex not conjugated) for the
   rest; table-routed, native in the vendor-free build.

**Uniform panels.** A batch shares one set of gemm sub-views, but a Bunch-Kaufman panel ends per
item. Every panel is the range [k0, k0 + 31) and attempts its last column, so an item ends at k0 + 31
or k0 + 32 (a 2 x 2 block); W has 32 columns, uncomputed ones zero, and the gemm's k is 32. An item
that ended late starts the next panel one column late (`state` = {first, end}). Pivots equal
`?sytf2`'s in exact arithmetic, so ipiv matches `wg` and cuSOLVER on data without near ties
(`BlockedMatchesWgAndVendor`). Caveat: a non-finite entry in a not-yet-factored column k0 + 31 meets
a zero W column (Inf x 0 = NaN); such an item's factor is non-finite anyway.

**Constants** (`sytrf_native.hh`, untimed): `kSytrfBlockedNb` = 31, diagonal blocks of 128, batch
chunked by getrf's L2 rule (`tuning::blocked_chunk_items`). Workspace (W, state, info fallback, gemm
pointer arrays) through `BumpAllocator::measuring()`.

## sytrf: termination

Every device loop is bounded by n (k advances by 1 or 2). The one way to hang is a barrier some lanes
skip: per-lane pivot decisions split a work-group when lanes read different bits (a NaN compare, or
items aliased by a stride shorter than the matrix). The decisions are lane 0's, broadcast
(`Group::bcast`, `pivot_stage`/`pivot_choice`), so the barrier count is uniform whatever the data.

**Known gap:** `NonFiniteAndAliasedItemsTerminate` (NaN, Inf, all-NaN and aliased items on cta, wg,
blocked) provokes no lane divergence on the 4090 box, so removing the broadcast leaves it green; no
input that hung the old kernels was kept. `SytrsMalformedPivotsGiveNaN` guards the ipiv check. The
ctest entries carry `TIMEOUT 1800`; the suite is heavy-tier.

## sytrs: the converted solve {#sytrs-the-converted-solve}

`wg` is LAPACK `?sytrs` (interchanges one pivot at a time between eliminations). `cta` and `blocked`
are `?sytrs2` (`sytrs_device.hh`): the factor is first **converted** (`?syconv`: each interchange
applied to the rows of the earlier columns of L, 2 x 2 off-diagonals moved out of L), giving
\f$A = P L D L^T P^T\f$ with one permutation P. The solve is gather \f$y = P^T b\f$, \f$L w = y\f$,
\f$w := D^{-1} w\f$, \f$L^T z = w\f$, scatter \f$x = P z\f$, in logical lower form
([reversal map](#sytrf-the-reversal-map)). The pivot list is validated once per item first.

* **`cta`**: one work-group per item; the lower triangle is staged and converted in local memory
  (ld = n | 1). Each sub-group owns right-hand sides sgid, sgid + nsg, ... (nsg = min(nrhs, 8)); lane
  i holds rows i, i + 32, ... in registers (at most 4, so n <= 128); both sweeps are broadcast + axpy
  (no reduction, no barrier); D^-1 reads a 2 x 2 partner across the lane-31 / lane-0 seam. cdouble
  stops near n = 76 on sm_89 (local memory).
* **`blocked`**: per chunk of items, a prep kernel writes the converted factor (`convert_gather`,
  through the composed permutation) and the gathered B to workspace; then `trsm` (`Left`, `NoTrans`,
  `Unit`), a D^-1 kernel, `trsm` with `Trans` (not conjugated), and the scatter. The chunk is getrf's
  L2 rule capped by half the scratch lease and the grid (`blocked_chunk`). Any n, any nrhs.
* **sysv `cta`**: sub-group 0 factors (bit-identical to sytrf `cta`), then one sub-group per
  right-hand side (at most 8) runs `cta`'s solve.

## sytrs: measured {#sytrs-measured}

`benchmarks/sytrs_benchmark` (`BM_SYTRS`, `BM_SYSV`; family = route pin), RTX 4090, 2026-10-26,
`--warmup=3 --min_iters=8`, two passes in reversed order, median ms:

| dtype | n | nrhs | batch | `wg` | `cta` | `blocked` |
|---|---|---|---|---|---|---|
| float | 8 | 16 | 8192 | 0.297 | **0.065** | 0.067 |
| float | 32 | 64 | 8192 | 4.35 | **0.486** | 1.01 |
| float | 64 | 1 | 8192 | **0.243** | 0.602 | 0.726 |
| float | 128 | 1 | 8192 | **0.869** | 10.2 | 3.20 |
| float | 128 | 64 | 8192 | 44.2 | 19.8 | **7.09** |
| cfloat | 64 | 64 | 8192 | 19.8 | **3.16** | 6.91 |
| cfloat | 512 | 16 | 1024 | 25.0 | - | **12.7** |

* `cta` wins n <= 64 at nrhs >= 4 and nrhs = 1 up to n = 32; at n = 128 its staged factor fills an SM.
* `blocked` wins from n = 128 at nrhs >= 4, and at every n for nrhs >= 16, batch >= 1024.
  `convert_gather` beat one swap per pivot step 1.0-2.3x from n = 64.
* `wg` stays for nrhs = 1 from n = 64 and nrhs <= 4 from n = 256.
* sysv `cta` (one sub-group per rhs) vs the old serial-rhs fused kernel: 1.0-1.1x at nrhs = 1,
  3.5-11.8x at nrhs = 64.
* Dropped: `sg`, one sub-group per (item, rhs) on the unconverted factor, best in 12 of 162 cells by
  at most 1.35x over `wg`.

Evidence: `/home/jonaslacour/.claude/jobs/ad7932ae/tmp/perf-sytrs/` on the 4090 box.

## sytrf: vendor

cuSOLVER 13.2 has `cusolverDn<t>sytrf` (legacy, int32 `ipiv`, one matrix per call) and
`cusolverDnXsytrs` (int64 `ipiv`); no `Xsytrf`, no batched sytrf or sytrs. The vendor family is a
per-item loop on one stream with one shared work buffer; its `ipiv` feeds the native `sytrs`
directly (`VendorFactorSolvesWithNativeSytrs`). `Xsytrs` is not wired (int64 pivot copy, still per
item), so CUDA has no sytrs vendor family.

## sytrf: tests

`tests/sytrf_tests.cc`: every route pinned, padded `ld` and stride, pads and the unreferenced
triangle poisoned, complex with nonzero imaginary parts. n straddles 31 | 32, 32 | 33, 62 | 63,
128 | 129 and 256 | 257; generators force interchanges and 2 x 2 blocks (asserted in ipiv first).
Bit-identity guards at batch 1024 (`SaturatingBatchIsBitIdenticalToItemZero`, its sytrs twin,
n = 170 for `blocked`); `WgPivotsMatchVendor` and `BlockedMatchesWgAndVendor` check ipiv against
cuSOLVER, which a residual cannot see. `SytrsPinnedFamiliesAreReached` checks each pin's
`reached,sytrs` coverage row.

### sytrf: deliberate breaks

RTX 4090 box, 2026-10-25..27, float and cfloat CUDA (2 x 2 interchange: all eight instantiations),
every break reverted. Logs: `break-<name>.log` under the evidence directory above and
`/home/jonaslacour/.claude/jobs/ad7932ae/tmp/leftovers/sytrs-2/`.

| break | red set |
|---|---|
| trailing kernel skips column k0 + 31 | `blocked` rows, `BlockedMatchesWgAndVendor`, n = 170 saturating |
| step kernel skips `ldl_undo` | `blocked` residuals only |
| alpha = (1 + sqrt 17) / 7 | `WgPivotsMatchVendor`, vendor half of `BlockedMatchesWgAndVendor` |
| `ldl_factor` reports the last zero column | `SingularItemReportsInfo...`, `wg` rows of `SingularColumnsReportTheFirstInLoopOrder` |
| blocked tail info drops the Lower offset | `SingularColumnsReportTheFirstInLoopOrder` `blocked` Lower {95} |
| `sytrs` drops the Upper reversal / the rhs column offset | every Upper / nrhs >= 2 `sytrs` row |
| sysv `compose` or `cta` ignores the skip list | `SingularItemReportsInfo...` only |
| `ldl_solve` pivot validation off | `SytrsMalformedPivotsGiveNaN` (illegal address) |
| `cta` D^-1 drops the lane-31 partner | pinned `cta` rows, `SytrsTwoByTwoAcrossTheLaneBoundary` |
| `?syconv` walk skips swaps | pinned `cta` and sysv `cta` rows (`CtaAndWgAgreeBitForBit` green: shared walk) |
| `convert_gather` never composes sigma | pinned `blocked` rows |
| `blocked` chunk loop drops later parts' pivot offset | `SytrsBlockedChunksKeepEachItemsPivots` only |
| `build_perm` or `ldl_solve` swaps row k, not k + 1, at a 2 x 2 step | `SytrsTwoByTwoWithInterchangeEveryFamily` and most `Sytrs*` rows |
| decisions not broadcast | nothing ([termination](#sytrf-termination)) |

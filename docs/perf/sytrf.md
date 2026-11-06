# LDL^T: sytrf, sytrs, sysv {#perf_sytrf}

> **Status:** current · sytrf routes untimed; sytrs and sysv timed on the RTX 4090 box (sm_89, float and cfloat), sm_120 transcribed · CUDA 13.2 · 2026-10-26

Batched symmetric indefinite factorization by Bunch-Kaufman pivoting with LAPACK `?sytrf` semantics,
its solve `?sytrs`, and the composed `?sysv`. Consumers: KKT systems \f$\begin{bmatrix} H & A^T \\ A & 0 \end{bmatrix}\f$ (optimize)
and `dsysv` (gfn2). Complex is symmetric, not Hermitian (`csytrf`).

## sytrf: contract

* **Storage.** Only the `uplo` triangle is read and written; it receives \f$D\f$ and the multipliers of \f$L\f$
  (Lower) or \f$U\f$ (Upper) as LAPACK stores them.
* **Pivots.** LAPACK's `ipiv`, 1-based int32 packed into the first half of the int64 span, item \f$b\f$ at
  `as_span<int>()[b n + k]` (getrf's packing). `ipiv[k] > 0`: \f$1 \times 1\f$ block, \f$k\f$ and `ipiv[k] - 1`
  interchanged. Lower `ipiv[k] = ipiv[k+1] = -p`: \f$2 \times 2\f$ block at \f$(k, k+1)\f$, rows \f$k+1\f$ and \f$p-1\f$
  interchanged. Upper `ipiv[k-1] = ipiv[k] = -p`: \f$2 \times 2\f$ block at \f$(k-1, k)\f$, rows \f$k-1\f$ and \f$p-1\f$.
* **info.** Per item: 0, or \f$i > 0\f$ when \f$D(i,i)\f$ (1-based) is exactly zero, `?sytf2`'s rule (the first
  column in loop order whose `max(|a_kk|, colmax)` is zero: the zero matrix gives 1 for Lower, \f$n\f$ for
  Upper). The factorization still completes.
* **Pivot metric.** `cabs1 = |re| + |im|`, as `csytf2`.

## sysv and sytrs: contract

* `sytrs` reads \f$A\f$, `uplo` and `ipiv` as `sytrf` left them and overwrites \f$B\f$ with \f$X\f$. No singularity
  check (a singular item gives Inf or NaN). Workspace `sytrs_buffer_size()` bytes, nonzero for
  `blocked` only. A pivot list `sytrf` cannot have written (an entry 0 or outside \f$[-n, n]\f$, a \f$2 \times 2\f$
  entry without its partner) gives that item an all-NaN \f$X\f$; LAPACKE's `?sytrs` (NETLIB) does no check.
* `sysv` is `sytrf` then `sytrs`. An item with `info > 0` keeps its \f$B\f$ (LAPACK `?sysv`); every route,
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

| op | \f$n \le 32\f$ | \f$33 \le n \le 128\f$ | \f$n \ge 256\f$ |
|---|---|---|---|
| `sytrf` | `cta`, `wg`, `blocked`, `vendor` | `wg`, `blocked`, `vendor` | `blocked`, `wg`, `vendor` |
| `sysv` | `cta`, `compose` | `compose` | `compose` |

The order is a hypothesis: native first because the vendor arm is a serial per-item loop; `blocked`
first above \f$n = 128\f$ because it reads the trailing matrix once per panel, not once per column. The
128 | 256 edge is a guess. **Open:** time sytrf (`tools/tune/sytrf_spec.cc`, Lower only) with
`batchlas_tune sytrf --tier preview --dtype float,cfloat`, then `scripts/sweep_to_table.py --op sytrf`.

sysv sm_89 float and cfloat are a `preview` sweep (run `20261026T152523-JRHNBR-1179869`, 691 rows):
`compose` wins much of \f$n \le 32\f$ from \f$\mathrm{nrhs} = 16\f$ (\f$8 \times 16 \times 8192\f$: 0.133 vs 0.169 ms for `cta`). double,
cdouble and sm_120 keep the transcribed rows.

### sytrs: the tuned tables {#sytrs-selection}

`tuned/sytrs.sm_89.txt` float and cfloat: `preview` sweep of `tools/tune/sytrs_spec.cc` (A factored
once by the routed sytrf, untimed; Lower only; verified by `solve_residual`), run
`20261026T143047-JRHNBR-1000130`, batch 128..32768, 1256 cells. double, cdouble and all of sm_120
copy that order row for row (`transcribed-sytrs-d67ad754`; FP64 is not tuned on 1:64 parts), and
`can_run` drops `cta` where cdouble's factor does not fit local memory. Regions as in
[sytrs: measured](#sytrs-measured).

## sytrf: the reversal map

The kernels implement `?sytf2` and `?sytrs` once, in lower form (`include/batchlas/device/lapack/sytrf.hh`). Upper runs the
same code on the index-reversed matrix, logical \f$i\f$ = physical \f$n - 1 - i\f$: `?sytf2`'s upper loop becomes its
lower loop step for step, the factor lands in the upper triangle and `ipiv` is written at the physical
index. The one asymmetry is `I?AMAX`'s tie-break (first maximum in storage order): Upper takes the
largest logical index on a tie. The argmax reductions are a sub-group butterfly plus sub-groups in
fixed order, deterministic and uniform across lanes (NaN magnitude folded to +inf).

## sytrf: the blocked route

LAPACK `?sytrf`'s structure in lower form: `?lasyf` on a panel, the trailing update
\f$A_{22} \mathrel{-}= L_{21} W_{21}^T\f$, the next panel, `?sytf2` on the last. Per panel
(`sytrf_kernels.cc`):

1. **Step kernel** (one work-group per matrix, `ldl_panel`): the previous panel's `ldl_undo`
   (`?lasyf`'s closing loop), then the panel, each column formed in \f$W\f$ and pivoted as in `?sytf2`.
2. **Trailing kernel**: the lower triangle of each 128-wide diagonal block.
3. **Public `gemm`** (`NoTrans`, `Trans`, \f$\alpha = -1, \beta = 1\f$, complex not conjugated) for the
   rest; table-routed, native in the vendor-free build.

**Uniform panels.** A batch shares one set of gemm sub-views, but a Bunch-Kaufman panel ends per
item. Every panel is the range \f$[k_0, k_0 + 31)\f$ and attempts its last column, so an item ends at \f$k_0 + 31\f$
or \f$k_0 + 32\f$ (a \f$2 \times 2\f$ block); \f$W\f$ has 32 columns, uncomputed ones zero, and the gemm's \f$k\f$ is 32. An item
that ended late starts the next panel one column late (`state` = {first, end}). Pivots equal
`?sytf2`'s in exact arithmetic, so ipiv matches `wg` and cuSOLVER on data without near ties
(`BlockedMatchesWgAndVendor`). Caveat: a non-finite entry in a not-yet-factored column \f$k_0 + 31\f$ meets
a zero \f$W\f$ column (\f$\infty \times 0 = \mathrm{NaN}\f$); such an item's factor is non-finite anyway.

**Constants** (`sytrf_native.hh`, untimed): `kSytrfBlockedNb` = 31, diagonal blocks of 128, batch
chunked by getrf's L2 rule (`tuning::blocked_chunk_items`). Workspace (\f$W\f$, state, info fallback, gemm
pointer arrays) through `BumpAllocator::measuring()`.

## sytrf: termination

Every device loop is bounded by \f$n\f$ (\f$k\f$ advances by 1 or 2). The one way to hang is a barrier some lanes
skip: per-lane pivot decisions split a work-group when lanes read different bits (a NaN compare, or
items aliased by a stride shorter than the matrix). The decisions are lane 0's, broadcast
(`Group::bcast`, `pivot_stage`/`pivot_choice`), so the barrier count is uniform whatever the data.

**Known gap:** `NonFiniteAndAliasedItemsTerminate` (NaN, Inf, all-NaN and aliased items on cta, wg,
blocked) provokes no lane divergence on the 4090 box, so removing the broadcast leaves it green; no
input that hung the old kernels was kept. `SytrsMalformedPivotsGiveNaN` guards the ipiv check. The
ctest entries carry `TIMEOUT 1800`; the suite is heavy-tier.

## sytrs: the converted solve {#sytrs-the-converted-solve}

`wg` is LAPACK `?sytrs` (interchanges one pivot at a time between eliminations). `cta` and `blocked`
are `?sytrs2` (`include/batchlas/device/lapack/sytrs.hh`): the factor is first **converted** (`?syconv`: each interchange
applied to the rows of the earlier columns of \f$L\f$, \f$2 \times 2\f$ off-diagonals moved out of \f$L\f$), giving
\f$A = P L D L^T P^T\f$ with one permutation \f$P\f$. The solve is gather \f$y = P^T b\f$, \f$L w = y\f$,
\f$w := D^{-1} w\f$, \f$L^T z = w\f$, scatter \f$x = P z\f$, in logical lower form
([reversal map](#sytrf-the-reversal-map)). The pivot list is validated once per item first.

* **`cta`**: one work-group per item; the lower triangle is staged and converted in local memory
  (`ld = n | 1`). Each sub-group owns right-hand sides \f$\mathrm{sgid}, \mathrm{sgid} + \mathrm{nsg}, \dots\f$ (\f$\mathrm{nsg} = \min(\mathrm{nrhs}, 8)\f$); lane
  \f$i\f$ holds rows \f$i, i + 32, \dots\f$ in registers (at most 4, so \f$n \le 128\f$); both sweeps are broadcast + axpy
  (no reduction, no barrier); \f$D^{-1}\f$ reads a \f$2 \times 2\f$ partner across the lane-31 / lane-0 seam. cdouble
  stops near \f$n = 76\f$ on sm_89 (local memory).
* **`blocked`**: per chunk of items, a prep kernel writes the converted factor (`convert_gather`,
  through the composed permutation) and the gathered \f$B\f$ to workspace; then `trsm` (`Left`, `NoTrans`,
  `Unit`), a \f$D^{-1}\f$ kernel, `trsm` with `Trans` (not conjugated), and the scatter. The chunk is getrf's
  L2 rule capped by half the scratch lease and the grid (`blocked_chunk`). Any \f$n\f$, any \f$\mathrm{nrhs}\f$.
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

* `cta` wins \f$n \le 64\f$ at \f$\mathrm{nrhs} \ge 4\f$ and \f$\mathrm{nrhs} = 1\f$ up to \f$n = 32\f$; at \f$n = 128\f$ its staged factor fills an SM.
* `blocked` wins from \f$n = 128\f$ at \f$\mathrm{nrhs} \ge 4\f$, and at every \f$n\f$ for \f$\mathrm{nrhs} \ge 16\f$, \f$\mathrm{batch} \ge 1024\f$.
  `convert_gather` beat one swap per pivot step 1.0-2.3x from \f$n = 64\f$.
* `wg` stays for \f$\mathrm{nrhs} = 1\f$ from \f$n = 64\f$ and \f$\mathrm{nrhs} \le 4\f$ from \f$n = 256\f$.
* sysv `cta` (one sub-group per rhs) vs the old serial-rhs fused kernel: 1.0-1.1x at \f$\mathrm{nrhs} = 1\f$,
  3.5-11.8x at \f$\mathrm{nrhs} = 64\f$.
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
triangle poisoned, complex with nonzero imaginary parts. \f$n\f$ straddles 31 | 32, 32 | 33, 62 | 63,
128 | 129 and 256 | 257; generators force interchanges and \f$2 \times 2\f$ blocks (asserted in ipiv first).
Bit-identity guards at batch 1024 (`SaturatingBatchIsBitIdenticalToItemZero`, its sytrs twin,
\f$n = 170\f$ for `blocked`); `WgPivotsMatchVendor` and `BlockedMatchesWgAndVendor` check ipiv against
cuSOLVER, which a residual cannot see. `SytrsPinnedFamiliesAreReached` checks each pin's
`reached,sytrs` coverage row.

### sytrf: deliberate breaks

RTX 4090 box, 2026-10-25..27, float and cfloat CUDA (\f$2 \times 2\f$ interchange: all eight instantiations),
every break reverted. Logs: `break-<name>.log` under the evidence directory above and
`/home/jonaslacour/.claude/jobs/ad7932ae/tmp/leftovers/sytrs-2/`.

| break | red set |
|---|---|
| trailing kernel skips column \f$k_0 + 31\f$ | `blocked` rows, `BlockedMatchesWgAndVendor`, \f$n = 170\f$ saturating |
| step kernel skips `ldl_undo` | `blocked` residuals only |
| \f$\alpha = (1 + \sqrt{17}) / 7\f$ | `WgPivotsMatchVendor`, vendor half of `BlockedMatchesWgAndVendor` |
| `ldl_factor` reports the last zero column | `SingularItemReportsInfo...`, `wg` rows of `SingularColumnsReportTheFirstInLoopOrder` |
| blocked tail info drops the Lower offset | `SingularColumnsReportTheFirstInLoopOrder` `blocked` Lower {95} |
| `sytrs` drops the Upper reversal / the rhs column offset | every Upper / \f$\mathrm{nrhs} \ge 2\f$ `sytrs` row |
| sysv `compose` or `cta` ignores the skip list | `SingularItemReportsInfo...` only |
| `ldl_solve` pivot validation off | `SytrsMalformedPivotsGiveNaN` (illegal address) |
| `cta` \f$D^{-1}\f$ drops the lane-31 partner | pinned `cta` rows, `SytrsTwoByTwoAcrossTheLaneBoundary` |
| `?syconv` walk skips swaps | pinned `cta` and sysv `cta` rows (`CtaAndWgAgreeBitForBit` green: shared walk) |
| `convert_gather` never composes \f$\sigma\f$ | pinned `blocked` rows |
| `blocked` chunk loop drops later parts' pivot offset | `SytrsBlockedChunksKeepEachItemsPivots` only |
| `build_perm` or `ldl_solve` swaps row \f$k\f$, not \f$k + 1\f$, at a \f$2 \times 2\f$ step | `SytrsTwoByTwoWithInterchangeEveryFamily` and most `Sytrs*` rows |
| decisions not broadcast | nothing ([termination](#sytrf-termination)) |

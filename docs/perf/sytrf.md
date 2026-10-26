# LDL^T: sytrf, sytrs, sysv {#perf_sytrf}

> **Status:** current · sytrf routes untimed; sytrs and sysv timed on the RTX 4090 box (sm_89, float and cfloat), sm_120 transcribed · CUDA 13.2 · 2026-10-26

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
  singularity check: a singular item gives infinities or NaNs. It takes a workspace of
  `sytrs_buffer_size()` bytes (nonzero for the `blocked` route only).
* `sysv` is `sytrf` then `sytrs`. An item with `info > 0` keeps its B (LAPACK `?sysv`); the other
  items are solved. Every route honours this, the fused `cta` kernel included.

## Selection

| op | family | driver | `can_run` |
|---|---|---|---|
| `sytrf` | `cta` | `sytrf_kernels.cc` `cta_launch`: one sub-group per matrix, logical lower triangle in local memory | native term, `n <= 32` (`sytrf.cc`, `can_run`) |
| `sytrf` | `wg` | `sytrf_kernels.cc` `sytrf_wg_dispatch`: one work-group (32..256) per matrix, unblocked `?sytf2` in place in global memory | native term |
| `sytrf` | `blocked` | `sytrf_kernels.cc` `sytrf_blocked_dispatch`: `?lasyf` panels of 31 columns (one work-group per matrix) and the trailing update through the public `gemm`; `?sytf2` on the last panel | native term (the gemm selects for itself) |
| `sytrf` | `vendor` | `cusolverDn<t>sytrf` per item (`cusolver_sytrf.cc`), LAPACKE `?sytrf` (`netlib_lapack.cc`) | CUDA with cuSOLVER or NETLIB with LAPACKE, `d.has_vendor`, homogeneous batch. rocSOLVER is not wired |
| `sytrs` | `cta` | `sytrs_kernels.cc` `sytrs_cta_dispatch`: [the converted solve](#sytrs-the-converted-solve), factor in local memory, one sub-group per right-hand side | native term, `n <= 128`, `sytrs_cta_slm_bytes<T>(n) <= slm_budget` (`sytrs.hh`, `can_run`) |
| `sytrs` | `blocked` | `sytrs_blocked_dispatch`: converted factor and gathered B in workspace per chunk, two public `trsm` around a D^-1 kernel | native term (the trsm selects for itself) |
| `sytrs` | `wg` | `sytrf_kernels.cc` `sytrs_native_dispatch`: one work-group per right-hand side, `?sytrs` step by step (the first sytrs) | native term |
| `sytrs` | `vendor` | LAPACKE `?sytrs` per item (`netlib_lapack.cc`) | NETLIB with LAPACKE, homogeneous batch |
| `sysv` | `cta` | `sysv_cta_dispatch`: the `cta` factor, then sytrs `cta`'s solve with one sub-group per right-hand side, one launch | GPU, sub-group 32, homogeneous, `1 <= n <= 32` |
| `sysv` | `compose` | public `sytrf` (routed), then the routed `sytrs` with the per-item status as a skip list | homogeneous batch |

* **Native term:** `B != NETLIB && d.is_gpu && d.has_sg32 && max_wg >= 32`, homogeneous batch,
  `n, batch >= 1` (sytrs: `nrhs >= 1`, A and B homogeneous).
* **sytrs** is `Op::sytrs`, `src/ops/sytrs/` (`choice.hh`, `sytrs.hh` shared with sysv's compose,
  `sytrs.cc`); key `n:log:2 nrhs:log batch:log`. cuSOLVER's `Xsytrs` is not wired (int64 pivots,
  per item), so CUDA has no sytrs vendor family.
* **Last resort** (no table): sytrf `wg`, then `vendor`; sytrs `blocked`, then `vendor`; sysv `compose`.
* **Pins:** `BATCHLAS_SYTRF_ROUTE` (`cta`, `wg`, `blocked`, `vendor`, `native`, `auto`),
  `BATCHLAS_SYTRS_ROUTE` (`cta`, `blocked`, `wg`, `vendor`, `native`, `auto`),
  `BATCHLAS_SYSV_ROUTE` (`cta`, `compose`). A sytrf `cta` pin above n = 32 and a sytrs `cta` pin
  above n = 128 (or past local memory) throw.

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

**sysv sm_89, re-swept 2026-10-26** (its `cta` solve and its `compose` child changed with sytrs):
float and cfloat `preview`, run `20261026T152523-JRHNBR-1179869`, 691 measured rows (first entry
`cta` 94 / 165, `compose` 185 / 247 for float / cfloat); `compose` wins much of n <= 32 too from
nrhs = 16 (8 x 16 x 8192: 0.133 vs 0.169 ms), since its sytrs is now the routed `cta` or `blocked`.
double, cdouble and sm_120 keep the transcribed rows above.

### sytrs: the tuned tables {#sytrs-selection}

`tuned/sytrs.sm_89.txt` float and cfloat are a measured `preview` sweep (`tools/tune/sytrs_spec.cc`:
random symmetric A factored once by the routed sytrf, untimed; Lower only, as sytrf; every family
verified by `solve_residual` against A), run `20261026T143047-JRHNBR-1000130`, GPU 1 of the 4090 box,
`--batch-ladder off` (batch 128 .. 32768), 1256 cells with refinement. double and cdouble
(sm_89) and all four dtypes on sm_120 are run `transcribed-sytrs-d67ad754`: the measured float /
cfloat order copied row for row (FP64 is not tuned on 1:64 parts; no sm_120 box measured it), and
`can_run` drops `cta` where cdouble's factor does not fit local memory. Measured first entries
(float / cfloat, of 614 / 642 rows): `cta` 341 / 325, `blocked` 196 / 211, `wg` 77 / 106. The same
regions as the benchmark below: `cta` at small n, `blocked` at large n x nrhs, `wg` for one or a few
right-hand sides at n >= 64.

```sh
build/tools/tune/batchlas_tune sytrs --tier preview --dtype float,cfloat --devices 1 --batch-ladder off
python3 scripts/sweep_to_table.py --op sytrs --device sm_89
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

## sytrs: the converted solve {#sytrs-the-converted-solve}

The first `sytrs` (now the `wg` family) was LAPACK `?sytrs`: one work-group per right-hand side,
the interchanges applied one pivot at a time between the eliminations, 2-3 barriers per step and a
work-group sum per column of the backward pass, L re-read once per right-hand side. `cta` and
`blocked` are LAPACK `?sytrs2` instead (`sytrs_device.hh`): the factor is first **converted**
(`?syconv`): each step's interchange is applied to the rows of the *earlier* columns of L, in step
order (a 2 x 2 step at k swaps row k + 1 and touches columns < k), and the 2 x 2 off-diagonals move
out of L. Then \f$A = P L D L^T P^T\f$ with L unit lower and one permutation P, so the solve is a
gather \f$y = P^T b\f$, \f$L w = y\f$, \f$w := D^{-1} w\f$, \f$L^T z = w\f$ and a scatter
\f$x = P z\f$. Both run in the logical lower form; Upper through [the reversal map](#sytrf-the-reversal-map).
The pivot list is validated once per item first, as before: a list `sytrf` cannot have written gives
that item an all-NaN X.

* **`cta`** (`sytrs_kernels.cc`, `cta_launch`): one work-group per item. The logical lower triangle
  is staged once in local memory (ld = n | 1) and converted there (one lane per column walks the
  pivot list; no barrier inside the walk); lane 0 composes P and the block kinds. Then each
  sub-group owns right-hand sides sgid, sgid + nsg, ... (nsg = min(nrhs, 8)): lane i holds logical
  rows i, i + 32, ... in registers (R = ceil(n / 32) <= 4, so n <= 128), the gather and the scatter
  use the same physical row per lane, both sweeps are broadcast form (`group_broadcast` of row k,
  then an axpy over the lane's rows: no reduction, no barrier), and D^-1 reads a 2 x 2 partner by a
  shift, across the lane-31 / lane-0 seam between register slots. `can_run`: n <= 128 and the
  staged factor fits local memory (complex<double> stops near n = 76 on sm_89).
* **`blocked`** (`sytrs_blocked_dispatch`): per chunk of items, a prep kernel (one work-group per
  item) writes the converted factor into workspace (n x n per item, the uplo triangle only, diagonal
  unused) and the gathered B (n x nrhs); then the public `trsm` (`Left`, uplo, `NoTrans`, `Unit`), a
  D^-1 kernel over (row, rhs, item), the public `trsm` with `Trans` (complex is not conjugated), and
  the scatter back to B. The chunk is getrf's L2 rule on the per-item workspace, capped by half the
  scratch lease and by the grid (`blocked_chunk`), so the workspace is per chunk, never
  n^2 x batch. Any n, any nrhs; the vendor-free build serves it with the native trsm.
* **sysv `cta`**: the fused factor is sub-group 0's (sub-group barriers and broadcasts only, so the
  factor is bit-identical to sytrf `cta`); the work-group then has one sub-group per right-hand
  side (at most 8) and solves with exactly `cta`'s converted solve, in parallel.
* **skip list**: every family keeps the B of an item whose entry is nonzero (sysv after a singular
  factor); sysv `compose` now calls the routed sytrs with its status as that list.

## sytrs: measured {#sytrs-measured}

`benchmarks/sytrs_benchmark` (`BM_SYTRS`, `BM_SYSV`; Args n, nrhs, batch; the family is the route
pin), RTX 4090 (GPU 1, idle, `CUDA_VISIBLE_DEVICES=1`), 2026-10-26, `--warmup=3 --min_iters=8`, two
passes in reversed family order, median. Time per call in ms; `wg` is the first sytrs, unchanged.

| dtype | n | nrhs | batch | `wg` (old) | `cta` | `blocked` | best / old |
|---|---|---|---|---|---|---|---|
| float | 8 | 16 | 8192 | 0.297 | 0.065 | 0.067 | 4.6x |
| float | 32 | 4 | 8192 | 0.303 | 0.083 | 0.397 | 3.6x |
| float | 32 | 64 | 8192 | 4.35 | 0.486 | 1.01 | 9.0x |
| float | 64 | 16 | 8192 | 3.35 | 0.657 | 1.10 | 5.1x |
| float | 64 | 1 | 8192 | 0.243 | 0.602 | 0.726 | 0.40x (old kept) |
| float | 128 | 1 | 8192 | 0.869 | 10.2 | 3.20 | 0.27x (old kept) |
| float | 128 | 16 | 8192 | 11.2 | 5.87 | 3.87 | 2.9x |
| float | 128 | 64 | 8192 | 44.2 | 19.8 | 7.09 | 6.2x |
| float | 512 | 64 | 1024 | 56.5 | - | 12.4 | 4.6x |
| cfloat | 32 | 64 | 1024 | 1.29 | 0.135 | 0.377 | 9.6x |
| cfloat | 64 | 64 | 8192 | 19.8 | 3.16 | 6.91 | 6.3x |
| cfloat | 128 | 64 | 8192 | 76.1 | - | 17.3 | 4.4x |
| cfloat | 512 | 16 | 1024 | 25.0 | - | 12.7 | 2.0x |

* **`cta`** is the best family for n <= 64 at nrhs >= 4 (2-10x over `wg`) and at nrhs = 1 up to
  n = 32 (1.1-1.7x). Its local memory is the staged factor, so at n = 128 one work-group fills an SM
  (66 KB float): it loses there except at batch 128 or nrhs >= 16.
* **`blocked`** wins from n = 128 at nrhs >= 4 and at every n for nrhs >= 16 with batch >= 1024 at
  small n (8 x 64 x 8192: 0.127 vs 0.168 ms). Its first version converted by one swap per pivot
  step in global memory (n^2 / 2 strided swaps per item); converting by a gather through the
  composed permutation (`convert_gather`) made it 1.0-2.3x faster from n = 64 (median 1.55x) and
  left it unchanged below (`blocked-swap-vs-gather.txt` in the evidence below).
* **`wg` stays** where it wins: nrhs = 1 from n = 64 (batch >= 1024) and nrhs <= 4 from n = 256. A
  work-group per right-hand side has the memory parallelism a single-rhs solve needs there.
* **Measured dead end, `sg`**: one sub-group per (item, rhs) on the factor as stored (no conversion,
  no local memory, next column prefetched). Best in 12 of 162 cells, by at most 1.35x over `wg`
  (32 x 1 x 8192, 64 x 1 x 1024) plus cells nothing else ran; not kept.
* **sysv `cta`** (fused factor plus the converted solve, one sub-group per rhs) against the old
  fused kernel (one sub-group, rhs one after another, `?sytrs` step by step): 1.0-1.1x at nrhs = 1,
  1.6-2.0x at nrhs = 4, 1.3-4.3x at 16 and 3.5-11.8x at 64 (float and cfloat, n = 8..32, batch 128..8192).
* **Vendor**: cuSOLVER has no batched sytrs and `Xsytrs` is not wired (int64 pivots, per item), so
  there is no vendor arm to measure on CUDA.

Evidence: `/home/jonaslacour/.claude/jobs/ad7932ae/tmp/perf-sytrs/` (`families-2pass.txt`,
`sysv-cta-old-vs-new.txt`, `blocked-swap-vs-gather.txt`, the raw CSVs) on the 4090 box.

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
* **sytrs**, every family pinned (`solve_routes`: cta where it fits, sg to n = 256, blocked, wg;
  LAPACKE on NETLIB): `SytrsEveryFamily` (random and KKT factors, both uplo, n straddling 32 | 33,
  64 | 65, 128 | 129, 256 | 257, nrhs 1, 8, 9: the per-item sub-group count 8 | 9),
  `SytrsWideRightHandSides` (nrhs = 300), `SytrsTwoByTwoAcrossTheLaneBoundary` (a 2 x 2 block forced
  at logical 31 | 32, 63 | 64, ... on dominant data, asserted present first),
  `SytrsTwoByTwoWithInterchangeEveryFamily` (2 x 2 steps forced at (1, n - 1 - item) and
  (n / 2, n / 2 + 3), n = 8, 33, 65, 129: row k + 1 interchanged with a row p != k + 1, the partner
  asserted in ipiv first; `SytrsEveryFamily` likewise asserts a 2 x 2 block in every KKT item),
  `SytrsSaturatingBatchIsBitIdenticalToItemZero` (batch 1024, heavy tier: cta n = 32 and 100, sg
  n = 200, blocked n = 150, wg n = 40), `SytrsCeilingsStraddled` (pins at 128 | 129 and 256 | 257,
  Auto on both sides), `SytrsSkipListEveryFamily` (sysv compose with each sytrs family pinned: the
  singular item keeps B bit for bit), `SytrsBlockedChunksKeepEachItemsPivots` (chunk forced to 7 of 23
  items, in and out of order, a malformed list in the last part, sysv's skip in the second),
  `SytrsMalformedPivotsGiveNaN` (every family) and `SytrsPinnedFamiliesAreReached` (each pin leaves its
  own `reached,sytrs` coverage row with the call's key).

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

sytrs `cta` / `blocked` and sysv `cta` (2026-10-26, float and cfloat CUDA, the shipped tables, every
break reverted; "pinned rows" are the `Sytrs*` rows naming a family, "Auto" the sytrf and sysv tests
whose solve the table sends to that family):

| break | red set |
|---|---|
| `cta` D^-1 drops the lane-31 partner from the next register slot | pinned `cta` rows of `SytrsEveryFamily`, `SytrsTwoByTwoAcrossTheLaneBoundary`, `SytrsWideRightHandSides`, `SytrsSaturating...` (float), `SytrsCtaCeilingStraddled`; Auto: `FactorThenSolveEveryRoute`, `SysvEveryRoute`, `BlockedMatchesWgAndVendor` |
| `?syconv` column walk skips its swaps (`cta`, sysv `cta`) | every pinned `cta` row with interchanges (`SytrsEveryFamily`, `SytrsMalformed...`, `SytrsSkipList...`, `SytrsWide...`, `SytrsSaturating...`, `SytrsCtaCeiling...`), the sysv `cta` rows of `SysvEveryRoute` and `SaturatingBatch...`; Auto as above plus `VendorFactorSolvesWithNativeSytrs`. `CtaAndWgAgreeBitForBit` stays green: sysv `cta` and sytrs `cta` share the walk, so both are wrong alike |
| `cta` gathers B by t instead of perm[t] | the same set as the column-walk break |
| `cta` ignores the skip list | `SytrsSkipListEveryFamily` (`cta`), `SingularItemReportsInfo...` (`compose`) only |
| sysv `cta` solves a singular item | `SingularItemReportsInfo...` only |
| `convert_gather` never composes sigma (`blocked`) | pinned `blocked` rows of `SytrsEveryFamily`, `SytrsMalformedPivotsGiveNaN`, `SytrsSkipList...`, `SytrsWide...`, `SytrsSaturating...`, `SytrsBlockedChunks...`; Auto: cfloat `SaturatingBatchIsBitIdentical...` (n = 170) |
| `blocked` D^-1 does not write the second row of a 2 x 2 block | the same plus `SytrsTwoByTwoAcrossTheLaneBoundary` (`blocked`) |
| `blocked` scatter ignores P | the same set as the sigma break |
| `blocked` chunk loop drops the pivot offset of parts after the first | `SytrsBlockedChunksKeepEachItemsPivots` only |
| `blocked` scatter ignores the skip list | `SytrsSkipListEveryFamily` (`blocked`), `SytrsBlockedChunks...` only |

The 2 x 2 interchange (2026-10-26, all eight typed instantiations, every break reverted). `cta` and
`blocked` gather B through `build_perm`, `wg` swaps in `ldl_solve`; no break is narrow, because KKT
factors also interchange their 2 x 2 rows:

| break | red set |
|---|---|
| `build_perm` swaps row k instead of k + 1 at a 2 x 2 step | `SytrsTwoByTwoWithInterchangeEveryFamily` (`cta`, `blocked`), `SytrsTwoByTwoAcrossTheLaneBoundary`, `SytrsEveryFamily`, `SytrsWide...`, `SytrsSaturating...`, `SytrsSkipList...`, `SytrsBlockedChunks...`, `SytrsMalformed...`, `SytrsCtaCeilingStraddled` (2 of 8); Auto: `FactorThenSolveEveryRoute`, `SysvEveryRoute`, `BlockedMatchesWgAndVendor`, `SaturatingBatch...`, `VendorFactorSolvesWithNativeSytrs`; `SytrsPinnedFamiliesAreReached` green |
| `build_perm` drops the 2 x 2 interchange (p := k + 1) | the same set less `SytrsTwoByTwoAcrossTheLaneBoundary`, whose forced blocks have p = k + 1 |
| `ldl_solve` (`wg`) swaps row k instead of k + 1 | `SytrsTwoByTwoWithInterchangeEveryFamily`, `SytrsEveryFamily`, `SytrsMalformed...`, `SytrsSaturating...`, `SytrsSkipList...` (`wg` rows), `SytrsCtaCeilingStraddled` (Auto at 129); Auto: `FactorThenSolveEveryRoute`, `SysvEveryRoute` (`compose`), `BlockedMatchesWgAndVendor`, `SaturatingBatch...` (6 of 8); `SytrsTwoByTwoAcrossTheLaneBoundary` green |

Logs: `break-<name>.log` in the evidence directory under [sytrs: measured](#sytrs-measured); the
2 x 2 interchange breaks in `/home/jonaslacour/.claude/jobs/ad7932ae/tmp/leftovers/sytrs-2/`
(`break-swap-row-k.log`, `break-drop-2x2-interchange.log`, `break-wg-swap-row-k.log`).

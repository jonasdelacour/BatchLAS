# sygst and sygvd {#perf_sygvd}

> **Status:** current · sygst measured on sm_89 (float, cfloat; 2026-10-26); sygvd routes and tables are still untimed hypotheses (RTX 4090 sm_89, CUDA 13.2)

`sygvd` solves the generalized Hermitian-definite eigenproblem LAPACK `?sygvd` / `?hegvd` solves, for a
batch. `sygst` is its reduction step, also public. The consumer is gfn2-xTB's SCC loop: every iteration
solves \f$ H C = S C \varepsilon \f$ (itype 1, `jobz = V`, `uplo = U`) per molecule, for n of about
240 to 1200, with the overlap S fixed per geometry.

## The API {#sygvd-api}

- **`sygst(A, B, itype, uplo, ws)`**: the `uplo` triangle of `B` holds `potrf`'s factor. itype 1 gives
  \f$ L^{-1} A L^{-H} \f$ (Lower) or \f$ U^{-H} A U^{-1} \f$ (Upper); itypes 2 and 3 give
  \f$ L^H A L \f$ or \f$ U A U^H \f$. Only the `uplo` triangle of the result is defined; `B` is read-only.
  `hegst` is the same call.
- **`sygvd(A, B, W, itype, jobz, uplo, ws, info)`**: LAPACK's three problem types. itype 1 is
  \f$ A x = \lambda B x \f$, itype 2 is \f$ A B x = \lambda x \f$, and itype 3 is \f$ B A x = \lambda x \f$.
  `A` gets the eigenvectors, normalized \f$ X^H B X = I \f$ (itype 3: \f$ X^H B^{-1} X = I \f$).
  `B` gets its Cholesky factor. `hegvd` is the same call.
- **`sygvd_factored(A, F, W, itype, jobz, uplo, ws, info)`**: the S-fixed case. Factor `B` once with
  `potrf` and call this every iteration. `F` is read and never written. It always composes, because no
  vendor library takes a factored B.
- **`hegst`, `hegvd`** are the complex spellings, with the same backend-deducing overloads
  (`hegvd(ctx, ...)`).
- **`info`** follows LAPACK. 0 is success. i in 1..n means syev did not converge. n + i means the leading
  minor of order i of `B` is not positive definite. For that item `A`, `B` and `W` are then unspecified;
  the other items are unaffected (bit-identical to a batch without the failure, per family). On
  `compose` the failed item's `B` triangle is replaced by the identity before `sygst`, so its eigenvalues
  are those of `A` and finite. A partial factor (NaN from a negative pivot) would otherwise flow into
  `syev`. `sygvd_factored` reports only syev's status: potrf's own `info` is where a bad `B` shows.


## Routes {#sygvd-routes}

`homogeneous` below means one shape, `n >= 1` and no batch mask (`can_run` in `src/ops/sygst/sygst.cc`).

| op | family | what runs | `can_run` |
|---|---|---|---|
| sygst | `blocked:nb` (32, 64, 128, 256) | LAPACK's blocked `?sygst`/`?hegst` ([the blocked family](#sygst-the-blocked-family)) | homogeneous, GPU, compiled `nb`, `max_wg >= 128`, the diagonal kernel's SLM fits |
| sygst | `twosided` | `A.hermitize(uplo)`, then two public `trsm` (itype 1) or two public `trmm` through an `n x n` workspace product (itypes 2, 3) | homogeneous |
| sygst | `vendor` | LAPACKE `?sygst` / `?hegst` per item | vendor linked; homogeneous. cuSOLVER 13.2 has no public `?sygst` |
| sygvd | `compose` | `potrf(B)` (skipped by `sygvd_factored`), the failed-item identity, `sygst`, `syev`, then a `trsm` back-transform (itypes 1, 2) or `trmm` plus a copy (itype 3), then the info merge | homogeneous |
| sygvd | `vendor` | cuSOLVER `cusolverDn?sygvd` per item on one stream with one shared work buffer (`cusolver_sygvd.cc`); LAPACKE `?sygvd` per item on the host | vendor linked; homogeneous; not `sygvd_factored` |

Every child of `blocked`, `twosided` and `compose` is a public routed op, so `BATCHLAS_SYEV_ROUTE`,
`BATCHLAS_TRSM_ROUTE` and the rest apply inside them, and the vendor-free build serves `compose`
wherever its children are native. rocSOLVER's batched `?sygvd` is not wired, so ROCm runs `compose`.

Tables: keys `dtype`, `n:log:3`, `batch:log` (sygvd adds `jobz:exact` first). `tuned/sygst.sm_89.txt`
float and cfloat rows are measured, the rest transcribed ([sygst: measured](#sygst-measured)).
`tuned/sygvd.{sm_89,sm_120}.txt` rank `compose` first in every row, `transcribed` (run
`transcribed-ab235065`). The CPU has no table; the last-resort order puts `vendor` first.

> **Warning:** `compose` before `vendor` on the GPU (sygvd) is an untimed hypothesis from the shape of
> the vendor call (a per-item loop at batch >= 128). The vendor may win at small batch and large n.

## sygst: the blocked family {#sygst-the-blocked-family}

`blocked:nb=<32|64|128|256>` is LAPACK's `dsygst` / `zhegst` (`src/ops/sygst/kernels/sygst_blocked.cc`, `blocked_range`).
It works in place on the `uplo` triangle of `A` and never hermitizes it: every read goes through the
named triangles, so the other triangle, `ld` pads and inter-item gaps keep their bits. Per block of
`nb` columns (`kb` the block's order, `m` the trailing order):

- **itype 1** (\f$ L^{-1} A L^{-H} \f$; Upper is the transposed schedule): the diagonal kernel reduces
  \f$ A_{11} \f$ and writes the Hermitian \f$ C_{11} \f$ to workspace `H`; `trsm` (Right,
  \f$ L_{11}^{-H} \f$) on \f$ P = A_{21} \f$; `gemm` \f$ P \mathrel{-}= \tfrac12 L_{21} H \f$;
  `her2k` / `syr2k` on \f$ A_{22} \f$; the same `gemm`; `trsm` (Left, \f$ L_{22} \f$) on the panel.
- **itypes 2 and 3** (\f$ L^H A L \f$): a mirror kernel copies \f$ A_{22} \f$ into `H`; `trmm` writes
  \f$ W = A_{21} L_{11} \f$ out of place (n x nb workspace); `gemm` \f$ W \mathrel{+}= \tfrac12 H L_{21} \f$,
  `her2k`, the `gemm` again; `trmm` writes \f$ A_{21} = L_{22}^H W \f$; the diagonal kernel reduces
  \f$ A_{22} \f$.

LAPACK's `hemm` / `symm` became `gemm` against `H`: the `symm` table sends the tall float shape to
cuBLAS's per-item loop (float n = 128, batch 256: 2.23 ms against 0.44 ms with `gemm`). A diagonal
block wider than 32 recurses at a quarter of its order (at least 32). `nb` is a family field because
the best block grows with n.

**The diagonal kernel** (`src/ops/sygst/kernels/sygst_diag.cc`) gives each column of a kb <= 32 block one
work-item with the column in registers; `F` and a transpose buffer sit in shared memory, read as
16-byte packs. Up to 128 columns share a work-group (several items at kb < 32); orders 32, 16 and 8
are compile-time instances (float n = 32, batch 8192: 0.20 ms, `twosided` 0.21 ms).

**Chunking:** potrf's L2 chunk (`blocked_chunk`, `BLOCKED_L2_CHUNK_PERCENT`), but never fewer than
4 x CU items per pass (`kBlockedMinChunkCus`, [measured](#sygst-measured)).

`twosided` stays as the second native family: the measured tables rank it first in about 20% of
float and cfloat itype-1 rows (small n, or batch 128) and in 128 of 292 float itype 2/3 rows, where
two `trmm` run at gemm speed and `blocked`'s many small child calls do not pay.

## sygst: measured {#sygst-measured}

RTX 4090 (sm_89), CUDA 13.2, 2026-10-27. Tables from `batchlas_tune sygst --tier deep --dtype float`
(run `20261026T230652-JRHNBR-1807408`) and `--dtype cfloat` (`20261026T230955-JRHNBR-1810656`), over
itype 1, 2, 3 x uplo L, U, cold protocol, then `scripts/sweep_to_table.py`. itype and uplo are keys
because the itype-1-only table sent itype 2/3 cells to `blocked` where `twosided` was 1.2x faster.

Auto against `twosided` (`sygst_benchmark` warm, 4 alternating reps, spread <= 8.8%), twosided / Auto:

| dtype | itype | n 64 b 1024 | n 256 b 1024 | n 480 b 1024 | n 1024 b 128 | n 1024 b 1024 |
|---|---|---:|---:|---:|---:|---:|
| float | 1 | 1.00x ts | 1.29-1.33x b64 | 1.62-1.76x b128 | 1.44-1.68x b256 | 2.27-2.87x b256 |
| float | 2, 3 | 1.00x ts | 1.00x ts | 0.91-1.09x b128 | 1.25-1.42x | 2.03-2.30x |
| cfloat | 1 | 1.44-1.49x b32 | 1.63x b64 | 1.58-1.62x b128 | 1.65-1.78x b256 | - (no fit) |
| cfloat | 2, 3 | 1.17-1.22x b32 | 1.08-1.15x | 1.40-1.52x | 1.74-1.87x b256 | - |

Remaining loss: float itypes 2 and 3 Lower at n = 480, 0.90-0.91x warm; cold the tuner measured a tie
(batch 128: 1.776 against 1.799 ms), and the table keeps its first entry. `blocked` loses itypes 2/3 at
small n because it issues dozens of small child calls per pass (nsys, float n = 256 batch 1024 itype 2:
1.11 ms in 60 `sgemm` launches, 0.74 `syr2k`, 0.50 diagonal, 3.28 ms total against `twosided`'s 2.69);
no single child is cheap to remove.

**The chunk floor.** potrf's chunk split float n = 256..480 at batch 1024 into 128-256-item parts. Best
`blocked` ms (cold, 16 reps):

| dtype | itype | n | batch | potrf's chunk | 512 items | one pass |
|---|---:|---:|---:|---:|---:|---:|
| float | 1 | 480 | 1024 | 22.02 | 14.04 | 13.71 |
| cfloat | 1 | 256 | 1024 | 10.72 | 7.772 | 7.988 |
| cfloat | 1 | 128 | 4096 | 5.748 | 5.728 | 7.03 |

One pass loses where items are small; 512 items is never more than 7% behind the better of the
other two. Hence the 4 x CU floor (512 on the 4090); float n = 128 (chunk 896) keeps its chunk (at 512
it lost up to 26%). `BATCHLAS_TUNE_L2_CHUNK_ITEMS` forces the size.

**Transcribed rows.** double, cdouble (1:64 FP64, not tuned), every sm_120 row and cells past the
tuner's memory cap are run `transcribed-7d0b0f84`, written by `scripts/transcribe/sygst_transcribe.py`
as a copy of the nearest sm_89 `deep` float or cfloat row with the same itype and uplo;
`TunedTables.SygstTablesAreMeasuredOnSm89AndTranscribedFromIt` checks that relation. Its input table is
`sweep_to_table.py --ledger DIR` on a copy of the sm_89 ledger holding only the float and cfloat lines,
without their transcribed run lines. Raw data was not committed.

## sygst: the deliberate breaks {#sygst-the-deliberate-breaks}

Each applied alone, `sygvd_tests` run on CUDA and NETLIB, all four types (2026-10-26/27). NETLIB skips
`blocked` and stays green. "Definition" is `SygstBlockedMatchesTheDefinitionAcrossBlockEdges`,
"saturating" is `SaturatingSygstBlockedIsBitIdenticalToItemZero`.

| break | red set |
|---|---|
| diagonal solve `<` to `<=` / product `>` to `>=` (`sygst_diag.cc`) | definition for that itype only; saturating; sygvd. complex<double> green (one element per pack cannot reach the guard) |
| mirror kernel without `conj` | complex only: definition itypes 2, 3 (n >= 48), itype 1 (n >= 65, nb >= 64 recursion) |
| a half update's sign, or a `trmm` transpose, per itype and triangle | definition for that itype and triangle at n >= 33 (48) only; saturating |
| the wide diagonal block recurses at kb - 1 | definition only, nb 64-256 |
| a pass offsets B by A's batch stride | aborts in saturating's forced 100-item passes only |
| the diagonal kernel stores the whole block | definition and saturating, from the poison check |
| packed work-groups overlap by one item | saturating and sygvd's bit-identity case; definition for types with few items per group |

## Workspace {#sygvd-workspace}

All sizes come from `BumpAllocator::measuring()`; `*_buffer_size` makes the same choices as the call.
`compose` holds potrf's per-item status and, for itype 3 with vectors, an `n x n` product per item, plus
one scratch region of `max(potrf, sygst, syev)` bytes the children take in turn. `blocked` needs per pass
two pointer arrays, an nb x nb `H` and (itypes 2, 3) an n x nb `W` per item; a single diagonal block
needs none. `twosided` itype 1 needs none; itypes 2 and 3 one `n x n` product per item. The cuSOLVER
vendor needs one `lwork` buffer and, without a caller `info`, an int per item.

## To time {#sygvd-to-time}

- `batchlas_tune sygvd --dtype float,cfloat` (spec `tools/tune/sygvd_spec.cc`; itype 1 Upper,
  gfn2-xTB's call) to replace the transcribed rows.
- The gfn2 shapes (n = 480, batch 1819; n = 1200, batch 76) as `compose` against `vendor`, pinned
  with `BATCHLAS_SYGVD_ROUTE`; also `sygvd_factored` against `sygvd` to price the potrf it saves.

## What the tests pin {#sygvd-tests}

`tests/sygvd_tests.cc`:

- `sygst` against its definition for every family, itype and triangle; `blocked` at every `nb` at
  orders straddling each edge (8, 16, 32, nb, a two-level recursion at n = 260), on a badly scaled
  B = D S D against a slack of 32. A's storage outside its triangle and all of B keep their bits.
- 1024 identical items bit-identical to item 0, forced 100-item passes
  (`BATCHLAS_TUNE_L2_CHUNK_ITEMS` bypasses the 4 x CU floor), exact workspace, a workspace one byte
  short throws.
- `CoverageRowNamesEachPinnedSygstFamily`: each pinned call's `reached` row names its family.
- A `NoRouteError` under a pin fails, except where `can_run` is documented false (skip).
- `sygvd` residuals and B-normalization for every itype, jobz, triangle and pinned family, graded B,
  complex data with nonzero imaginary parts, poisoned other triangle, padded ld and stride.
- A non-positive-definite item: `info = n + 4`, the other items bit-identical.
- `sygvd_factored`: B never written, the factor reused, bit-identical to `compose`.
- On GPU backends potrf is pinned `native`: cuSOLVER's Upper potrf overwrites the lower triangle,
  which the tests poison.

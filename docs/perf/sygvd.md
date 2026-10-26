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

Assumptions (the plan text is lost; these are this page's choices). `itype` is a plain `int` (1, 2, 3),
as in LAPACK, not an enum. The factored entry is a separate function, not a flag, so its `sig::` alias
and its workspace query stay distinct. Neither `itype` nor `uplo` is a table key.

## Routes {#sygvd-routes}

| op | family | what runs | `can_run` |
|---|---|---|---|
| sygst | `blocked:nb` (32, 64, 128, 256) | LAPACK's blocked `?sygst`/`?hegst` on the `uplo` triangles: per block the SLM diagonal kernel, then `trsm` (itype 1) or `trmm` (2, 3), `gemm` and `her2k`/`syr2k` ([the blocked family](#sygst-the-blocked-family)) | homogeneous, `n >= 1`, GPU, compiled `nb`, `max_wg >= 128`, the diagonal kernel's SLM fits (`sygst.cc:57-60`) |
| sygst | `twosided` | `A.hermitize(uplo)`, then two public `trsm` (itype 1) or two public `trmm` through an `n x n` workspace product (itypes 2, 3) | homogeneous batch, `n >= 1` (`sygst.cc:62`) |
| sygst | `vendor` | LAPACKE `?sygst` / `?hegst` per item | NETLIB with LAPACKE; homogeneous. cuSOLVER 13.2 has no public `?sygst` |
| sygvd | `compose` | `potrf(B)` (skipped by `sygvd_factored`), the failed-item identity (`sygvd_native.cc`), `sygst`, `syev`, then a `trsm` back-transform (itypes 1, 2) or `trmm` plus a copy (itype 3), then the info merge | homogeneous batch, `n >= 1` |
| sygvd | `vendor` | cuSOLVER `cusolverDn?sygvd` per item on one stream with one shared work buffer (`cusolver_sygvd.cc`); LAPACKE `?sygvd` per item on the host | CUDA with cuSOLVER or NETLIB with LAPACKE; homogeneous; not `sygvd_factored` |

Every child of `blocked`, `twosided` and `compose` is a public routed op, so `BATCHLAS_SYEV_ROUTE`,
`BATCHLAS_TRSM_ROUTE` and the rest apply inside them, and the vendor-free build serves `compose`
wherever its children are native. rocSOLVER's batched `?sygvd` is not wired (no ROCm headers on this box),
so ROCm runs `compose`.

Tables: keys `dtype`, `n:log:3`, `batch:log` (sygvd adds `jobz:exact` first, as syev does).
`tuned/sygst.sm_89.txt` float and cfloat rows are measured, the rest transcribed from them
([sygst: measured](#sygst-measured)). `tuned/sygvd.{sm_89,sm_120}.txt` rank `compose` first in every
row, `transcribed` from `benchmarks/results/tuning/ledger/sygvd.<device>.jsonl` (run
`transcribed-ab235065`) by `scripts/sweep_to_table.py`. The CPU has no table and never borrows. The last-resort order puts
`vendor` first, so a host queue takes one LAPACKE call per item.

> **Warning:** `compose` before `vendor` on the GPU (sygvd) is a hypothesis from the shape of the vendor call, a
> per-item loop of one-matrix solves at batch >= 128, against batched children. It was not timed. The
> vendor may win at small batch and large n, where one `?sygvd` per item already fills the GPU.

## sygst: the blocked family {#sygst-the-blocked-family}

`blocked:nb=<32|64|128|256>` is LAPACK's `dsygst` / `zhegst` (`src/ops/sygst/sygst.cc`, `blocked_range`).
It works in place on the `uplo` triangle of `A` and never hermitizes it. Every read of `A` and `B` goes
through the named triangles, so the other triangle of `A`, its `ld` pads and the gaps between items
keep their bits (the tests poison them; defect-14 rule). For each block of `nb` columns, with `kb` the
block's order and `m` the trailing order, the steps are:

- **itype 1** (\f$ L^{-1} A L^{-H} \f$; Upper is the transposed schedule):
  1. the diagonal kernel reduces \f$ A_{11} \f$ to \f$ C_{11} \f$ and also writes the whole Hermitian
     \f$ C_{11} \f$ to workspace `H`;
  2. `trsm` (Right, \f$ L_{11}^{-H} \f$) on the panel \f$ P = A_{21} \f$;
  3. `gemm` \f$ P \mathrel{-}= \tfrac12 L_{21} H \f$;
  4. `her2k` / `syr2k` \f$ A_{22} \mathrel{-}= P L_{21}^H + L_{21} P^H \f$, on the triangle only;
  5. the same `gemm` again;
  6. `trsm` (Left, \f$ L_{22} \f$, order m) on the panel.
- **itypes 2 and 3** (\f$ L^H A L \f$): before each block's own reduction, the reduced leading k x k
  block grows by this block row. A mirror kernel copies the unreduced \f$ A_{22} \f$ whole into `H`. Then
  `trmm` writes \f$ W = A_{21} L_{11} \f$ out of place (`trmm` is out of place, so `W` is an n x nb
  workspace), followed by `gemm` \f$ W \mathrel{+}= \tfrac12 H L_{21} \f$, `her2k`
  \f$ A_{11} \mathrel{+}= W^H L_{21} + L_{21}^H W \f$ and the `gemm` again. Finally `trmm` writes
  \f$ A_{21} = L_{22}^H W \f$ back into `A`, and the diagonal kernel reduces \f$ A_{22} \f$.

Every child is the public, table-routed op. LAPACK's `hemm` / `symm` became `gemm` against `H`: the
transcribed `symm` table sends the tall float shape (m = 96, n = 32) to cuBLAS's per-item `?symm` loop.
That cost float n = 128 at batch 256 2.23 ms; with `gemm` it is 0.44 ms (tuner custom screen, 8 reps).
A diagonal block wider than the kernel's 32 recurses at a quarter of its order (at least 32): it is the
same reduction with that block's own factor. `nb` is a family field because the best block grows with n
(table rows below).

**The diagonal kernel** (`src/ops/sygst/kernels/sygst_diag.cc`) gives each column of a kb <= 32 block one
work-item and keeps the column in registers. `F` (L, or U^H for Upper) and a transpose buffer sit in
shared memory, with `F` stored so that each pass reads its lines as 16-byte packs. Pass one is a
triangular solve with F (itype 1) or a product with F^H (2, 3). Pass two applies the same operation to
the conjugate transpose of the result, which is C itself. Up to 128 columns share a work-group (several
items at kb < 32). Orders 32, 16 and 8 are compile-time instances. At float n = 32, batch 8192, the
first version (one 32-wide group per item, a runtime order, scalar `F` loads) took 0.33 ms; this one
takes 0.20 ms, against `twosided`'s 0.21 ms.

**Workspace**, per pass of the L2 chunking that potrf uses (`blocked_chunk`, `BLOCKED_L2_CHUNK_PERCENT`):
two pointer arrays for the vendor `trsm`, `H` (nb x nb per item) and, for itypes 2 and 3, `W` (n x nb per
item). There is no n x n buffer. A single diagonal block (n <= min(nb, 32)) needs none.

**can_run** (`src/ops/sygst/sygst.cc:57-60`): homogeneous batch, `n >= 1`, a GPU, a compiled `nb`,
`max_wg >= 128`, and the diagonal kernel's shared memory at order min(n, 32) within `slm_budget`. All
of these are correctness conditions. `twosided` (`:62`) stays as the second native family. The measured
tables rank it first in 33 of 192 rows: float n = 27..76 (every batch at n = 33, batch >= 8192 at
n = 27..54, batch <= 2048 at n = 64, 76) and n = 128..351 at batch 128 (128 also at 512), and cfloat
n = 70..277 at batch 128 and a few n <= 16 cells at batch >= 8192. There the trailing BLAS-3 calls of
`blocked` are too small to pay for their launches, and `twosided`'s two large `trsm` win.

## sygst: measured {#sygst-measured}

RTX 4090 (sm_89, GPU 1, headless), CUDA 13.2, DPC++ `/opt/dpcpp-cuda`, 2026-10-26. Itype 1 Lower, the
tuner's spec (`tools/tune/sygst_spec.cc`) under the cold protocol (L2 scrub before every sample),
`batchlas_tune sygst --reps 16 --passes 2 --no-refine --cap-gib 8`. Each arm is pinned and verified;
pass 2 runs the arms in reverse order. Time is pass 2's median. "Old" is `twosided`, the only CUDA route
before this family. "Auto" is the first entry of the shipped table's nearest row. Raw data:
`final_ab/` in the session's evidence directory, not committed.

| dtype | n | batch | twosided (old) ms | best blocked ms (nb) | Auto picks | Auto ms | old / Auto | reps |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| float | 8 | 128 | 0.05544 | 0.01878 (32) | `blocked:nb=32` | 0.01878 | 2.95x | 16 |
| float | 8 | 1024 | 0.01799 | 0.009045 (64) | `blocked:nb=32` | 0.009173 | 1.96x | 16 |
| float | 16 | 128 | 0.04115 | 0.01817 (64) | `blocked:nb=32` | 0.01846 | 2.23x | 16 |
| float | 16 | 1024 | 0.03961 | 0.01959 (32) | `blocked:nb=64` | 0.02017 | 1.96x | 16 |
| float | 32 | 128 | 0.04529 | 0.02486 (32) | `blocked:nb=32` | 0.02486 | 1.82x | 16 |
| float | 32 | 1024 | 0.05059 | 0.03466 (32) | `blocked:nb=32` | 0.03466 | 1.46x | 16 |
| float | 64 | 128 | 0.09255 | 0.1059 (64) | `twosided` | 0.09255 | 1.00x | 16 |
| float | 64 | 1024 | 0.1765 | 0.1965 (32) | `twosided` | 0.1765 | 1.00x | 16 |
| float | 128 | 128 | 0.2104 | 0.3196 (64) | `twosided` | 0.2104 | 1.00x | 16 |
| float | 128 | 1024 | 0.7401 | 0.7642 (64) | `twosided` | 0.7401 | 1.00x | 16 |
| float | 256 | 128 | 0.578 | 0.9317 (128) | `twosided` | 0.578 | 1.00x | 16 |
| float | 256 | 1024 | 4.951 | 4.46 (128) | `blocked:nb=128` | 4.46 | 1.11x | 16 |
| float | 480 | 128 | 2.997 | 2.73 (128) | `blocked:nb=128` | 2.73 | 1.10x | 16 |
| float | 480 | 1024 | 24.76 | 22.07 (128) | `blocked:nb=128` | 22.07 | 1.12x | 16 |
| float | 1024 | 128 | 21.73 | 11.65 (256) | `blocked:nb=256` | 11.65 | 1.87x | 16 |
| cfloat | 8 | 128 | 0.08174 | 0.01889 (64) | `blocked:nb=32` | 0.02081 | 3.93x | 16 |
| cfloat | 8 | 1024 | 0.0782 | 0.01581 (128) | `blocked:nb=32` | 0.02053 | 3.81x | 16 |
| cfloat | 16 | 128 | 0.07553 | 0.02452 (32) | `blocked:nb=32` | 0.02452 | 3.08x | 16 |
| cfloat | 16 | 1024 | 0.08469 | 0.03524 (32) | `blocked:nb=32` | 0.03524 | 2.40x | 16 |
| cfloat | 32 | 128 | 0.1011 | 0.0333 (32) | `blocked:nb=32` | 0.0333 | 3.03x | 16 |
| cfloat | 32 | 1024 | 0.1898 | 0.07843 (32) | `blocked:nb=32` | 0.07843 | 2.42x | 16 |
| cfloat | 64 | 128 | 0.1602 | 0.1401 (128) | `blocked:nb=32` | 0.1408 | 1.14x | 16 |
| cfloat | 64 | 1024 | 0.4503 | 0.3113 (32) | `blocked:nb=32` | 0.3113 | 1.45x | 16 |
| cfloat | 128 | 128 | 0.38 | 0.3984 (64) | `twosided` | 0.38 | 1.00x | 16 |
| cfloat | 128 | 1024 | 2.586 | 1.579 (64) | `blocked:nb=64` | 1.579 | 1.64x | 16 |
| cfloat | 256 | 128 | 1.3 | 1.359 (128) | `twosided` | 1.3 | 1.00x | 16 |
| cfloat | 256 | 1024 | 12.67 | 10.77 (128) | `blocked:nb=64` | 11.43 | 1.11x | 16 |
| cfloat | 480 | 128 | 6.796 | 5.262 (128) | `blocked:nb=128` | 5.262 | 1.29x | 16 |
| cfloat | 480 | 1024 | 56.02 | 41.57 (128) | `blocked:nb=128` | 41.57 | 1.35x | 16 |
| cfloat | 1024 | 128 | 48.23 | 27.14 (256) | `blocked:nb=256` | 27.14 | 1.78x | 16 |

The tables come from `batchlas_tune sygst --tier deep --dtype float` (run
`20261026T141406-JRHNBR-940530`) and `--dtype cfloat` (`20261026T142225-JRHNBR-969703`), 96 cells each,
`scripts/sweep_to_table.py`. double and cdouble are not timed on a 1:64-FP64 GPU (agent guide §10).
Their sm_89 rows, every sm_120 row and the sm_89 float/cfloat grid cells past the tuner's memory cap
are run `transcribed-fd79f23c` (untimed): each is a copy of the nearest sm_89 `deep` float or cfloat row,
written by a one-off script that is not committed. `TunedTables.SygstTablesAreMeasuredOnSm89AndTranscribedFromIt` checks that relation.

## sygst: the deliberate breaks {#sygst-the-deliberate-breaks}

Each break was applied alone to the committed code, then `sygvd_tests` (every case, CUDA and NETLIB,
all four types) was rebuilt and run, and the source restored (2026-10-26, after the table commit, so
`compose` on Auto reaches `blocked` wherever the tables rank it first). The NETLIB configurations
skip `blocked` (it is a GPU family) and stayed green under every break. "Definition" is
`SygstBlockedMatchesTheDefinitionAcrossBlockEdges`, "saturating" is
`SaturatingSygstBlockedIsBitIdenticalToItemZero`, and "sygvd" is the Auto `compose` cases
(`SygvdSolvesEveryItypeOnEveryFamily`, `FactoredEntryReusesTheFactorAndMatchesCompose`).

| break (file) | red set |
|---|---|
| diagonal solve: `p0 + u < i` to `<= i` (`sygst_diag.cc`, `apply`) | definition itype 1 only, both triangles, every n >= 7, every nb; saturating (itype 1, its chunked part); sygvd. float, double, cfloat. complex<double> is green: at one element per pack the guard cannot reach the diagonal, so the break does not change its code |
| diagonal product: `p0 + u > i` to `>= i` (`apply`) | definition itypes 2 and 3 only, every n >= 1; saturating; sygvd. complex<double> green for the same reason |
| mirror kernel without `conj` (`sygst_mirror`) | complex only: definition itypes 2, 3 at n >= 48 and itype 1 at n >= 65 (the nb >= 64 recursion mirrors its block); saturating chunked itype 2 |
| itype 1 Lower, first half update `-1/2` to `+1/2` (`blocked_range`) | definition itype 1 Lower at n >= 33 only; saturating; complex<double> sygvd |
| itype 1 Upper, second half update `-1/2` to `+1/2` | definition itype 1 Upper at n >= 33 only (33, 40, 48, ..., 260); saturating; complex<double> sygvd. (Giving its `her2k` NoTrans instead throws `her2k: incompatible matrix dimensions` at the first multi-block Upper call) |
| itypes 2/3 Lower, first `trmm` NoTrans to ConjTrans | definition itypes 2, 3 Lower at n >= 33 only; saturating chunked itype 2 Lower; complex<double> sygvd |
| itypes 2/3 Upper, last `trmm` ConjTrans to NoTrans | definition itypes 2, 3 Upper at n >= 48 only; saturating chunked itype 2 Upper; complex<double> factored |
| the wide diagonal block recurses at order kb - 1 | definition only, every itype and triangle at n >= 33, nb 64, 128, 256 only (nb 32 never recurses) |
| a pass offsets B by A's batch stride (`blocked`) | aborts (CUDA illegal address) in saturating's forced 100-item passes, the first case with more than one pass; every earlier case green |
| the diagonal kernel stores the whole block, not the triangle | definition and saturating only, from the poison check (`A written at ...`), every itype, triangle, n and nb |
| packed work-groups overlap by one item (`first = group * (ipw - 1)`) | saturating only, both its bit-identity loop and sygvd's `SaturatingBatchIsBitIdenticalToItemZeroAndWorkspaceIsExact`: batch 3 fits one group, so no definition case can see it |

Per-break logs: `breaks/B*.txt` in the session's evidence directory.

## Workspace {#sygvd-workspace}

All sizes come from `BumpAllocator::measuring()`; `*_buffer_size` makes the same choices as the call.
`compose` holds potrf's per-item status (kept until the merge) and, for itype 3 with vectors, an
`n x n` product per item. It also holds one scratch region of `max(potrf, sygst, syev)` bytes, which the
children take in turn on an in-order queue. `blocked` needs per pass two pointer arrays, an nb x nb
block and (itypes 2, 3) an n x nb panel product per item ([the blocked family](#sygst-the-blocked-family)).
`twosided` itype 1 needs no workspace; itypes 2 and 3 need one `n x n` product per item. The cuSOLVER vendor needs one `lwork` buffer and, without a caller `info`,
an int per item.

## To time {#sygvd-to-time}

sygst is timed ([sygst: measured](#sygst-measured)); `benchmarks/sygst_benchmark` (ARGS n batch itype
uplo, `BATCHLAS_SYGST_ROUTE` pins) times one family. Still to time, on a quiet GPU:

- `batchlas_tune sygvd --dtype float,cfloat`.
  The specs are `tools/tune/{sygst,sygvd}_spec.cc`, with jobz a key for sygvd. sygvd times itype 1
  Upper, gfn2-xTB's call: uplo is not a key, and potrf (inside `compose`) routes Upper apart from Lower
  (vendor rows at n 96..384 on sm_89). sygst times itype 1 Lower; its verify forms \f$ L C L^H \f$, and
  an Upper sygst is a pair of trsm with the other transposes. These replace the transcribed rows.
- The gfn2 shapes (n = 480, batch 1819; n = 1200, batch 76) as `compose` against `vendor`, both pinned
  with `BATCHLAS_SYGVD_ROUTE`; also `sygvd_factored` against `sygvd` to price the potrf it saves.

## What the tests pin {#sygvd-tests}

`tests/sygvd_tests.cc` checks:

- `sygst` against its definition for every family and itype, both triangles: \f$ A_0 = L C L^H \f$ and
  the rest via `similarity_residual`.
- `blocked` at every `nb`, itype and triangle, at orders straddling each edge (the kernel's compile-time
  orders 8, 16, 32; nb 32 to 256; a two-level recursion at n = 260), on B with cond ~1e3 (float) or ~1e6
  (double), against a slack of 32 (the reduction's error on a graded B follows the scaled S, measured at
  most 9x the bound). A's storage outside its triangle and all of B keep their bits.
- 1024 identical items on `blocked`, bit-identical to item 0 (n = 5 packs 25 items per work-group, 16, 20,
  40), forced 100-item passes (`BATCHLAS_TUNE_L2_CHUNK_ITEMS`), and an exact workspace query.
- `CoverageRowNamesEachPinnedSygstFamily`: the `reached` coverage row of each pinned call names its family.
- `sygvd` residuals and B-normalization for every itype, both jobz, both triangles and every pinned
  family, at n through the syev CTA width. B is graded (cond about 1e2 float, 1e4 double) and complex
  data has nonzero imaginary parts. The other triangle is poisoned with a large finite value, and ld and
  stride are padded.
- A non-positive-definite item: `info = n + 4`, the other items bit-identical.
- `sygvd_factored`: B is never written, a second A reuses the factor, and the result is bit-identical to
  `compose`.
- B's other triangle holds poison when the code under test reads B. On GPU backends potrf is pinned
  `native`: cuSOLVER's Upper potrf overwrites the lower triangle, and the unpinned table sends Upper
  n = 33 and 70 to it. The tests' own factorizations re-poison it, and `compose` must leave it intact.
- 1024 identical items, bit-identical to item 0, and a workspace one byte short throws.

The deliberate breaks, each with its narrow red set, are named above the tests.

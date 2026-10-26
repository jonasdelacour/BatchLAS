# sygst and sygvd {#perf_sygvd}

> **Status:** current · routes and tables are hypotheses: nothing here is timed (RTX 4090 sm_89, CUDA 13.2, 2026-10-25)

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
- **`sygvd_masked`, `sygvd_factored_masked`** solve only the items whose mask entry is nonzero and leave
  the rest bit-identical. They are for the SCC loop, whose molecules converge at different iterations
  (@ref design_masked_batches). Their buffer-size query is the same for every mask, so it is made once.
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
| sygst | `twosided` | `A.hermitize(uplo)`, then two public `trsm` (itype 1) or two public `trmm` through an `n x n` workspace product (itypes 2, 3) | homogeneous batch, `n >= 1` |
| sygst | `vendor` | LAPACKE `?sygst` / `?hegst` per item | NETLIB with LAPACKE; homogeneous. cuSOLVER 13.2 has no public `?sygst` |
| sygvd | `compose` | `potrf(B)` (skipped by `sygvd_factored`), the failed-item identity (`sygvd_native.cc`), `sygst`, `syev`, then a `trsm` back-transform (itypes 1, 2) or `trmm` plus a copy (itype 3), then the info merge | homogeneous batch, `n >= 1` |
| sygvd | `vendor` | cuSOLVER `cusolverDn?sygvd` per item on one stream with one shared work buffer (`cusolver_sygvd.cc`); LAPACKE `?sygvd` per item on the host | CUDA with cuSOLVER or NETLIB with LAPACKE; homogeneous; not `sygvd_factored` |

Every child of `twosided` and `compose` is a public routed op, so `BATCHLAS_SYEV_ROUTE`,
`BATCHLAS_TRSM_ROUTE` and the rest apply inside them, and the vendor-free build serves `compose`
wherever its children are native. rocSOLVER's batched `?sygvd` is not wired (no ROCm headers on this box),
so ROCm runs `compose`.

Tables: keys `dtype`, `n:log:3`, `batch:log` (sygvd adds `jobz:exact` first, as syev does).
`tuned/sygst.{sm_89,sm_120}.txt` rank `twosided` first, and `tuned/sygvd.{sm_89,sm_120}.txt` rank
`compose` first, in every row. Both are `transcribed` from the ledgers
`benchmarks/results/tuning/ledger/{sygst,sygvd}.<device>.jsonl` (run `transcribed-ab235065`) by
`scripts/sweep_to_table.py`. The CPU has no table and never borrows. The last-resort order puts
`vendor` first, so a host queue takes one LAPACKE call per item.

> **Warning:** `compose` before `vendor` on the GPU is a hypothesis from the shape of the vendor call, a
> per-item loop of one-matrix solves at batch >= 128, against batched children. It was not timed. The
> vendor may win at small batch and large n, where one `?sygvd` per item already fills the GPU.

## Workspace {#sygvd-workspace}

All sizes come from `BumpAllocator::measuring()`; `*_buffer_size` makes the same choices as the call.
`compose` holds potrf's per-item status (kept until the merge) and, for itype 3 with vectors, an
`n x n` product per item. It also holds one scratch region of `max(potrf, sygst, syev)` bytes, which the
children take in turn on an in-order queue. `twosided` itype 1 needs no workspace; itypes 2 and 3 need
one `n x n` product per item. The cuSOLVER vendor needs one `lwork` buffer and, without a caller `info`,
an int per item.

## To time {#sygvd-to-time}

Nothing is timed: the box was shared during the work. On a quiet GPU:

- `tools/tune/batchlas_tune sygst --dtype float,cfloat` and `batchlas_tune sygvd --dtype float,cfloat`.
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

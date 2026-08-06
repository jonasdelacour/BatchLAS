# Can a specialised level-3 op replace a GEMM in `src/extensions`?

Survey of all 73 `gemm` call sites under `src/extensions`, asking where `trmm`,
`symm`, `hemm`, `syrk`, `syr2k`, `herk` or `her2k` expresses the same thing, and
then measuring whether it is actually faster **at large batch**.

One substitution survived measurement. It is the one that was already scaffolded
and switched off.

Hardware: RTX 4090 / sm_89, CUDA 13.2, `RelWithDebInfo`, one GPU, no other
compute processes resident. Harness: `benchmarks/gemm_vs_level3_benchmark.cc`,
whose shapes are lifted from the call sites rather than invented.

---

## The one change: `sytrd_blocked` trailing update -> `syr2k`

`src/extensions/sytrd_blocked.cc` updates `A22 -= V W^H + W V^H` with two full
`n2 x n2` GEMMs. That is one `syr2k`, and `syr2k` touches only the triangle the
panel loop goes on to read.

The route already existed behind `BATCHLAS_SYTRD_TRAILING_UPDATE=syr2k`,
gated to CUDA + float, defaulting off. **This survey flips that default on.**

The update in isolation (float, `n2 x ib` operands into `n2 x n2`):

| n2 | ib | batch | 2x GEMM | syr2k | syr2k+symmetrize | speedup |
|-----|-----|-------|---------|-------|------------------|---------|
| 256 | 32 | 1024 | 1.476 ms | 0.427 | 0.886 | **1.66x** |
| 256 | 64 | 1024 | 1.531 | 0.512 | 0.957 | **1.60x** |
| 512 | 32 | 512 | 2.580 | 0.724 | 1.640 | **1.57x** |
| 512 | 64 | 512 | 2.682 | 0.814 | 1.761 | **1.52x** |
| 1024 | 32 | 128 | 2.482 | 0.693 | 1.749 | **1.42x** |
| 1024 | 64 | 128 | 2.544 | 0.730 | 2.039 | **1.25x** |
| 2048 | 64 | 32 | 2.504 | 0.692 | 1.682 | **1.49x** |

The `syr2k` itself is 3.4-3.6x faster. The middle column is what it cost to hand
the other triangle back afterwards -- a bandwidth-bound `n^2` pass that ate over
half the win. **That symmetrize turned out to be unnecessary and is now gone**;
see "Does anything read the upper triangle?" below. The 3.4-3.6x column is the
one that applies.

End to end through `sytrd_blocked_benchmark`, at the batch sizes its own grid
pairs with each `n`:

| n | batch | nb | GEMM | syr2k | speedup |
|---|-------|----|------|-------|---------|
| 512 | 1024 | 16 | 263.97 ms | 227.51 | 1.16x |
| 512 | 1024 | 24 | 253.05 | 227.32 | 1.11x |
| 512 | 1024 | 32 | 248.30 | 231.64 | 1.07x |
| 256 | 2048 | 16 | 34.347 | 27.040 | 1.27x |
| 256 | 2048 | 24 | 34.641 | 30.612 | 1.13x |
| 256 | 2048 | 32 | 36.995 | 34.028 | 1.09x |

Best-configuration to best-configuration: 248.30 -> 227.32 ms at n=512, and
34.35 -> 27.04 ms at n=256. Run-to-run stddev was 0.02-0.4 ms except one 2.3 ms
outlier, so these are well clear of noise.

### Test coverage this needed first

The trailing update only runs when the trailing block is wider than 128;
narrower and `sytrd_blocked` takes `update_vw_lower_small`. Every pre-existing
case in `tests/sytrd_blocked_tests.cc` is `n <= 128`, so **the syr2k route had no
test coverage at all** -- flipping the default on the strength of the benchmarks
alone would have been flipping an unexercised branch.

`SytrdBlockedTest.TrailingUpdateRoutesAgree` (n=320, nb=32, so `n2 = 288` on the
first panel and above 128 for six of them) runs both routes and checks each
against the netlib reference spectrum.

It was checked for teeth rather than assumed to have them:

- forcing `alpha = -0.5` in the syr2k call fails it loudly -- worst eigenvalue
  error 2.777 against a 3.2e-3 bound, with the GEMM route reported at 2.6e-6.
- the backward-error bound alone (`4 n eps ||A||` = 3.2e-3) is ~1000x looser
  than the error either route actually incurs, so it would pass almost anything.
  The assertion that does the work is the relative one: syr2k must land within
  `4 * (GEMM route error) + 8 eps ||A||`.

## Does anything read the upper triangle?

Disabling the symmetrize did not fail the test, which had two possible readings:
the pass is unnecessary, or the test does not reach whatever needs it. Reading
every consumer settles it -- **nothing in the `sytrd_blocked` pipeline reads A's
upper triangle**, so the symmetrize is now removed.

The symmetric matvec is the only place tempted to cross the diagonal, and all
three `latrd_lower_panel` variants split it at `c == r`, taking `Ab(r,c)` for
`c <= r` and `conj(Ab(c,r))` for `c > r`:

| reader | how it stays below the diagonal |
|---|---|
| `latrd_lower_panel`, legacy | explicit split at `c == r`, documented in place |
| `latrd_lower_panel`, grid | same split, same code shape |
| `latrd_lower_panel`, device | `device::hemv<Uplo::Lower>`, which mirrors rather than reads across -- for `row < col` it loads `a(col,row)` and applies `symmetric_mirror`, in both the tiled and the generic path |
| fused trailing update, legacy + grid | `if (r < c) continue` |
| fused trailing update, device | `device::her2k<Uplo::Lower>` |
| `restore_tridiag_lower` | reads the diagonal; the superdiagonal it touches is a *write* |

The GEMM pair happened to leave a valid upper triangle behind as a side effect.
Nothing depended on it, so that was never a contract -- which is exactly the kind
of thing that is invisible until an op that respects the triangle replaces one
that does not.

Verified rather than argued: with the symmetrize removed, the suite passes on the
legacy impl, on `BATCHLAS_SYTRD_IMPL=device`, and on the device impl with the
grid variant forced (`BATCHLAS_LATRD_GRID_GROUPS=4`).

The device impl was the only path that had been paying it, and it recovers the
full win -- at n=512 batch=1024 it goes 243.8/239.5/239.8 ms -> 226.5/228.0/231.1
at nb=16/24/32, landing on the legacy path's numbers to within noise (1.16x over
the GEMM baseline, up from 1.08x).

(One thing to know before forcing grid parameters while bisecting:
`SytrdBlockedLatrdGridCudaTest.LargeNBatchOneMatchesNetlibReference` and
`.GridMatchesLegacyTridiagonal` fail under an externally forced
`BATCHLAS_LATRD_GRID_GROUPS`, at 4 and at 8, and they do so with the GEMM route
and no syr2k anywhere. They pick their own grid parameters and an outer override
collides with them. Unrelated to any of this, but it looks alarming.)

### Why it stays CUDA + float only

`syrk`/`syr2k` reach a batched kernel **only** through the custom float route
added in PR 60. Every other type and backend falls through to
`syr2k_vendor_impl` / `syrk_vendor_impl` in `src/backends/cublas.cc`, which is a
host loop issuing one `cublasXsyr2k` per batch member -- there is no batched
cuBLAS syrk. At large batch that inverts the result completely. Same shapes, in
double:

| n2 | ib | batch | 2x GEMM | syr2k | |
|-----|-----|-------|---------|-------|---|
| 256 | 32 | 1024 | 7.56 ms | 58.52 | **7.7x slower** |
| 256 | 64 | 1024 | 14.34 | 106.79 | **7.4x slower** |
| 512 | 64 | 512 | 28.57 | 53.61 | **1.9x slower** |
| 2048 | 64 | 32 | 28.54 | 18.45 | 1.55x faster |

Double only wins where the batch is small enough that the per-item launch cost
amortises against a large per-item problem -- the opposite of the regime this
survey cares about.

---

## Measured and rejected: `syrk`/`herk` for `ortho`'s Gram matrix

`src/extensions/ortho.cc` builds `C = A^H A` three times (lines 129, 198, 232 --
the Cholesky, shifted-Cholesky and SVQB paths) with
`gemm(A, A, C, transA=inv_trans, transB=transA)`. This is textbook `syrk`/`herk`,
and it is safe on the consumer side: `potrf`, `trsm` and `syev` all default to
`Uplo::Lower`, so a one-triangle result is exactly what they read.

It is still the wrong call, because of the shape `ortho` is given. `A` is
`m x k`, `C` is `k x k`, and the win depends entirely on `k`:

| m | k | batch | GEMM | syrk | |
|---|---|-------|------|------|---|
| 256 | 32 | 2048 | 0.350 ms | 33.73 | **96x slower** |
| 512 | 32 | 2048 | 0.759 | 61.33 | **81x slower** |
| 512 | 64 | 1024 | 0.411 | 32.07 | **78x slower** |
| 1024 | 64 | 1024 | 0.742 | 60.67 | **82x slower** |
| 1024 | 128 | 512 | 0.425 | 30.93 | **73x slower** |
| 2048 | 128 | 256 | 0.400 | 29.76 | **74x slower** |
| 256 | 256 | 512 | 0.443 | 0.462 | 0.96x |
| 512 | 512 | 128 | 0.813 | 0.589 | 1.38x |
| 512 | 512 | 512 | 2.888 | 2.508 | 1.15x |
| 1024 | 1024 | 32 | 1.512 | 0.999 | 1.51x |
| 1024 | 1024 | 128 | 5.671 | 3.393 | **1.67x** |

The cliff is the same host loop as above. `syrk_use_cuda_custom` takes the
batched route only if `syrk_prefer_triangular_tiles` (needs `n >= ~384` so the
tile grid is at least three 128-wide tiles a side) or
`syrk_prefer_cuda_custom_heuristic` (needs `min_dim * 2 >= max_dim`, i.e. an
aspect ratio no worse than 2:1) says yes. A Gram matrix with `k = 32` and a
reduction depth of `m = 256` fails both, and drops to one `cublasSsyrk` launch
per batch member.

The winning column is `k >= 512` and square-ish. `ortho`'s callers do not live
there: `syevx_lobpcg`, `syevx_filtered` and `lanczos` all pass `k` = the block
size, which is tens of vectors, not hundreds. The only caller that orthogonalises
a wide square block is `src/extra/random_cond.cc`, a test-matrix generator.

So at the batch sizes that matter, substituting `syrk` here is a 70-100x
regression on the shapes that actually occur. Left as `gemm`.

(`svqb_alg` has a second, independent problem: it scales the full `k x k` before
handing it to `syev`, so a one-triangle `syrk` would leave it multiplying
uninitialised workspace. Moot given the above, but it would need a mirror pass.)

---

## Measured and rejected: `trmm` for the WY block factor

`ormqr_blocked.cc:442,458` and `ormbr.cc:549,559` compute `W2 = T^H W1` where `T`
is the `ib x ib` upper-triangular factor from `larft`. The GEMM multiplies
through `T`'s zero half.

`trmm` loses at every shape (float, `T` is `ib x ib`, `W1` is `ib x nC`):

| ib | nC | batch | GEMM | trmm |
|----|-----|-------|------|------|
| 32 | 256 | 2048 | 0.195 ms | 0.238 |
| 64 | 256 | 2048 | 0.374 | 0.560 |
| 64 | 512 | 1024 | 0.348 | 0.498 |
| 128 | 512 | 1024 | 0.704 | 1.076 |
| 128 | 1024 | 512 | 0.670 | 0.989 |
| 256 | 1024 | 256 | 0.779 | 1.152 |

This one is structural rather than a tuning accident. `src/extensions/trmm.cc`
recurses only until `n <= recursion_stop_size = 256`, and then calls the very
GEMM it was meant to replace. `ormqr`'s `nb` is one of {16, 32, 64, 128, 256}, so
the triangular structure is *never* exploited at these sizes -- `trmm` is the
same GEMM plus dispatch, plus a zeroing pass for the `beta = 1` accumulate.
Left as `gemm`.

---

## Rejected on inspection, no measurement needed

- **`syevx_lobpcg.cc:647,1225`, `syevx_filtered.cc:418`** -- `X^H A X`. The
  result is symmetric but it is a product of two *different* matrices; no BLAS
  op expresses it. `syr2k` is not this.
- **`syevx_lobpcg.cc:624,638,1212`, `syevx_filtered.cc:241`, `lanczos.cc:112`,
  `ritz_values.cc:59`** -- `A X` with symmetric `A`, nominally `symm`/`hemm`.
  But `symm` in this tree *expands the triangle into scratch and then GEMMs*
  (`src/extensions/symm.cc`, and the same shape in `symm_custom_dispatch`). `A`
  is already stored full at these call sites, so `symm` would add a copy and a
  symmetrize to reach the identical GEMM. Worse by construction, not by tuning.
- **`gebrd_blocked.cc:364,365`** -- `a22 -= v2 y2^H; a22 -= x2 u2`. Looks like
  `syr2k`, is not: this is bidiagonal reduction of a *general* matrix, and `a22`
  is not symmetric.
- **`stedc.cc:465,487,500`, `steqr_legacy.cc:412,463,512`, `ortho.cc:293,399,400`,
  `ormqr_blocked.cc:439,444,455,460`, `ormbr.cc:548,550,558,560`** -- general
  products of distinct matrices with no symmetry or triangularity to exploit.

---

## The recurring shape of the negative results

Two of the three rejections have the same root cause, and it is worth stating
plainly: **in this tree a specialised level-3 op is only faster than the GEMM it
replaces when it reaches a batched custom kernel.** `syrk`, `syr2k`, `herk` and
`her2k` have exactly one such path each -- CUDA, float, and only inside the shape
window the PR 60 routers were tuned for. Outside that window they degrade to a
host loop over the vendor call, which at batch 1024+ is one to two orders of
magnitude off a batched GEMM.

So the flop count is the wrong thing to reason from. "This does half the
arithmetic" predicts nothing here; what predicts the result is whether the router
takes the batched branch. Any future substitution should check that first.

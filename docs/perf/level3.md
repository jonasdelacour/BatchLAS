# Level-3: symm, hemm, syrk, herk, syr2k, her2k, trmm

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, `RelWithDebInfo` · measured 2026-09-05; tables transcribed 2026-10-06

Routing and measurements for the seven level-3 ops. `symm`, `syrk`, `syr2k` and `trmm` choose from
tuned tables. `hemm`, `herk` and `her2k` choose by fixed rules in the cuBLAS backend. The table rows
for the four are transcribed from the deleted hand-written gates (at `ff340fc6`) and are untimed.

## Level-3: how the table-selected four choose

`src/ops/<op>/choice.hh` lists the families. `src/ops/<op>/<op>.cc` validates shapes (throwing
`std::invalid_argument` on every backend), returns a no-op event for an empty problem, and calls
`select::run`, which takes the first runnable entry of the nearest row of
`tuned/<op>.<dtype>.<device>.txt`. @ref selection_tables lists the families per op.

### Level-3: the families and their can_run terms

Every native family requires `B == Backend::CUDA`, `d.is_gpu`, homogeneous operands, extents and batch
>= 1, and batch <= 65535 (the CUDA grid z; 65536 throws `Number of work-groups exceed limit`). The
vendor family needs `d.has_vendor` and refuses heterogeneous operands, since each vendor loop runs
every item at the top-level extents.

| op | spelling | driver | further `can_run` terms | dtypes |
|---|---|---|---|---|
| `symm` | `expand` | `expand_mirrored`, then the public `gemm` | `max_wg >= 256`; `expansion_fits` (`symm.cc:52-56`) | float, double |
| `symm` | `vendor` | `backend::symm_vendor`: `?symm` per item | | float, double |
| `syrk` | `gram` | `syrk_gram_tiles`: one tile covers C | n <= `kGramMaxTile` (128); `max_wg >= gram_threads(n)`; tile fits SLM (`syrk.cc:58-59`) | float, double |
| `syrk` | `triangular` | `syrk_triangular_tiles`: 128x128x8 tiles of one half | `max_wg >= 256`; T(T+1)/2 <= 65535, T = ceil(n/128) (`syrk.cc:62-63`) | float |
| `syrk` | `vendor` | `backend::syrk_vendor`: `?syrk` per item | | float, double |
| `syr2k` | `triangular` | `syr2k_triangular_tiles`: both products fused | `max_wg >= 256`; `transA != ConjTrans`; tile ceiling (`syr2k.cc:53-55`) | float |
| `syr2k` | `vendor` | `backend::syr2k_vendor`: `?syr2k` per item | | float, double |
| `trmm` | `triangular` | `trmm_triangular_tiles`: row tiles that skip the zero half | `max_wg >= 256`; `Side::Left`; `trmm_tile_groups` <= 65535 (`trmm.cc:56-61`) | all four |
| `trmm` | `expand` | `expand_triangular`, then `gemm` at beta 0 | `max_wg >= 256`; `expansion_fits` (`trmm.cc:63-68`) | all four |
| `trmm` | `vendor` | `backend::trmm_vendor`: `cublas?trmm` per item | | all four |

A real `ConjTrans` is `Trans` to both syrk tile kernels, so the exact `trans` key keeps it on the
vendor. The last resort when nothing runs is `expand, vendor` (symm), `triangular, gram, vendor`
(syrk), `triangular, vendor` (syr2k) and `expand, triangular, vendor` (trmm).

### Level-3: what the shipped tables rank first

The twenty tables (`tuned/{symm,syrk,syr2k}.{float,double}.*`, `tuned/trmm.{float,double,cfloat,cdouble}.*`,
for `sm_89` and `sm_120`, identical because the old rules read no architecture) are
`source=transcribed:ff340fc6`. Per grid cell they put the old Auto choice first. From the `sm_89` rows:

| op, dtype | ranked first |
|---|---|
| `syrk` float | `gram` at every n <= 128; `triangular` on every `sq` row above n = 128, and on `tall`/`wide` rows with n >= 257, k >= 8 and batch x T(T+1)/2 >= 160; `vendor` on the other `tall`/`wide` rows and every trans C row |
| `syrk` double | `gram` at every n <= 128 (N and T); `vendor` everywhere else |
| `syr2k` float | `triangular` at batch >= 2; `vendor` at batch 1 |
| `syr2k` double | `vendor` (the only candidate) |
| `symm` float | `expand` on `sq` rows with batch >= 4 or max(m, n) >= 256; `vendor` on the other `sq` rows and every `tall`/`wide` row |
| `symm` double | `vendor` everywhere, `expand` second |
| `trmm`, all four | `triangular` on every `side=L` row, `expand` on every `side=R` row, `vendor` last |

A vendor-free build therefore serves every float `syrk` GPU shape, double `syrk` at n <= 128, float
`syr2k` at batch 1, float and double `symm`, and `trmm` in every dtype. The transcription matches the
old Auto choice on 100% of 2500-3000 random off-grid points per op, dtype and device.

### Level-3: pinning a family

`BATCHLAS_<OP>_ROUTE` takes `auto`, `native`, `vendor` or a spelling from the table above. A spelling
that is not compiled for the dtype, or cannot run the shape, throws `std::invalid_argument`. Pin words
are listed at [the level-3 pin words](dispatch.md#the-level-3-pin-words).

- `BATCHLAS_TRMM_ROUTE=vendor` is always the `cublas?trmm` loop, even where a tile kernel ranks first.
  The old meaning (expand plus gemm) is now `expand`.
- `herk` reaches the Gram kernel only through `BATCHLAS_SYRK_ROUTE=gram` (`ops::syrk::herk_gram_pinned()`).
- `symm` does not read `BATCHLAS_EXPAND_ROUTE`; hemm, herk and her2k do.

### Level-3: hemm, herk and her2k still choose by fixed rules

These three have no `choice.hh` and no table. Their entry points (`src/ops/level3/level3.cc`) call
`backend::<op>_vendor` directly, or throw `NoRouteError` when no level-3 library is compiled in. The
cuBLAS backend then picks between one batched GEMM into scratch and the per-item vendor loop:

```cpp
// hemm_vendor (src/backends/cublas.cc:192-193); expansion_preferred is triangular_expand.hh:31-42
expansion_fits(ctx, k, batch, bytes) && expansion_preferred(max(m, n, k), batch)  // batch >= 4 || max_dim >= 256
// herk_vendor (cublas.cc:366); herk_gemm_preferred is cublas.cc:257-263
herk_gemm_preferred(n, batch) && expansion_fits(ctx, n, batch, bytes)             // batch >= 4 && n <= 768
// her2k_vendor (cublas.cc:436); her2k_gemm_preferred is src/expansion_budget.hh:65-69
her2k_gemm_preferred(n, batch) && expansion_fits(ctx, n, batch, bytes)            // batch >= 2 || n >= 128
```

`BATCHLAS_EXPAND_ROUTE=expand|loop` overrides the preference, not the fit. The host backend runs the
per-item `cblas_?hemm`/`?herk`/`?her2k` loops; ROCm instantiates none of the three.

### Level-3: dtype reach of the native families

`gram` and `triangular` admit non-float scalars, so double `syrk` at n <= 128 and `trmm` in every dtype
run vendor-free; `syr2k` has no non-float family. `level3_tile_route_available`
(`src/select/vendor.hh:94-96`) is deliberately kept float-or-cuBLAS, because `sytrd_blocked`, `ortho`,
`ormqr_blocked` and the coverage census read it; widening it would move their routes.

### Level-3: before flat selection

Up to `ff340fc6` these four ops chose by hand-written gates in `src/backends/*_custom_dispatch.cc`. The
gates were gate-only, so the transcriber evaluated the whole old decision per grid cell. The tables hold
the effective window, not the predicates.

## Boundaries and their evidence

The windows below were measured against the old gates. The transcribed tables reproduce them cell by cell.

### syrk triangular tiles

128x128x8 tiles over the triangular tile set. Grid: float, n 64..2048 x batch 1..512, against the full
n x n batched GEMM.

| table edge | admit side | bracketing non-winner |
|---|---|---|
| T >= 3 tiles a side (n >= 257) | 1.45x at n=512 batch 512; 1.63x at n=1024 batch 64; 1.71x at n=2048 batch 16 | n = 256: **0.84x–1.22x** (depends on the grid's wave boundary) |
| batch x T(T+1)/2 >= 160 | won from 168 blocks up | 144 blocks 1.14x slower; 136 blocks 1.25x slower |
| k | does not enter | not swept here (`experiments/syrk_kskew.sh`) |

At n = 256 a diagonal tile is computed whole and then masked, so only one tile in four is saved; from
n = 384 every saturated shape won. The grid must fill the 128 SMs (two 256-thread blocks each), so below
~160 blocks the route lost. The k >= 8 edge is the old gate's `kTriangularTileK`, not a measured one.

### syrk: the squareish rows below the triangular window

The `sq` form key carries the old cuBLASDx-sized heuristic (`min_dim*2 >= max_dim && tiled_work >= 8`).
One `cublasSsyrk` launch per batch member costs about **9 us**, so every `sq` row from n = 129 up ranks
`triangular` first. The two boundary records differ: the exploration note gives `0.89x at n = 256,
1.28x at 512, 1.57x at 1024`; the old gate's comment gives `0.84x–1.22x at 256, 1.45x at 512 batch 512,
1.63x at 1024 batch 64`. Neither holds a cell in **257 <= n <= 383**, which the rows admit.

### syrk gram tiles

A single-tile kernel sized to n: one shared tile covers all of C, so A crosses the bus once. It serves
n <= 128, and the rows rank `gram` first there in float and double. Float, ms (`m` is the reduction depth,
`before` is the host loop over `cublasSsyrk`):

| m | n | batch | gemm | before | now | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.334 | 33.588 | 0.0780 | 4.29x | 431x |
| 1024 | 64 | 1024 | 0.668 | 60.397 | 0.3326 | 2.01x | 182x |
| 1024 | 128 | 512 | 0.409 | 30.296 | 0.4146 | **0.99x** | 73x |

At n = 32 the kernel reads 71 MB in 78 us (933 GB/s, the memory roofline). At n = 128 the prize is 1.37x,
not 2x. Double, same shapes (gemm / before / now, ms): 0.901 / 115.40 / **0.837** (1.08x vs gemm),
3.444 / 112.45 / **1.934** (1.78x), 13.72 / 110.91 / **6.521** (2.10x). The win grows in double because
FP64 at 1/64 rate makes the Gram product compute bound.

### syr2k triangular tiles

One pass fusing both rank-k products into the same accumulators. Grid: float, n 8..3072 x k 4..2048 x
batch 1..1024. Batch is the only key that decides, because the vendor route is one launch per batch member.

- **batch >= 2:** won every shape: 1.06x at n=3072, 1.12x at n=1024, 1.3–1.4x through the middle, up to
  226x where the whole cost is the launch.
- **batch 1:** the vendor wins 1.18–1.60x below n=1280 and 1.16x at n=3072; the kernel wins 1.02–1.71x
  between; the vendor wins 4–10x on a deep k with a small n, where cuBLAS splits the reduction.

Issuing the two products sequentially rather than interleaved is worth **1.53x** (3.34 vs 5.11 ms at
n=512 batch 512).

### trmm tiles have no threshold

The old gate admitted the tile kernel only for `m <= 64 || m >= 512`, read off a trmm-vs-gemm column. The
real choice is between the tile kernel and **the expansion**. Float, saturating batch, ms:

| m | nC | batch | tile | expand + gemm |
|---|---|---|---|---|
| 128 | 512 | 1024 | **0.698** | 0.784 |
| 256 | 256 | 512 | **0.536** | 0.692 |
| 256 | 1024 | 256 | 0.915 | **0.855** |

The gate cost up to **1.29x** on the shapes it meant to protect. The single 7% loss is left unfitted.
Every `side=L` row of every trmm table ranks `triangular` first.

### trmm tile vs gemm by dtype

This is the caller's question, whether to spell a product as trmm at all. Ratios are trmm against the
GEMM spelling of the same product; **bold** is a win.

| m | nC | batch | float | double | `complex<float>` | `complex<double>` |
|---|---|---|---|---|---|---|
| 32 | 256 | 2048 | **1.13x** | **1.29x** | 0.88x | **1.05x** |
| 128 | 512 | 1024 | **1.03x** | **1.48x** | 0.69x | **1.42x** |
| 256 | 1024 | 256 | 0.91x | **1.77x** | 0.93x | **1.71x** |
| 512 | 512 | 128 | **1.15x** | **1.91x** | **1.10x** | **1.85x** |
| 1024 | 1024 | 32 | **1.26x** | **2.02x** | **1.23x** | **1.95x** |

- `double` and `complex<double>` win every shape. `float` loses at m = 256, where cuBLAS SGEMM is near its
  ~45 TFLOP/s peak and this kernel reaches ~57% of it. `complex<float>` loses below m = 512 on a register
  ceiling: a complex accumulator doubles the registers.
- The saving is not 2x. With `R = m / TileM` row tiles the reduction shrinks to `(R+1)/2R`: 1.0x, 1.33x and
  1.78x at R = 1, 2 and 8. See [choosing the row tile](#trmm-choosing-the-row-tile-by-scalar-type).

### symm and hemm expansion crossover

Measured against a per-batch loop over the vendor's triangular primitive: float `cublas?symm` over n 16..2048
x batch 1..512, and complex64 `cublas?hemm` over n 16..512 x batch 1..16. The expansion wins **1.2x–72x**
everywhere except `batch <= 2 && n <= 128`, where the call is launch-bound and the expansion loses by up to **2.5x**.

The shipped edge is more conservative than that loss region: `kExpandMinBatch = 4` and `kExpandMinDim = 256`
(`src/backends/triangular_expand.hh:28-29`), used by the float `symm` rows and by `hemm_vendor`. Batch 3 at every
n, and batch <= 2 with `129 <= n <= 255`, take the loop although they lie outside the loss region. That slack
is deliberate; the exploration numbers would support 2 and 128. `symm` ranks `vendor` first on every `tall` and
`wide` row (no bracketing cell exists). Double `symm` ranks `vendor` first everywhere, and `expand` for double is unmeasured.

trmm's `expand` is first on every Right row, because `cublas?trmm` has a flat ~110 us floor: **1.15x–162x** over
49 square cells and **1.22x–32x** over 64 skewed cells. No cell goes the other way.

### herk and her2k: the GEMM-plus-fold crossovers

Both replace a per-batch vendor loop with one strided-batched GEMM into scratch plus a fold, because cuBLAS has
no batched `?herk` or `?her2k`. The rule is a conjunction, not a disjunction. herk's predicate is
`batch >= 4 && n <= 768`; the mirrored expansion's is `batch >= 4 || max_dim >= 256`. The expansion has no
large-n ceiling because it is one bandwidth-bound kernel. herk's GEMM computes both triangles and keeps one,
so it needs the ceiling where one `cublas?herk` already saturates the device.

| op | predicate | admit side | bracketing non-winner |
|---|---|---|---|
| `herk` | `batch >= 4 && n <= 768` | 1.6x–72x for batch >= 4 at n <= 512 | batch <= 2 is a wash or a loss at every n; **0.82x–0.93x from n = 896 up** |
| `her2k` | `batch >= 2 \|\| n >= 128` | 1.4x–128x | batch 1 at n <= 64 only: **0.74x at n = 32, 0.89x at n = 64** |

Grid: complex64, n 32..1024 x batch 1..256 (`experiments/herk_crossover.sh`). Gaps: batch 3 is unmeasured
(the grid steps 1, 2, 4, 8, 16, 64, 256); **769 <= n <= 895 is unmeasured**, since the wash band runs to 768
and the loss starts at 896; the batch-1 side of her2k's `n >= 128` has no quoted cell. `her2k_gemm_preferred`
lives in `src/expansion_budget.hh` so that `sytrd_blocked.cc` can evaluate the whole route condition
(`her2k_takes_gemm_route`) before it calls her2k.

## Level-3: kernel design notes

The reasons behind the tile kernels' shapes. The headers keep the invariants and point here.

### syrk gram tiles: kernel design

`src/backends/syrk_gram_tiles.hh`: a single-tile batched SYRK for the tall skinny A and small square C that
`ortho` supplies.

- **Why a second kernel.** The 128x128 triangular kernel cannot serve n <= 128. A 128-wide tile spends
  128*128*k arithmetic on an n*n/2 answer, a **32x overcharge at n = 32**.
- **One shared tile.** Both operands of `A^T A` are the same columns of A, so A crosses the bus once. The
  intensity is n/4 flop per byte (8 at n = 32) against a ridge of ~40, so the kernel is bandwidth bound.
- **Only the triangle's thread tiles are carried.** 136 of 256 tiles meet the triangle at n = 128, so 160
  threads instead of 256. Masking the epilogue alone left the block doing a GEMM's arithmetic.
- **Layout.** The shared tile is `[k][n]` with stride exactly n, so fragment loads are 16-byte aligned and
  issue as LDS.128. A padded stride has no headroom.
- **Packet swizzle.** Packet q of reduction row kk lives at `q ^ (kk/4)`, which makes staging writes and
  fragment loads conflict-free.
- **Contiguous thread rows are a correctness invariant**; see [the band-split syrk bug](#the-band-split-syrk-bug).
- **Thread tile by scalar.** Float uses a 4-wide tile (544 threads), the fastest measured. A complex scalar
  needs **205 registers per work-item**, and 544 x 205 is past the 65,536 a work-group gets. An 8-wide
  tile (160 threads) fits.
- **Rejected: a 128-bit staging load.** Slower everywhere (n = 64 batch 1024: **0.333 -> 0.384 ms**).

### syrk and syr2k triangular tiles: kernel design

`src/backends/syrk_triangular_tiles.hh` and `src/backends/syr2k_triangular_tiles.hh`, sharing the grid in
`triangular_tiles.hh`.

- **The saving.** A GEMM-based batched SYRK computes every tile outside the triangle and discards it. The
  grid is indexed over the triangular tile set (`triangular_tile_decode`), so those tiles are never launched.
  Diagonal tiles are computed whole and their epilogue drops the unreferenced half.
- **Inner loop.** An aligned shared stride (LDS.128), operands staged `[k][row]`, and the 8x8 thread tile
  split into two 4-wide bands 64 apart so each LDS.128 is bank-conflict free.
- **syr2k is fused.** Two batched GEMMs aimed at C would clobber the caller's other half and stream C three
  times. Both products accumulate into one set of accumulators, four staged 128x8 tiles per k step (16 KB).
- **Diagonal aliasing.** On the diagonal the bj-side tile pointers alias the bi-side ones. `on_diagonal` is
  uniform across the block, so this costs no divergence.

### trmm triangular tiles: kernel design

`src/backends/trmm_triangular_tiles.hh`.

- **The triangle was never exploited before.** `src/extensions/trmm.cc` recursed to a 256-wide block and then
  called the GEMM it was meant to replace. On CUDA the `expand` family (now second on every Left row) ran a full GEMM.
- **A loop bound, not a mask.** For `C = alpha*op(A)*B` with op(A) upper triangular, output row i touches only
  p >= i, so an output tile rooted at row m0 starts its reduction at p = m0. Only the k-tile straddling the
  diagonal is masked. Which end is skipped is `lower_eff = (uplo == Lower) != transposed`.
- **The tile is sized to m.** ormqr's `W2 = T^H W1` has m = ib in the tens, where a 128-row tile would spend
  four times the arithmetic.
- **Lanes.** A 128-wide side gets 16 lanes of 8. Narrower sides keep the 4-wide band and drop lanes, so
  ThreadRows must stay a multiple of 4.
- **Complex thread tile at TileM >= 64.** A complex scalar halves what the shared path delivers per clock. A
  4x8 thread tile needs 2.67 MAC per load against a capability of 2; a 4x16 tile reaches 3.2. Not at TileM 32:
  halving the columns there measured **0.374 -> 0.424 ms** at m = 32.
- **Staging follows op(A)'s contiguous direction.** A warp walking a transposed A strides by lda, and each lane
  pulls its own 32-byte sector; that was the dominant cost before the split.

### trmm: choosing the row tile by scalar type

`trmm_row_tile` trades arithmetic against re-reading B: each row tile stages its own copy of the reduction
range, up to `(R+1)/2` times. Float is bandwidth bound (ridge near 40 flop per byte), so the widest tile that
fits wins. Double and complex run at 1/64 rate (ridge near 1.4), so the narrowest tile wins.

Tile16 against tile32, through `ormqr_blocked_benchmark` (n 256/512, batch 128-256, ib 16/32/64):

| type | tile16 / tile32 | verdict |
|---|---|---|
| `double` | 1.007x - 1.022x | wins at every ib, including 64 |
| `complex<double>` | 0.995x - 1.040x | wins to ib 32, a wash at 64 |
| `complex<float>` | 0.993x - 1.026x | wins to ib 32, a wash at 64 |
| `float` | 0.966x - 0.997x | loses everywhere (the B re-read) |

Shipped thresholds: wide types take 16 through m = 64; complex takes 16 only through m = 32; float takes 32
through m = 32, 64 through m = 512, and 128 above. **Wide types never take 128**: an 8x8 tile in
`complex<double>` is 256 accumulator registers, and 256 threads x 256 is the whole 65,536 a work-group gets,
so the runtime rejects the launch. `BATCHLAS_TRMM_TILE_M` pins the tile for sweeps.

### Level-3: the complex MAC and return-by-value rules

`src/backends/triangular_tiles.hh`, shared by every tile kernel.

- **`accumulate` writes the complex MAC out by hand.** `std::complex<float>::operator*` lowers to the
  `__mulsc3` libcall (C99 Annex G), with a branch on Inf and NaN around every multiply. The first complex Gram
  build ran at **1.2 TFLOP/s against float's 13.8**. Four real multiplies and two adds give the four FMAs a
  complex MAC should be.
- **Return the accumulator by value.** A `T&` out-parameter into the register-resident accumulator array moves
  the whole array to local memory: **43%** on float at m = 512 (0.659 -> 0.944 ms).
- **`tile_load4` is 128-bit only for float.** Four doubles (32 bytes) and four `complex<double>` (64 bytes) have
  no load form. The reinterpret asserts an alignment `sycl::local_accessor` never promised, and it fails silently.

## Negative results {#level3-negative-results}

A specialised level-3 op beats the GEMM it replaces only when it reaches a batched native kernel. Where the row
or rule does not reach one, the op becomes a host loop one to two orders of magnitude slower at batch 1024 and
up. Check which family ran with `BATCHLAS_SELECT_TRACE=1`.

### syrk for the ortho gram matrix

`src/extensions/ortho.cc` builds `C = A^H A` three times. Plain syrk was a **70–100x regression** at these
shapes, because a skinny Gram matrix failed both routers and dropped to one `cublasSsyrk` per batch member
(m=256 k=32 batch 2048: GEMM 0.350 ms, syrk 33.73 ms). That motivated `syrk_gram_tiles`. Once it existed the
substitution was taken: at k = 32 float 1.450 -> **0.895** ms (1.62x), and ShiftChol3 1.998 -> **1.298** ms
(1.54x) at m = 1024 batch 512. `gram_max_k` is 64 for float and 128 for double (`src/extensions/ortho.cc:143`):
float k=128 is 0.96x, double k=128 is 1.34x. `svqb_alg` keeps its GEMM, since a one-triangle result would
multiply uninitialised workspace. The precision table is at
[the Gram matrix through syrk, per precision](ortho.md#ortho-the-gram-matrix-through-syrk-per-precision).

### herk on the gram tile kernel

Built and rejected. In complex float the conjugating Gram route loses to GEMM-plus-Hermitian-fold at every shape:
0.217 vs **0.206** ms at n=32 batch 2048, and 2.08 vs **1.57** ms at n=128 batch 512. A complex multiply is four
real ones, so herk is compute bound where real syrk is bandwidth bound. The route stays reachable as
`BATCHLAS_SYRK_ROUTE=gram` so the conjugation stays under test.

### trmm for the WY block factor

Before the tile kernel, trmm for `W2 = T^H W1` lost at every shape. Re-measured with the tile kernel against
`BATCHLAS_ORMQR_WY=gemm` (ABBA order, `Side::Left`, `ConjTrans`, two agreeing passes), gemm/trmm over all cells:

| type | gemm/trmm | verdict |
|---|---|---|
| `float` | 1.006x–1.046x | wins everywhere |
| `double` | 1.004x–1.016x | wins everywhere; newly enabled |
| `complex<float>` | 0.944x–0.995x | loses everywhere; excluded |
| `complex<double>` | 0.958x–1.010x | loses at ib = 16; excluded |
| `netlib float` / `netlib double` | 0.336x–1.199x / 0.379x–1.064x | excluded (per-batch cblas loops both ways) |

`wy_trmm_applicable` (`src/extensions/ormqr_blocked.cc:52`) is therefore per type, plus `ib <= 64`. The effect
on syev is small (1.036x at n=64 batch 2048, 1.003x at n=512). The 16-row tile moved `double` to 1.013x–1.036x
and closed `complex<double>`'s ib = 16 hole to parity; it is kept for double.

### Rejected on inspection

- `X^H A X` (`syevx_lobpcg.cc`, `syevx_filtered.cc`): a product of two different matrices; `syr2k` does not express it.
- `A X` with symmetric A (`syevx_lobpcg.cc`, `lanczos.cc`, `ritz_values.cc`): symm would expand and then GEMM, adding a copy.
- `gebrd_blocked.cc:364,365`: looks like syr2k, but `a22` is not symmetric.
- Transcribing the thresholds into the old `RouteTable::preferred` was killed: a `tiles_per_side >= 3` rule rejected
  the tile route for `129 <= n <= 383`, sending n = 256 to a kernel that wrote **both triangles**.

## Correctness findings {#level3-correctness-findings}

### The band-split syrk bug

The 128-wide Gram tile first split each thread's 8 rows into two 4-wide bands 64 apart, as the square kernels
do. That is incompatible with taking the triangle at thread-tile granularity: thread (0,1) owns element (64,4),
in the lower triangle while its tile is not, so nothing wrote it. It was silent and appeared only at n > 64.
`SyrkTest.NarrowShapesMatchGemmReference` (`tests/syrk_tests.cc:91`) sweeps n in {24,32,48,64,96,128} x trans x
uplo at k = 200 and mirrors both sides before comparing.

### The herk test that could not fail

`HerkTest` could not catch conjugating the wrong operand, which returns `conj(C)`: still Hermitian and consistent.
`HerkTest.MatchesGemmReference` (`tests/herk_tests.cc:314`) compares against a GEMM and fails when the conjugation flips.

### The trmm poison test

Other trmm tests used `RandomTriangular`, already zeroed opposite the triangle, which passes an implementation that
ignores `uplo` and `diag`. `TrmmTest.IgnoresUnreferencedTriangleAndUnitDiagonal` (`tests/trmm_tests.cc:132`)
poisons the forbidden storage and compares against a GEMM on the clean A, over both sides, uplo, diag, three
transposes and ragged shapes. On CUDA it re-runs with `BATCHLAS_EXPAND_MAX_BYTES=0` to reach the `cublas?trmm` loop.

### The syr2k trailing-update test

`SytrdBlockedTest.TrailingUpdateRoutesAgree` (n=320, nb=32) checks the syr2k route against the GEMM route. Forcing
`alpha = -0.5` fails it (worst eigenvalue error 2.777 against a 3.2e-3 bound; the GEMM route is at 2.6e-6). The
assertion is relative: `4 * (GEMM route error) + 8 eps ||A||`. The same change removed a symmetrize pass: float
n=512 batch 1024 goes 263.97 -> **227.51** ms (nb=16). The gate stays CUDA with `float` or `complex<float>`
(`rank2k_trailing_update_supported`, `sytrd_blocked.cc:785-787`). In double it **inverts**: 7.56 vs 58.52 ms at
n2=256 batch 1024 (**7.7x slower**), because double syr2k has only the per-batch `vendor` loop.

### Level-3: generic syrk and syr2k fallbacks fold one triangle

The generic fallbacks in `src/extensions/syrk.cc` and `syr2k.cc` (used where no native `?syrk`/`?syr2k` is wired,
MKL) once wrote **both** triangles and overwrote the half the caller owns. The product now goes to scratch, and
`fold_symmetric_product_into_triangle` (`src/extensions/symmetric_product_fold.hh`) folds only the named triangle:
`C(i,j) := product(i,j) + beta*C(i,j)`. With beta == 0, C is not read, so a poisoned C cannot produce NaN.

### her2k alpha alignment fault

cuBLASLt, reached through `cublas?her2k`, reads the host `alpha` with a 16-byte aligned vector load.
`std::complex<double>` is 8-byte aligned, so the call faults when the address lands at 8 mod 16. It is
shape-dependent and reproduces against cuBLAS 13.2 with no BatchLAS present. The fix is
`alignas(16) T alpha_aligned = alpha` in `her2k_vendor` (`cublas.cc`).

## Level-3: code-structure decisions

Each item is a decision a future edit could undo by accident.

### Level-3: one set of shape validators

`src/backends/level3_shape.hh` holds the shape contracts once: squareness first, then batch, then shapes. Only
the exception type differs by backend (template parameter `E`): cuBLAS and rocBLAS throw `std::invalid_argument`,
netlib's host tasks throw `std::runtime_error`.

- **Namespace trap.** The validators live in `batchlas::backend::shape`, not `batchlas::backend::detail`. In
  `netlib_lapack.cc` a `backend::detail` makes the unqualified `detail::` resolve wrongly and `submit_host_task`
  stops being found.
- **The vendor family never calls the public entry.** A fallback that did would re-enter the same gate and recurse
  without bound.
- **One route parse per variable.** `BATCHLAS_EXPAND_ROUTE` comes from one Settings field that `expansion_route_pin`
  and `expansion_preferred` both read, so the two cannot disagree.

### Level-3: scratch expansions and their ceilings

`src/backends/triangular_expand.hh` and `src/expansion_budget.hh`. Consumers are symm's and trmm's `expand` families
(`expansion_fits` in `can_run`) and the fixed rules of `hemm_vendor`, `herk_vendor` and `her2k_vendor`.

- **Why an expansion.** SYMM, HEMM and TRMM must not touch the unreferenced triangle (TRMM under `Diag::Unit` must not
  touch the diagonal; HEMM not its imaginary part), and a GEMM aimed at the caller's A would read all of them. The
  scratch is a workspace lease, not a `Matrix`: a `Matrix` migrates pages on first touch, which at n = 512 batch 512
  costs several times the GEMM it feeds, and it would be freed while kernels are still enqueued.
- **Mirrored expansion.** It stages one tile pair per work-group through local memory: 1.5 n^2 of traffic, against
  3 n^2 for a copy plus an in-place symmetrize.
- **Two hard ceilings.** SYCL linearises the global id into an `int`, so one work-item per element fails at 2^31
  elements (a thrown `sycl::exception` at n = 2048 batch 512). The scratch also shares the device with A, B and C,
  so the budget is a quarter of global memory. `BATCHLAS_EXPAND_MAX_BYTES` lowers the memory ceiling. A shape past
  either ceiling fails `can_run`, and the row's next entry runs.
- **Why `expansion_budget.hh` is outside `src/backends/`.** `src/extensions/sytrd_blocked.cc` must know whether her2k
  takes its GEMM route, and it is also built for ROCm and the host. There is one definition, because a reimplemented
  ceiling drifts silently.
- **A guard must model the whole predicate.** The her2k guard in `sytrd_blocked` once replicated only the size ceiling,
  and under `BATCHLAS_EXPAND_ROUTE=loop` it assumed the GEMM route while the loop ran. Hence `her2k_takes_gemm_route`
  and `expansion_route_pin` (`1` = expand, `0` = loop, `-1` = unset).

### trmm: the generic recursion reads the whole square of A

`src/extensions/trmm.cc` (the MKL instantiation) ends its recursion at n <= 256 in a plain `gemm` on the diagonal block
with `beta = 1`. Read as written, the base case uses A's unreferenced triangle and its stored diagonal under
`Diag::Unit`, and accumulates into C rather than overwriting it. It matches BLAS only when the caller zeroed the
opposite triangle and C. This was found by reading and is **not confirmed by a test**. A poison test in the style of
[the trmm poison test](#the-trmm-poison-test) against the MKL backend would settle it.

## Open debts {#level3-open-debts}

- **trmm tile kernel is `Side::Left` only.** Every Right row ranks `expand` first. `ormbr` has the same WY update, is
  not wired, and feeds gesvd.
- **ROCm has no native level-3 path** and no hemm, herk or her2k; its `syrk`, `syr2k` and `trmm` take the rocBLAS loop.
- **`her2k_gemm_preferred` was swept only over square rank-k shapes.** The narrow panel case (k = nb in {16,24,32},
  n2 up to 480) awaits an A/B at n2 in {224,480}, `complex<float>`. `complex<double>` is unmeasured.
- **The tables are transcribed, not timed.** Suspects: `sq` rows that rank `triangular` from n = 129 (only n >= 384
  was measured to win); double `symm` rows that rank `vendor` first; `complex<float>` Left `trmm` rows that rank
  `triangular` first (it loses below m = 512). Retune by sweeping the families on the saturated grid (`tools/tune`).
- **Unverified boundaries:** `257 <= n <= 383` for `syrk` `triangular`; batch 3 and `769 <= n <= 895` for herk; batch 1
  with `n >= 128` for her2k; the slack between the symm/hemm loss region and the shipped constants.
- **Vendor-free gaps.** Still `NoRouteError` in a vendor-free build: hemm, herk, her2k, double syrk above n = 128, and
  double syr2k. Their rows hold only `vendor`. Read the per-op census, not the pass count.
- **Instrumentation.** `symm_benchmark`, `syrk_benchmark` and `syr2k_benchmark` abort on the host backend at tiny shapes
  (SYCL assertion `adjustNDRangePerKernel: NDR.LocalSize[0] == 0`). Recorded 2026-09, not re-checked.
- **Kernel headroom.** The trmm tile kernel runs at ~70% of cuBLAS's per-flop rate; closing that would turn m = 128..256
  into a ~1.2x win. syrk at n = 256 (0.84x–1.22x) is a pre-existing loss.
- **Not the lever for syev.** At n = 512 with eigenvectors the two-stage path splits `backtransform_q2` 46.4%,
  `sb2st_hh` 25.5%, `stedc_eigvecs` 10.6%; none is a level-3 triangular op.

## Level-3: raw evidence

Raw data is at the git tag `perf-evidence/vendor-independence` (`git show <tag>:<path>`). Per-cell CSV captures were
not committed; the numbers on this page come from the distilled notes and the route comments.

| topic | path |
|---|---|
| syrk gram tiles; trmm tiles; per-dtype tables; the 16-row tile; ortho wiring; ormqr WY; syev traces | `experiments/TRMM_SYRK_BATCHED_KERNELS.md` |
| the 73-call-site substitution survey; sytrd syr2k trailing update; ortho gram and WY-factor rejections | `experiments/GEMM_TO_LEVEL3_SURVEY.md` |
| syrk and syr2k crossover drivers | `experiments/syrk_sweep.sh`, `experiments/syrk_kskew.sh`, `experiments/syr2k_sweep.sh` |
| herk/her2k expand-vs-loop driver; the exclusive-GPU guard | `experiments/herk_crossover.sh`, `experiments/gpu_guard.sh` |
| WP1 design pass; vendor-free failing set and `NoRouteError` census | `WP1_LEVEL3_SPEC.md`, `VENDOR_FREE_BASELINE.md` |

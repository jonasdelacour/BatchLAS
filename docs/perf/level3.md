# Level-3: symm, hemm, syrk, herk, syr2k, her2k, trmm {#perf_level3}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, `RelWithDebInfo` · measured 2026-09-05 and 2026-10-22/26; sm_89 tables transcribed, sm_120 tables deep

Routing, kernel design and the measurements behind the seven level-3 ops. All seven choose from tuned tables.
The sm_89 rows for `symm`, `syrk`, `syr2k` and `trmm` are transcribed from the deleted hand-written gates (at
`ff340fc6`) and are untimed except where a section below says `custom`; `hemm`, `herk` and `her2k` transcribe the
deleted cuBLAS-backend rules (at `409c2fb4`). The sm_120 tables are deep-measured (`tuned/README.md`). gemm itself
is on @ref perf_gemm, trsm on @ref perf_trsm.

## Level-3: how the table-selected ops choose

`src/ops/<op>/choice.hh` lists the families; `src/ops/<op>/<op>.cc` validates shapes (`std::invalid_argument` on
every backend), returns a no-op event for an empty problem and calls `select::run` on `tuned/<op>.<device>.txt`.
@ref selection_tables lists the families per op.

### Level-3: the families and their can_run terms

Every native family needs a native GPU backend (`CUDA` or `SYCL`; the gemm-based `expand` and `fold` also
`ROCM`), `d.is_gpu`, homogeneous operands and batch \f$\ge 1\f$. Every native launch splits the batch into sub-views of at
most `Device::max_groups_yz` items (`for_each_chunk`, `src/ops/level3/batched.hh`; grid z is 65535 on CUDA), so
`can_run` has no batch term. `vendor` needs `d.has_vendor` and homogeneous operands; hemm's, herk's and her2k's also
refuse ROCm.

| op | spelling | driver | further `can_run` terms | dtypes |
|---|---|---|---|---|
| symm, hemm | `mirror` | `symm_mirror<T, Herm>`: wide v2's gemm tile staging A from its named triangle | `max_wg >= 256`; `mirror::slm_bytes<T>()` fits; \f$\lceil m / \mathrm{tile} \rceil \le\f$ `max_groups_yz` | symm all four (double: 16x16 tiled kernel); hemm complex |
| symm, hemm | `expand` | `expand_mirrored[<Conjugate>]`, then the public `gemm`, per chunk | `max_wg >= 256`; one item's scratch fits | symm all four (complex is complex-symmetric); hemm complex |
| syrk | `gram` | `syrk_gram_tiles`: one tile covers C | \f$n \le\f$ `kGramMaxTile` (128); `max_wg >= gram_threads(n)`; tile fits SLM | all four |
| syrk | `triangular` | `syrk_triangular_tiles`: 128x128x8 tiles of one half | `max_wg >= 256`; two staged tiles in SLM; \f$T(T+1)/2 \le 65535\f$, \f$T = \lceil n/128 \rceil\f$ | all but cdouble |
| syr2k | `triangular` | `syr2k_triangular_tiles`: both products fused | as syrk's, four staged tiles; real `ConjTrans` runs as `Trans` | all but cdouble |
| herk | `gram` | `syrk_gram_tiles<T, Conjugate>` | as syrk's `gram` (8-wide thread tile at \f$n = 128\f$) | complex |
| herk, her2k | `triangular` | `syrk_` / `syr2k_triangular_tiles<T, Conjugate>` (her2k stages \f$A\f$ times \f$\alpha\f$) | as syrk's / syr2k's | cfloat |
| herk, her2k | `fold` | the public `gemm` of the whole product into \f$n \times n\f$ scratch, then `fold_hermitian` (two-sided for her2k), per chunk | `max_wg >= 256`; one item's scratch fits | complex |
| trmm | `triangular` | `trmm_triangular_tiles`: row tiles that skip the zero half | `max_wg >= 256`; `Side::Left`; `trmm_tile_groups` \f$\le 65535\f$ | all four |
| trmm | `expand` | `expand_triangular`, then `gemm` at \f$\beta = 0\f$, per chunk | `max_wg >= 256`; one item's scratch fits | all four |
| all | `vendor` | `backend::<op>_vendor`: `?<op>` per item | above | all (ROCm symm/syrk/syr2k real only) |

A real `ConjTrans` is `Trans` to both syrk tile kernels, so the exact `trans` key keeps it on the vendor. The last
resort when no row entry runs is `mirror, expand, vendor` (symm, hemm), `triangular, gram, vendor` (syrk),
`triangular, vendor` (syr2k), `expand, triangular, vendor` (trmm), `fold, triangular, gram, vendor` (herk) and
`fold, triangular, vendor` (her2k). complex<double> has no `triangular`: 64 accumulators of 16 bytes fill the
255-register file, and her2k's four staged tiles would need 64 KB of SLM.

### Level-3: what the shipped tables rank first

From the sm_89 rows (float and double for symm, syrk, syr2k; all four for trmm). The transcription matches the old
Auto choice on 100% of 2500-3000 random off-grid points per op, dtype and device.

| op, dtype | ranked first |
|---|---|
| `syrk` float | `gram` at \f$n \le 128\f$; `triangular` on every `sq` row above \f$n = 128\f$ and on `tall`/`wide` rows with \f$n \ge 257\f$, \f$k \ge 8\f$, \f$\mathrm{batch} \times T(T+1)/2 \ge 160\f$; else `vendor` |
| `syrk` double | `gram` at \f$n \le 128\f$; `vendor` elsewhere |
| `syr2k` float | `triangular` at batch \f$\ge 2\f$; `vendor` at batch 1 |
| `syr2k` double | `vendor` (the only candidate) |
| `symm` float | `expand` on `sq` rows with batch \f$\ge 4\f$ or \f$\max(m, n) \ge 256\f$, `vendor` on the other `sq` and every `tall`/`wide` row; 80 `custom` rows at [the mirror grid points](#level-3-mirror-measured-sm89) |
| `symm` double | `vendor`, `expand` second |
| `trmm`, all four | `triangular` on every `side=L` row, `expand` on every `side=R` row, `vendor` last |
| `hemm` | `expand` iff batch \f$\ge 4\f$ or \f$\max(m, n) \ge 256\f$ (edges m, n 255\|256; batch 3\|4), else `vendor`; 76 cfloat `custom` rows at the mirror grid points |
| `herk` | `fold` iff batch \f$\ge 4\f$ and \f$n \le 768\f$ (edges n 768\|769, batch 3\|4), else `vendor`; cfloat `triangular` in [the measured windows](#level3-hermitian-families-measured-sm89) |
| `her2k` | `fold` iff batch \f$\ge 2\f$ or \f$n \ge 128\f$ (edges n 127\|128, batch 1\|2), else `vendor`; cfloat `triangular` in the measured windows |

Complex-symmetric `syrk`/`syr2k` cfloat rows: `gram` first at \f$n \le 64\f$, `triangular` up to \f$n = 768\f$, at batch \f$\ge 4\f$
(syr2k \f$\ge 2\f$), ranked from the herk/her2k timings (same kernels without the conjugate). Complex `symm` rows copy
hemm's. A vendor-free build, `Backend::SYCL` and ROCm walk past `vendor` to the first native entry.

### Level-3: pinning a family

`BATCHLAS_<OP>_ROUTE` takes `auto`, `native`, `vendor` or a spelling above; one not compiled for the dtype or unable
to run the shape throws `std::invalid_argument`. Pin words: [the level-3 pin words](dispatch.md#the-level-3-pin-words).
`BATCHLAS_TRMM_ROUTE=vendor` is always the `cublas?trmm` loop; herk's Gram kernel is `BATCHLAS_HERK_ROUTE=gram`.

### Level-3: dtype reach of the native families

Every GPU shape of every level-3 op has a native route (vendor-free included) except an item too large for its own
scratch (`expand`/`fold` refuse it; `trmm` Left and `syrk`/`syr2k`/`herk` still have their tile kernels).
`level3_tile_substitution_enabled` (`src/select/vendor.hh`) stays float-or-CUDA-with-cuBLAS because `sytrd_blocked`,
`ortho`, `ormqr_blocked` and the coverage census read it; widening it moves their routes.

### Level-3: hemm, herk and her2k families measured on sm_89 {#level3-hermitian-families-measured-sm89}

`batchlas_tune <op> --dtype cfloat --reps 8 --warm 0.5 --no-refine`, every arm verified, RTX 4090, vendor build,
2026-10-22; ms for the whole batch, median of the second pass. The per-item vendor loop loses 5x-200x at every
batch \f$\ge 128\f$ cell below \f$n = 1024\f$. ConjTrans herk matches NoTrans within 3%.

| op, shape | batch | winner | runner-up | other |
|---|---|---|---|---|
| herk N \f$n=128\f$ \f$k=8\f$ | 1024 | `triangular` 0.353 | `fold` 0.420 | vendor 3.79 |
| herk N \f$n=256\f$ \f$k=64\f$ | 1024 | `triangular` 1.64 | `fold` 1.75 | vendor 7.64 |
| herk N \f$n=256\f$ \f$k=512\f$ | 1024 | `fold` 5.97 | `triangular` 6.68 | vendor 34.5 |
| herk N \f$n=512\f$ \f$k=8\f$ | 128 | `triangular` 0.480 | vendor 0.699 | `fold` 0.774 |
| herk N \f$n=768\f$ \f$k=8\f$ | 128 | `triangular` 0.920 | vendor 1.03 | `fold` 1.70 |
| herk N \f$n=1024\f$ \f$k=8\f$ | 128 | vendor 1.415 | `triangular` 1.509 | `fold` 2.96 |
| her2k \f$n=256\f$ \f$k=8\f$ | 1024 | `triangular` 1.19 | `fold` 1.87 | vendor 6.99 |
| her2k \f$n=256\f$ \f$k=512\f$ | 1024 | `fold` 6.40 | `triangular` 11.3 | vendor 63.6 |
| her2k \f$n=1024\f$ \f$k=64\f$ | 128 | `triangular` 3.67 | vendor 3.70 | `fold` 4.01 |

- **herk cfloat:** `triangular` first at \f$256 \le n \le 768\f$ (except \f$n = 256\f$, \f$k \ge 512\f$, where `fold` wins by 12%) and
  at \f$n \le 128\f$ with \f$k \le 8\f$ and batch \f$\ge 1024\f$; \f$n > 768\f$ keeps `vendor` first.
- **her2k cfloat:** `triangular` is two fused products and pays for k: first only at \f$k \le 8\f$ (\f$n \ge 256\f$, or \f$n \ge 128\f$
  at batch \f$\ge 1024\f$) and at \f$n \ge 1024\f$ with \f$k \le 64\f$; `fold` elsewhere.

complex<double> rows keep the transcribed rule (FP64 is not tuned on a 1:64 card). Vendor-free, `fold` and
`expand` feed the native gemm; tested, not timed.

## Level-3: the mirror family {#level-3-the-mirror-family}

`mirror` (symm and hemm) computes the product without the \f$n \times n\f$ scratch `expand` writes. It serves
`syev_two_stage`'s sy2sb update \f$Y = A_{22} P\f$ (Side::Left, Lower, \f$P\f$ \f$\mathrm{pn} \times \mathrm{pk}\f$ with \f$\mathrm{pk} \le 32\f$), which reads only \f$A_{22}\f$'s
lower triangle (defect 14).

### Level-3: mirror's tile loader {#level-3-mirror-tile-loader}

`src/ops/symm/kernels/symm_mirror.cc` is wide v2's 64x64 double-buffered gemm (`register_64x64_k16_wide.hh`, see
@ref perf_gemm) with its own loader for the symmetric operand (A on the Left, the gemm's B on the Right). Per slab,
uniformly over the work-group:

- strictly inside the named triangle: the gemm's natural load;
- strictly inside the other triangle: the other staging map, reading the mirror `A(kk, i)` contiguously (hemm
  negates the imaginary parts in registers);
- touching the diagonal: element by element, the address chosen per element, hemm's diagonal imaginary part zeroed.

The gemm headers are not edited (sm_120's deep gemm records hash them). complex<double> takes the \f$k = 8\f$ slab
(32 KiB); double has no wide v2 tile and runs gemm's 16x16 tiled kernel with the same mirrored read. Limits are in
`src/ops/symm/mirror_tile.hh`.

### Level-3: mirror measured on sm_89 {#level-3-mirror-measured-sm89}

`batchlas_tune {symm --dtype float, hemm --dtype cfloat} --reps 8 --warm 0.5 --no-refine`, verified, 2026-10-26,
median ms, \f$n = 32\f$ (the sy2sb panel):

| op | m | batch | mirror | expand | vendor |
|---|---|---|---|---|---|
| symm float | 256 | 1024 | **0.281** | 0.871 | 19.9 |
| symm float | 2048 | 128 | **2.15** | 6.72 | 17.7 |
| hemm cfloat | 1024 | 128 | **1.36** | 3.13 | 6.61 |
| hemm cfloat | 512 | 8 | 0.068 | **0.052** | 0.214 |

On the level-3 grid `mirror` wins every float cell with \f$m\f$ or \f$n \le 128\f$ and every `tall` one; `expand` wins the square
float cells from 512 up by 1.15-1.25x (\f$1024 \times 1024 \times 128\f$: 9.06 vs 7.75 ms) and hemm cfloat at \f$512 \times 512\f$ by 1.06x;
hemm at batch 8, \f$255 \le m \le 512\f$ goes to `expand`. The rows are `custom` on \f$32 \le m \le 2048\f$, \f$8 \le n \le 1024\f$, batch 8/128/1024;
double and cdouble keep transcribed rows, and sm_120's deep table predates `mirror`.

### Level-3: mirror's deliberate breaks {#level-3-mirror-deliberate-breaks}

`tests/symm_mirror_tests.cc` pins `mirror` per scalar with poisoned A, B, C at \f$\mathrm{ld} = \mathrm{rows} + 3\f$, the unnamed triangle
poisoned, complex \f$\alpha\f$/\f$\beta\f$, extents straddling the 64-row tile and 16-deep slab, \f$\beta = 0\f$ over NaN, batch 1024
bit-identical, grid z 65535 | 65536 and grid y 65535 | 65536 row tiles. Each break below was one edit, run on sm_89:

| break | red (of 6 cases per scalar) |
|---|---|
| mirrored slab not conjugated (hemm) | cfloat, cdouble: 4 |
| hemm's diagonal imaginary part kept | cfloat, cdouble: 5 |
| Right side's triangle not transposed | float, cfloat, cdouble: 4 |
| diagonal slab classed inside the named triangle | float, cfloat, cdouble: 5 |
| batch stride derived as \f$\mathrm{ld} \times k\f$ | float, cfloat, cdouble: 5 |
| double's tiled read ignoring `uplo` | double: 4 |

## Boundaries and their evidence

Measured against the old gates (2026-09-05, float unless stated); the transcribed sm_89 tables reproduce them.

| route edge | admit side | bracketing non-winner |
|---|---|---|
| syrk `triangular`: \f$T \ge 3\f$ tiles a side (\f$n \ge 257\f$) | 1.45x \f$n=512\f$ \f$b = 512\f$; 1.63x \f$n=1024\f$ \f$b = 64\f$; every saturated shape from \f$n = 384\f$ | \f$n = 256\f$: 0.84x-1.22x (a diagonal tile is computed whole, so one tile in four is saved) |
| syrk `triangular`: \f$\mathrm{batch} \times T(T+1)/2 \ge 160\f$ | won from 168 blocks | 144 blocks 1.14x slower; 136 blocks 1.25x slower |
| syrk `gram`: \f$n \le 128\f$ | 4.29x vs gemm at \f$n=32\f$ \f$b = 2048\f$ (933 GB/s, the roofline); 2.01x at \f$n=64\f$ | \f$n = 128\f$: 0.99x float, 2.10x double (FP64 makes it compute bound) |
| syr2k `triangular`: batch \f$\ge 2\f$ | 1.06x-226x every shape | batch 1: vendor 1.18-1.60x below \f$n = 1280\f$, 4-10x at deep k, small n |
| trmm `triangular` on every Left row | \f$m=128\f$ \f$n_C=512\f$ \f$b = 1024\f$: 0.698 vs 0.784 ms expand | \f$m=256\f$ \f$n_C=1024\f$ \f$b = 256\f$: 0.915 vs **0.855** (left unfitted) |
| symm/hemm `expand`: batch \f$\ge 4\f$ or \f$\max(m, n) \ge 256\f$ | 1.2x-72x over the per-item loop | batch \f$\le 2\f$, \f$n \le 128\f$: loses by up to 2.5x |
| trmm `expand` on every Right row | 1.15x-162x (49 square cells), 1.22x-32x (64 skewed) | none (`cublas?trmm` has a ~110 us floor) |
| herk `fold`: batch \f$\ge 4\f$ and \f$n \le 768\f$ | 1.6x-72x at \f$n \le 512\f$ | batch \f$\le 2\f$ a wash or loss; 0.82x-0.93x from \f$n = 896\f$ |
| her2k `fold`: batch \f$\ge 2\f$ or \f$n \ge 128\f$ | 1.4x-128x | batch 1: 0.74x at \f$n=32\f$, 0.89x at \f$n=64\f$ |

The \f$k \ge 8\f$ syrk edge is the old gate's `kTriangularTileK`, not measured. The symm/hemm shipped edge is more
conservative than its loss region (the data would support 2 and 128). herk/her2k grid: complex64, \f$32 \le n \le 1024\f$ by
\f$1 \le \mathrm{batch} \le 256\f$ (`experiments/herk_crossover.sh`). herk needs the \f$n\f$ ceiling because its gemm computes both triangles,
and one `cublas?herk` already saturates the device there. Whether to spell a product as trmm at all (trmm vs gemm per
dtype: double and cdouble win 1.05x-2.02x everywhere, float loses at \f$m = 256\f$, cfloat below \f$m = 512\f$) is in
`experiments/TRMM_SYRK_BATCHED_KERNELS.md`.

## Level-3: kernel design notes

### syrk gram tiles: kernel design

`src/ops/syrk/kernels/syrk_gram_tiles.hh`: a single-tile batched SYRK for the tall skinny A and small square C of
`ortho`.

- **Why a second kernel.** The 128-wide triangular tile spends \f$128 \cdot 128 \cdot k\f$ arithmetic on an \f$n^2/2\f$ answer, a 32x
  overcharge at \f$n = 32\f$.
- **One shared tile.** Both operands of \f$A^T A\f$ are the same columns, so \f$A\f$ crosses the bus once; \f$n/4\f$ flop per byte
  against a ridge of ~40, so the kernel is bandwidth bound.
- **Only the triangle's thread tiles are carried**: 136 of 256 at \f$n = 128\f$. Masking only the epilogue left a GEMM's
  arithmetic.
- **Layout.** The shared tile is `[k][n]` with stride exactly \f$n\f$ (16-byte aligned LDS.128); packet q of row kk lives at
  `q ^ (kk/4)`, conflict-free for staging and fragment loads.
- **Contiguous thread rows are a correctness invariant**; see [the band-split syrk bug](#the-band-split-syrk-bug).
- **Thread tile by scalar.** Float uses a 4-wide tile (544 threads). Complex needs 205 registers per work-item, and
  \f$544 \times 205\f$ exceeds the 65,536 per work-group; an 8-wide tile (160 threads) fits.
- **Rejected:** a 128-bit staging load (\f$n = 64\f$ \f$b = 1024\f$: 0.333 -> 0.384 ms).

### syrk and syr2k triangular tiles: kernel design

`syrk_triangular_tiles.hh` and `syr2k_triangular_tiles.hh`, sharing the grid in `src/kernels/common/triangular_tiles.hh`.

- **The saving.** The grid is indexed over the triangular tile set (`triangular_tile_decode`), so tiles outside the
  triangle are never launched; diagonal tiles are computed whole and their epilogue drops the other half.
- **Inner loop.** Operands staged `[k][row]` at an aligned stride; the \f$8 \times 8\f$ thread tile split into two 4-wide bands 64
  apart, so each LDS.128 is bank-conflict free.
- **syr2k is fused.** Both products accumulate into one set of accumulators (four staged \f$128 \times 8\f$ tiles, 16 KB per k
  step); two batched GEMMs would clobber the caller's other half and stream C three times. Issuing the two products
  sequentially rather than interleaved is worth 1.53x (3.34 vs 5.11 ms, \f$n = 512\f$ \f$b = 512\f$).
- **Diagonal aliasing.** On the diagonal the bj-side pointers alias the bi-side ones; `on_diagonal` is block-uniform.

### trmm triangular tiles: kernel design

`src/ops/trmm/kernels/trmm_triangular_tiles.hh`.

- **A loop bound, not a mask.** With \f$\mathrm{op}(A)\f$ upper triangular, an output tile rooted at row \f$m_0\f$ starts its reduction at
  \f$p = m_0\f$; only the k-tile straddling the diagonal is masked. The skipped end is
  `lower_eff = (uplo == Lower) != transposed`.
- **The tile is sized to m.** ormqr's \f$W_2 = T^H W_1\f$ has \f$m = \mathrm{ib}\f$ in the tens, where a 128-row tile spends 4x the
  arithmetic. The saving is not 2x: with \f$R = m / \mathrm{TileM}\f$ row tiles the reduction shrinks to \f$\frac{R+1}{2R}\f$.
- **Lanes.** A 128-wide side gets 16 lanes of 8; narrower sides keep the 4-wide band and drop lanes, so ThreadRows
  stays a multiple of 4.
- **Complex thread tile at \f$\mathrm{TileM} \ge 64\f$** is 4x16 (3.2 MAC per load against 4x8's 2.67 and a capability of 2). Not at
  \f$\mathrm{TileM} = 32\f$: halving the columns there measured 0.374 -> 0.424 ms at \f$m = 32\f$.
- **Staging follows op(A)'s contiguous direction**; a warp walking a transposed A by lda pulls one sector per lane.

### trmm: choosing the row tile by scalar type

`trmm_row_tile` trades arithmetic against re-reading B (each row tile restages its reduction range, up to \f$(R+1)/2\f$
times). Float is bandwidth bound, so the widest tile that fits wins; double and complex run at 1/64 rate, so the
narrowest wins. Tile16 / tile32 through `ormqr_blocked_benchmark` (n 256/512, batch 128-256, ib 16/32/64):

| type | tile16 / tile32 | verdict |
|---|---|---|
| `double` | 1.007x - 1.022x | 16 wins at every ib |
| `complex<double>`, `complex<float>` | 0.993x - 1.040x | 16 wins to ib 32, a wash at 64 |
| `float` | 0.966x - 0.997x | 16 loses everywhere |

Shipped: double takes 16 through \f$m = 64\f$; complex takes 16 through \f$m = 32\f$, 32 through \f$m = 64\f$; both take 64 above; float takes
32 through \f$m = 32\f$, 64 through \f$m = 512\f$, 128 above. Wide types never take 128: an \f$8 \times 8\f$ complex<double> tile is 256
registers, x 256 threads is the whole 65,536 of a work-group, and the launch is rejected. `BATCHLAS_TRMM_TILE_M` pins
the tile.

### Level-3: the complex MAC and return-by-value rules

`src/kernels/common/triangular_tiles.hh`, shared by every tile kernel.

- **`accumulate` writes the complex MAC by hand.** `std::complex<float>::operator*` lowers to `__mulsc3` (C99
  Annex G) with an Inf/NaN branch per multiply; the first complex Gram build ran at 1.2 TFLOP/s against float's 13.8.
- **Return the accumulator by value.** A `T&` out-parameter spills the accumulator array to local memory: 43% on
  float at \f$m = 512\f$ (0.659 -> 0.944 ms).
- **`tile_load4` is 128-bit only for float.** Wider scalars have no load form, and the reinterpret asserts an
  alignment `sycl::local_accessor` never promised, failing silently.

## Negative results {#level3-negative-results}

A specialised level-3 op beats the GEMM it replaces only where it reaches a batched native kernel; otherwise it is a
host loop 10-100x slower at batch \f$\ge 1024\f$. Check with `BATCHLAS_SELECT_TRACE=1`.

| Alternative | Result | Verdict |
|---|---|---|
| herk on the Gram tile kernel (cfloat) | loses to gemm + fold: 0.217 vs 0.206 ms (\f$n=32\f$ \f$b = 2048\f$), 2.08 vs 1.57 (\f$n=128\f$ \f$b = 512\f$); complex is compute bound | kept reachable as `gram`, ranked last |
| \f$X^H A X\f$ as syr2k; \f$A X\f$ (symmetric \f$A\f$) as symm; `gebrd_blocked` \f$A_{22}\f$ update as syr2k | two different matrices; adds a copy; \f$A_{22}\f$ not symmetric | rejected on inspection |
| WY block factor \f$W_2 = T^H W_1\f$ as trmm | per type, see [the ormqr WY trmm gate](qr.md#ormqr-wy-trmm-gate) | real types only |
| sytrd trailing update as syr2k | see [sytrd: the rank-2k trailing update](sytrd.md#sytrd-the-rank-2k-trailing-update-in-the-blocked-reduction) | float, cfloat only |

### syrk for the ortho gram matrix

`src/ops/ortho/ortho.cc` builds \f$C = A^H A\f$; through the per-item `cublasSsyrk` loop that was a 70-100x regression
(\f$m=256\f$ \f$k=32\f$ \f$b = 2048\f$: gemm 0.350, syrk 33.73 ms), which motivated `syrk_gram_tiles`. With it, float \f$k = 32\f$ goes
1.450 -> 0.895 ms and ShiftChol3 1.998 -> 1.298 ms (\f$m = 1024\f$ \f$b = 512\f$). `gram_max_k` is 64 for float (\f$k = 128\f$:
0.96x) and 128 for double (\f$k = 128\f$: 1.34x). `svqb_alg` keeps its GEMM: a one-triangle result would multiply
uninitialised workspace. Per precision: [ortho](ortho.md#ortho-the-gram-matrix-through-syrk-per-precision).

## Correctness findings {#level3-correctness-findings}

### The band-split syrk bug

Splitting each Gram-tile thread's 8 rows into two 4-wide bands 64 apart (as the square kernels do) breaks taking the
triangle at thread-tile granularity: thread (0,1) owns (64,4), in the lower triangle while its tile is not, so
nothing wrote it (silent, \f$n > 64\f$ only). `SyrkTest.NarrowShapesMatchGemmReference` sweeps \f$n \in \{24, 32, 48, 64, 96, 128\}\f$ for both `trans` and both
`uplo` and mirrors both sides before comparing.

### Level-3: generic syrk and syr2k fallbacks fold one triangle

`fold_symmetric_product_into_triangle` (`src/ops/potrf/kernels/symmetric_product_fold.hh`, used by `potrf_blocked`)
folds only the named triangle, \f$C_{ij} := \mathrm{product}_{ij} + \beta\, C_{ij}\f$; with \f$\beta = 0\f$ \f$C\f$ is not read, so poison cannot
produce NaN. The deleted `src/extensions/syrk.cc` and `syr2k.cc` wrote both triangles.

### her2k alpha alignment fault

cuBLASLt (through `cublas?her2k`) vector-loads the host `alpha` at 16-byte alignment; `std::complex<double>` is
8-byte aligned, so the call faults at addresses \f$\equiv 8 \pmod{16}\f$ (reproduces on cuBLAS 13.2 alone). The vendor wrappers copy
alpha/beta into `alignas(16)` locals (`src/backends/cublas.cc`).

## Level-3: code-structure decisions

### Level-3: one set of shape validators

`src/backends/level3_shape.hh` holds the shape contracts once (squareness, then batch, then shapes); only the exception
type differs (template `E`: `batchlas::invalid_argument` for cuBLAS, rocBLAS and the public entries,
`batchlas::internal_error` for netlib host tasks).

- **Namespace trap.** They live in `batchlas::backend::shape`: a `backend::detail` makes `netlib_lapack.cc`'s
  unqualified `detail::` resolve wrongly and `submit_host_task` stops being found.
- **The vendor family never calls the public entry**, which would re-enter the same selection and recurse.

### Level-3: scratch expansions and their ceilings

`src/kernels/common/triangular_expand.hh`, `src/expansion_budget.hh`, `src/ops/level3/batched.hh`; used by the
`expand` families of symm, hemm, trmm and the `fold` families of herk, her2k.

- **Why an expansion.** symm, hemm and trmm must not touch the unreferenced triangle (trmm `Diag::Unit` not the
  diagonal; hemm not its imaginary part). The scratch is a workspace lease, not a `Matrix`: a `Matrix` migrates
  pages on first touch (several times the GEMM at \f$n = 512\f$ \f$b = 512\f$) and could be freed while kernels are enqueued.
- **Mirrored expansion** stages one tile pair per work-group through local memory: \f$1.5\,n^2\f$ traffic against \f$3\,n^2\f$
  for copy plus in-place symmetrize.
- **Three ceilings, met by chunking:** grid z (65535 on CUDA); SYCL's `int` global id (\f$2^{31}\f$ elements, a thrown
  `sycl::exception` at \f$n = 2048\f$ \f$b = 512\f$); a scratch budget of a quarter of global memory (`BATCHLAS_EXPAND_MAX_BYTES`
  lowers it). `expansion_chunk` gives the most items one launch takes under all three; the launch leases that once
  and walks the batch in batch / chunk steps. `can_run` refuses only a single item that does not fit.
- **A guard must model the whole predicate.** `sytrd_blocked` and `sytrd_sy2sb` ask her2k's own selection
  (`ops::her2k::picks_batched_route`, `src/ops/her2k/route.hh`) rather than replicating its size ceiling.

## Open debts {#level3-open-debts}

- **trmm tile kernel is `Side::Left` only**; Right rows rank `expand`. `ormbr` has the same WY update, unwired.
- **ROCm** has only the gemm-based families; its syrk, syr2k and trmm Left take the rocBLAS loop. Never run here.
- **sm_89 tables are transcribed, not timed** (except `custom` rows). Suspects: `sq` syrk rows ranking `triangular`
  from \f$n = 129\f$ (only \f$n \ge 384\f$ measured to win; \f$257 \le n \le 383\f$ has no cell); double `symm` ranking `vendor`;
  cfloat Left `trmm` ranking `triangular` (loses to gemm below \f$m = 512\f$).
- **Unmeasured edges:** herk batch 3 and \f$769 \le n \le 895\f$; her2k batch 1 with \f$n \ge 128\f$; her2k narrow panels (\f$k = \mathrm{nb}\f$ in
  \f$\{16, 24, 32\}\f$, \f$\mathrm{n2}\f$ up to 480); the symm/hemm slack.
- **Vendor-free gaps:** complex<double> syr2k (vendor only), complex<double> syrk above \f$n = 128\f$, complex-symmetric
  syrk/syr2k on ROCm.
- **Instrumentation:** `symm_benchmark`, `syrk_benchmark`, `syr2k_benchmark` abort on the host backend at tiny shapes
  (`adjustNDRangePerKernel: NDR.LocalSize[0] == 0`; 2026-09, not re-checked).
- **Kernel headroom:** trmm tiles run at ~70% of cuBLAS's per-flop rate; closing it would make \f$m = 128\ldots256\f$ a ~1.2x win.

## Level-3: raw evidence

Raw data is at the git tag `perf-evidence/vendor-independence` (`git show <tag>:<path>`); per-cell CSVs were not
committed.

| topic | path |
|---|---|
| syrk gram tiles; trmm tiles; per-dtype tables; the 16-row tile; ortho wiring | `experiments/TRMM_SYRK_BATCHED_KERNELS.md` |
| substitution survey; sytrd syr2k trailing update; ortho rejections | `experiments/GEMM_TO_LEVEL3_SURVEY.md` |
| syrk, syr2k, herk/her2k crossover drivers | `experiments/{syrk_sweep,syrk_kskew,syr2k_sweep,herk_crossover}.sh` |

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
`ROCM`), `d.is_gpu`, homogeneous operands and batch >= 1. Every native launch splits the batch into sub-views of at
most `Device::max_groups_yz` items (`for_each_chunk`, `src/ops/level3/batched.hh`; grid z is 65535 on CUDA), so
`can_run` has no batch term. `vendor` needs `d.has_vendor` and homogeneous operands; hemm's, herk's and her2k's also
refuse ROCm.

| op | spelling | driver | further `can_run` terms | dtypes |
|---|---|---|---|---|
| symm, hemm | `mirror` | `symm_mirror<T, Herm>`: wide v2's gemm tile staging A from its named triangle | `max_wg >= 256`; `mirror::slm_bytes<T>()` fits; ceil(m / tile) <= `max_groups_yz` | symm all four (double: 16x16 tiled kernel); hemm complex |
| symm, hemm | `expand` | `expand_mirrored[<Conjugate>]`, then the public `gemm`, per chunk | `max_wg >= 256`; one item's scratch fits | symm all four (complex is complex-symmetric); hemm complex |
| syrk | `gram` | `syrk_gram_tiles`: one tile covers C | n <= `kGramMaxTile` (128); `max_wg >= gram_threads(n)`; tile fits SLM | all four |
| syrk | `triangular` | `syrk_triangular_tiles`: 128x128x8 tiles of one half | `max_wg >= 256`; two staged tiles in SLM; T(T+1)/2 <= 65535, T = ceil(n/128) | all but cdouble |
| syr2k | `triangular` | `syr2k_triangular_tiles`: both products fused | as syrk's, four staged tiles; real `ConjTrans` runs as `Trans` | all but cdouble |
| herk | `gram` | `syrk_gram_tiles<T, Conjugate>` | as syrk's `gram` (8-wide thread tile at n = 128) | complex |
| herk, her2k | `triangular` | `syrk_` / `syr2k_triangular_tiles<T, Conjugate>` (her2k stages A times alpha) | as syrk's / syr2k's | cfloat |
| herk, her2k | `fold` | the public `gemm` of the whole product into n x n scratch, then `fold_hermitian` (two-sided for her2k), per chunk | `max_wg >= 256`; one item's scratch fits | complex |
| trmm | `triangular` | `trmm_triangular_tiles`: row tiles that skip the zero half | `max_wg >= 256`; `Side::Left`; `trmm_tile_groups` <= 65535 | all four |
| trmm | `expand` | `expand_triangular`, then `gemm` at beta 0, per chunk | `max_wg >= 256`; one item's scratch fits | all four |
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
| `syrk` float | `gram` at n <= 128; `triangular` on every `sq` row above n = 128 and on `tall`/`wide` rows with n >= 257, k >= 8, batch x T(T+1)/2 >= 160; else `vendor` |
| `syrk` double | `gram` at n <= 128; `vendor` elsewhere |
| `syr2k` float | `triangular` at batch >= 2; `vendor` at batch 1 |
| `syr2k` double | `vendor` (the only candidate) |
| `symm` float | `expand` on `sq` rows with batch >= 4 or max(m, n) >= 256, `vendor` on the other `sq` and every `tall`/`wide` row; 80 `custom` rows at [the mirror grid points](#level-3-mirror-measured-sm89) |
| `symm` double | `vendor`, `expand` second |
| `trmm`, all four | `triangular` on every `side=L` row, `expand` on every `side=R` row, `vendor` last |
| `hemm` | `expand` iff batch >= 4 or max(m, n) >= 256 (edges m, n 255\|256; batch 3\|4), else `vendor`; 76 cfloat `custom` rows at the mirror grid points |
| `herk` | `fold` iff batch >= 4 and n <= 768 (edges n 768\|769, batch 3\|4), else `vendor`; cfloat `triangular` in [the measured windows](#level3-hermitian-families-measured-sm89) |
| `her2k` | `fold` iff batch >= 2 or n >= 128 (edges n 127\|128, batch 1\|2), else `vendor`; cfloat `triangular` in the measured windows |

Complex-symmetric `syrk`/`syr2k` cfloat rows: `gram` first at n <= 64, `triangular` up to n = 768, at batch >= 4
(syr2k >= 2), ranked from the herk/her2k timings (same kernels without the conjugate). Complex `symm` rows copy
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
batch >= 128 cell below n = 1024. ConjTrans herk matches NoTrans within 3%.

| op, shape | batch | winner | runner-up | other |
|---|---|---|---|---|
| herk N n=128 k=8 | 1024 | `triangular` 0.353 | `fold` 0.420 | vendor 3.79 |
| herk N n=256 k=64 | 1024 | `triangular` 1.64 | `fold` 1.75 | vendor 7.64 |
| herk N n=256 k=512 | 1024 | `fold` 5.97 | `triangular` 6.68 | vendor 34.5 |
| herk N n=512 k=8 | 128 | `triangular` 0.480 | vendor 0.699 | `fold` 0.774 |
| herk N n=768 k=8 | 128 | `triangular` 0.920 | vendor 1.03 | `fold` 1.70 |
| herk N n=1024 k=8 | 128 | vendor 1.415 | `triangular` 1.509 | `fold` 2.96 |
| her2k n=256 k=8 | 1024 | `triangular` 1.19 | `fold` 1.87 | vendor 6.99 |
| her2k n=256 k=512 | 1024 | `fold` 6.40 | `triangular` 11.3 | vendor 63.6 |
| her2k n=1024 k=64 | 128 | `triangular` 3.67 | vendor 3.70 | `fold` 4.01 |

- **herk cfloat:** `triangular` first at 256 <= n <= 768 (except n = 256, k >= 512, where `fold` wins by 12%) and
  at n <= 128 with k <= 8 and batch >= 1024; n > 768 keeps `vendor` first.
- **her2k cfloat:** `triangular` is two fused products and pays for k: first only at k <= 8 (n >= 256, or n >= 128
  at batch >= 1024) and at n >= 1024 with k <= 64; `fold` elsewhere.

complex<double> rows keep the transcribed rule (FP64 is not tuned on a 1:64 card). Vendor-free, `fold` and
`expand` feed the native gemm; tested, not timed.

## Level-3: the mirror family {#level-3-the-mirror-family}

`mirror` (symm and hemm) computes the product without the n x n scratch `expand` writes. It serves
`syev_two_stage`'s sy2sb update `Y = A22 P` (Side::Left, Lower, P pn x pk with pk <= 32), which reads only A22's
lower triangle (defect 14).

### Level-3: mirror's tile loader {#level-3-mirror-tile-loader}

`src/ops/symm/kernels/symm_mirror.cc` is wide v2's 64x64 double-buffered gemm (`register_64x64_k16_wide.hh`, see
@ref perf_gemm) with its own loader for the symmetric operand (A on the Left, the gemm's B on the Right). Per slab,
uniformly over the work-group:

- strictly inside the named triangle: the gemm's natural load;
- strictly inside the other triangle: the other staging map, reading the mirror `A(kk, i)` contiguously (hemm
  negates the imaginary parts in registers);
- touching the diagonal: element by element, the address chosen per element, hemm's diagonal imaginary part zeroed.

The gemm headers are not edited (sm_120's deep gemm records hash them). complex<double> takes the k = 8 slab
(32 KiB); double has no wide v2 tile and runs gemm's 16x16 tiled kernel with the same mirrored read. Limits are in
`src/ops/symm/mirror_tile.hh`.

### Level-3: mirror measured on sm_89 {#level-3-mirror-measured-sm89}

`batchlas_tune {symm --dtype float, hemm --dtype cfloat} --reps 8 --warm 0.5 --no-refine`, verified, 2026-10-26,
median ms, n = 32 (the sy2sb panel):

| op | m | batch | mirror | expand | vendor |
|---|---|---|---|---|---|
| symm float | 256 | 1024 | **0.281** | 0.871 | 19.9 |
| symm float | 2048 | 128 | **2.15** | 6.72 | 17.7 |
| hemm cfloat | 1024 | 128 | **1.36** | 3.13 | 6.61 |
| hemm cfloat | 512 | 8 | 0.068 | **0.052** | 0.214 |

On the level-3 grid `mirror` wins every float cell with m or n <= 128 and every `tall` one; `expand` wins the square
float cells from 512 up by 1.15-1.25x (1024 x 1024 x 128: 9.06 vs 7.75 ms) and hemm cfloat at 512 x 512 by 1.06x;
hemm at batch 8, m 255..512 goes to `expand`. The rows are `custom` on m 32..2048, n 8..1024, batch 8/128/1024;
double and cdouble keep transcribed rows, and sm_120's deep table predates `mirror`.

### Level-3: mirror's deliberate breaks {#level-3-mirror-deliberate-breaks}

`tests/symm_mirror_tests.cc` pins `mirror` per scalar with poisoned A, B, C at ld = rows + 3, the unnamed triangle
poisoned, complex alpha/beta, extents straddling the 64-row tile and 16-deep slab, beta = 0 over NaN, batch 1024
bit-identical, grid z 65535 | 65536 and grid y 65535 | 65536 row tiles. Each break below was one edit, run on sm_89:

| break | red (of 6 cases per scalar) |
|---|---|
| mirrored slab not conjugated (hemm) | cfloat, cdouble: 4 |
| hemm's diagonal imaginary part kept | cfloat, cdouble: 5 |
| Right side's triangle not transposed | float, cfloat, cdouble: 4 |
| diagonal slab classed inside the named triangle | float, cfloat, cdouble: 5 |
| batch stride derived as ld x k | float, cfloat, cdouble: 5 |
| double's tiled read ignoring `uplo` | double: 4 |

## Boundaries and their evidence

Measured against the old gates (2026-09-05, float unless stated); the transcribed sm_89 tables reproduce them.

| route edge | admit side | bracketing non-winner |
|---|---|---|
| syrk `triangular`: T >= 3 tiles a side (n >= 257) | 1.45x n=512 b=512; 1.63x n=1024 b=64; every saturated shape from n = 384 | n = 256: 0.84x-1.22x (a diagonal tile is computed whole, so one tile in four is saved) |
| syrk `triangular`: batch x T(T+1)/2 >= 160 | won from 168 blocks | 144 blocks 1.14x slower; 136 blocks 1.25x slower |
| syrk `gram`: n <= 128 | 4.29x vs gemm at n=32 b=2048 (933 GB/s, the roofline); 2.01x at n=64 | n = 128: 0.99x float, 2.10x double (FP64 makes it compute bound) |
| syr2k `triangular`: batch >= 2 | 1.06x-226x every shape | batch 1: vendor 1.18-1.60x below n = 1280, 4-10x at deep k, small n |
| trmm `triangular` on every Left row | m=128 nC=512 b=1024: 0.698 vs 0.784 ms expand | m=256 nC=1024 b=256: 0.915 vs **0.855** (left unfitted) |
| symm/hemm `expand`: batch >= 4 or max(m, n) >= 256 | 1.2x-72x over the per-item loop | batch <= 2, n <= 128: loses by up to 2.5x |
| trmm `expand` on every Right row | 1.15x-162x (49 square cells), 1.22x-32x (64 skewed) | none (`cublas?trmm` has a ~110 us floor) |
| herk `fold`: batch >= 4 and n <= 768 | 1.6x-72x at n <= 512 | batch <= 2 a wash or loss; 0.82x-0.93x from n = 896 |
| her2k `fold`: batch >= 2 or n >= 128 | 1.4x-128x | batch 1: 0.74x at n=32, 0.89x at n=64 |

The k >= 8 syrk edge is the old gate's `kTriangularTileK`, not measured. The symm/hemm shipped edge is more
conservative than its loss region (the data would support 2 and 128). herk/her2k grid: complex64, n 32..1024 x
batch 1..256 (`experiments/herk_crossover.sh`). herk needs the n ceiling because its gemm computes both triangles,
and one `cublas?herk` already saturates the device there. Whether to spell a product as trmm at all (trmm vs gemm per
dtype: double and cdouble win 1.05x-2.02x everywhere, float loses at m = 256, cfloat below m = 512) is in
`experiments/TRMM_SYRK_BATCHED_KERNELS.md`.

## Level-3: kernel design notes

### syrk gram tiles: kernel design

`src/ops/syrk/kernels/syrk_gram_tiles.hh`: a single-tile batched SYRK for the tall skinny A and small square C of
`ortho`.

- **Why a second kernel.** The 128-wide triangular tile spends 128*128*k arithmetic on an n*n/2 answer, a 32x
  overcharge at n = 32.
- **One shared tile.** Both operands of `A^T A` are the same columns, so A crosses the bus once; n/4 flop per byte
  against a ridge of ~40, so the kernel is bandwidth bound.
- **Only the triangle's thread tiles are carried**: 136 of 256 at n = 128. Masking only the epilogue left a GEMM's
  arithmetic.
- **Layout.** The shared tile is `[k][n]` with stride exactly n (16-byte aligned LDS.128); packet q of row kk lives at
  `q ^ (kk/4)`, conflict-free for staging and fragment loads.
- **Contiguous thread rows are a correctness invariant**; see [the band-split syrk bug](#the-band-split-syrk-bug).
- **Thread tile by scalar.** Float uses a 4-wide tile (544 threads). Complex needs 205 registers per work-item, and
  544 x 205 exceeds the 65,536 per work-group; an 8-wide tile (160 threads) fits.
- **Rejected:** a 128-bit staging load (n = 64 b = 1024: 0.333 -> 0.384 ms).

### syrk and syr2k triangular tiles: kernel design

`syrk_triangular_tiles.hh` and `syr2k_triangular_tiles.hh`, sharing the grid in `src/kernels/common/triangular_tiles.hh`.

- **The saving.** The grid is indexed over the triangular tile set (`triangular_tile_decode`), so tiles outside the
  triangle are never launched; diagonal tiles are computed whole and their epilogue drops the other half.
- **Inner loop.** Operands staged `[k][row]` at an aligned stride; the 8x8 thread tile split into two 4-wide bands 64
  apart, so each LDS.128 is bank-conflict free.
- **syr2k is fused.** Both products accumulate into one set of accumulators (four staged 128x8 tiles, 16 KB per k
  step); two batched GEMMs would clobber the caller's other half and stream C three times. Issuing the two products
  sequentially rather than interleaved is worth 1.53x (3.34 vs 5.11 ms, n = 512 b = 512).
- **Diagonal aliasing.** On the diagonal the bj-side pointers alias the bi-side ones; `on_diagonal` is block-uniform.

### trmm triangular tiles: kernel design

`src/ops/trmm/kernels/trmm_triangular_tiles.hh`.

- **A loop bound, not a mask.** With op(A) upper triangular, an output tile rooted at row m0 starts its reduction at
  p = m0; only the k-tile straddling the diagonal is masked. The skipped end is
  `lower_eff = (uplo == Lower) != transposed`.
- **The tile is sized to m.** ormqr's `W2 = T^H W1` has m = ib in the tens, where a 128-row tile spends 4x the
  arithmetic. The saving is not 2x: with R = m / TileM row tiles the reduction shrinks to (R+1)/2R.
- **Lanes.** A 128-wide side gets 16 lanes of 8; narrower sides keep the 4-wide band and drop lanes, so ThreadRows
  stays a multiple of 4.
- **Complex thread tile at TileM >= 64** is 4x16 (3.2 MAC per load against 4x8's 2.67 and a capability of 2). Not at
  TileM 32: halving the columns there measured 0.374 -> 0.424 ms at m = 32.
- **Staging follows op(A)'s contiguous direction**; a warp walking a transposed A by lda pulls one sector per lane.

### trmm: choosing the row tile by scalar type

`trmm_row_tile` trades arithmetic against re-reading B (each row tile restages its reduction range, up to (R+1)/2
times). Float is bandwidth bound, so the widest tile that fits wins; double and complex run at 1/64 rate, so the
narrowest wins. Tile16 / tile32 through `ormqr_blocked_benchmark` (n 256/512, batch 128-256, ib 16/32/64):

| type | tile16 / tile32 | verdict |
|---|---|---|
| `double` | 1.007x - 1.022x | 16 wins at every ib |
| `complex<double>`, `complex<float>` | 0.993x - 1.040x | 16 wins to ib 32, a wash at 64 |
| `float` | 0.966x - 0.997x | 16 loses everywhere |

Shipped: double takes 16 through m = 64; complex takes 16 through m = 32, 32 through m = 64; both take 64 above; float takes
32 through m = 32, 64 through m = 512, 128 above. Wide types never take 128: an 8x8 complex<double> tile is 256
registers, x 256 threads is the whole 65,536 of a work-group, and the launch is rejected. `BATCHLAS_TRMM_TILE_M` pins
the tile.

### Level-3: the complex MAC and return-by-value rules

`src/kernels/common/triangular_tiles.hh`, shared by every tile kernel.

- **`accumulate` writes the complex MAC by hand.** `std::complex<float>::operator*` lowers to `__mulsc3` (C99
  Annex G) with an Inf/NaN branch per multiply; the first complex Gram build ran at 1.2 TFLOP/s against float's 13.8.
- **Return the accumulator by value.** A `T&` out-parameter spills the accumulator array to local memory: 43% on
  float at m = 512 (0.659 -> 0.944 ms).
- **`tile_load4` is 128-bit only for float.** Wider scalars have no load form, and the reinterpret asserts an
  alignment `sycl::local_accessor` never promised, failing silently.

## Negative results {#level3-negative-results}

A specialised level-3 op beats the GEMM it replaces only where it reaches a batched native kernel; otherwise it is a
host loop 10-100x slower at batch >= 1024. Check with `BATCHLAS_SELECT_TRACE=1`.

| Alternative | Result | Verdict |
|---|---|---|
| herk on the Gram tile kernel (cfloat) | loses to gemm + fold: 0.217 vs 0.206 ms (n=32 b=2048), 2.08 vs 1.57 (n=128 b=512); complex is compute bound | kept reachable as `gram`, ranked last |
| `X^H A X` as syr2k; `A X` (symmetric A) as symm; `gebrd_blocked` a22 update as syr2k | two different matrices; adds a copy; a22 not symmetric | rejected on inspection |
| WY block factor `W2 = T^H W1` as trmm | per type, see [the ormqr WY trmm gate](qr.md#ormqr-wy-trmm-gate) | real types only |
| sytrd trailing update as syr2k | see [sytrd: the rank-2k trailing update](sytrd.md#sytrd-the-rank-2k-trailing-update-in-the-blocked-reduction) | float, cfloat only |

### syrk for the ortho gram matrix

`src/ops/ortho/ortho.cc` builds `C = A^H A`; through the per-item `cublasSsyrk` loop that was a 70-100x regression
(m=256 k=32 b=2048: gemm 0.350, syrk 33.73 ms), which motivated `syrk_gram_tiles`. With it, float k = 32 goes
1.450 -> 0.895 ms and ShiftChol3 1.998 -> 1.298 ms (m = 1024 b = 512). `gram_max_k` is 64 for float (k = 128:
0.96x) and 128 for double (k = 128: 1.34x). `svqb_alg` keeps its GEMM: a one-triangle result would multiply
uninitialised workspace. Per precision: [ortho](ortho.md#ortho-the-gram-matrix-through-syrk-per-precision).

## Correctness findings {#level3-correctness-findings}

### The band-split syrk bug

Splitting each Gram-tile thread's 8 rows into two 4-wide bands 64 apart (as the square kernels do) breaks taking the
triangle at thread-tile granularity: thread (0,1) owns (64,4), in the lower triangle while its tile is not, so
nothing wrote it (silent, n > 64 only). `SyrkTest.NarrowShapesMatchGemmReference` sweeps n in {24..128} x trans x
uplo and mirrors both sides before comparing.

### Level-3: generic syrk and syr2k fallbacks fold one triangle

`fold_symmetric_product_into_triangle` (`src/ops/potrf/kernels/symmetric_product_fold.hh`, used by `potrf_blocked`)
folds only the named triangle, `C(i,j) := product(i,j) + beta*C(i,j)`; with beta == 0 C is not read, so poison cannot
produce NaN. The deleted `src/extensions/syrk.cc` and `syr2k.cc` wrote both triangles.

### her2k alpha alignment fault

cuBLASLt (through `cublas?her2k`) vector-loads the host `alpha` at 16-byte alignment; `std::complex<double>` is
8-byte aligned, so the call faults at addresses 8 mod 16 (reproduces on cuBLAS 13.2 alone). The vendor wrappers copy
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
  pages on first touch (several times the GEMM at n = 512 b = 512) and could be freed while kernels are enqueued.
- **Mirrored expansion** stages one tile pair per work-group through local memory: 1.5 n^2 traffic against 3 n^2
  for copy plus in-place symmetrize.
- **Three ceilings, met by chunking:** grid z (65535 on CUDA); SYCL's `int` global id (2^31 elements, a thrown
  `sycl::exception` at n = 2048 b = 512); a scratch budget of a quarter of global memory (`BATCHLAS_EXPAND_MAX_BYTES`
  lowers it). `expansion_chunk` gives the most items one launch takes under all three; the launch leases that once
  and walks the batch in batch / chunk steps. `can_run` refuses only a single item that does not fit.
- **A guard must model the whole predicate.** `sytrd_blocked` and `sytrd_sy2sb` ask her2k's own selection
  (`ops::her2k::picks_batched_route`, `src/ops/her2k/route.hh`) rather than replicating its size ceiling.

## Open debts {#level3-open-debts}

- **trmm tile kernel is `Side::Left` only**; Right rows rank `expand`. `ormbr` has the same WY update, unwired.
- **ROCm** has only the gemm-based families; its syrk, syr2k and trmm Left take the rocBLAS loop. Never run here.
- **sm_89 tables are transcribed, not timed** (except `custom` rows). Suspects: `sq` syrk rows ranking `triangular`
  from n = 129 (only n >= 384 measured to win; 257 <= n <= 383 has no cell); double `symm` ranking `vendor`;
  cfloat Left `trmm` ranking `triangular` (loses to gemm below m = 512).
- **Unmeasured edges:** herk batch 3 and 769 <= n <= 895; her2k batch 1 with n >= 128; her2k narrow panels (k = nb in
  {16, 24, 32}, n2 up to 480); the symm/hemm slack.
- **Vendor-free gaps:** complex<double> syr2k (vendor only), complex<double> syrk above n = 128, complex-symmetric
  syrk/syr2k on ROCm.
- **Instrumentation:** `symm_benchmark`, `syrk_benchmark`, `syr2k_benchmark` abort on the host backend at tiny shapes
  (`adjustNDRangePerKernel: NDR.LocalSize[0] == 0`; 2026-09, not re-checked).
- **Kernel headroom:** trmm tiles run at ~70% of cuBLAS's per-flop rate; closing it would make m = 128..256 a ~1.2x win.

## Level-3: raw evidence

Raw data is at the git tag `perf-evidence/vendor-independence` (`git show <tag>:<path>`); per-cell CSVs were not
committed.

| topic | path |
|---|---|
| syrk gram tiles; trmm tiles; per-dtype tables; the 16-row tile; ortho wiring | `experiments/TRMM_SYRK_BATCHED_KERNELS.md` |
| substitution survey; sytrd syr2k trailing update; ortho rejections | `experiments/GEMM_TO_LEVEL3_SURVEY.md` |
| syrk, syr2k, herk/her2k crossover drivers | `experiments/{syrk_sweep,syrk_kskew,syr2k_sweep,herk_crossover}.sh` |

# The level-3 tile ops: symm, hemm, syrk, herk, syr2k, her2k, trmm

> **Covers:** the native kernels behind `symm`, `syrk`, `syr2k` and `trmm`, which select from tuned
> tables, and behind `hemm`, `herk` and `her2k`, which still choose by fixed rules in the cuBLAS
> backend: each family, what its `can_run` admits, what the shipped tables rank first, the
> measurements those rankings come from, the kernel design, negative results and correctness
> findings.
> **Status:** current. The four table-selected ops' tables are transcribed from the deleted
> hand-written gates at `ff340fc6` and are untimed; the measurements on this page are what those
> gates were drawn from. [Before flat selection](#level-3-before-flat-selection) summarises the
> deleted mechanism; nothing else here describes deleted code as current.
> **Machine:** RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda, `RelWithDebInfo`, one dedicated GPU
> (`experiments/gpu_guard.sh`), saturating batch unless a cell says otherwise.
> **Measured:** WP1 and the vendor-independence campaign, distilled here on 2026-09-05 (raw notes at
> the tag `perf-evidence/vendor-independence`, see [Level-3: raw evidence](#level-3-raw-evidence)). The level-3
> flat-selection wave (2026-10-06) moved the decisions into tables and measured nothing.

A specialised level-3 op beats the GEMM it replaces only where it reaches a batched native kernel;
everywhere else the vendor is a host loop, one launch per batch item. So the tables put `gram` first
for every `syrk` with n <= 128, the 128-wide `triangular` tiles first for float `syrk` and `syr2k`
wherever the grid fills the device or a loop would cost more, `triangular` first for every
`Side::Left` `trmm` and `expand` for every `Side::Right` one, and `expand` first for float `symm` in
the square, batched window. The exploration notes (`experiments/TRMM_SYRK_BATCHED_KERNELS.md`,
`experiments/GEMM_TO_LEVEL3_SURVEY.md`, `WP1_LEVEL3_SPEC.md`) are the authority on *why*; where they
disagree with the shipped code, the disagreement is named in place.

## Level-3: how the table-selected four choose

`symm`, `syrk`, `syr2k` and `trmm` follow the flat-selection layout (@ref design_flat_selection,
[§12 "Level-3 four"](../design/flat-kernel-selection.md#level-3-four-symm-syrk-syr2k-trmm)):
`src/ops/<op>/choice.hh` lists the families, `src/ops/<op>/<op>.cc` validates the shapes (throwing
`std::invalid_argument` on every backend), returns a no-op event for an empty problem, and calls
`select::run`, which takes the first entry of the nearest row of `tuned/<op>.<dtype>.<device>.txt`
that `can_run` admits. @ref selection_tables lists the families per op.

### Level-3: the families and their can_run terms

Every native family shares one term: `B == Backend::CUDA` (the kernels are portable SYCL, wired for
CUDA only; ROCm and the host take the vendor), `d.is_gpu`, every operand homogeneous, extents and
batch >= 1, and batch <= 65535 (the batch is CUDA grid z; 65536 throws `Number of work-groups exceed
limit`). It is `src/ops/symm/symm.cc:49-50`, `src/ops/syrk/syrk.cc:54-55`,
`src/ops/syr2k/syr2k.cc:53-55` and `src/ops/trmm/trmm.cc:56-57`. The vendor family needs
`d.has_vendor` and also refuses a heterogeneous operand, because every vendor loop (cuBLAS, rocBLAS,
netlib) runs each item at the top-level extents, which is a wrong answer for a ragged batch.

| op | spelling | driver | further `can_run` terms | dtypes |
|---|---|---|---|---|
| `symm` | `expand` | `expand_mirrored<T, false>` into a queue-arena lease, then the public `gemm` | `max_wg >= 256` (`symm.cc:49`); `expansion_fits` (`symm.cc:52-56`) | float, double |
| `symm` | `vendor` | `backend::symm_vendor`: `cublas?symm` / `cblas_?symm` per item | | float, double |
| `syrk` | `gram` | `syrk_gram_tiles<T, false>`: one tile covers C | `syrk_gram_supported` (n <= `kGramMaxTile` = 128), `max_wg >= gram_threads(n)`, the staged tile fits SLM (`syrk.cc:58-59`) | float, double |
| `syrk` | `triangular` | `syrk_triangular_tiles<float>`: 128x128x8 tiles of one half | float; `max_wg >= 256`; T(T+1)/2 <= 65535 tiles, T = ceil(n/128) (`syrk.cc:62-63`) | float |
| `syrk` | `vendor` | `backend::syrk_vendor`: `?syrk` per item | | float, double |
| `syr2k` | `triangular` | `syr2k_triangular_tiles<float>`: both products fused | float; `max_wg >= 256`; `transA != ConjTrans`; the same tile ceiling (`syr2k.cc:53-55`) | float |
| `syr2k` | `vendor` | `backend::syr2k_vendor`: `?syr2k` per item | | float, double |
| `trmm` | `triangular` | `trmm_triangular_tiles<T>`: row tiles that skip the zero half | `max_wg >= 256` (`trmm.cc:56`); `trmm_tiles_supported` (`Side::Left`); `trmm_tile_groups` <= 65535 (`trmm.cc:60-61`) | all four |
| `trmm` | `expand` | `expand_triangular<T>` into a lease, then the public `gemm` at beta 0 | `max_wg >= 256`; `expansion_fits` (`trmm.cc:63-68`) | all four |
| `trmm` | `vendor` | `backend::trmm_vendor`: `cublas?trmm` per item | | all four |

A real `ConjTrans` is `Trans` to both `syrk` tile kernels, so it is no `can_run` term there; the exact
`trans` key keeps it on the vendor. For `syr2k` it is a term (`transA != ConjTrans`), because opening
it would be a routing change. When nothing in the row runs, the last resort is `expand, vendor`
(symm), `triangular, gram, vendor` (syrk), `triangular, vendor` (syr2k) and
`expand, triangular, vendor` (trmm).

### Level-3: what the shipped tables rank first

All twenty tables (`tuned/{symm,syrk,syr2k}.{float,double}.*` and
`tuned/trmm.{float,double,cfloat,cdouble}.*`, each for `sm_89` and `sm_120`) are
`source=transcribed:ff340fc6`: per grid cell, the old vendor-present Auto choice first, then every
other structurally runnable candidate, `vendor` last unless it was the Auto choice. The `sm_89` and
`sm_120` files hold identical rows, because the old rules read no architecture. They are untimed; a
row is replaced when the tuner sweeps that device (`tuned/README.md`). Read off the `sm_89` rows
(counting only rows whose extents agree with their `form` key; a cell that contradicts its form holds
the decision of the form's representative shape):

| op, dtype | ranked first |
|---|---|
| `syrk` float | `gram` at every n <= 128 (trans N and T); `triangular` on every `sq` row above n = 128, and on `tall`/`wide` rows with n >= 257, k >= 8 and batch x T(T+1)/2 >= 160; `vendor` on the other `tall`/`wide` rows (129 <= n <= 256, k < 8, or too few tiles) and on every trans C row |
| `syrk` double | `gram` at every n <= 128 (N and T); `vendor` everywhere else |
| `syr2k` float | `triangular` at batch >= 2, `vendor` at batch 1, at every n and k |
| `syr2k` double | `vendor` (the only candidate) |
| `symm` float | `expand` on `sq` rows with batch >= 4 or max(m, n) >= 256; `vendor` on the other `sq` rows and on every `tall`/`wide` row |
| `symm` double | `vendor` everywhere, `expand` second |
| `trmm`, all four | `triangular` on every `side=L` row, `expand` on every `side=R` row, `vendor` last |

So a vendor-free build serves every float `syrk` GPU shape, double `syrk` at n <= 128, float `syr2k`
at batch 1 too, float and double `symm`, and `trmm` in every dtype on both sides: the rows carry the
native entry the old gates skipped. The transcription's fidelity (100% agreement with the old Auto
choice on 2500-3000 random off-grid points per op, dtype and device) is recorded in
[§12 "Level-3 four"](../design/flat-kernel-selection.md#level-3-four-symm-syrk-syr2k-trmm).

### Level-3: pinning a family

`BATCHLAS_<OP>_ROUTE` takes `auto`, `native`, `vendor` or a spelling from the table above, for
example `BATCHLAS_SYRK_ROUTE=triangular` or `BATCHLAS_TRMM_ROUTE=expand`. A spelling that is not a
compiled candidate for the dtype, or cannot run the shape, throws `std::invalid_argument`:
`triangular` on double `syrk` or `syr2k`, `gram` at n > 128 (the kernel would answer wrongly there,
not throw), and every removed word (`cublasdx`, `tiles`, `narrow`, `gemm`, `custom`). The full list is
[the level-3 pin words](dispatch.md#the-level-3-pin-words). `BATCHLAS_TRMM_ROUTE=vendor` is now the
`cublas?trmm` loop; its old meaning (expand plus gemm when the scratch fit) is `expand`. `herk`
reaches the Gram kernel only through `BATCHLAS_SYRK_ROUTE=gram` (`ops::syrk::herk_gram_pinned()`,
`src/ops/syrk/choice.hh`). `symm` no longer reads `BATCHLAS_EXPAND_ROUTE`; hemm, herk and her2k do.

### Level-3: hemm, herk and her2k still choose by fixed rules

They have no `choice.hh` and no table. Their public entry points (`src/ops/level3/level3.cc`) call
`backend::<op>_vendor` directly, or throw `NoRouteError` when no level-3 library is compiled in. The
cuBLAS backend then picks between one batched GEMM into scratch and the per-item vendor loop (condensed):

```cpp
// hemm_vendor (src/backends/cublas.cc:192-193); expansion_preferred is triangular_expand.hh:31-42
expansion_fits(ctx, k, batch, bytes) && expansion_preferred(max(m, n, k), batch)  // batch >= 4 || max_dim >= 256
// herk_vendor (cublas.cc:366); herk_gemm_preferred is cublas.cc:257-263
herk_gemm_preferred(n, batch) && expansion_fits(ctx, n, batch, bytes)              // batch >= 4 && n <= 768
// her2k_vendor (cublas.cc:436); her2k_gemm_preferred is src/expansion_budget.hh:65-69
her2k_gemm_preferred(n, batch) && expansion_fits(ctx, n, batch, bytes)             // batch >= 2 || n >= 128
```

`BATCHLAS_EXPAND_ROUTE=expand|loop` overrides the preference (not the fit). The host backend runs the
per-item `cblas_?hemm`/`?herk`/`?her2k` loops; ROCm instantiates none of the three. The crossovers
are [symm and hemm expansion crossover](#symm-and-hemm-expansion-crossover) and
[herk and her2k: the GEMM-plus-fold crossovers](#herk-and-her2k-the-gemm-plus-fold-crossovers).

### Level-3: dtype reach of the native families

Before flat selection two native arms for non-float scalars lived inside the cuBLAS backend, and so
were reachable only when cuBLAS was compiled: `syrk` gram tiles in `syrk_vendor` (n <= 128) and the
`trmm` tile kernel in `trmm_vendor` (`Side::Left`, homogeneous). They are now the `gram` and
`triangular` families, so double `syrk` at n <= 128 and `trmm` in every dtype run vendor-free.
`syr2k` still has nothing non-float. `level3_tile_route_available` (`src/select/vendor.hh:94-96`) is
deliberately kept at its old value, float or cuBLAS present, because `sytrd_blocked`, `ortho`,
`ormqr_blocked` and the coverage census read it, and widening it would move their routes:

```cpp
template <Backend B, typename T>
inline constexpr bool level3_tile_route_available =
    B == Backend::CUDA && (std::is_same_v<T, float> || bool(BATCHLAS_HAS_CUBLAS));
```

Why the non-float rows rank a native family first at every cell, with no threshold:

* **syrk, non-float -> gram tiles only.** It is the one level-3 kernel whose staging and fragment
  loads are not written around a 128-bit packet, so it is the one that generalises; the 128x128
  triangular kernel stays float. Below `kGramMaxTile` the alternative is the vendor's host loop, one
  `cublas?syrk` per batch member, two orders of magnitude off anything batched at large batch.
* **trmm, non-float -> tile kernel wherever it fits.** The kernel is type-generic; only its routing
  was ever float. The alternative is the expansion plus GEMM, which is strictly more work than the
  GEMM it wraps, so there is nothing to weigh.

### Level-3: before flat selection

Up to `ff340fc6` these four ops chose by hand-written rules in `src/backends/*_custom_dispatch.cc`,
called from the public entry points in `src/ops/level3/level3.cc` only for float on CUDA, with arms
named `Route{Origin, Algorithm}` (`{Native, ExpandGemm}`, `{Native, GramTiles}`,
`{Native, TriangularTiles}`, `{Vendor, FusedDevice}`, `{Vendor, Auto}`), the vocabulary the
measurement notes below still use. `{Vendor, FusedDevice}` was the cuBLASDx fused kernel, which never
ran here (MathDx is absent, so every "cublasdx" measurement was its fallback); `{Vendor, DiagFullGemm}`
was a deliberately **wrong** route that stored both triangles, kept to measure the arithmetic the
triangular kernels save. The thresholds were **gate-only**: once `syrk_use_cuda_custom` returned true
for any reason, Auto took the triangular tiles after the gram test failed with no second preference
check, so a square n = 256 shape reached the tile kernel even though the tile predicate read
n >= 257. That is why these four had no table and were instrumented at each terminal instead, and why
the flat-selection transcriber evaluated the whole old decision per grid cell rather than the
predicates (`tuned/README.md`): the tables hold the effective window, not the predicate. The level-3
flat-selection wave deleted the gates, `level3_coverage.hh`, the cuBLASDx paths and `DiagFullGemm`
(@ref design_flat_selection).

## Boundaries and their evidence

The windows below were measured against the old gates; the transcribed tables reproduce them cell by
cell, so each one now reads "the rows rank X first where ...".

### syrk triangular tiles

128x128x8, indexed over the triangular tile set so a tile outside the requested half is never
launched. Grid: float, n in 64..2048 x batch in 1..512, against the full n x n batched GEMM. The float
rows rank `triangular` first on `tall`/`wide` rows where n >= 257, k >= 8 and
batch x T(T+1)/2 >= 160, T = ceil(n/128), and on every `sq` row above n = 128 (see
[the squareish rows](#syrk-the-squareish-rows-below-the-triangular-window)).

| table edge | admit side | bracketing non-winner |
|---|---|---|
| T >= 3 tiles a side (n >= 257) | 1.45x at n=512 batch 512; 1.63x at n=1024 batch 64; 1.71x at n=2048 batch 16 | n = 256 measured **0.84x–1.22x**, depending where its grid fell against a wave boundary |
| batch x T(T+1)/2 >= 160 | won from 168 blocks up (recorded as `0.71x`, inverting its own convention mid-sentence) | 144 blocks 1.14x slower; 136 blocks 1.25x slower |
| k | does not enter — it only deepens both routes' reduction | not swept here; `experiments/syrk_kskew.sh` exists to sweep k free |

Why these two conditions (as argued in the deleted gate's comment):

* **n past 256.** A grid narrower than three 128-wide tiles a side is more than half diagonal, and
  a diagonal tile is computed whole and then masked, so at n = 256 only one tile in four is saved.
  That does not cover the tile kernel's lower per-tile rate against cuBLAS. From n = 384 up every
  saturated shape won, and the win grows with n as the diagonal thins out.
* **The grid has to fill the device.** The 128 SMs hold two of these 256-thread blocks apiece;
  below ~160 blocks the triangular route lost.
* **k does not enter**: it only deepens each block's reduction, which moves both routes together.
  The table's k >= 8 edge is the old gate's `kTriangularTileK` (one k step), not a measured one.

### syrk: the squareish rows below the triangular window

The old gate also admitted a cuBLASDx-sized heuristic (`min_dim*2 >= max_dim && tiled_work >= 8`,
n >= 16), which the `form` key now carries as `sq`. Below the triangular window the question is no
longer "does it beat the full GEMM" but "does it beat a host loop over `cublasSsyrk`", and one
launch per batch member costs about **9 us**, so anything with a batch at all is better off on the
tile kernel even where the grid is half diagonal. This is how n = 256 reaches the tile kernel: every
`sq` row from n = 129 up ranks `triangular` first.

**Two records of the same boundary differ in framing.** The exploration note reports `0.89x at
n = 256, 1.28x at 512, 1.57x at 1024`; the deleted gate's comment reports `0.84x–1.22x at 256, 1.45x
at 512 batch 512, 1.63x at 1024 batch 64` — different batches, same conclusion at 256 and the same
direction above it. Neither holds a cell in **257 <= n <= 383**, which the rows admit (the grid has
points at 257, 384 and 385).

### syrk gram tiles

The single-tile kernel sized to n, so both operands of `A^T A` are the same columns of A: one
shared tile, A crosses the bus once. Serves n <= 128, exactly what the triangular grid cannot; no
threshold to tune, because below 128 the alternative is a host loop over `cublasSsyrk`, and the rows
rank `gram` first at every n <= 128 in float and double. Float, `m` is the reduction depth, `before`
is that loop (ms):

| m | n | batch | gemm | before | now | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.334 | 33.588 | 0.0780 | 4.29x | 431x |
| 512 | 64 | 1024 | 0.345 | 31.911 | 0.1875 | 1.84x | 170x |
| 1024 | 64 | 1024 | 0.668 | 60.397 | 0.3326 | 2.01x | 182x |
| 1024 | 128 | 512 | 0.409 | 30.296 | 0.4146 | **0.99x** | 73x |
| 2048 | 128 | 256 | 0.404 | 29.370 | 0.3820 | 1.06x | 77x |

At n = 32 it reads 71 MB in 78 us — 933 GB/s, at the memory roofline; at n = 128 reading A once
costs 298 us against GEMM's measured 409, so the whole prize there was 1.37x, not 2x. Double,
same three shapes: 0.901 / 115.40 / **0.837** (1.08x vs gemm), 3.444 / 112.45 / **1.934**
(1.78x), 13.72 / 110.91 / **6.521** (2.10x) — the win *grows* with n in double and shrinks in
float, because FP64 at 1/64 rate makes the Gram product compute bound.

### syr2k triangular tiles

One pass fusing both rank-k products into the same accumulators. Grid: float, n in 8..3072 x
k in 4..2048 x batch in 1..1024. The float rows rank `triangular` first at batch >= 2 and `vendor` at
batch 1, at every n and k.

Why batch is the only key that decides: the vendor route is a host loop over `cublasSsyr2k`, one
launch per batch member, against one launch for the whole batch here, so the two are only ever close
at a batch of one and the vendor pays double from two members up. Neither n nor k nor the tile count
changes which side of that per-launch difference a shape falls on.

* **batch >= 2**: won *every* shape — 1.06x at n=3072, 1.12x at n=1024, 1.3–1.4x through the
  middle, up to 226x where n is small enough that the whole cost is the launch.
* **batch 1**: does not sort by anything. Vendor wins 1.18–1.60x below n=1280 and 1.16x at
  n=3072; the kernel wins 1.02–1.71x between; the vendor wins 4–10x on a deep k with a small n,
  where the kernel has a single block and cuBLAS splits the reduction.

Bracketed on both sides, and the batch-1 side is genuinely unsortable — hence no edge in n. Issuing
the two products sequentially rather than interleaved keeps one pair of 8-wide fragments live over
the 64 accumulators, worth **1.53x** (3.34 vs 5.11 ms at n=512 batch 512).

### trmm tiles have no threshold

The first router gated the tile kernel to `m <= 64 || m >= 512`, read off a trmm-vs-gemm column.
Wrong comparison: the choice is between the tile kernel and **the expansion**. Measured tile against
the expansion-plus-gemm (then the `vendor` arm), float, saturating batch (ms): m=128 nC=512 batch
1024 **0.698** vs 0.784; m=128 nC=1024 batch 512 0.686 vs 0.687; m=256 nC=256 batch 512 **0.536**
vs 0.692; m=256 nC=1024 batch 256 0.915 vs **0.855**. The gate cost up to **1.29x** on exactly the
shapes it was meant to protect; the single 7% loss is left unfitted, since a clause that narrow would
need re-tuning whenever either route moved. Every `side=L` row of every trmm table ranks `triangular`
first, whatever the order, q and batch.

### trmm tile vs gemm by dtype

The *caller's* question — whether to spell a product as trmm at all — not the table's. Ratios
are trmm against the GEMM spelling of the same product; **bold** is a win.

| m | nC | batch | float | double | complex&lt;float&gt; | complex&lt;double&gt; |
|---|---|---|---|---|---|---|
| 32 | 256 | 2048 | **1.13x** | **1.29x** | 0.88x | **1.05x** |
| 128 | 512 | 1024 | **1.03x** | **1.48x** | 0.69x | **1.42x** |
| 256 | 1024 | 256 | 0.91x | **1.77x** | 0.93x | **1.71x** |
| 512 | 512 | 128 | **1.15x** | **1.91x** | **1.10x** | **1.85x** |
| 1024 | 1024 | 32 | **1.26x** | **2.02x** | **1.23x** | **1.95x** |

`double` and `complex<double>` win at every shape, 1.05x–2.02x — the ceiling, since FP64 at 1/64
rate makes the halved arithmetic land in full and grow with m. `float` wins 8 of the 10 cells in
the full grid; the exceptions are m = 256, where cuBLAS SGEMM runs at its 45 TFLOP/s peak and this
kernel reaches ~57% of it, so a 1.6x arithmetic saving cannot cover a 1.75x rate deficit.
`complex<float>` loses below m = 512 on a register-file ceiling, not a tuning miss: a complex
accumulator doubles the registers, so a competitive fragment-to-FMA thread tile does not fit at
usable occupancy. The saving is **not** the textbook 2x — with `R = m / TileM` row tiles the
reduction shrinks to `(R+1)/2R`, so 1.0x at R = 1, 1.33x at R = 2, 1.78x by R = 8 — which is why
`trmm_row_tile` (`trmm_triangular_tiles.hh`, see
[choosing the row tile](#trmm-choosing-the-row-tile-by-scalar-type)) picks the tile from the scalar type: float
TileM 32 / 64 / 128 measured 0.663 / **0.658** / 0.699 at m=128 nC=512 batch 1024 against a GEMM's
0.674, so 64 through m = 512 and 128 above it (at m = 1024, 1.146 vs 1.180). Wide types take 16
through m = 64, complex through m = 32, never 128 — an 8x8 tile in `complex<double>` is 256
accumulator registers, which the runtime rejects rather than spilling.

### symm and hemm expansion crossover

Measured against a per-batch loop over the vendor's own triangular primitive: `cublas?symm` in
float over n in 16..2048 x batch in 1..512, and `cublas?hemm` in complex64 over n in 16..512 x
batch in 1..16. Both put the crossover in the same place — the expansion wins **1.2x–72x**
everywhere except `batch <= 2 && n <= 128`, where the call is launch-bound and the extra kernel
costs more than the loop it replaces; there it loses by up to **2.5x**.

**The shipped edge is not the complement of that loss region and is strictly more conservative.**
The float `symm` rows rank `expand` first on `sq` rows with batch >= 4 or max(m, n) >= 256, and
`hemm_vendor` applies the same constants through `expansion_preferred`
(`kExpandMinBatch = 4`, `kExpandMinDim = 256`, `src/backends/triangular_expand.hh:28-29`). So batch 3
at every n and batch <= 2 with `129 <= n <= 255` take the loop despite lying outside the measured
loss region (the symm grid straddles both edges: batch 3|4, n 255|256). `WP1_LEVEL3_SPEC.md`
correction 4 records the constants as 4 and 256 and says to preserve them; the exploration numbers
would have supported 2 and 128. The code wins — that gap is deliberate slack, not a measured
boundary. `symm` additionally ranks `vendor` first on every `tall` and `wide` row, because the old
gate required `squareish` (`min_dim*2 >= max_dim`) where `hemm`'s never did, on the argument that a
full k x k expansion stops paying once k dwarfs m and n — with no bracketing cell for it in either
source. The double `symm` rows rank `vendor` first everywhere, as the float-only old gate left them;
`expand` serves double, unmeasured.

**trmm's `expand` is first on every Right row, batch 1 included**, because `cublas?trmm` has a flat
~110 us floor whatever the shape: **49 square cells** (k in 16..1024 x batch in 1..512) at
**1.15x–162x** for the expansion, and **64 skewed cells** (k in 256..2048 against 1..128 right-hand
sides) at **1.22x–32x**, with not one cell going the other way. No boundary to bracket there; the
`vendor` loop is second, for a scratch that does not fit.

### herk and her2k: the GEMM-plus-fold crossovers

Both replace a per-batch vendor loop with one strided-batched GEMM into scratch plus a fold. These
two still choose by the fixed rules in
[hemm, herk and her2k still choose by fixed rules](#level-3-hemm-herk-and-her2k-still-choose-by-fixed-rules).
cuBLAS has no batched or strided-batched `?herk`/`?her2k` (only the single `cublasCherk`,
`cublasZherk`, `cublasCher2k`, `cublasZher2k`), so a batch is a host loop over launches; the
alternative is two launches in total, one GEMM over the whole n x n product and
`accumulate_hermitian`. The same holds for `?hemm`, which is why `hemm_vendor` expands.

**Conjunction, not disjunction.** herk's predicate is `batch >= 4 && n <= 768` where the mirrored
expansion's (`batch >= 4 || max_dim >= 256`) is a disjunction. The expansion has no large-n
ceiling because expanding an operand is one bandwidth-bound kernel followed by exactly the vendor's
own work, so it never pays twice for the arithmetic; herk's GEMM does pay twice, so it needs the
ceiling where one `cublas?herk` already saturates the device.

| op | predicate | admit side | bracketing non-winner |
|---|---|---|---|
| `herk` | `batch >= 4 && n <= 768` | 1.6x–72x for batch >= 4 at n <= 512 | batch <= 2 is a wash or a loss at every n; **0.82x–0.93x from n = 896 up**, where one `cublas?herk` already saturates the device |
| `her2k` | `batch >= 2 \|\| n >= 128` | 1.4x–128x | batch 1 at n <= 64 only: **0.74x at n = 32, 0.89x at n = 64** |

herk grid: complex64, n in 32..1024 x batch in 1..256 (`experiments/herk_crossover.sh` drives it),
reported as loop time over GEMM-route time.
`herk` starts from **twice** a rank-k update's arithmetic — the GEMM computes both triangles and
keeps one — so it wins only where the loop is launch bound; `her2k` starts from **half**, because
`alpha*A*B^H` and `conj(alpha)*B*A^H` are conjugate transposes and the mirrored read manufactures
the second term from the first. Two gaps: the grid steps batch 1, 2, 4, 8, 16, 64, 256, so
**batch 3 is unmeasured**, and with a wash from n = 640 to 768 and a loss from 896,
**769 <= n <= 895 is unmeasured** — the threshold sits at the top of the wash band. For her2k the
winning side of the `n >= 128` disjunct at batch 1 has no quoted cell. her2k's loss cells are
where the fold's own launch is not repaid.

`her2k_gemm_preferred` lives in `src/expansion_budget.hh`, not in `cublas.cc`, because
`sytrd_blocked.cc` must evaluate the whole route condition (`her2k_takes_gemm_route`) before it
decides to call her2k; see
[the scratch-expansion section](#level-3-scratch-expansions-and-their-ceilings).

## Level-3: kernel design notes

The reasoning behind the tile kernels' shapes, moved out of the kernel headers. The code keeps the
invariants and points here. Machine as at the top of the page.

### syrk gram tiles: kernel design

`src/backends/syrk_gram_tiles.hh`. A single-tile batched SYRK for the shape `ortho` hands it: a
tall, skinny A and a small square C.

**Why a second kernel.** The 128x128 triangular kernel is built for n in the hundreds, where
skipping tiles outside the triangle is the point. It cannot serve n <= 128: one tile is the whole
matrix, so there is nothing to skip, and a 128-wide tile spends 128*128*k arithmetic on an
n*n/2 answer, a **32x overcharge at n = 32**. The old router (correctly) refused it, so every Gram
matrix in `ortho` fell through to a host loop over `cublasSsyrk` and lost by up to two orders of
magnitude at large batch (see [syrk for the ortho gram matrix](#syrk-for-the-ortho-gram-matrix)).

**The design** sizes the tile to n and takes the triangle at thread-tile granularity:

* One tile covers all of C, so the two operands of `A^T A` are the same columns of A: one shared
  tile, and A crosses the bus exactly once. At the skinny end the arithmetic intensity is n/4 flop
  per byte, 8 at n = 32 against the 4090's ridge of ~40, so the kernel is bandwidth bound by 5x
  and the measured **933 GB/s** is the whole story.
* Only the `Lanes*(Lanes+1)/2` thread tiles that meet the requested triangle are carried: 136 of
  256 at n = 128, so 160 threads instead of 256. Merely masking the epilogue leaves the block doing
  a GEMM's arithmetic, which is why the first cut matched a GEMM at n = 128 and never beat one.
  The tail of the last warp stages and reaches every barrier but computes nothing.
* The shared tile is `[k][n]` with the stride exactly n, so fragment loads stay 16-byte aligned
  and issue as LDS.128. An 8x8 thread tile reads 16 floats per k-step and issues 64 FFMAs; the SM's
  32-floats-per-clock shared path against its 128-FFMA-per-clock math path puts that exactly on
  the balance point, so there is no headroom for a padded stride.

**Staging by transpose mode.** Reaching that aligned layout without losing coalescing on the
global side or colliding on the shared side needs opposite assignments. `Trans` (A is k x n,
contiguous down k): one thread takes four adjacent reduction rows of one column, so eight lanes
span a 128-byte run; the shared side pays a bank conflict for it, the cheaper half, since the tile
is written once and read KC times. `NoTrans` (A is n x k, contiguous across n): one thread takes
four adjacent columns of one reduction row.

**The packet swizzle.** Packet q of reduction row kk lives at `q ^ (kk/4)`. With an aligned stride
alone every lane of a staging write lands in the same bank: the rows a lane writes are 4 apart and
the stride is a multiple of 32, so only the column separates them. Rotating by whole 4-wide
packets restores the spread and keeps the fragment loads LDS.128; both sides come out
conflict-free.

**Contiguous thread rows are a correctness invariant**, not a layout choice; see
[the band-split syrk bug](#the-band-split-syrk-bug).

**Thread tile by scalar.** A 4-wide thread tile puts 528 tiles (544 threads) on the 128-wide case,
which is what float wants and measured fastest. A complex scalar cannot afford that block: two
components per accumulator took it to **205 registers per work-item**, and 544 x 205 is past the
65,536 a work-group gets, which the runtime rejects outright rather than spilling. An 8-wide thread
tile quarters the thread count to 160 and the same accumulators fit.

**Rejected: a 128-bit staging load.** Taking a staging thread's four elements as one 128-bit load
was measured slower everywhere (n = 64 batch 1024: **0.333 -> 0.384 ms**). The four predicated
scalar loads land in one cache line, so the merge saves no traffic, and the branch it needs costs
more than the instructions it removes.

### syrk and syr2k triangular tiles: kernel design

`src/backends/syrk_triangular_tiles.hh`, `src/backends/syr2k_triangular_tiles.hh`, sharing the
grid in `triangular_tiles.hh`.

**The saving.** Routing SYRK at a batched GEMM is correct but does twice the arithmetic BLAS
specifies: every 128x128 output tile strictly outside the named triangle is computed and
discarded. The grid is indexed over the triangular tile set instead (`triangular_tile_decode`), so
such a tile is never launched. Diagonal tiles are the only ones computed in full, and their
epilogue drops the elements in the unreferenced half (BLAS forbids writing them and, with
beta != 0, reading them), so the diagonal tile also gives up the 128-bit store form. Keeping the
decode, the tile geometry and the 128-bit vector type in one header is what stops the two kernels
drifting apart on which half of the grid they visit.

**The inner loop** is that of `src/sycl/gemm/register_128x128.hh`, for the same reasons: an
aligned shared stride so fragment loads become LDS.128, operands staged `[k][row]` so a thread's
8 values are contiguous, and the 8x8 thread tile split into two 4-wide bands 64 apart so an LDS.128
is bank-conflict free. Staging follows the transpose mode: `NoTrans` stages with one vector load
per thread; `Trans` stages transposed, four consecutive k per thread scattered into shared.

**syr2k: why fused, not two GEMMs.** Decomposing `C = alpha*A*B^T + alpha*B*A^T + beta*C` into
two batched GEMMs aimed at C is wrong twice over: each writes the whole n x n, clobbering the
caller's half, and the pair streams C three times (written with beta, read back, written again)
for an output touched once here. Both products land in the same accumulators
(`accum += A[bi]*B[bj]^T` and `accum += B[bi]*A[bj]^T`), needing four staged tiles per k step and
paying with twice the arithmetic: 8 LDS.128 against 128 FFMA, the same 16:1 ratio as a plain GEMM
tile. The four 128x8 tiles are 16 KB of shared memory, inside what lets two blocks share an SM.

**syr2k: sequential, not interleaved, products.** Only one pair of 8-wide fragments is live at a
time on top of the 64 accumulators. Holding both pairs fits in a thread but not inside the
register budget that leaves room for a second block per SM; that occupancy is worth **1.53x**
(5.11 vs 3.34 ms at n = 512 batch 512).

**syr2k: diagonal aliasing.** On the diagonal the two row offsets coincide, so the kernel aliases
the bj-side tile pointers to the bi-side ones rather than staging the same rows twice.
`on_diagonal` is uniform across the block, so this costs no divergence.

### trmm triangular tiles: kernel design

`src/backends/trmm_triangular_tiles.hh`.

**History: nothing used to exploit the triangle.** `src/extensions/trmm.cc` recurses only until
the block is 256 wide and then calls the very GEMM it is meant to replace, so for every ib in
{16,32,64,128,256} that `ormqr`/`ormbr` use the triangle was never exploited. On CUDA the operation
did not reach that recursion: `trmm_vendor_impl` expanded the triangle into a k x k x batch scratch
and ran a full GEMM, paying an extra write and read for twice the arithmetic (that path is now the
`expand` family, second to `triangular` on every Left row). PR #61 measured the
consequence and refused to use trmm anywhere (see
[trmm for the WY block factor](#trmm-for-the-wy-block-factor)).

**A loop bound, not a mask.** For `C = alpha*op(A)*B` with op(A) upper triangular, output row i
only touches `op(A)_{i,p}` for p >= i, so an output tile rooted at row m0 starts its reduction at
p = m0. The arithmetic is not done and discarded; only the k-tile straddling the diagonal is
masked, in the A staging, where the tile is small and read once. Which end is skipped comes from
uplo and trans together: `lower_eff = (uplo == Lower) != transposed`.

**The saving is not 2x.** With R = m/TileM row tiles the reduction shrinks to (R+1)/2R of the
square: 1.0x at R = 1, 1.33x at R = 2, 1.78x by R = 8. Getting R above 1 is what decides whether
this beats a GEMM, which is why the tile is sized to m rather than fixed at 128 (ormqr's
`W2 = T^H W1` has m = ib in the tens, and a 128-row tile would spend four times the arithmetic).
Shared layout, staging and packet swizzle follow [the gram kernel](#syrk-gram-tiles-kernel-design).

**Lanes.** A 128-wide side gets 16 lanes of 8; narrower sides keep the 4-wide band the vectorized
fragment load is built on and drop lanes instead. That fixes the 16-row tile at 4 lanes: ThreadRows
must stay a multiple of 4.

**Complex thread tile at TileM >= 64.** A complex scalar halves how many elements the shared path
delivers per clock while leaving the FMA rate per element alone. A 4x8 thread tile reads 12
complex and issues 32 complex MACs, needing 2.67 MAC per load against a capability of 2; a 4x16
thread tile takes it to 3.2 and puts the kernel back on the FMA pipe, paid for by halving the
column lanes (the row tiling, where the saving lives, is untouched). Not at the 32-row tile: the
block is already 64 threads there and halving again costs more in outstanding loads than the ratio
buys (that end is bandwidth bound): measured **0.374 -> 0.424 ms** at m = 32.

**Staging follows op(A)'s contiguous direction.** A warp walking i over a transposed A strides by
lda and each lane pulls its own 32-byte sector, which measured as the dominant cost before the
transposed and non-transposed staging were split. In the non-transposed staging the four rows a
thread stages go back as one 128-bit store: single-element stores would put every lane on one of
only eight banks (the row stride is a multiple of 32 and drops out).

### trmm: choosing the row tile by scalar type

`trmm_row_tile` is the kernel's one real trade-off. A smaller tile does strictly less arithmetic
((R+1)/2R: 1.0x at R = 1, 0.75x at R = 2, 0.5625x at R = 8) but re-reads B, since each row tile
stages its own copy of the reduction range, up to (R+1)/2 times over (much of it L2 hits). Which
dominates is decided by the scalar type, not the shape:

* **float** is bandwidth bound at these sizes (intensity m/4 flop per byte against a ridge near
  40), so paying B twice to save a quarter of the arithmetic loses; the widest tile that fits wins.
* **double** runs at 1/64 rate, putting the ridge near 1.4 flop per byte and the problem far on the
  compute side; the arithmetic is the whole cost and B's re-read is close to free, so the narrowest
  tile wins. Complex is the same argument (a complex multiply is four real ones).

So one threshold cannot serve both, and the first version (tuned on float alone, one tile for
m <= 128) could not beat a GEMM at m = 128 in any type: R = 1 saves nothing.

**The 16-row tile.** 32 rows used to be the floor, which left ormqr's WY update (m = ib, in the
tens) at R = 1 with no saving, and at m = 16 with half the tile masked off after the arithmetic
was issued. Tile16 against tile32, through `ormqr_blocked_benchmark` (n 256/512, batch 128-256,
ib 16/32/64):

| type | tile16 / tile32 | verdict |
|---|---|---|
| `double` | 1.007x - 1.022x | wins at every ib, including 64 |
| `complex<double>` | 0.995x - 1.040x | wins to ib 32, a wash at 64 |
| `complex<float>` | 0.993x - 1.026x | wins to ib 32, a wash at 64 |
| `float` | 0.966x - 0.997x | loses everywhere, as predicted (the B re-read) |

The shipped thresholds are that table: wide types take 16 through m = 64, complex only through
m = 32 (its 64-row cell was a wash, 0.993x in `complex<float>`, and the wider tile keeps more of the
block: at TileM 16 the launch is down to 64 threads). Float takes 32 through m = 32, 64 through
m = 512 and 128 above (the float TileM 32/64/128 cells are in
[trmm tile vs gemm by dtype](#trmm-tile-vs-gemm-by-dtype)). Wide types **never take 128**: a
128x128 tile is an 8x8 thread tile, which in `complex<double>` is 256 accumulator registers, and
256 threads x 256 is the whole 65,536 a work-group gets; the runtime rejects the launch rather than
spilling. `BATCHLAS_TRMM_TILE_M` pins the tile for sweeps; the `triangular` family's grid-ceiling
term (`trmm_tile_groups`) reads the same tile, pin included.

### Level-3: the complex MAC and return-by-value rules

`src/backends/triangular_tiles.hh`, shared by every tile kernel above.

* **`accumulate` writes the complex MAC out by hand.** `std::complex<float>::operator*` lowers to
  the `__mulsc3` libcall (C99 Annex G), a branch on Inf and NaN around every multiply, invisible in
  the source. The first complex build of the Gram kernel ran at **1.2 TFLOP/s against float's
  13.8**, and at n = 128 took **38 ms where a cuBLAS GEMM took 1.5**. Four real multiplies and two
  adds fold to the four FMAs a complex MAC should be; a NaN input is already a NaN answer, so no
  exceptional case is worth a branch.
* **Return the accumulator by value.** Taking the address of an element of the register-resident
  accumulator array (a `T&` out-parameter) makes the compiler move the whole array to local memory.
  It cost **43%** on float at m = 512 (0.659 -> 0.944 ms) with no other change. `tile_load4`
  returns by value for the same reason.
* **`tile_load4` is 128-bit only for float.** Four doubles are 32 bytes and four `complex<double>`
  are 64, which no load form covers, and the reinterpret asserts an alignment
  `sycl::local_accessor` never promised (it aligns to T, not 4*sizeof(T)). This branch is the only
  thing between these kernels and a misaligned access in double, and it is silent when wrong.

## Negative results

A specialised level-3 op beats the GEMM it replaces only when it reaches a batched custom kernel.
Flop count predicts nothing: `syrk`, `syr2k`, `herk` and `her2k` have only their native families
(`gram`/`triangular`, `triangular`) or batched GEMM route, and where the table row or the fixed rule
does not take one they degrade to a host loop over the vendor call, which at batch 1024+ is one to
two orders of magnitude off a batched GEMM. Check which family ran first (`BATCHLAS_SELECT_TRACE=1`).

### syrk for the ortho gram matrix

`src/extensions/ortho.cc` builds `C = A^H A` three times. Textbook syrk, safe on the consumer
side, and a **70–100x regression** at the shapes that occur, because a skinny Gram matrix failed
both PR-60 routers and dropped to one `cublasSsyrk` launch per batch member:

| m | k | batch | GEMM | syrk | |
|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.350 | 33.73 | 96x slower |
| 1024 | 128 | 512 | 0.425 | 30.93 | 73x slower |
| 512 | 512 | 128 | 0.813 | 0.589 | 1.38x |
| 1024 | 1024 | 128 | 5.671 | 3.393 | **1.67x** |

The winning column is `k >= 512` and square-ish; `ortho`'s callers (`syevx_lobpcg`,
`syevx_filtered`, `lanczos`) all pass `k` = a block size in the tens. **This is what motivated
`syrk_gram_tiles`**, after which the substitution was re-measured and taken: at k = 32, float
1.450 -> **0.895** ms (1.62x) and ShiftChol3 1.998 -> **1.298** (1.54x), m = 1024 batch 512.
`gram_max_k` is 64 for float and 128 for double (`src/extensions/ortho.cc:143`) because of one losing cell: float
k=128 is 8.813 -> 9.156 (**0.96x**), double k=128 is 57.39 -> **42.94** (1.34x). `svqb_alg` keeps
its GEMM: it scales the whole k x k before `syev`, so a one-triangle result would multiply
uninitialised workspace. The full k x precision table (float 1.62/1.12/0.96x, double
1.02/1.20/1.34x at k = 32/64/128) is on @ref perf_ortho, under
[the Gram matrix through syrk, per precision](ortho.md#ortho-the-gram-matrix-through-syrk-per-precision).

### herk on the gram tile kernel

The conjugating path through the same kernel was built and **measured and rejected**: in complex
float it loses to the existing GEMM-plus-Hermitian-fold at every Gram shape — 0.217 vs **0.206**
ms at n=32 batch 2048, 2.08 vs **1.57** at n=128 batch 512. A complex multiply is four real ones,
so herk is compute bound where real syrk is bandwidth bound, and cuBLAS's cgemm is better at
compute. The route stays reachable as `BATCHLAS_SYRK_ROUTE=gram` so it stays measurable and the
conjugation stays under test (`ops::syrk::herk_gram_pinned()`, `src/ops/syrk/choice.hh`, read by
`herk_vendor` in `cublas.cc`).

### trmm for the WY block factor

Before the tile kernel existed, substituting `trmm` for `W2 = T^H W1` lost at **every** shape
(float, ms): 0.195 -> 0.238 at ib=32 nC=256 batch 2048, 0.348 -> 0.498 at ib=64 nC=512 batch 1024,
0.779 -> 1.152 at ib=256 nC=1024 batch 256. Structural, not tuning: `src/extensions/trmm.cc`
recurses only to `n <= 256` and then calls the very GEMM it was meant to replace, so the
triangular structure was never exploited at any `ib` ormqr uses. Re-measured after the tile kernel
against `BATCHLAS_ORMQR_WY=gemm` (ABBA-ordered, `Side::Left`, `ConjTrans`, batch 256 / 128 for
`complex<double>`, nb in {16,32,64}, two agreeing passes) and partly taken:

| type | gemm/trmm over all cells | verdict |
|---|---|---|
| `float` | 1.006x–1.046x | wins everywhere |
| `double` | 1.004x–1.016x | wins everywhere — newly enabled |
| `complex<float>` | 0.944x–0.995x | loses everywhere — excluded |
| `complex<double>` | 0.958x–1.010x | loses at ib = 16 — excluded |
| `netlib float` | 0.336x–1.199x | 0.34x at n=128 ib=16 — excluded |
| `netlib double` | 0.379x–1.064x | 0.38x at n=128 ib=16 — excluded |

`wy_trmm_applicable` (`src/extensions/ormqr_blocked.cc:52`) is therefore **per type**, not
per-precision, plus `ib <= 64` (past it the tile kernel measured 0.83x–0.97x in float). netlib is
out because its trmm and its gemm are both per-batch cblas loops; ROCm because `rocblas_?trmm` is
a per-batch loop against a strided-batched GEMM. **And it barely moves syev**: 1.036x at n=64
batch 2048 nb=16 down to 1.003x at n=512 — consistent across eight cells and several times the
stddev, so real, but 3.6% at best. The trace said so in advance:
`trmm_cuda_custom.triangular_tiles` is **1.0%** of traced syev time at n = 512 (8.58 ms of 868,
over 240 calls), so a 10% faster op cannot return more than 0.1%.

### The 16-row trmm tile

Built to give `ib = 32` `R = 2` and `ib = 16` an exact fit, on the hypothesis that complex lost
because those cells ran at R = 1. **Confirmed in direction, refuted in magnitude.** Against the
GEMM, tile16 takes `double` from 1.004x–1.016x to **1.013x–1.036x**, `complex<double>` from
0.958x–1.010x to 0.996x–1.018x and `complex<float>` from 0.944x–0.995x to 0.946x–0.983x, while
`float` drops from 1.006x–1.046x to 1.004x–1.028x (tile16 vs tile32 is 0.966x–0.997x there).
Every type moves as the R argument says — narrower helps where the kernel is compute bound and
hurts float, which is bandwidth bound and pays in B's re-read — and `complex<double>`'s ib = 16
hole closes from 0.958x to 0.996x, confirming it was the masked-off half tile. But it closes to
*parity*, and `complex<float>` stays 2–5% behind. The tile is kept because **double** wants it;
`wy_trmm_applicable` is unchanged.

### Rejected on inspection, and the transcription that was killed

* `X^H A X` (`syevx_lobpcg.cc:528,1225`, `syevx_filtered.cc:418`) — symmetric result, but a
  product of two *different* matrices; no BLAS op expresses it, and `syr2k` is not this.
* `A X` with symmetric A (`syevx_lobpcg.cc:509,638,1212`, `lanczos.cc:112`, `ritz_values.cc:59`)
  — nominally `symm`/`hemm`, but symm here expands then GEMMs and A is already stored full, so it
  would add a copy to reach the identical GEMM. Worse by construction.
* `gebrd_blocked.cc:364,365` — looks like syr2k, is not: bidiagonal reduction of a *general*
  matrix, `a22` is not symmetric.
* Transcribing the four ops' thresholds into the old `RouteTable::preferred` was killed by a
  **confirmed silent route change**: a transcribed `tiles_per_side >= 3` rule rejected the tile route
  for `129 <= n <= 383` at every batch, sending n = 256 to `DiagFullGemm`, which wrote **both
  triangles** — a shape `tests/syrk_tests.cc` names explicitly, found independently by two judges.
  The flat-selection transcriber avoided it by evaluating the whole old decision per grid cell, not
  the predicate (see [before flat selection](#level-3-before-flat-selection)): the n = 256 `sq` rows
  rank `triangular` first, and `DiagFullGemm` is deleted.

## Correctness findings

### The band-split syrk bug

The 128-wide Gram tile initially split each thread's 8 rows into two 4-wide bands 64 apart — what
the square 128x128 kernels do to spread banks. That is incompatible with taking the triangle at
*thread-tile* granularity, which decides a whole thread tile is inside the requested half from
its tile indices alone: thread (0,1) then owns element (64,4), lower triangle while its tile is
not, so **nothing wrote it**. Silent, and only at n > 64. It hid because `syrk_tests` pinned one
shape, n = 96, reaching only one of the kernel's three tile widths.
`SyrkTest.NarrowShapesMatchGemmReference` (`tests/syrk_tests.cc:91`) now sweeps
n in {24,32,48,64,96,128} x trans x uplo at k = 200 (not a multiple of the k chunk), and mirrors
both sides before comparing so an element left at its input value is caught.

### The herk test that could not fail

`HerkTest` checked that the unreferenced triangle stays untouched and that the two uplo runs
agree. **Neither can catch conjugating the wrong operand** — the likeliest defect in a shared
syrk/herk kernel: conjugating the row index instead of the column returns `conj(C)`, still
Hermitian and still consistent across both triangles. `HerkTest.MatchesGemmReference`
(`tests/herk_tests.cc:314`) compares against a GEMM and was confirmed to fail when the
conjugation is flipped.

### The trmm poison test

trmm carries a documented prior incident where the tempting 8x "fix" was the wrong-answer one and
the guarding test could not fail by construction: every other test built A with
`RandomTriangular` — already zeroed opposite the triangle, already ones on a unit diagonal — and
validated against a full gemm on that same A, which passes for an implementation ignoring `uplo`
and `diag` entirely. `TrmmTest.IgnoresUnreferencedTriangleAndUnitDiagonal`
(`tests/trmm_tests.cc:132`) poisons the forbidden storage and differences against a gemm on the
clean A over both sides, uplo, diag, three transposes, ragged and non-square shapes; on CUDA it
re-runs with `BATCHLAS_EXPAND_MAX_BYTES=0`, which makes `expand`'s `can_run` refuse every shape, so
Right-side calls reach the `cublas?trmm` loop that no test shape otherwise reaches.
`uplo`/`side`/`diag` are in the coverage key (`coverage::Shape`) for the same reason.

### The syr2k trailing-update test

The `sytrd_blocked` trailing update runs only when the trailing block exceeds 128, and every
pre-existing case in `tests/sytrd_blocked_tests.cc` was n <= 128 — so **the syr2k route had no
test coverage at all**, and flipping its default on benchmark strength alone would have flipped
an unexercised branch. `SytrdBlockedTest.TrailingUpdateRoutesAgree` (n=320, nb=32) was added and
checked for teeth: forcing `alpha = -0.5` fails it loudly (worst eigenvalue error 2.777 against a
3.2e-3 bound, GEMM route at 2.6e-6), and the backward-error bound alone is ~1000x looser than
either route's error, so the assertion doing the work is the *relative* one — syr2k within
`4 * (GEMM route error) + 8 eps ||A||`.

The same work removed a symmetrize pass that had eaten over half the win, because **nothing in the
`sytrd_blocked` pipeline reads A's upper triangle**: all three `latrd_lower_panel` variants split
at `c == r`, both fused trailing updates skip `r < c`, and `restore_tridiag_lower` only *writes*
the superdiagonal. The GEMM pair happened to leave a valid upper triangle behind and nothing
depended on it, so it was never a contract — invisible until an op respecting the triangle replaces
one that does not. Verified on the legacy impl, on `BATCHLAS_SYTRD_IMPL=device`, and with the grid
variant forced. End to end (float, ms) n=512 batch 1024 goes 263.97 -> **227.51** at nb=16 and
248.30 -> **231.64** at nb=32, n=256 batch 2048 goes 34.347 -> **27.040** and 36.995 -> **34.028**,
and the update alone is 3.4–3.6x. The gate stays CUDA + `float`/`complex<float>`
(`rank2k_trailing_update_supported`, `sytrd_blocked.cc:785-787`) because in double it **inverts**:
double `syr2k` has only the `vendor` family, a per-batch loop, 7.56 vs 58.52 ms at n2=256 ib=32
batch 1024, **7.7x slower**.

### The coverage instrument itself

Three defects in the level-3 coverage recorder of the WP1 era (`level3_coverage.hh`, deleted with
the hand-written gates) surfaced only by *using* it, each looking healthy while reporting almost
nothing: the gate-declined half was unrecorded, so a shape moving *off* a native kernel was
invisible; `uplo`/`side`/`diag` were not in the key, so calls differing only in `uplo` collapsed into
one row; and `emit()` opened with `"w"`, so each of 53 test binaries truncated the last one's output.
`native_supported` was a **tri-state** because on a gate decline the caller cannot tell "no native
route serves this shape" from "one does, but the heuristic preferred the vendor". The lessons carried
over: the four ops now record through `select::run` like every other op, `native_supported` is the
tri-state `NativeFacts` (`src/select/select.hh`), and `coverage::Shape` carries `uplo`, `side` and
`diag`.

### The her2k alpha alignment fault

`cublas?her2k` dispatches to cuBLASLt, which reads the host `alpha` with a 16-byte aligned vector
load; `std::complex<double>` is 8-byte aligned, so passing the parameter's address faults whenever
it lands 8 mod 16 — shape-dependent, so most calls survive. Reproducible against cuBLAS 13.2 with
none of BatchLAS present; fixed with `alignas(16) T alpha_aligned = alpha` (`her2k_vendor` in `cublas.cc`).

### Level-3: generic syrk and syr2k fallbacks fold one triangle

The generic fallbacks in `src/extensions/syrk.cc` and `syr2k.cc` (used where no native
`?syrk`/`?syr2k` is wired here: MKL) used to be one or two GEMMs straight into C with `uplo` an
unnamed parameter, so they wrote **both** triangles and silently overwrote the half the caller
owns. A GEMM computes the whole symmetric product, so it cannot be aimed at C (the same reason the
CUDA herk path computes into scratch). The product now goes to scratch and
`fold_symmetric_product_into_triangle` (`src/extensions/symmetric_product_fold.hh`) folds only the
named triangle back: `C(i,j) := product(i,j) + beta*C(i,j)` for (i,j) in the uplo triangle, with
beta == 0 meaning C is not read, so an uninitialised or poisoned C cannot turn the result into NaN.

## Level-3: code-structure decisions

Why the level-3 family is laid out the way it is. Each item is a decision a future edit could
undo by accident.

### Level-3: lessons of the deleted gate layer

Three structures of the WP1 era were deleted with the hand-written gates; each left a rule the flat
layout still follows.

* **Instrumented rather than routed.** WP1 changed where symm/syrk/syr2k/trmm terminate, and the only
  acceptable outcome on a vendor-present box was that no decision moved. Reading the diff cannot show
  that (a dispatcher hides the decision from the call site); timing cannot (an unsaturated
  benchmark's ratios are overhead, and routing a shape to cuBLAS may be *faster*, so a perf gate
  cannot flag a wrong route); the kernel trace cannot (its record holds a `sycl::event`, so a
  vendor-to-vendor change is invisible). The shared selector did not see these four, so each terminal
  recorded the branch actually taken, beside a `return` and never in place of one. Today
  `select::run` records them like every other op, with `chosen_algo` the spelling, so
  `scripts/route_diff.sh` sees a level-3 route change directly. A table was **not** equivalent while
  the gates were gate-only ([before flat selection](#level-3-before-flat-selection)): measuring first,
  transcribing per cell later, is what made the move safe.
* **The cuBLASDx fused-tail hook.** The fused kernels lived behind a link-time hook
  (`level3_fused_cuda.cc`, compiled only with cuBLAS, against `level3_fused_absent.cc`) because an
  `if constexpr` cannot discard a file-scope `#include`; only `#if` can, and fencing in place would
  have left the dispatchers striped with preprocessor and reaching `<cuda_runtime_api.h>`. The hook
  answered three outcomes (`Ran`, `NoKernel`, `DeviceUnsupported`). MathDx was never present
  (`BATCHLAS_HAS_CUBLASDX 0`), so no tail was ever reachable and no test here could have caught a
  mistake in it; that, and syr2k's fused fallback taking no `uplo` (it wrote both triangles, and every
  level-3 test uses a single uplo per call), is why the paths were deleted rather than kept. A
  `cublasdx` pin now throws as an unknown family.
* **The sideways vendor seam.** Making the old dispatchers' vendor fallbacks call the *public* entry
  point recursed without bound: every fallback site is reached *after* a gate that already returned
  true, so the public call re-entered the same gate with the same environment and views
  (reachable with `BATCHLAS_SYMM_ROUTE=cublasdx` on a CPU queue). The seam forwarded to the vendor or
  threw `NoRouteError` instead. Under flat selection the `vendor` family calls
  `backend::<op>_vendor<B, T>` under `has_library<B>(spec.vendor)` and never the public entry, so
  that recursion cannot happen. The seam's other lesson stands in the headers: `sig::<op>_vendor`
  signatures are spelled out, not generated from the public declarations, because the vendor and
  public forms disagree (trsm's vendor form takes `alpha` last).

### Level-3: one set of shape validators

`src/backends/level3_shape.hh` replaced seventeen hand-written copies: symm/hemm/trmm, syrk/herk
and syr2k/her2k impose the same three shape contracts on every backend, and cuBLAS and rocBLAS (at
the top of the vendor wrapper) and netlib (inside the deferred host task) each wrote all three out.
The predicates were identical and in the same order (squareness first, since a non-square A makes
"order k" meaningless; then batch; then shapes), so only the exception type stays per backend,
hence the template parameter `E`: netlib's checks run inside `submit_host_task`, at a different
time from the caller's stack, and throw `std::runtime_error`, while cuBLAS and rocBLAS throw
`std::invalid_argument` from the call itself, which `options_api_tests` pins with `EXPECT_THROW`.
Collapsing the two would change what a caller catches. Since flat selection the four table-selected
ops also validate at their public entry with `std::invalid_argument`, before `select::run`, so for
them every backend (netlib included) and a vendor-free build throw that; netlib's host-task
`std::runtime_error` is what a malformed hemm, herk or her2k call still meets on the host. The
validators return the dimensions they
derive, because the recomputed m/n/k were the other half of the duplication. The batch message
names every batch size, since which operand is the odd one out is otherwise invisible.

**Namespace trap.** The validators live in `batchlas::backend::shape`, not
`batchlas::backend::detail`: `netlib_lapack.cc` writes most of its file in `namespace batchlas` and
reaches its own `batchlas::detail` helpers unqualified, and introducing a
`batchlas::backend::detail` into that TU makes the unqualified `detail::` inside its
`namespace backend` blocks resolve to the wrong one (`submit_host_task` stops being found).

### Level-3: one route parse per variable

`BATCHLAS_<OP>_ROUTE` for these four is parsed by `src/select` as for every other op (`pin_text`,
then `resolve_pin` in `src/select/select.hh`), and a spelling the op does not take throws
([the level-3 pin words](dispatch.md#the-level-3-pin-words)). `BATCHLAS_EXPAND_ROUTE`, which only
hemm, herk and her2k still read, comes from one Settings field that `expansion_route_pin`
(`src/expansion_budget.hh`) and `expansion_preferred` (`triangular_expand.hh`) both read, so the two
cannot disagree.

History. Before WP0, `BATCHLAS_TRMM_VARIANT` was read by two parsers that disagreed on its
vocabulary: one understood `vendor`/`cublasdx|dx|custom`/`auto` and returned Auto for anything else,
the other looked for `triangular|tiles`, so `=triangular` was simultaneously "no opinion" and "pin
the tile kernel". WP0 folded them into one parse per op; phase 5 dropped the `_VARIANT` spellings,
and the level-3 flat-selection wave moved the parse into `src/select`. Two legacy words did **not**
mean what the canonical vocabulary read them as: `custom` was the fused cuBLASDx kernel, and `gemm`
was the both-triangles `DiagFullGemm` measurement route. Both now throw
(`*CudaCustomTest.RemovedRouteWordsThrow`). `=vendor` must keep meaning the vendor even where a tile
kernel ranks first, because it is the only "before" a measurement can be taken against.

### Level-3: scratch expansions and their ceilings

`src/backends/triangular_expand.hh` and `src/expansion_budget.hh`. Consumers: `symm`'s and
`trmm`'s `expand` families (`symm.cc:52-56`, `trmm.cc:63-68` check `expansion_fits` in `can_run`)
and the fixed rules of `hemm_vendor`, `herk_vendor` and `her2k_vendor` in `cublas.cc`.

**Why an expansion.** SYMM, HEMM and TRMM must not touch the unreferenced triangle (nor, for TRMM
under `Diag::Unit`, the diagonal; nor, for HEMM, the diagonal's imaginary part), so pointing a GEMM
at the caller's A is wrong even when the caller happened to zero it. The expansion materialises the
dense operand in scratch. The scratch is a workspace lease, not a fresh `Matrix`: a `Matrix` is a
managed allocation whose pages migrate on first touch, which at n = 512 batch = 512 costs several
times the GEMM it feeds (an order of magnitude, per the symm note), and it would be freed on return
while the kernels reading it are only enqueued.

**The mirrored expansion** stages one tile pair per work-group through local memory: the mirror of
a coalesced column read is a row write, one cache line per element, and the expansion is pure
bandwidth. Going through a tile keeps the read and both writes coalesced and moves 1.5 n^2 of
traffic against the 3 n^2 of a copy plus in-place symmetrize. Half the groups (the unreferenced
side) retire empty, which is cheaper than an integer square root per work-item.

**`expansion_fits`: two hard ceilings, not tuned ones.** SYCL linearises the global id and rejects
a range whose product does not fit an `int`; one work-item per element hits that at 2^31 elements,
measured as a thrown `sycl::exception` at n = 2048 batch = 512. And the scratch shares the device
with A, B and C (about three times its size for a square problem), so the budget is a quarter of
global memory: at n = 2048 batch = 256 that is 4.3 GB of scratch inside 17 GB of live operands,
which runs. `BATCHLAS_EXPAND_MAX_BYTES` lowers the memory ceiling, for sharing a device and for
reaching the no-scratch fallback from a test without allocating gigabytes. `expanded_ld` packs the
columns and pads only to 16 bytes, the alignment the vendor and native GEMMs want for packet loads;
the caller's ld is irrelevant because the expansion writes every element. A shape past either
ceiling makes `expand` fail `can_run`, so the row's next entry (the vendor loop) runs.

**Why `expansion_budget.hh` lives outside `src/backends/`.** `src/extensions/sytrd_blocked.cc`
must know whether her2k will take its batched-GEMM route or its per-batch host loop before it
decides to call her2k at all, and it could not include `triangular_expand.hh` (that pulled
`cublasdx_dispatch_common.hh`, whose `<cuda_runtime_api.h>` was unguarded, while `sytrd_blocked.cc`
is also built for ROCm and the host; `cublasdx_dispatch_common.hh` is deleted now, but the split
stays). One definition, not a copy: a reimplemented ceiling drifts
silently, and the caller believes it got the fast route and gets the host loop.

**The bug that put `expansion_route_pin` there too.** `sytrd_blocked`'s her2k guard originally
replicated only the size ceiling, so under `BATCHLAS_EXPAND_ROUTE=loop` it concluded her2k would
take its batched-GEMM route while `her2k_gemm_preferred` returned false and sent it to the
per-batch `cublas?her2k` loop: one sequential launch per batch member, for every panel with
n2 > 128. A guard that models only half the predicate it guards against is worse than none,
because it reads as checked. Hence `her2k_takes_gemm_route`, the whole condition, and
`expansion_route_pin` (`1` = expand, `0` = loop, `-1` = unset), which `cublas.cc`'s
`rankk_route_pin` delegates to and which reads the same Settings field as
`triangular_expand.hh`'s `expansion_preferred`.

## Open debts

### Forced-route defects

1. ~~`BATCHLAS_SYRK_ROUTE=native` produced a wrong answer~~ (it fell through to the
   both-triangles `DiagFullGemm` route) and
2. ~~`BATCHLAS_SYR2K_ROUTE=native` threw a cuBLASDx message it did not ask for~~: fixed in flat
   selection phase 5, which replaced the route vocabulary with per-op words
   ([dispatch.md](dispatch.md#the-level-3-pin-words)) and deleted `DiagFullGemm`. `native` takes
   the first native entry of the row; `SyrkCudaCustomTest.AutoAndNativeRoutesLeaveTheOtherHalfUntouched`
   holds it.

### Routing and reachability

3. ~~`symm` has no `expansion_fits` ceiling~~: closed by the level-3 flat-selection wave; symm's
   `expand` family checks it in `can_run`, and a shape over the budget takes the vendor.
4. ~~`double` symm has no expansion route at all~~: `expand` serves double too (the double rows
   still rank `vendor` first, as the old rule did; vendor-free it runs `expand`).
5. ~~Heterogeneous `symm` is unmeasured~~: measured on `ff340fc6`; a heterogeneous B or C made the
   expansion's gemm throw and a heterogeneous A ran at the storage order. `expand` now refuses
   every heterogeneous operand, and so does the vendor (its loop answers at the storage order).
6. **`trmm`'s tile kernel is `Side::Left` only** — every Right row ranks `expand` first. syev
   uses Left only; `ormbr` has the same WY update, is not wired, and feeds gesvd.
7. **ROCm has no native level-3 path and no `hemm`, `herk` or `her2k`.** The native families'
   `can_run` requires `Backend::CUDA`, so ROCm `syrk`, `syr2k` and `trmm` take the rocBLAS loop
   (`rocblas.cc` instantiates only gemm, gemv, trsm, trmm, syrk and syr2k vendor wrappers); ROCm and
   MKL `symm` is `src/extensions/symm.cc`, outside selection. Wiring the trmm tile kernel there is
   where `wy_trmm_applicable` would be re-measured.
8. **`her2k_gemm_preferred` was swept over square rank-k shapes**, but `sytrd_blocked`'s panel
   loop issues a narrow one — k = nb in {16,24,32} against n2 up to 480 — where the GEMM is near
   bandwidth bound and the fold adds an `n2^2 * batch` write plus read the direct GEMMs never pay.
   Awaits an A/B at n2 in {224,480}, `complex<float>`; `complex<double>` is left out entirely,
   same route but 16 bytes per element halves the fit headroom, unmeasured.
9. **The level-3 tables are transcribed, not timed.** Every row reproduces the old gate's choice at
   its grid point; none was measured as a ranking. Retuning means sweeping the families on the
   saturated grid (`tools/tune`) and replacing the rows. Known suspects: the `sq` rows that rank
   `triangular` from n = 129 although only n >= 384 was measured to win against the GEMM; the double
   `symm` rows that rank `vendor` first although `expand` now runs double (unmeasured); and the
   `complex<float>` Left `trmm` rows that rank `triangular` first although the tile kernel loses to a
   plain GEMM below m = 512 ([trmm tile vs gemm by dtype](#trmm-tile-vs-gemm-by-dtype)); tile against
   `expand` was measured in float only.

### trmm: the generic recursion reads the whole square of A

`src/extensions/trmm.cc` (the MKL instantiation) ends its recursion at n <= 256 in a plain `gemm`
on the diagonal block with `beta = 1`; a triangularize call there was commented out (and removed
as a dead comment on 2026-09-30). Read as written, the base case uses A's unreferenced triangle and
its stored diagonal under `Diag::Unit`, and accumulates into C rather than overwriting it, so it
matches BLAS only when the caller zeroed the opposite triangle and C. Found by reading during the
2026-09-30 comment pass; **not confirmed by a test**. A poison test in the style of
[the trmm poison test](#the-trmm-poison-test) against the MKL backend would settle it.

### Unverified boundaries

`257 <= n <= 383` for the `syrk` `triangular` rows; batch 3 and `769 <= n <= 895` for herk's rule;
batch 1 with `n >= 128` for her2k's; the `symm` rows' `sq`-only `expand` (the old squareish
exclusion); and the deliberate slack between the symm/hemm measured loss region
(`batch <= 2 && n <= 128`) and the shipped constants (4 and 256).

### Vendor-free gaps still open

Post-WP8 `NoRouteError` census over `ctest -LE slow`: `trmm` 16, `herk` 16, `syrk` 12, `her2k` 12,
`hemm` 12, `syr2k` 10, `symm` 8 — the double and complex arms trapped in `cublas.cc` plus the ops
with no native arm. The suite pass count cannot show movement here; read the per-op census. The
level-3 flat-selection wave closed trmm (every dtype, both sides), symm (float and double), float
syrk and syr2k entirely and double syrk at n <= 128; hemm, herk, her2k, double syrk above n = 128
and double syr2k remain (their rows hold only `vendor`, so a vendor-free build throws `NoRouteError`).

### Instrumentation and harness

* (Recorded at WP1, not re-checked since.) `symm_benchmark`, `syrk_benchmark` and
  `syr2k_benchmark` **aborted before printing anything** — a
  SYCL scheduler assertion (`adjustNDRangePerKernel: NDR.LocalSize[0] == 0`) on the host backend
  at tiny shapes, attributed by revert-and-rebuild to something pre-existing. Every measurement at
  the WP1 line needed a standalone harness.
* **A false win was nearly reported.** `syr2k` at n = 1024 looked 10.9% faster after the GEMM
  terminal moved to the public entry point; repeating the prior step there gave a 5.65–6.40 ms
  spread. It was noise.

### Kernel-level headroom

* The trmm tile kernel runs at ~70% of cuBLAS's per-flop rate; closing that would turn the
  m = 128..256 band from a loss into a ~1.2x win. Complex needs that rate, not a better R.
* syrk at n = 256 (0.84x–1.22x on `syrk_triangular_tiles`) is a pre-existing loss, untouched.
* The two-stage syev path's `sy2sb` trailing update has no syr2k and n >= 512 syev goes through
  it — separate, larger work. **Level-3 substitution is not the lever for syev**: at n = 512 with
  eigenvectors the split is `backtransform_q2` 46.4%, `sb2st_hh` 25.5%, `stedc_eigvecs` 10.6%,
  none a level-3 triangular op.

## Level-3: raw evidence

Raw data is preserved at the git tag `perf-evidence/vendor-independence`, retrievable with
`git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| syrk gram tiles; trmm tiles; per-dtype tables; the 16-row tile; ortho wiring; ormqr WY; syev traces | `experiments/TRMM_SYRK_BATCHED_KERNELS.md` |
| the 73-call-site substitution survey; sytrd syr2k trailing update; ortho gram rejection; WY-factor rejection | `experiments/GEMM_TO_LEVEL3_SURVEY.md` |
| syrk crossover drivers (route x shape on `avg_ms`; the second frees k) | `experiments/syrk_sweep.sh`, `experiments/syrk_kskew.sh` |
| syr2k crossover driver; herk/her2k expand-vs-loop driver | `experiments/syr2k_sweep.sh`, `experiments/herk_crossover.sh` |
| the exclusive-GPU guard every sweep above runs through | `experiments/gpu_guard.sh` |
| WP1 design pass, its five corrections to the plan, the eight steps | `WP1_LEVEL3_SPEC.md` |
| vendor-free failing set and the per-op `NoRouteError` census | `VENDOR_FREE_BASELINE.md` |
| campaign context, Class B analysis, WP1-complete summary | `VENDOR_INDEPENDENCE_PLAN.md` |

**The per-cell CSV captures for these sweeps were not committed.** The tag holds the four sweep
scripts and the two distilled notes; the numbers here come from those notes and from the shipped
route comments, written against the runs. Re-deriving a boundary means re-running the script.

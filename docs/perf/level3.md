# The level-3 tile ops: symm, hemm, syrk, herk, syr2k, her2k, trmm

The shipped code is the authority on *what* ships; the exploration notes
(`experiments/TRMM_SYRK_BATCHED_KERNELS.md`, `experiments/GEMM_TO_LEVEL3_SURVEY.md`,
`WP1_LEVEL3_SPEC.md`) are the authority on *why*; where they disagree, the disagreement is named
in place. All measurements: RTX 4090 / sm_89, CUDA 13.2, `RelWithDebInfo`, one dedicated GPU
(`experiments/gpu_guard.sh`), at saturating batch unless the cell says otherwise.

## What ships

### Route arms

| op | Origin x Algorithm arms | scalar types served |
|---|---|---|
| `symm` | `{Native, ExpandGemm}`, `{Vendor, FusedDevice}`, `{Vendor, Auto}` | **float only** |
| `hemm` | expand-then-gemm inside `hemm_vendor`; per-batch `cublas?hemm` loop | complex only (BLAS has no real `?hemm`) |
| `syrk` | `{Native, GramTiles}`, `{Native, TriangularTiles}`, `{Vendor, FusedDevice}`, `{Vendor, DiagFullGemm}`, `{Vendor, Auto}` | float via the facade; `GramTiles` also double/complex, but only from `cublas.cc` |
| `herk` | GEMM-into-scratch + `accumulate_hermitian<false>`; per-batch `cublas?herk` loop; opt-in `GramTiles` | complex only |
| `syr2k` | `{Native, TriangularTiles}`, `{Vendor, FusedDevice}`, `{Vendor, DiagFullGemm}`, `{Vendor, Auto}` | **float only** — `syr2k_triangular_tiles` has one call site in the tree |
| `her2k` | GEMM-into-scratch + `accumulate_hermitian<true>`; per-batch `cublas?her2k` loop | complex only |
| `trmm` | `{Native, TriangularTiles}`, `{Vendor, FusedDevice}`, `{Vendor, Auto}` (expand-then-gemm) | float via the facade; the tile kernel is type-generic, double/complex reach it only from `cublas.cc` |

`{Vendor, FusedDevice}` is the cuBLASDx fused kernel and **has never run in this build**: MathDx
is absent (`BATCHLAS_HAS_CUBLASDX 0`), so `cublasdx_variant_needs_fallback` is unconditionally
true (`level3_fused.hh:14-18`) and every "cublasdx" route ever measured here is its fallback.
`{Vendor, DiagFullGemm}` is a deliberately **wrong** route that stores both triangles, kept only
so the arithmetic the triangular kernels save is measurable; `Auto` must never select it
(`route.hh:38-86`).

### The shipped predicates

Quoted as implemented, not as the notes describe them.

```cpp
// syrk_custom_dispatch.cc:109-118, n and k taken from C.rows() and the transA-selected extent
if (detail::triangular_tiles_per_side(n) < 3 || k < detail::kTriangularTileK) return false;
return (long long)A.batch_size() * detail::triangular_tile_count(n) >= 160;
bool syrk_prefer_gram_tiles(C) { return C.rows() <= detail::kGramMaxTile; }   // :125-127, == 128
return min_dim*2 >= max_dim && tiled_work >= 8;   // :129-144, and n >= 16

bool syr2k_prefer_triangular_tiles(A) { return A.batch_size() >= 2; }  // syr2k_...cc:95-97

// trmm_custom_dispatch.cc:141-157 -- there is NO shape threshold
if (detail::is_gpu_queue(ctx) && trmm_triangular_supported(A, B, C, side) &&
    (trmm_triangular_requested() || !dispatch::is_plain_vendor(trmm_route_request()))) return true;

return batch >= kExpandMinBatch /*4*/ || max_dim >= kExpandMinDim /*256*/;  // expand.hh:44-45,59
return batch >= 4 && n <= 768;    // cublas.cc:387             herk
return batch >= 2 || n >= 128;    // expansion_budget.hh:115   her2k
```

`triangular_tiles_per_side(n) = ceil(n/128)`, `kTriangularTileK = 8`
(`triangular_tiles.hh:118-128`), so `>= 3` means **n >= 257**, not the n >= 384 that the same
function's comment cites as the measured win. `expansion_fits` (`expansion_budget.hh:66-79`) is
two hard ceilings, not a tuned one: the SYCL global range must fit an `int` (fails at 2^31
elements — a thrown `sycl::exception` at n=2048 batch=512), and the scratch must fit a quarter of
global memory.

### Where the decision actually happens

**The four gates are float-and-CUDA only** — `src/dispatch/entry_points/level3.cc:188, 379, 417,
456` each wrap the gate in `if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>)`.
**And the thresholds above are GATE-ONLY, so the effective route is wider than the predicate
reads**: once `syrk_use_cuda_custom` returns true for any reason, `syrk_cuda_custom`'s `Auto` arm
takes `syrk_triangular_tiles` unconditionally after the gram test fails
(`syrk_custom_dispatch.cc:237-241`), with no second preference check — so a square n = 256 shape
passes the third disjunct, reaches the tile kernel, and `n >= 257` never runs. Reading these as a
`preferred()` window is wrong in both directions (`level3_coverage.hh:29-37`), and is why these
four ops have **no `RouteTable`** and are instrumented at each terminal instead.

### Non-float routes

Two native arms exist for non-float, reachable **only** when cuBLAS is compiled because they live
in `cublas.cc`: `syrk` gram tiles at `:751-764` (n <= 128 only) and `trmm` triangular tiles at
`:997-1006` (`Side::Left`, homogeneous). `syr2k` has **nothing** non-float. Hence WP1 S7 refused
to flip `level3_tile_kernels_compiled` to a bare `true` (`route_compiled.hh:63-64`):

```cpp
template <Backend B, typename T>
inline constexpr bool level3_tile_route_available =
    B == Backend::CUDA && (std::is_same_v<T, float> || bool(BATCHLAS_HAS_CUBLAS));
```

## Boundaries and their evidence

### syrk triangular tiles

128x128x8, indexed over the triangular tile set so a tile outside the requested half is never
launched. Grid: float, n in 64..2048 x batch in 1..512, against the full n x n batched GEMM
(`syrk_custom_dispatch.cc:88-108`).

| boundary | admit side | bracketing non-winner |
|---|---|---|
| `tiles_per_side >= 3` (n >= 257) | 1.45x at n=512 batch 512; 1.63x at n=1024 batch 64; 1.71x at n=2048 batch 16 | n = 256 measured **0.84x–1.22x**, depending where its grid fell against a wave boundary |
| `batch * tile_count >= 160` | won from 168 blocks up (recorded as `0.71x`, inverting its own convention mid-sentence) | 144 blocks 1.14x slower; 136 blocks 1.25x slower |
| k | does not enter — it only deepens both routes' reduction | not swept here; `experiments/syrk_kskew.sh` exists to sweep k free |

**Two records of the same boundary differ in framing.** The exploration note reports `0.89x at
n = 256, 1.28x at 512, 1.57x at 1024`; the shipped comment reports `0.84x–1.22x at 256, 1.45x at
512 batch 512, 1.63x at 1024 batch 64` — different batches, same conclusion at 256 and the same
direction above it. Neither holds a cell in **257 <= n <= 383**, which the predicate admits.

### syrk gram tiles

The single-tile kernel sized to n, so both operands of `A^T A` are the same columns of A: one
shared tile, A crosses the bus once. Serves n <= 128, exactly what the triangular grid cannot; no
threshold to tune, because below 128 the alternative is a host loop over `cublasSsyrk`. Float,
`m` is the reduction depth, `before` is that loop (ms):

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
k in 4..2048 x batch in 1..1024 (`syr2k_custom_dispatch.cc:83-94`).

* **batch >= 2**: won *every* shape — 1.06x at n=3072, 1.12x at n=1024, 1.3–1.4x through the
  middle, up to 226x where n is small enough that the whole cost is the launch.
* **batch 1**: does not sort by anything. Vendor wins 1.18–1.60x below n=1280 and 1.16x at
  n=3072; the kernel wins 1.02–1.71x between; the vendor wins 4–10x on a deep k with a small n,
  where the kernel has a single block and cuBLAS splits the reduction.

Bracketed on both sides, and the batch-1 side is genuinely unsortable — hence no threshold in n.
Issuing the two products sequentially rather than interleaved keeps one pair of 8-wide fragments
live over the 64 accumulators, worth **1.53x** (3.34 vs 5.11 ms at n=512 batch 512).

### trmm tiles have no threshold

The first router gated the tile kernel to `m <= 64 || m >= 512`, read off a trmm-vs-gemm column.
Wrong comparison: the router chooses between the tile kernel and **the expansion**. Measured tile
against vendor, float, saturating batch (`trmm_custom_dispatch.cc:84-94`, ms): m=128 nC=512 batch
1024 **0.698** vs 0.784; m=128 nC=1024 batch 512 0.686 vs 0.687; m=256 nC=256 batch 512 **0.536**
vs 0.692; m=256 nC=1024 batch 256 0.915 vs **0.855**. The gate cost up to **1.29x** on
exactly the shapes it was meant to protect; the single 7% loss is left unfitted, since a clause
that narrow would need re-tuning whenever either route moved.

### trmm tile vs gemm by dtype

The *caller's* question — whether to spell a product as trmm at all — not the router's. Ratios
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
`trmm_row_tile` (`trmm_triangular_tiles.hh:412-438`) picks the tile from the scalar type: float
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

**The shipped guard is not the complement of that loss region and is strictly more
conservative**: with `batch >= 4 || max_dim >= 256`, batch 3 at every n and batch <= 2 with
`129 <= n <= 255` are refused despite lying outside the measured loss region.
`WP1_LEVEL3_SPEC.md` correction 4 records the constants as 4 and 256 and says to preserve them;
the exploration numbers would have supported 2 and 128. The code wins — that gap is deliberate
slack, not a measured boundary. `symm` additionally requires `squareish` (`min_dim*2 >= max_dim`)
and `shared_dim == k` (`symm_custom_dispatch.cc:59-77`) where `hemm` does not, on the argument
that a full k x k expansion stops paying once k dwarfs m and n — with no bracketing cell for it
in either source.

**`trmm` deliberately does not consult this predicate.** `trmm_vendor_impl` (`cublas.cc:908-915`)
expands wherever `expansion_fits` allows, batch 1 included, because `cublas?trmm` has a flat
~110 us floor whatever the shape: **49 square cells** (k in 16..1024 x batch in 1..512) at
**1.15x–162x**, and **64 skewed cells** (k in 256..2048 against 1..128 right-hand sides) at
**1.22x–32x**, with not one cell going the other way. No boundary to bracket there.

### herk and her2k crossovers

Both replace a per-batch vendor loop with one strided-batched GEMM into scratch plus a fold.

| op | predicate | admit side | bracketing non-winner |
|---|---|---|---|
| `herk` | `batch >= 4 && n <= 768` | 1.6x–72x for batch >= 4 at n <= 512 | batch <= 2 is a wash or a loss at every n; **0.82x–0.93x from n = 896 up**, where one `cublas?herk` already saturates the device |
| `her2k` | `batch >= 2 \|\| n >= 128` | 1.4x–128x | batch 1 at n <= 64 only: **0.74x at n = 32, 0.89x at n = 64** |

herk grid: complex64, n in 32..1024 x batch in 1..256 (`experiments/herk_crossover.sh` drives it).
`herk` starts from **twice** a rank-k update's arithmetic — the GEMM computes both triangles and
keeps one — so it wins only where the loop is launch bound; `her2k` starts from **half**, because
`alpha*A*B^H` and `conj(alpha)*B*A^H` are conjugate transposes and the mirrored read manufactures
the second term from the first. Two gaps: the grid steps batch 1, 2, 4, 8, 16, 64, 256, so
**batch 3 is unmeasured**, and with a wash from n = 640 to 768 and a loss from 896,
**769 <= n <= 895 is unmeasured** — the threshold sits at the top of the wash band. For her2k the
winning side of the `n >= 128` disjunct at batch 1 has no quoted cell.

## Negative results

A specialised level-3 op beats the GEMM it replaces only when it reaches a batched custom kernel.
Flop count predicts nothing: `syrk`, `syr2k`, `herk` and `her2k` have exactly one batched path
each, and outside its window they degrade to a host loop over the vendor call, which at batch
1024+ is one to two orders of magnitude off a batched GEMM. Check the branch first.

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
`gram_max_k` is 64 for float and 128 for double (`ortho.cc:176`) because of one losing cell: float
k=128 is 8.813 -> 9.156 (**0.96x**), double k=128 is 57.39 -> **42.94** (1.34x). `svqb_alg` keeps
its GEMM: it scales the whole k x k before `syev`, so a one-triangle result would multiply
uninitialised workspace.

### herk on the gram tile kernel

The conjugating path through the same kernel was built and **measured and rejected**: in complex
float it loses to the existing GEMM-plus-Hermitian-fold at every Gram shape — 0.217 vs **0.206**
ms at n=32 batch 2048, 2.08 vs **1.57** at n=128 batch 512. A complex multiply is four real ones,
so herk is compute bound where real syrk is bandwidth bound, and cuBLAS's cgemm is better at
compute. The route stays reachable as `BATCHLAS_SYRK_VARIANT=gram` so it stays measurable and the
conjugation stays under test (`syrk_custom_dispatch.hh:16-24`).

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

`wy_trmm_applicable` (`src/extensions/ormqr_blocked.cc:50`) is therefore per-**type**, not
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
* Transcribing the four ops' thresholds into `RouteTable::preferred` was killed by a **confirmed
  silent route change**: a transcribed `tiles_per_side >= 3` rule rejects the tile route for
  `129 <= n <= 383` at every batch, sending n = 256 to `DiagFullGemm`, which writes **both
  triangles** — a shape `tests/syrk_tests.cc` names explicitly, found independently by two judges.
  See [where the decision actually happens](#where-the-decision-actually-happens).

## Correctness findings

### The band-split syrk bug

The 128-wide Gram tile initially split each thread's 8 rows into two 4-wide bands 64 apart — what
the square 128x128 kernels do to spread banks. That is incompatible with taking the triangle at
*thread-tile* granularity, which decides a whole thread tile is inside the requested half from
its tile indices alone: thread (0,1) then owns element (64,4), lower triangle while its tile is
not, so **nothing wrote it**. Silent, and only at n > 64. It hid because `syrk_tests` pinned one
shape, n = 96, reaching only one of the kernel's three tile widths.
`SyrkTest.NarrowShapesMatchGemmReference` (`tests/syrk_tests.cc:113`) now sweeps
n in {24,32,48,64,96,128} x trans x uplo at k = 200 (not a multiple of the k chunk), and mirrors
both sides before comparing so an element left at its input value is caught.

### The herk test that could not fail

`HerkTest` checked that the unreferenced triangle stays untouched and that the two uplo runs
agree. **Neither can catch conjugating the wrong operand** — the likeliest defect in a shared
syrk/herk kernel: conjugating the row index instead of the column returns `conj(C)`, still
Hermitian and still consistent across both triangles. `HerkTest.MatchesGemmReference`
(`tests/herk_tests.cc:337`) compares against a GEMM and was confirmed to fail when the
conjugation is flipped.

### The trmm poison test

trmm carries a documented prior incident where the tempting 8x "fix" was the wrong-answer one and
the guarding test could not fail by construction: every other test built A with
`RandomTriangular` — already zeroed opposite the triangle, already ones on a unit diagonal — and
validated against a full gemm on that same A, which passes for an implementation ignoring `uplo`
and `diag` entirely. `TrmmTest.IgnoresUnreferencedTriangleAndUnitDiagonal`
(`tests/trmm_tests.cc:200`) poisons the forbidden storage and differences against a gemm on the
clean A over both sides, uplo, diag, three transposes, ragged and non-square shapes; on CUDA it
re-runs with `BATCHLAS_EXPAND_MAX_BYTES=0` to reach the no-scratch route no test shape otherwise
reaches. `uplo`/`side`/`diag` are in the coverage key for the same reason.

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
(`sytrd_blocked.cc:817-822`) because in double it **inverts**: the route falls to
`syr2k_vendor_impl`'s per-batch loop, 7.56 vs 58.52 ms at n2=256 ib=32 batch 1024, **7.7x
slower**.

### The coverage instrument itself

Three defects in the S0/S7 coverage tool surfaced only by *using* it, each looking healthy while
reporting almost nothing: the gate-declined half was unrecorded, so a shape moving *off* a native
kernel was invisible; `uplo`/`side`/`diag` were not in the key, so calls differing only in `uplo`
collapsed into one row; and `emit()` opened with `"w"`, so each of 53 test binaries truncated the
last one's output. `native_supported` is a **tri-state** because on a gate decline the caller
cannot tell "no native route serves this shape" from "one does, but the heuristic preferred the
vendor" (`level3_coverage.hh:47-61`).

### The her2k alpha alignment fault

`cublas?her2k` dispatches to cuBLASLt, which reads the host `alpha` with a 16-byte aligned vector
load; `std::complex<double>` is 8-byte aligned, so passing the parameter's address faults whenever
it lands 8 mod 16 — shape-dependent, so most calls survive. Reproducible against cuBLAS 13.2 with
none of BatchLAS present; fixed with `alignas(16) T alpha_aligned = alpha` (`cublas.cc:640`).

## Open debts

### Forced-route defects

1. **`BATCHLAS_SYRK_ROUTE=native` produces a wrong answer.** `{Native, Auto}` passes
   `syrk_use_cuda_custom`, then fails every arm inside `syrk_cuda_custom` (`gram` requires
   `origin == Auto`; the tile arm requires `algo == TriangularTiles || origin == Auto`) and lands
   on `syrk_cublasdx_fallback_gemm` at `syrk_custom_dispatch.cc:261-262` — the `DiagFullGemm`
   route, which **writes both triangles**. `WP1_LEVEL3_SPEC.md` describes this fall-through as
   landing "into raw cuBLAS"; after WP1 S2 the terminal is the public `gemm`, so the note's
   destination is stale, but the defect is unchanged and unfixed.
2. **`BATCHLAS_SYR2K_ROUTE=native` throws a cuBLASDx message it did not ask for**
   (`syr2k_custom_dispatch.cc:199-211`); the throw is not guarded by `forced`. Pre-existing,
   preserved exactly rather than quietly improved.

### Routing and reachability

3. **`symm` has no `expansion_fits` ceiling** where hemm, herk and her2k all have one:
   `symm_cublasdx_fallback_gemm` allocates the k x k x batch scratch unconditionally
   (`symm_custom_dispatch.cc:95-99`), so a large enough symm hits the 2^31-element SYCL range
   failure instead of falling back. Adding the check *is* a route change and needs measuring.
4. **`double` symm has no expansion route at all** — the facade gate is float-only and
   `symm_vendor` forwards to a per-batch `cublasDsymm` loop, while complex `hemm` and float
   `symm` both get the expansion. Pre-existing.
5. **Heterogeneous `symm` is unmeasured** — `symm_problem_supported` does not reject it, unlike
   syrk's and syr2k's, so after WP1 S2 its expanded GEMM reaches `gemm_heterogeneous_vendor_impl`
   rather than a strided-batched call on max dims. Probably a correctness *improvement*, untested.
6. **`trmm`'s tile kernel is `Side::Left` only** — the right-side branch still expands. syev uses
   Left only; `ormbr` has the same WY update, is not wired, and feeds gesvd.
7. **ROCm has no `symm`, `hemm`, `herk` or `her2k`** — `rocblas.cc` instantiates only gemm, gemv,
   trsm, syrk, syr2k, trmm (`entry_points/level3.cc:398-535`); wiring the trmm tile kernel there
   is where `wy_trmm_applicable` would be re-measured.
8. **`her2k_gemm_preferred` was swept over square rank-k shapes**, but `sytrd_blocked`'s panel
   loop issues a narrow one — k = nb in {16,24,32} against n2 up to 480 — where the GEMM is near
   bandwidth bound and the fold adds an `n2^2 * batch` write plus read the direct GEMMs never pay.
   Awaits an A/B at n2 in {224,480}, `complex<float>`; `complex<double>` is left out entirely,
   same route but 16 bytes per element halves the fit headroom, unmeasured.

### Unverified boundaries

`257 <= n <= 383` for syrk triangular tiles; batch 3 and `769 <= n <= 895` for herk; batch 1 with
`n >= 128` for her2k; symm's `squareish` exclusion; and the deliberate slack between the
symm/hemm measured loss region (`batch <= 2 && n <= 128`) and the shipped constants (4 and 256).

### Vendor-free gaps still open

Post-WP8 `NoRouteError` census over `ctest -LE slow`: `trmm` 16, `herk` 16, `syrk` 12, `her2k` 12,
`hemm` 12, `syr2k` 10, `symm` 8 — the double and complex arms trapped in `cublas.cc` plus the ops
with no native arm. The suite pass count cannot show movement here; read the per-op census.

### Instrumentation and harness

* `symm_benchmark`, `syrk_benchmark` and `syr2k_benchmark` **abort before printing anything** — a
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

## Raw evidence

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

# Dispatch: the vendor gate, the level-3 route arms and the coverage instrument (WP0, WP1)

> **Covers:** the vendor-availability gate, the coverage instrument and the buffer-size rule every
> op shares; the level-3 four's families, `can_run` terms, pins and tuned rows; the measured windows
> those rows transcribe; and the WP0/WP1 measurements and negative results behind them.
> **Status:** current. Sections that measured the route era (`Route`, `RouteTable`, the facade, the
> hand-written level-3 gates, cuBLASDx; all deleted) say so and give today's equivalent. How
> table-selected ops choose: @ref design_flat_selection and @ref selection; per op, which family
> ranks first where: @ref selection_tables.
> **Machine:** RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda, unless a section says otherwise.
> **Measured:** WP0/WP1 (before 2026-09-30); the level-3 tables were transcribed on 2026-10-06.

The durable record of two work packages: **WP0**, which moved the public op definitions out of the vendor
translation units and built the coverage instrument, and **WP1**, which freed the four level-3 tile dispatchers
(`symm`/`syrk`/`syr2k`/`trmm`) from the cuBLAS object library.

**Before flat selection** WP0's vocabulary decided every op: a `Route{Origin, Algorithm}` pair per arm, a
`RouteTable` resolver per op (a support predicate, a vendor-present window and a vendor-free tier hook), legacy
`BATCHLAS_<OP>_VARIANT`/`_PROVIDER` spellings, and, for the level-3 four, hand-written gates in
`src/backends/*_custom_dispatch.cc` with cuBLASDx fused arms. Phase 5 of
[flat kernel selection](../design/flat-kernel-selection.md) deleted `include/batchlas/blas/dispatch/`, and the
level-3 wave (#147) deleted the gates and cuBLASDx. Every op, the level-3 four included, now chooses from tuned tables
(`src/ops/<op>/{choice.hh,<op>.cc}`, `tuned/<op>.<dtype>.<device>.txt`). Line citations of deleted files on this page
refer to `ff340fc6` (`git show ff340fc6:<path>`), the last commit that had them.

The shipped code is the authority on **what** ships; the exploration notes are the authority on **why**. Ops covered:
`symm`, `syrk`, `syr2k`, `trmm` (their families, `can_run` terms and the measured windows behind their tables), plus
the vendor gate and coverage instrument every op uses. The same four ops' kernel design and per-dtype tables live in
[`level3.md`](level3.md); this page is the selection half. Measured windows for `gemm`, `trsm`, `potrf`,
`geqrf`/`orgqr`, `getrf`/`getrs`/`getri`, `gemv` and `spmm` live on their own pages.

## What ships

### The vendor-availability gate

`src/select/vendor.hh` asks per **library**, not per device family, because the map is not uniform: on NVIDIA
`getrf`/`getri` come from cuBLAS while `potrf`/`syev` come from cuSOLVER, on AMD all from rocSOLVER
(`vendor.hh:23-27`):

```cpp
template <Backend B>  // gemm gemv trsm trmm symm syrk syr2k hemm herk her2k
inline constexpr bool level3_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCBLAS) :
    B == Backend::NETLIB ? kHasNetlib : false;
```

with `factorization_`, `solver_` and `sparse_` siblings, and `kHasNetlib = BATCHLAS_HAS_LAPACKE &&
BATCHLAS_HAS_CBLAS`. Each op's `OpSpec` names its library (`select::Lib::level3` for the level-3 four), selection
turns it into `select::Device::has_vendor`, which every `vendor` family's `can_run` reads, and the `vendor` launch
compiles the vendor call only under `if constexpr (select::has_library<B>(spec.vendor))`. When nothing serves a
call, `select::throw_no_vendor_route<T>` records a coverage miss and throws `batchlas::NoRouteError`
(`include/batchlas/no_route.hh`), whose message names op, scalar type and the switch that would restore it.

The spec's `src/dispatch/absent/*.cc` stub design was **declined**: it restates all 26 vendor signatures a second
time, and S5's two real bugs were signature divergence between restated copies. The shipped gate is an `if constexpr`
at the call site, so the vendor call is not compiled at all when the library is absent. The "is the kernel linked"
predicate four `src/extensions/` sites previously spelled `B == Backend::CUDA` is
`select::level3_tile_route_available<B, T> = B == Backend::CUDA && (std::is_same_v<T, float> ||
bool(BATCHLAS_HAS_CUBLAS))` (`vendor.hh:94-96`); its users are `ormqr_blocked.cc`, `ortho.cc` (twice),
`sytrd_blocked.cc` and the static coverage table. It deliberately keeps its pre-flat-selection value although double
syrk `gram`, symm `expand` and every trmm family now run vendor-free: widening it would move ortho's and ormqr's
vendor-free routes.

### Level-3 route arms

symm, syrk, syr2k and trmm select like every other op since the level-3 flat-selection wave (#147):
`src/ops/<op>/choice.hh` lists the families, `src/ops/<op>/<op>.cc` holds `key_of`, `can_run`, the launch and the
workspace, and `select::run` takes the first runnable entry of the nearest row of `tuned/<op>.<dtype>.<device>.txt`
(design record: `docs/design/flat-kernel-selection.md` §12 "Level-3 four"; per-op ranking on @ref selection_tables).
Every family is fieldless: the old routers chose no knob, and the derived ones (gram's tile width, the triangular
kernels' aligned/predicated leg, `trmm_row_tile`) stay derived. hemm, herk and her2k still choose by hand in
`cublas.cc` (vendor builds only), between an expansion or fold into the public `gemm` and the vendor loop, and have
no `BATCHLAS_<OP>_ROUTE`.

| op | candidates, tie-break order | keys | last resort |
|---|---|---|---|
| `symm` | `expand` (mirror A's triangle into scratch, then the public `gemm`), `vendor`; float and double | `form:exact m:log n:log batch:log` | `expand, vendor` |
| `syrk` | float `gram, triangular, vendor`; double `gram, vendor` | `form:exact trans:exact n:log:2 k:log batch:log` | `triangular, gram, vendor` |
| `syr2k` | float `triangular, vendor`; double `vendor` | `n:log:2 k:log batch:log` | `triangular, vendor` |
| `trmm` | `triangular` (`Side::Left` tile kernel), `expand` (expand the triangle, then the public `gemm`), `vendor` (the `cublas?trmm` loop); all four types | `side:exact order:log:2 q:log batch:log` | `expand, triangular, vendor` |

`form` is `sq` when `2*min >= max` over C's `m, n` (symm) or over `n, k` (syrk), else `tall` or `wide`; it lines the
old aspect-ratio test up with a table axis. The `expand` families' inner GEMMs go through the public `gemm`, so they
take the gemm table's choice like any other caller.

**`can_run`, correctness only.** Every native family needs `B == Backend::CUDA` (the old reach: ROCm and the host
stay vendor), a GPU queue, a homogeneous batch (every tile kernel indexes operands as `base + batch * stride`),
extents >= 1 and `batch <= 65535` (the batch is grid z, or y for gram; 65536 throws `Number of work-groups exceed
limit`, confirmed by launch). Per family:

| op | family | further terms | where |
|---|---|---|---|
| `symm` | `expand` | `max_wg >= 256` (expand_mirrored's 8 x 32 group); `expansion_fits(q, k, batch, bytes)` (the int-linearised grid and the `BATCHLAS_EXPAND_MAX_BYTES` / quarter-of-memory budget) | `src/ops/symm/symm.cc:44-56` |
| `syrk` | `gram` | `syrk_gram_supported` (square C with `n <= 128`: the tile is all of C, and past it the kernel answers wrongly rather than throwing); `max_wg >= gram_threads(n)`; the SLM tile fits | `src/ops/syrk/syrk.cc:57-60` |
| `syrk` | `triangular` | float only; `max_wg >= 256`; `T(T+1)/2 <= 65535` tiles, `T = ceil(n/128)` (grid y; n = 46208 runs, 46209 throws) | `src/ops/syrk/syrk.cc:61-64` |
| `syr2k` | `triangular` | float only; `max_wg >= 256`; `transA != ConjTrans` (real ConjTrans stays on the vendor, as before); the same tile ceiling | `src/ops/syr2k/syr2k.cc:52-56` |
| `trmm` | `triangular` | `trmm_tiles_supported` (`Side::Left`); `ceil(m/tile_m) * ceil(q/128) <= 65535` tiles in grid y | `src/ops/trmm/trmm.cc:59-62` |
| `trmm` | `expand` | `max_wg >= 256`; `expansion_fits` | `src/ops/trmm/trmm.cc:63-68` |
| all four | `vendor` | the level-3 library is linked, **and a homogeneous batch**: every vendor loop (cuBLAS, rocBLAS, netlib) runs each item at the top-level extents, a wrong answer for a heterogeneous batch | `symm.cc:57`, `syrk.cc:65`, `syr2k.cc:57`, `trmm.cc:69` |

syrk is real-only, so `ConjTrans` is passed to its kernels as `Trans` and is not a `can_run` term; the exact `trans`
key keeps real `ConjTrans` on the vendor where the old rule put it.

**What the tables rank.** All level-3 tables are **transcribed, untimed** (`source=transcribed:ff340fc6`, entries
`<spelling> -`), identical for sm_89 and sm_120 because the old rules read no architecture, and ship for float and
double only for the three real-only ops (`tuned/README.md`). Each row is the old vendor-present Auto choice, then
every other candidate that is structurally runnable at the cell, `vendor` last unless it was the Auto choice, so a
vendor-free build takes a native kernel from the row. Read off `tuned/<op>.float.sm_89.txt`:

| op | rows that rank a native family first (float) | double |
|---|---|---|
| `symm` | `form=sq` rows with `batch >= 4` or `max(m, n) >= 256` rank `expand` first (e.g. `m=128 n=255` `vendor` at batch 3, `expand` at 4; `m=128 n=256` `expand` from batch 1); every reachable `tall`/`wide` row ranks `vendor` first (the
48 `tall`/`wide` rows at `m, n <= 2` rank `expand` at `batch >= 4`, but no call has that form there, since
`2*min >= max` holds) | `vendor` first everywhere |
| `syrk` | `n <= 128`: `gram`; above, `triangular` on every `form=sq` row, and on `tall`/`wide` rows only where `n >= 257`, `k >= 8` and `batch * T(T+1)/2 >= 160` (e.g. `tall n=512 k=64` `vendor` to batch 15, `triangular` from 16); `trans=C` rows `vendor` first | `gram` to n = 128, `vendor` above |
| `syr2k` | `batch >= 2`: `triangular`; `batch = 1`: `vendor` | `vendor` (the only candidate) |
| `trmm` | `side=L`: `triangular`; `side=R`: `expand` (all four types) | same |

The off-grid gate agreed with the old vendor-present choice on 100.00% of 2500-3000 random points per (op, dtype,
device) (§12, sm_120 box). The measurements behind each of these edges are in [Measured boundaries](#measured-boundaries).

**Before flat selection** the same decisions were hand-written gates in `src/backends/{symm,syrk,syr2k,trmm}_custom_dispatch.cc`,
float and CUDA only (other dtypes went straight to cuBLAS): symm `squareish && expansion_preferred(max_dim, batch)`;
syrk `prefer_gram || prefer_triangular || cublasdx_heuristic`; syr2k `batch >= 2`; trmm `trmm_triangular_supported(...)`
with no size threshold, Right going to a vendor arm that meant "expand + gemm when the scratch fits, else the loop".
Where the old Auto took a native kernel past batch 65535 it aborted; it now takes the vendor.

### The level-3 pin words

`BATCHLAS_<OP>_ROUTE` for these four ops is parsed by `src/select` like every other op's: `auto`,
`native`, `vendor` (class words: `native` takes the row's first runnable non-vendor entry, `vendor`
the vendor; with nothing of their class runnable they fall back to Auto with a warning, so `vendor`
in a vendor-free build means Auto) or a family spelling, case-folded and trimmed. A spelling that does
not parse, is not a compiled candidate for the dtype, or whose `can_run` refuses the call throws
`std::invalid_argument` ("cannot run this shape"), before anything is written
(`<Op>Candidates.UnknownPinsThrow`, `<Op>Candidates.CanRunFalsePinsThrow`,
`<Op>Candidates.ClassWordsAndSpellings` in `tests/{symm,syrk,syr2k,trmm}_candidates_tests.cc`). The value
comes from the settings snapshot, so a raw `setenv` mid-process is not seen until `detail::reload_settings()`.

| op | spellings | notes |
|---|---|---|
| `symm` | `expand`, `vendor` | `native` = `expand`; `expand` throws on a heterogeneous operand or when `expansion_fits` fails (e.g. `BATCHLAS_EXPAND_MAX_BYTES=0`) |
| `syrk` | `gram`, `triangular`, `vendor` | `triangular` on double throws (not a double candidate); `gram` at n > 128 throws (the kernel answers wrongly there; it used to run) |
| `syr2k` | `triangular`, `vendor` | `triangular` on double, or with real `ConjTrans`, throws |
| `trmm` | `triangular`, `expand`, `vendor` | `triangular` with `Side::Right` throws; `vendor` is the `cublas?trmm` loop. **Behaviour change:** the old `vendor` meant expand+gemm when the scratch fit; that is now `expand` |

Every word that is not one of the op's own spellings throws, including another level-3 op's spelling (`expand`
for syrk, `gram` for trmm), the removed `cublasdx` (cuBLASDx was deleted; it never ran here, MathDx being absent), the
legacy `tiles`/`narrow`/`gemm`/`custom`/`dx`/`fused` words, the old `DiagFullGemm` measurement route
(`diag_full_gemm`), and the `origin:algorithm` forms `native:auto` and `vendor:auto` (`SyrkCudaCustomTest.RemovedRouteWordsThrow`,
`SymmCudaCustomTest.RemovedRouteWordsThrow`, `TrmmCandidates.UnknownPinsThrow`). The old gates were float-only and
ignored every word for the other dtypes; a pin now applies, and is checked, for every dtype.

Two neighbouring variables are not pins of these four ops. `herk` reaches syrk's conjugating Gram kernel only when
`BATCHLAS_SYRK_ROUTE` is exactly `gram` (`ops::syrk::herk_gram_pinned()` in `src/ops/syrk/choice.hh`), and no
longer throws on a syrk word it does not understand. `BATCHLAS_EXPAND_ROUTE=expand|loop` is read by hemm, herk and
her2k only (symm stopped reading it in #147).

### The environment vocabulary

History. `src/backends/route_common.hh` was carved out as the backend-neutral half of the old
`cublasdx_dispatch_common.hh`, because that header included `<cuda_runtime_api.h>` for `cudaStream_t`, which made
every consumer (including `triangular_expand.hh` and the then hand-written symm/syrk/syr2k/trmm selectors)
CUDA-only. Before the pin words, the portable half carried `parse_cublasdx_variant_request`, which turned a
`BATCHLAS_<OP>_VARIANT` string into one of three per-op enum values and was the last of the five non-communicating
environment mechanisms the WP0 plan named (an **unset** variable meant Auto there but Vendor for GEMM). WP2 E6
removed that asymmetry, flat selection phase 5 removed the `_VARIANT` spellings and the route vocabulary, and
`Level3Pin` replaced it until the level-3 flat-selection wave deleted it with cuBLASDx and
`cublasdx_dispatch_common.hh`. Today `route_common.hh` holds only `ceil_div` and `is_gpu_queue`, and the level-3
pins are parsed by `src/select` like every other op's ([the level-3 pin words](#the-level-3-pin-words)).

## Measured boundaries

These measurements are the evidence behind the level-3 tables. The tables transcribe the old hand-written gates
cell by cell (see [What the tables rank](#level-3-route-arms)), so each boundary below is a row edge in
`tuned/<op>.float.sm_89.txt` today, and each window is still a route-era decision that nobody has re-timed: the
phase-4 retune would replace the transcribed rows with timed ones. Where a gate's source is quoted below it is at
`ff340fc6` (`git show ff340fc6:src/backends/<file>`). The kernels' own design and the per-dtype kernel tables are
owned by [`level3.md`](level3.md); where a figure appears on both pages, level3.md is the newer record.

All figures RTX 4090 / sm_89, CUDA 13.2, `RelWithDebInfo`, one dedicated GPU via `experiments/gpu_guard.sh`. Batch is
always large enough to saturate; batch = 1 is not a design target.

### Expansion crossover

The rule: expand when `batch >= 4 || max_dim >= 256` (`kExpandMinBatch`/`kExpandMinDim`
in `src/backends/triangular_expand.hh`). It survives as code only in `expansion_preferred` there, which
decides **hemm** and reads `BATCHLAS_EXPAND_ROUTE=expand|loop` **first**, so a pin overrides the window there. For
**symm** the same rule is data: the `form=sq` rows of `tuned/symm.float.sm_89.txt` rank `expand` first exactly where
it holds (the old gate also required the squareish shape), and symm no longer reads `BATCHLAS_EXPAND_ROUTE`.
Measured against a per-batch loop over the vendor's own triangular primitive — float `symm` over n 16..2048 × batch
1..512, complex64 `hemm` over n 16..512 × batch 1..16 — the expansion wins **1.2x to 72x** everywhere except the
bracketing region **batch ≤ 2 with n ≤ 128**, where it loses by **up to 2.5x**.

> The exploration notes (`VENDOR_INDEPENDENCE_PLAN.md` §WP1) quote the *loss* region, 2 and 128; the shipped constants
> are **4 and 256**, i.e. the guard is the complement over a wider region. `WP1_LEVEL3_SPEC.md` correction 4 records
> this. Consequence: batch = 3, or `128 < n < 256` at batch ≤ 3, is refused the expansion with **no bracketing
> measurement** — unverified, conservative in direction. The symm table carries the same refusal (`form=sq m=255
> n=255` ranks `vendor` first at batch 3, `expand` at 4).

`trmm` never consulted this: `cublas?trmm` has a flat ~110 µs floor whatever the shape, so the expansion beat it in
every cell measured, batch 1 included. Its table ranks `expand` first on every `side=R` row and `triangular` (then
`expand`) on every `side=L` row.

### `syrk` tile boundaries

The old triangular gate, `syrk_prefer_triangular_tiles` at `ff340fc6`, with `kTriangularTile = 128` and
`kTriangularTileK = 8`: at least 3 tiles a side (`n >= 257`), `k >= 8`, and `batch * T(T+1)/2 >= 160` with
`T = ceil(n/128)`. Today those are the `tall`/`wide` row edges of `tuned/syrk.float.sm_89.txt`: `n=256` rows rank
`vendor` first at every batch, `n=257` rows rank `triangular` first from batch 27 (6 tiles), `k=7` rows `vendor`,
`k=8` rows `triangular` (at batch 1024, n = 512), and `n=512 k=64` rows switch at batch 16 (10 tiles, 160 blocks).

| boundary | winner side | bracketing non-winner |
|---|---|---|
| tile grid ≥ 3 a side | 1.45x at n=512 batch 512; 1.63x at n=1024 batch 64; 1.71x at n=2048 batch 16 | n=256 measured **0.84x–1.22x** depending on where its grid fell against a wave boundary — no win at all |
| batch × tile count ≥ 160 | won from 168 blocks up | **1.14x slower at 144 blocks, 1.25x slower at 136** |

> `n >= 257` is the edge the rows carry, but the measurement supporting it says "from n = 384 up every saturated shape
> won" (`experiments/GEMM_TO_LEVEL3_SURVEY.md` says the router "needs n >= ~384"). The band **257 ≤ n ≤ 383 ranks
> `triangular` first in the tables and has no bracketing cell.** Worse than unmeasured: the kernel's unpredicated fast
> path additionally requires `n % 128 == 0`, `k % 8 == 0` and 4-element alignment on both operands
> (`src/backends/syrk_triangular_tiles.hh:36-44`), and 384 is the first multiple of 128 at or above 257 — so *every*
> shape in that band runs on the slower predicated path, which is the one the sweep never sampled. On `tall`/`wide`
> rows n = 256 (`tiles_per_side == 2`) ranks `vendor` first, and is a recorded 0.89x pre-existing loss on the tile
> route.

The squareish third disjunct (below) puts `triangular` first on every `form=sq` row above n = 128, including n = 256
and the 129..256 band the tile-count rule refuses.

The Gram gate was `C.rows() <= kGramMaxTile`, `kGramMaxTile = 128` (`src/backends/syrk_gram_tiles.hh:28`); today it is
both a `can_run` term (`syrk_gram_supported`, because past 128 the kernel answers wrongly) and the row edge: every
row with `n <= 128` ranks `gram` first, float and double. The single-tile Gram kernel serves exactly the range the
triangular grid cannot, and inside it the alternative is a host loop over `cublasSsyrk`, one to two orders of
magnitude off anything batched. Float, against the GEMM spelling (`experiments/TRMM_SYRK_BATCHED_KERNELS.md`):

| reduction m | n | batch | gemm (ms) | syrk before (ms) | syrk now (ms) | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.334 | 33.588 | 0.0780 | 4.29x | 431x |
| 512 | 64 | 1024 | 0.345 | 31.911 | 0.1875 | 1.84x | 170x |
| 1024 | 128 | 512 | 0.409 | 30.296 | 0.4146 | **0.99x** | 73x |
| 2048 | 128 | 256 | 0.404 | 29.370 | 0.3820 | 1.06x | 77x |

n = 128 is the bracketing cell: parity, and near the ceiling — reading A once costs 298 µs against the GEMM's measured
409, so the whole prize there was 1.37x, not 2x. At n = 32 the kernel reads 71 MB in 78 µs, 933 GB/s, i.e. at the
memory roofline. In `double` the win *grows* with n (FP64 at 1/64 rate is compute bound): 1.08x, 1.78x and 2.10x at
m/n/batch 256/32/2048, 512/64/1024 and 1024/128/512 — against 138x, 58x and 17x for the host loop (the double
table ranks `gram` first at `n <= 128` too). The old gate's third disjunct, `syrk_prefer_cuda_custom_heuristic` at
`ff340fc6` (a cuBLASDx heuristic), required `n >= 16`, aspect ratio `min_dim * 2 >= max_dim` over `(n, k)`, and
`tiled_work = batch × ⌈n/32⌉² × ⌈k/32⌉ >= 8` over a 32-wide tile. Because `tiled_work` carries the batch factor, at
any batch ≥ 8 that term is satisfied by every shape, so **in the regime this campaign tunes for the disjunct reduces to
`n >= 16 && min_dim * 2 >= max_dim`**. With MathDx absent its only effect was to admit shapes to the tile kernels, and
that effect is what the `form` key now carries: every `form=sq` row above n = 128 ranks `triangular` first. **Its
own crossover has no bracketing grid here** — unverified, and now a table edge rather than code.

### `syr2k` batch boundary

The old gate was `A.batch_size() >= 2` (`syr2k_prefer_triangular_tiles` at `ff340fc6`); today every
`tuned/syr2k.float.sm_89.txt` row with `batch >= 2` ranks `triangular` first and every `batch=1` row `vendor`, at
every n and k (e.g. `n=64 k=64`: `vendor` at batch 1, `triangular` at 2). Measured float over n 8..3072 × k 4..2048 ×
batch 1..1024:

* **from batch 2** the kernel won every shape in the grid: 1.06x at n=3072, 1.12x at n=1024, 1.3–1.4x through the
  middle, up to 226x where n is small enough that the whole cost is the launch;
* **batch 1 is the bracketing cell and does not sort by anything**: the vendor wins 1.18–1.60x below n=1280 and again
  1.16x at n=3072; the kernel wins 1.02–1.71x between; the vendor wins 4–10x on a deep k with a small n. No threshold
  in n exists, so batch 1 keeps the vendor.

Neither n nor k nor the tile count entered the gate, because none of them changes which side of the per-launch
difference a shape falls on; the table keys on them only because the transcription grid spans them (the tile count
is a `can_run` term now, as a launch limit, not a speed edge). double has no native family and ranks `vendor` alone.

### `trmm`: no threshold

There is deliberately no size threshold, and the table keeps it so: every `side=L` row of `tuned/trmm.*.sm_89.txt`
ranks `triangular` first at every order, `q` and batch, in all four types. The first router gated the tile kernel to
`m <= 64 || m >= 512`, read off a trmm-vs-<b>gemm</b> column — the wrong comparison, because the router chooses between
the tile kernel and the **vendor**. That gate sent m = 128..256 back to the expansion and cost **up to 1.29x on
exactly the shapes it was meant to protect**.

Float, tile against vendor, at saturating batch:

| m | nC | batch | tile (ms) | vendor (ms) |
|---|---|---|---|---|
| 128 | 512 | 1024 | 0.698 | 0.784 |
| 128 | 1024 | 512 | 0.686 | 0.687 |
| 256 | 256 | 512 | 0.536 | 0.692 |
| 256 | 1024 | 256 | **0.915** | **0.855** |

The last row is the single 7% loss and the bracketing cell; it is not worth a special case that would have to be
re-tuned every time either route changes. Against the GEMM spelling, by type, `trmm` wins everywhere in `double`
(1.29x–2.02x) and `complex<double>` (1.05x–1.95x), at 8 of 10 float shapes, and **loses below m = 512 in
`complex<float>`** (0.69x–0.93x) — a register-file ceiling, not a tuning miss: cuBLAS's cgemm runs at ~100% of FP32
FMA peak on these shapes while a complex accumulator costs twice the registers. The transcribed cfloat table still
ranks `triangular` first on every `side=L` row, as the old rule did: that GEMM comparison was against a plain gemm
spelling, not against the `expand` family (expansion + the public `gemm`), and `triangular` vs `expand` for cfloat
below m = 512 has not been timed. It is a cell for the phase-4 retune, not a measured loss of the shipped choice.

### `herk` and `her2k` crossovers

Not moved by WP1 or by flat selection — herk, her2k and hemm still choose by hand in the cuBLAS-gated TU and have no
tables or `BATCHLAS_<OP>_ROUTE` — but they are the same expansion decision and the constants are easy to confuse
with the ones above.

| predicate | shipped condition | evidence | bracketing non-winner |
|---|---|---|---|
| `herk_gemm_preferred` (`src/backends/cublas.cc`) | `batch >= 4 && n <= 768` | complex64, n 32..1024 × batch 1..256: 1.6x–72x for batch ≥ 4 at n ≤ 512 | a wash at n = 640..768; **0.82x–0.93x from n = 896 up**; batch ≤ 2 a wash or loss at every n |
| `her2k_gemm_preferred` (`src/expansion_budget.hh`) | `batch >= 2 \|\| n >= 128` | 1.4x–128x everywhere else | batch 1 at n ≤ 64: **0.74x at n = 32, 0.89x at n = 64** |

`herk`'s is a conjunction with a large-n ceiling because its GEMM computes both triangles and keeps one; the mirrored
expansion's is a disjunction with no ceiling, because expanding costs one bandwidth-bound kernel and then does exactly
the vendor's work (the comment above `herk_gemm_preferred`). Both predicates check `BATCHLAS_EXPAND_ROUTE` **before** their window
(`expansion_route_pin` in `src/expansion_budget.hh`, read by `rankk_route_pin` in `cublas.cc`), so a pin overrides the
measurement — which is exactly the seam the `sytrd_blocked` half-guard bug below fell through. Each also needs
`expansion_fits` for its scratch, else it takes the vendor loop.

**An open A/B, not a settled window.** `her2k_gemm_preferred` was swept over *square* rank-k shapes, but the
`sytrd_blocked` panel loop issues narrow ones — `k = ib = nb ∈ {16,24,32}` against `n2` up to 480 — where the GEMM is
near bandwidth-bound and the fold adds an `n2²·batch` write plus read the two direct GEMMs never pay. The halved
arithmetic may not survive that. `complex<double>` is deliberately excluded from that route for the same reason: it
would reach the same fast path, but its scratch is 16 bytes per element and none of it has been measured
(`src/extensions/sytrd_blocked.cc:777-782`; the her2k call and its fit check are at `:830-845`). Guessing is how the
7.8x double inversion below got written down in the first place.

## Negative results

Built, measured, rejected. These cost as much to establish as the wins. Items 1, 8 and 9 concern the
deleted route layer (`RouteTable`, `route_compiled.hh`, `resolve_route`) and are kept as history; the
lesson of each carried over to flat selection.

1. **The `split-tu` WP1 design** — split each level-3 TU into portable and CUDA halves, transcribing the gate
   thresholds into `RouteTable::preferred`. Killed by a *confirmed silent route change*: the live thresholds were
   **gate-only** (`syrk_cuda_custom`'s Auto arm took `syrk_triangular_tiles` unconditionally once the gram test
   failed, with no second preference check), so a transcribed `>= 3` rule rejected the tile route for **129 ≤ n ≤ 383
   at every batch**, sending n = 256 to a route that wrote both triangles. Two judges found it independently; it scored
   3/4/6 against the shipped `retarget-only` design's 9/7/8. The lesson is why the flat-selection transcription
   evaluated byte-for-byte copies of the **whole** old gate per grid cell (`--fidelity` gates against
   `git show ff340fc6:`), added the `form` key so the squareish disjunct is an axis, and was accepted only on a 100.00%
   off-grid agreement gate ([flat-kernel-selection.md](../design/flat-kernel-selection.md), §12).
2. **Making the sideways vendor fallback the public entry point.** Every `*_vendor_cuda_raw` site was reached *after*
   a gate that already returned true, so a public call from there re-entered the same gate with the same environment
   and views: unbounded recursion, reachable with a forced fused pin on a CPU queue. The route-era fix was a dedicated
   seam (`level3_vendor_fallback.hh`, deleted). Under flat selection each `vendor` family calls
   `backend::<op>_vendor<B, T>` directly, never the public entry, so the recursion cannot happen.
3. **`syrk`/`herk` for `ortho`'s Gram matrix, pre-kernel.** 73x–96x **slower** at the shapes `ortho` actually issues
   (m 256..2048, k 32..128, batch 256..2048), because k < 384 failed both router disjuncts and dropped to one
   `cublasSsyrk` launch per batch member. The winning column was k ≥ 512 and square-ish (1.15x–1.67x), where `ortho`'s
   callers do not live — `syevx_lobpcg`, `syevx_filtered` and `lanczos` all pass k = the block size. Reversed only
   after the Gram-tile kernel was written; see [syrk-tile-boundaries](#syrk-tile-boundaries).
4. **`trmm` for the WY block factor, pre-kernel.** Lost at every shape (0.195 → 0.238 ms at ib=32/nC=256/batch 2048,
   up to 0.779 → 1.152 at ib=256). Structural: `src/extensions/trmm.cc` recurses only to `n <= 256` and then calls the
   GEMM it was meant to replace, so for every `ib ∈ {16,32,64,128,256}` the triangular structure was never exploited.
   Re-measured after the tile kernel: `float` 1.006x–1.046x and `double` 1.004x–1.016x win everywhere,
   `complex<float>` **loses everywhere** (0.944x–0.995x), `complex<double>` loses at ib = 16 (0.958x–1.010x), netlib
   float/double 0.336x/0.379x at n = 128, ib = 16 — so only the double half of the gate was stale, and the split is
   **per type, not per precision**. A 16-row tile was then built to give ib = 32 a real R saving: it confirmed the
   `(R+1)/2R` argument (`complex<double>`'s ib = 16 hole closed 0.958x → 0.996x) but only to *parity*,
   `complex<float>` stayed 2–5% behind, and it is kept solely because it takes `double` to 1.013x–1.036x. Complex
   still takes the GEMM.
5. **Complex Gram tiles (`herk`).** Loses to the existing GEMM-plus-Hermitian-fold at every Gram shape: 0.217 vs 0.206
   ms at n=32/batch 2048; 2.08 vs 1.57 at n=128/batch 512. A complex multiply is four real ones, so herk is compute
   bound where real syrk is bandwidth bound. `herk` keeps its GEMM-plus-fold; the conjugating path stays reachable,
   only when `BATCHLAS_SYRK_ROUTE` is exactly `gram` (`ops::syrk::herk_gram_pinned()`), so it stays measurable and
   tested.
6. **`syr2k` for the `sytrd_blocked` trailing update, in `double`.** 7.7x and 7.4x slower at n2=256/batch 1024, 1.9x
   slower at n2=512/batch 512; 1.55x *faster* only at n2=2048/batch 32. Double wins only where the batch is small
   enough that per-item launch cost amortises — the opposite of the regime that matters. The tile kernel stays CUDA +
   float: `triangular` is a float-only syr2k candidate, and `sytrd_blocked` issues the rank-2k update only for float
   and `complex<float>` (`src/extensions/sytrd_blocked.cc:777-789`).
   (In float it is a **1.25x–1.66x** win on the update — the low cell is n2=1024/ib=64 — and 1.07x–1.27x end to end, and
it shipped. The update figure only became that good after a second correction: the first cut handed the other triangle
back with a bandwidth-bound `n²` symmetrize pass that **ate over half the win**, dropping 3.4–3.6x on the `syr2k`
itself to 1.25–1.66x. Reading every consumer showed that *nothing* in the `sytrd_blocked` pipeline reads `A`'s upper
triangle — all three `latrd_lower_panel` variants split at `c == r`, the fused trailing update skips `r < c`, and
`restore_tridiag_lower` only writes the superdiagonal — so the pass was removed. The GEMM pair had been leaving a
valid upper triangle behind as a side effect that nothing depended on and no one had written down: exactly the kind of
contract that is invisible until an op that respects the triangle replaces one that does not.)
7. **A false win nearly reported.** `syr2k` at n = 1024 looked 10.9% faster after WP1 S2. Repeating the prior step at
   that shape gave a 5.65–6.40 ms spread: the "win" was noise. The flattering direction needs the same scepticism as
   the alarming one.
8. **`route_compiled.hh`'s own prediction.** It said that once WP1 freed the four TUs the flag "becomes true for every
   backend — and that is the only edit needed here". Wrong in two directions: too wide in **type** (only float moved;
   `syrk`'s non-float gram branch and `trmm`'s non-float tile branch stayed in `cublas.cc`, and `syr2k` has no
   non-float tile route at all) and too wide in **backend** (the facade gate was guarded on `Backend::CUDA`). It took a
   scalar parameter instead, and survives as `select::level3_tile_route_available<B, T>` (`src/select/vendor.hh:94-96`).
9. **A compile-time coverage gate.** `resolve_route` was an inline function template, so every TU instantiated its own
   weak copy, and ELF resolves the executable's weak symbols ahead of a shared library's. A test compiled without the
   macro interposed its uninstrumented copy over the library's instrumented one; the run produced a coverage file with
   a correct header and **zero `reached` rows**. The gate is now a runtime bool in exactly one TU
   (`src/select/coverage.cc`), and `cmake/BatchLASOptions.cmake:141` records that the option was deliberately never added.

## Correctness findings

Wrong answers found, how they hid, and what guards them now.

* **`ormqr`'s buffer size and call disagreed by 108x.** `cta`, `two_stage` and `jacobi` all parsed but matched no
  branch, so `ormqr_dispatch` ran on the vendor while `ormqr_buffer_size` returned the *blocked* size — 2560 bytes
  against the 276480 the call then demanded, so sizing a workspace with the public API and passing it to the public
  call threw deterministically on every GPU type. Structurally prevented now: an op and its `*_buffer_size` query
  call the same `choose()` (flat selection, rule R5).
* **`{Vendor, FusedDevice}` satisfied `is_vendor` but was not "the plain vendor call"** (route era). The level-3
  dispatchers' `request == Vendor` tests meant `cublasSsyrk` specifically; rendering them as `is_vendor()` made a
  forced cuBLASDx request answer yes to "did the caller ask for the vendor?". The route-era `Level3Pin` words kept
  `vendor` and `cublasdx` apart. Today the question cannot arise: `vendor` is one fieldless family per op, the only
  one that calls the vendor library, and cuBLASDx is deleted.
* **The order-walk fallback inverted GEMM's default.** Taking "the first merely supported route" picks Native, because
  the orders list natives first — moving an 8×8×8 batch-1 GEMM from vendor to native. Guarded by
  `tests/route_gemm_equivalence_tests.cc`, whose `ReplicaIsFaithful` case pinned the transcription itself, until P3.4
  deleted it with `route_gemm.hh`. gemm has no order walk now; its sm_89 tables are the old decision transcribed per
  grid cell, and the last resort (`direct`, then `vendor`) applies only when no table entry can run.
* **Two ROCm defects invisible to the CUDA build.** `scripts/rocm_syntax_check.sh` (the ROCm headers live under
  `/opt/rocm/include/roc*/roc*.h`, a subdirectory, which is why a naive probe reads them as absent) caught a `trsm`
  instantiation left in the old parameter order and four orphaned macro-continuation lines. Its gate is "exactly one
  expected error" — a `get_native<ext_oneapi_hip>` overload this CUDA-only DPC++ lacks — so any other diagnostic is a
  real defect.
* **Signature divergence between vendor TUs.** `trsm`'s vendor form takes `alpha` last while the public form takes it
  third; `symm`/`syrk`/`syr2k` were `RealScalar`-constrained everywhere except cuBLAS. Generating facade bodies from
  the public declarations would have passed `alpha` where `side` was expected on **every** backend. Bodies were lifted
  verbatim instead (into `level3_vendor_fallback.hh`, since deleted); today each `vendor` family's launch spells its
  `backend::<op>_vendor<B, T>(...)` call by hand in `src/ops/<op>/<op>.cc`, so the hazard is one call site per op.
* **An instantiation binds as hard as a definition.** `syev` and `ormqr` were already defined in headers, but their
  *instantiations* lived in `cusolver.cc`/`cublas.cc`, which is enough to make them vanish from a build without those
  libraries. Verified by symbol, not diff: `scripts/facade_symbol_check.sh` asserts each public op is **absent** from
  the cuBLAS component and **present** in the facade, matching Itanium mangling directly because `nm -C` silently
  fails to demangle concept-constrained templates and would report the constrained ops as missing when present.
* **The SYRK 128-wide tile wrote nothing to some elements.** The first cut split each thread's 8 rows into two 4-wide
  bands 64 apart — what the square kernels do to spread shared-memory banks — which is incompatible with taking the
  triangle at *thread-tile* granularity: thread (0,1) then owned element (64,4), inside the lower triangle while its
  tile was not. It failed quietly and only at n > 64, and `syrk_tests` pinned exactly one shape, n = 96, which caught
  it **by luck** while reaching only one of three tile widths. `SyrkTest.NarrowShapesMatchGemmReference` now sweeps n
  ∈ {24,32,48,64,96,128} × trans × uplo at k = 200 (deliberately not a multiple of the k chunk).
* **A `herk` test that could not have failed.** `HerkTest` checked that the unreferenced triangle stays untouched and
  that the two `uplo` runs agree. Neither catches conjugating the wrong operand: that returns `conj(C)`, still
  Hermitian and still consistent across both triangles. `MatchesGemmReference` was added and confirmed to fail when
  the conjugation is flipped.
* **An unexercised branch nearly flipped on.** The `sytrd_blocked` trailing update runs only when the trailing block
  is wider than 128, and every pre-existing case in `tests/sytrd_blocked_tests.cc` was n ≤ 128 — so the syr2k path
  had **no test coverage at all**. `SytrdBlockedTest.TrailingUpdateRoutesAgree` (n=320, nb=32) was added and checked
  for teeth: forcing `alpha = -0.5` fails it at worst eigenvalue error 2.777. Its backward-error bound alone (`4 n eps
  ||A||` = 3.2e-3) is **~1000x looser** than either route's actual error and would pass almost anything; the
  load-bearing assertion is the relative one, syr2k within `4 × (GEMM route error) + 8 eps ||A||`.
* **A guard that modelled half its predicate.** `sytrd_blocked`'s her2k guard replicated only the size ceiling, so
  under `BATCHLAS_EXPAND_ROUTE=loop` the call site concluded her2k would take its batched-GEMM route while
  `her2k_gemm_preferred` returned false and sent it to a per-batch loop — one sequential launch per batch member, for
  every panel with n2 > 128. Both halves now live together in `her2k_takes_gemm_route` (`src/expansion_budget.hh`),
  which the call site asks (`src/extensions/sytrd_blocked.cc:838`).

### Dispatch: the ormqr chooser that forced past supports

History (route era; `route_ormqr.hh` and its `choose_ormqr_provider` predecessor are deleted).
`RouteTable<Op::ormqr>` replaced the smallest of the three Provider-based choosers, and it is where the
cost of conflating "forced" with "supported" was easiest to see. `choose_ormqr_provider` opened with

    Provider chosen = normalize_ormqr_vendor_like(policy.forced);
    if (chosen != Provider::Auto) return chosen;

so a forced provider was returned **without ever being checked against `ormqr_supports_blocked`**. Two
defects followed, both fixed by construction rather than by remembering a check:

1. **Forcing could run an unsupported kernel.** `ormqr_supports_blocked` is false for complex with
   `Transpose::Trans` and on any non-GPU queue, but `ormqr_dispatch`'s tail was
   `if (chosen == Vendor) vendor else blocked`, so `BATCHLAS_ORMQR_PROVIDER=blocked` ran the blocked
   path on exactly the inputs the predicate exists to exclude.
2. **The buffer size and the call could disagree.** For a forced value that is neither Vendor nor
   Blocked (`cta`, `two_stage`, `jacobi`, all of which parsed), `ormqr_dispatch` fell into its `else`
   arm and reset to Vendor, while `ormqr_buffer_size_dispatch`'s tail
   (`if (chosen == Vendor) vendor_size; return blocked_size`) returned the blocked size; the caller then
   hit "ormqr: insufficient workspace for chosen provider" from the call it had just sized for (the
   2560 vs 276480 byte instance above).

Both share the Provider enum's root: "the user asked for this" and "this can serve the shape" were the
same value. Splitting `supports()` from the forced request made the first impossible; resolving once
through a pure table made the second impossible. Flat selection keeps both properties: a pin that
fails `can_run` throws (R6), and `ormqr_buffer_size` runs the same `select::pick` as the call (R5).

Two table details that went with it: the order `{Native, Blocked}, {Vendor, Auto}` is what the shared
`std::array<Provider, 6>` came to for ormqr (`BatchLAS_CTA`, `_TwoStage`, `_Jacobi` were listed but
matched no branch, so they were inert padding). And ormqr had **no measured window**: no shape ever
sent a supported blocked call to the vendor, so `preferred()` equalled "native and supported". Its
transcribed sm_89 tables today rank `blocked` then `vendor` in every row where `blocked` can structurally run, and
`vendor` alone on the complex `trans=T` rows, where the old support predicate refused `blocked` (and `blocked`'s
`can_run` still does; `src/ops/ormqr/choice.hh:14`).

## Dispatch: buffer-size queries and the route they size

**Current rule (flat selection R5, [flat-kernel-selection.md](../design/flat-kernel-selection.md)).** Every
`<op>_buffer_size` runs the same `select::pick` as its op and returns exactly the chosen family's need, through the
op's `workspace()` visit in `src/ops/<op>/<op>.cc`; a nested op adds its children's sizes by calling their public
sizing functions. Query and call agree because they run the same choice, not because the query over-allocates. Two
constraints carried over from the route era still bind:

1. **The vendor path sizes itself from the vendor query, never from the public one.** In `src/backends/cusolver.cc`
   `potrf_vendor` sized its workspace with an unqualified `potrf_buffer_size`, and lookup escaped
   `batchlas::backend` to find the **facade** query. While facade == vendor this was invisible; once the public query
   returned `max(native, vendor)` it handed a batch-1 cuSOLVER call the native workspace size, silently: the pool was
   sized by the same public query and both terms are alignment multiples, so `pool.allocate` fits exactly and only
   `cusolverDnXpotrf` sees the wrong number, as its workspace-size argument. The call is now spelled
   `backend::potrf_vendor_buffer_size`.
2. **Sizing is pure.** `band_reduction.cc` and `sytrd_sy2sb.cc` size `geqrf` once against an `(m_max x nb_max)` dummy
   view and call it on smaller sub-views, so they use `geqrf_buffer_size_bound` (the maximum over every candidate the
   device can run, `src/ops/geqrf/geqrf.cc`), and no `geqrf` sizer may dereference `A.data_ptr()` or `tau.data()`,
   both `nullptr` there. `getri_buffer_size` runs under `BumpAllocator::measuring()` (`inv.cc` replays its layout
   through it), so everything reachable from it must be pure with respect to the workspace.

**History: the route-era facade (`src/dispatch/entry_points/`, deleted in phase 5).** There a query and its call
resolved independently, so the facade sized the **max over every supported tier**, not the chosen route; a
chosen-only size under-allocated wherever they disagreed (the `ormqr` 108x disagreement under
[Correctness findings](#correctness-findings)). "A native tier answered" had to be a flag, not `native_need != 0`,
because the CTA and Tiny `geqrf` tiers and the native `spmm` legitimately need zero bytes. And a native-routed call
was not sized by the vendor: `orgqr`'s vendor arm is a per-item loop whose size is batch-<b>linear</b> (about <b>4.6 GB</b>
at cdouble n=64, batch 8192, for a shape the native arm serves in a few megabytes; see
[the orgqr latent defect](qr.md#the-orgqr_buffer_size-latent-defect)), and `spmm`'s vendor sizer builds an
`SpmmCsrBatchPlan` that walks the CSR row offsets from the host. R5 dissolves the first two rules; the third is now
simply what "exactly the chosen family's need" means.

## Dispatch: the coverage instrument

Two tables, answering different questions (`src/select/coverage.hh`), plus `miss` rows for calls nothing could
serve. **static** (`linked`) lists, per op and backend, the vendor gate and whether a native kernel is linked, with
no kernel run — exact, instant, no GPU needed — and answers *"is the kernel in the build"*, the planning
question. **dynamic** (`reached`) counts `(op, scalar, backend, shape_class)` and records the chosen family spelling plus
`native_route_existed` / `native_route_supported`, answering *"did a call get there"*, the burn-down question. Reading
either as the other is how `VENDOR_FREE_BASELINE.md` came to claim a working vendor-free `gemm`
(`src/select/coverage.cc`, `append_static_rows`). **Linked is not reachable**, and a symbol being present is never evidence it
runs.

`native_route_supported` is a **tri-state** (`1` yes, `0` no, `-1` the call site could not tell); the third value is
load-bearing: the old level-3 gates never entered `*_cuda_custom` on a decline, and so conflated "nothing native
serves this shape" with "something does but the heuristic preferred the vendor" (`level3_coverage.hh`, deleted). Every
op, the level-3 four included since their flat-selection wave, records its row from `select::TraceScope`, with the
choice spelling (`triangular`, `gram`, `expand`, `vendor`, ...) as the algorithm. `uplo`/`side`/`diag`/`transA` are
part of the level-3 coverage **key**, not decoration.

`scripts/route_diff.sh capture|compare` is the only tool that sees vendor-to-vendor route changes: the kernel trace
cannot (its `Record` holds a `sycl::event`) and timing cannot (an unsaturated ratio is overhead, and routing a shape
to cuBLAS may well be faster). It treats a capture with **zero `reached` rows as a hard error** rather than as
"nothing changed" — the instrument has produced a correct header with no rows twice, for unrelated reasons, and both
times it looked clean. `scripts/coverage_merge.sh` collapses the per-PID shards a 53-binary `ctest` run produces.

### Instrument defects

Five, each of which looked healthy while reporting almost nothing: (1) the gate-declined half was unrecorded, so a
shape moving *off* a native kernel was invisible; (2) `uplo`/`side`/`diag` were not in the key, so two calls differing
only in `uplo` collapsed into one first-writer-wins row; (3) `emit()` opened with `"w"`, so each of 53 test binaries
truncated the last; (4) the compile-time gate and weak-symbol interposition (see [negative-results](#negative-results)
9); (5) `route_diff.sh compare` applies no `backend != AUTO` filter, so pure-layer test shapes recorded with `backend
= AUTO` make a clean 65-decision move look like 240 lines of churn.

## Vendor-free baseline

`cmake -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` yields
`BATCHLAS_HAS_CUDA_BACKEND 1` with every CUDA math library at 0 — a CUDA device with no CUDA math libraries, a state
the pre-WP0 scheme could not express and could not link. That configuration **configures, compiles, links, loads and
runs**. That is WP0's deliverable; it is emphatically not green, and no dispatch mechanism could make it so, because
the gap is missing kernels.

| milestone | `ctest -LE slow` | note |
|---|---|---|
| WP0 S6 | 20 / 53 | failures are `NoRouteError`, not crashes and not link errors |
| WP1 | 24 / 53 | `gemm_tests` 48 → 167 of 184; `bdsdc`, `ritz_values`, `sytrd_cta`, `transpose` recovered; none newly failing |
| WP2 correctness track | 25 / 53 | `gemm_tests` 184 / 184 |
| WP8 (latest recorded) | 35 / 57 | the failing **set** is the reviewable artefact, not the count |

The milestones predate flat selection; since then the level-3 wave added vendor-free routes for float syr2k at batch 1,
double symm and syrk `n <= 128`, and every trmm dtype on both sides (flat-kernel-selection.md §12, "Vendor-free gains").

WP1 ran in eight steps, each reporting the **same 3016 distinct routing decisions vendor-present**, diffed per step
with `scripts/route_diff.sh`. S0 made the four routes measurable (a 4-suite run went from 96 rows for one op to 312
across five); S1's portable vendor seam replaced 10 `*_vendor_cuda_raw` calls with byte-identical capture CSVs; S2
retargeted the terminal GEMM at the public entry point with timings within noise at saturating batch; S3 left **no
CUDA symbol in any of the four `.o`** under `nm -C`; S4 moved the TUs out of the CUDA object library; S5 gave the
facade's `gemm` a native arm (`gemm_tests` 48/184 → 167/184); S6 moved the gates to the facade, making the tile
kernels **reached** vendor-free for the first time — 41 native rows where there had been 0; S7 gave the tile predicate
a scalar parameter, leaving the failing set byte-identical.

S4's non-obvious detail: the CMake gate could not be relaxed by deleting `if(BATCHLAS_HAS_CUBLAS)`, because
`BACKEND_CUDA_SOURCES` feeds an object library that is not *created* when no CUDA math library is present; the four
names moved to `BACKEND_COMMON_SOURCES` (`src/backends/CMakeLists.txt:136-141`).

## Open debts

1. ~~`BATCHLAS_SYRK_ROUTE=native` reaches a route that writes both triangles~~ and
2. ~~`BATCHLAS_SYR2K_ROUTE=native` throws a cuBLASDx message it did not ask for~~: fixed, first by the route-era
   `Level3Pin` words and then by flat selection (#147). `native` now takes the row's first runnable non-vendor family
   (`gram` or `triangular` for syrk, `triangular` for float syr2k), and
   `SyrkCudaCustomTest.AutoAndNativeRoutesLeaveTheOtherHalfUntouched` poisons the other triangle under it.
3. ~~The four level-3 ops still have no tables~~: they do (level-3 flat-selection wave), transcribed from the old
   windows, untimed; measuring them is the phase-4 retune. hemm, herk and her2k still have none.
4. ~~`symm` has no `expansion_fits()` ceiling~~: symm's `expand` family checks it in `can_run`.
5. ~~Heterogeneous `symm` is unmeasured and untested~~: measured on `ff340fc6` (a heterogeneous B or C made the
   expansion's gemm throw; a heterogeneous A ran at the storage order); `expand` and `vendor` now refuse any
   heterogeneous operand (`HeterogeneousBatchHasNoRoute`).
6. ~~MathDx-present boxes are untestable here~~: cuBLASDx is deleted.
7. ~~Level-3 non-float is still cuBLAS-only~~: double syrk `gram`, double symm `expand` and every trmm family run
   vendor-free. **`syr2k` still has no non-float native route** — `syr2k_triangular_tiles` is float only.
8. ~~The static coverage table's `trsm` row is hardcoded `false`~~: it reads `true` (WP3).
9. **A coverage row cannot confirm that a particular shape ran**: rows are keyed on a power-of-two `shape_class`,
   first-writer-wins, so the m/n/k/batch columns can report a *different* call's shape. Prove a shape with a break
   that is red only for it. (The gate itself is a runtime bool on `$BATCHLAS_COVERAGE_OUT`; there is deliberately no
   `BATCHLAS_ENABLE_COVERAGE` build option, `cmake/BatchLASOptions.cmake:141`.)
10. **`symm_benchmark`, `syrk_benchmark` and `syr2k_benchmark` abort before printing anything** — a SYCL scheduler
    assertion (`adjustNDRangePerKernel: NDR.LocalSize[0] == 0`) on the host backend at tiny shapes, attributed by
    revert-and-rebuild as pre-existing and not WP1's. WP1 S2 needed a standalone harness.
11. **`Backend::INTEL` is hard-wired FALSE and oneMKL cannot be tested here**; WP0 only removed the dead branch that
    produced undefined references. (The route-era `syev.hh` measured-grid guard that read `s.backend ==
    Backend::CUDA` as measurement *provenance* went with syev's migration to `tuned/syev.*`; no such guard remains.)
12. **Unverified windows, in one place.** Each is now a row edge of a transcribed level-3 table, carried over from
    the old gates without a bracketing grid in these sources: the 257 ≤ n ≤ 383 band that the `tall`/`wide` rows of
    `tuned/syrk.float.*` rank `triangular` first; the batch-3 and `128 < n < 256` `form=sq` cells that
    `tuned/symm.float.*` ranks `vendor` first (and the same constants in hemm's `expansion_preferred`); and the old
    squareish disjunct's 2:1 aspect ratio (`tiled_work >= 8` folded away at batch ≥ 8), which is the `form=sq` edge of
    the syrk table. The phase-4 retune of the level-3 tables is what would settle them.

## Raw evidence

Raw data is preserved at the git tag `perf-evidence/vendor-independence` and is retrievable with `git show
perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| Gram/tile kernel design, all dtype tables, the band-split bug, the trmm routing mistake, the 16-row tile | `experiments/TRMM_SYRK_BATCHED_KERNELS.md` |
| Which level-3 op can replace a GEMM in `src/extensions`, and the three rejections | `experiments/GEMM_TO_LEVEL3_SURVEY.md` |
| SYRK triangular-route crossover sweep (n × batch) | `experiments/syrk_sweep.sh` |
| SYRK crossover with k free, to show insensitivity to reduction depth | `experiments/syrk_kskew.sh` |
| SYR2K triangular-route crossover sweep (n × k × batch) | `experiments/syr2k_sweep.sh` |
| herk/her2k GEMM-route vs per-batch-loop crossover | `experiments/herk_crossover.sh` |
| the dedicated-GPU harness every sweep above ran under | `experiments/gpu_guard.sh` |

Superseded root documents, retained at the same tag: `WP0_DISPATCH_SPEC.md` (the 14-agent design, migration table and
CMake variable list), `WP1_LEVEL3_SPEC.md` (the 12-agent design and its five corrections), `VENDOR_FREE_BASELINE.md`
(failing-set records per package) and `VENDOR_INDEPENDENCE_PLAN.md` (status board).

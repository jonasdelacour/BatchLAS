# Dispatch: the vendor gate, the level-3 route arms and the coverage instrument (WP0, WP1)

The durable record of two work packages: **WP0**, which moved the public op definitions out of the vendor
translation units and built the coverage instrument, and **WP1**, which freed the four level-3 tile dispatchers
(`symm`/`syrk`/`syr2k`/`trmm`) from the cuBLAS object library. The `Route{Origin, Algorithm}` vocabulary, the
`RouteTable` resolver and the legacy `_VARIANT`/`_PROVIDER` spellings WP0 introduced are gone: every other op chooses
from tuned tables (`src/ops/<op>/`, docs/design/flat-kernel-selection.md), and phase 5 of that design deleted
`include/batchlas/blas/dispatch/`.

The shipped code is the authority on **what** ships; the exploration notes are the authority on **why**. Ops covered:
`symm`, `syrk`, `syr2k`, `trmm` (their route gates and windows), plus the vendor gate and coverage instrument every op
uses. The same four ops' kernel design and per-dtype tables live in [`level3.md`](level3.md); this page is the routing
half.

## What ships

### The vendor-availability gate

`src/select/vendor.hh` asks per **library**, not per device family, because the map is not uniform: on NVIDIA
`getrf`/`getri` come from cuBLAS while `potrf`/`syev` come from cuSOLVER, on AMD all from rocSOLVER.

```cpp
template <Backend B>  // gemm gemv trsm trmm symm syrk syr2k hemm herk her2k
inline constexpr bool level3_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCBLAS) :
    B == Backend::NETLIB ? kHasNetlib : false;
```

with `factorization_`, `solver_` and `sparse_` siblings, and `kHasNetlib = BATCHLAS_HAS_LAPACKE &&
BATCHLAS_HAS_CBLAS`. When nothing serves a call, `select::throw_no_vendor_route<T>` records a coverage miss and throws
`batchlas::NoRouteError` (`include/batchlas/no_route.hh`), whose message names op, scalar type and the switch that
would restore it.

The spec's `src/dispatch/absent/*.cc` stub design was **declined**: it restates all 26 vendor signatures a second
time, and S5's two real bugs were signature divergence between restated copies. The shipped gate is an `if constexpr`
at the call site, so the vendor call is not compiled at all when the library is absent. The "is the kernel linked"
predicate four `src/extensions/` sites previously spelled `B == Backend::CUDA` is
`select::level3_tile_route_available<B, T> = B == Backend::CUDA && (std::is_same_v<T, float> ||
bool(BATCHLAS_HAS_CUBLAS))`; its users are `ormqr_blocked.cc`, `ortho.cc` (twice), `sytrd_blocked.cc` and the static
coverage table.

### Level-3 route arms

symm, syrk, syr2k and trmm select like every other op since the level-3 flat-selection wave
(`src/ops/<op>/{choice.hh,<op>.cc}`; families, `can_run` terms, keys and tables in
`docs/design/flat-kernel-selection.md` §12 "Level-3 four"). Their tables are transcribed from the
hand-written gates described below, so on-grid Auto is unchanged; the gates themselves, the
`*_custom_dispatch.cc` files and the cuBLASDx fused arms are deleted. hemm, herk and her2k still
choose in `cublas.cc` (vendor builds only), between an expansion or fold into the public `gemm` and
the vendor loop.

| op | families (spelling) | old gate, float CUDA GPU (`ff340fc6`) | old source |
|---|---|---|---|
| `symm` | `expand` (mirrored expansion + public `gemm`, float and double), `vendor` | `squareish && expansion_preferred(max_dim, batch)` | `symm_custom_dispatch.cc` |
| `syrk` | `gram` (n <= 128, float and double), `triangular` (float), `vendor` | `prefer_gram \|\| prefer_triangular \|\| cublasdx_heuristic` | `syrk_custom_dispatch.cc` |
| `syr2k` | `triangular` (float), `vendor` | `batch >= 2` | `syr2k_custom_dispatch.cc` |
| `trmm` | `triangular` (`Side::Left`), `expand` (expansion + public `gemm`), `vendor` (the loop) | `trmm_triangular_supported(...)`, **no size threshold**; Right went to an expand-or-loop vendor | `trmm_custom_dispatch.cc`, `cublas.cc` |

Their inner GEMMs go through the public `gemm`, so they take the gemm table's choice like any other caller.

Correctness, now `can_run` terms: every native family needs a CUDA GPU queue, a homogeneous batch
(every tile kernel indexes operands as `base + batch * stride`), extents >= 1 and batch <= 65535
(the batch is the grid's z or y dimension). syrk/syr2k `triangular` also bound the tile count (grid
y), `gram` needs n <= 128, symm/trmm `expand` need `expansion_fits`, trmm `triangular` needs
`Side::Left`, and syr2k keeps real `ConjTrans` on the vendor.

### The level-3 pin words

`BATCHLAS_<OP>_ROUTE` for these four ops is parsed by `src/select` like every other op's: `auto`,
`native`, `vendor` (class words; with nothing of their class runnable they fall back to Auto with
a warning) or a family spelling, lowercased and trimmed. A spelling that is not a compiled candidate
for the dtype, or cannot run the shape, throws `std::invalid_argument`.

| op | spellings | notes |
|---|---|---|
| `symm` | `expand`, `vendor` | `native` = `expand` |
| `syrk` | `gram`, `triangular`, `vendor` | `triangular` on double throws (not a double candidate); `gram` at n > 128 throws (the kernel answers wrongly there) |
| `syr2k` | `triangular`, `vendor` | `triangular` on double throws |
| `trmm` | `triangular`, `expand`, `vendor` | `vendor` is the `cublas?trmm` loop; the old `vendor` (expand+gemm when it fit) is `expand` |

`cublasdx` is not a word any more: cuBLASDx was deleted (it never ran here, MathDx being absent), so
a `cublasdx` pin throws as an unknown family. The legacy `tiles`/`narrow`/`gemm`/`custom` words and
the old `DiagFullGemm` measurement route are gone too; each op's should-throw tests list them
(`*CudaCustomTest.RemovedRouteWordsThrow`, `<Op>Candidates.UnknownPinsThrow`). `herk` reaches
syrk's Gram kernel only through `BATCHLAS_SYRK_ROUTE=gram` (`ops::syrk::herk_gram_pinned()`).

## Measured boundaries

These measurements are the evidence behind the old hand-written gates; the transcribed level-3
tables reproduce those gates, so they still describe today's Auto. File:line citations into
`*_custom_dispatch.cc`, `level3_vendor_fallback.hh` and `level3_coverage.hh` on this page refer to
the commits measured (up to `ff340fc6`); those files are deleted.

All figures RTX 4090 / sm_89, CUDA 13.2, `RelWithDebInfo`, one dedicated GPU via `experiments/gpu_guard.sh`. Batch is
always large enough to saturate; batch = 1 is not a design target.

### Expansion crossover

`expansion_preferred(max_dim, batch)` is `batch >= 4 || max_dim >= 256` (`kExpandMinBatch`/`kExpandMinDim`,
`triangular_expand.hh:43-45`; the predicate at `:49-60`, which consults `BATCHLAS_EXPAND_ROUTE=expand|loop` **first**,
so a pin overrides the window entirely).
Measured against a per-batch loop over the vendor's own triangular primitive — float `symm` over n 16..2048 × batch
1..512, complex64 `hemm` over n 16..512 × batch 1..16 — the expansion wins **1.2x to 72x** everywhere except the
bracketing region **batch ≤ 2 with n ≤ 128**, where it loses by **up to 2.5x** (`triangular_expand.hh:33-39`).

> The exploration notes (`VENDOR_INDEPENDENCE_PLAN.md` §WP1) quote the *loss* region, 2 and 128; the shipped constants
> are **4 and 256**, i.e. the guard is the complement over a wider region. `WP1_LEVEL3_SPEC.md` correction 4 records
> this. Consequence: batch = 3, or `128 < n < 256` at batch ≤ 3, is refused the expansion with **no bracketing
> measurement** — unverified, conservative in direction.

`trmm` does not consult this: `cublas?trmm` has a flat ~110 µs floor whatever the shape, so the expansion beat it in
every cell measured, batch 1 included (`triangular_expand.hh:41-44`).

### `syrk` tile boundaries

`syrk_prefer_triangular_tiles` (`syrk_custom_dispatch.cc:109-118`), with `kTriangularTile = 128` and `kTriangularTileK
= 8` (`triangular_tiles.hh:118-119`):

```cpp
if (detail::triangular_tiles_per_side(n) < 3 || k < detail::kTriangularTileK) return false;
return static_cast<long long>(A.batch_size()) * detail::triangular_tile_count(n) >= 160;
```

| boundary | winner side | bracketing non-winner |
|---|---|---|
| tile grid ≥ 3 a side | 1.45x at n=512 batch 512; 1.63x at n=1024 batch 64; 1.71x at n=2048 batch 16 | n=256 measured **0.84x–1.22x** depending on where its grid fell against a wave boundary — no win at all |
| batch × tile count ≥ 160 | won from 168 blocks up | **1.14x slower at 144 blocks, 1.25x slower at 136** |

Source for both: `syrk_custom_dispatch.cc:88-107`.

> `triangular_tiles_per_side(n) >= 3` is **n ≥ 257**, but the measurement supporting it says "from n = 384 up every
> saturated shape won" (`syrk_custom_dispatch.cc:98-99`; `experiments/GEMM_TO_LEVEL3_SURVEY.md` says the router "needs
> n >= ~384"). The band **257 ≤ n ≤ 383 is admitted by the shipped predicate and has no bracketing cell.** Worse than
> unmeasured: the kernel's unpredicated fast path additionally requires `n % 128 == 0`, `k % 8 == 0` and 4-element
> alignment on both operands (`syrk_triangular_tiles.hh:52-66`), and 384 is the first multiple of 128 at or above 257
> — so *every* shape in that band runs on the slower predicated path, which is the one the sweep never sampled.
> n = 256 (`tiles_per_side == 2`) is refused, and is a recorded 0.89x pre-existing loss on that route.

`syrk_prefer_gram_tiles(C)` is `C.rows() <= kGramMaxTile`, `kGramMaxTile = 128` (`syrk_custom_dispatch.cc:125-127`;
`syrk_gram_tiles.hh:65`): the single-tile Gram kernel serves exactly the range the triangular grid cannot, and inside
it the alternative is a host loop over `cublasSsyrk`, one to two orders of magnitude off anything batched
(`:120-124`). Float, against the GEMM spelling (`experiments/TRMM_SYRK_BATCHED_KERNELS.md`):

| reduction m | n | batch | gemm (ms) | syrk before (ms) | syrk now (ms) | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.334 | 33.588 | 0.0780 | 4.29x | 431x |
| 512 | 64 | 1024 | 0.345 | 31.911 | 0.1875 | 1.84x | 170x |
| 1024 | 128 | 512 | 0.409 | 30.296 | 0.4146 | **0.99x** | 73x |
| 2048 | 128 | 256 | 0.404 | 29.370 | 0.3820 | 1.06x | 77x |

n = 128 is the bracketing cell: parity, and near the ceiling — reading A once costs 298 µs against the GEMM's measured
409, so the whole prize there was 1.37x, not 2x. At n = 32 the kernel reads 71 MB in 78 µs, 933 GB/s, i.e. at the
memory roofline. In `double` the win *grows* with n (FP64 at 1/64 rate is compute bound): 1.08x, 1.78x and 2.10x at
m/n/batch 256/32/2048, 512/64/1024 and 1024/128/512 — against 138x, 58x and 17x for the host loop. The third disjunct,
`syrk_prefer_cuda_custom_heuristic` (`:129-144`), requires `n >= 16`, aspect ratio `min_dim * 2 >= max_dim`, and
`tiled_work = batch × ⌈n/32⌉² × ⌈k/32⌉ >= 8` over a 32-wide tile (`kSyrkCublasDxTile`, `:31`). Because `tiled_work`
carries the batch factor, at any batch ≥ 8 that term is satisfied by every shape, so **in the regime this campaign
tunes for the disjunct reduces to `n >= 16 && min_dim * 2 >= max_dim`**. With MathDx absent its only effect is to
admit shapes to the tile kernels, and **its own crossover has no bracketing grid here** — unverified.

### `syr2k` batch boundary

`syr2k_prefer_triangular_tiles(A)` is `A.batch_size() >= 2` (`syr2k_custom_dispatch.cc:95-97`). Measured float over n
8..3072 × k 4..2048 × batch 1..1024 (`:78-94`):

* **from batch 2** the kernel won every shape in the grid: 1.06x at n=3072, 1.12x at n=1024, 1.3–1.4x through the
  middle, up to 226x where n is small enough that the whole cost is the launch;
* **batch 1 is the bracketing cell and does not sort by anything**: the vendor wins 1.18–1.60x below n=1280 and again
  1.16x at n=3072; the kernel wins 1.02–1.71x between; the vendor wins 4–10x on a deep k with a small n. No threshold
  in n exists, so batch 1 keeps the vendor.

Neither n nor k nor the tile count enters the predicate, because none of them changes which side of the per-launch
difference a shape falls on.

### `trmm`: no threshold

There is deliberately no size threshold. The first router gated the tile kernel to `m <= 64 || m >= 512`, read off a
trmm-vs-**gemm** column — the wrong comparison, because the router chooses between the tile kernel and the **vendor**.
That gate sent m = 128..256 back to the expansion and cost **up to 1.29x on exactly the shapes it was meant to
protect** (`trmm_custom_dispatch.cc:76-94`).

Float, tile against vendor, at saturating batch (`trmm_custom_dispatch.cc:84-91`):

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
FMA peak on these shapes while a complex accumulator costs twice the registers.

### `herk` and `her2k` crossovers

Not moved by WP1 — both still live inside the cuBLAS-gated TU — but they are the same expansion decision and the
constants are easy to confuse with the ones above.

| predicate | shipped condition | evidence | bracketing non-winner |
|---|---|---|---|
| `herk_gemm_preferred` (`cublas.cc:382-388`) | `batch >= 4 && n <= 768` | complex64, n 32..1024 × batch 1..256: 1.6x–72x for batch ≥ 4 at n ≤ 512 | a wash at n = 640..768; **0.82x–0.93x from n = 896 up**; batch ≤ 2 a wash or loss at every n |
| `her2k_gemm_preferred` (`expansion_budget.hh:112-116`) | `batch >= 2 \|\| n >= 128` | 1.4x–128x everywhere else | batch 1 at n ≤ 64: **0.74x at n = 32, 0.89x at n = 64** |

`herk`'s is a conjunction with a large-n ceiling because its GEMM computes both triangles and keeps one; the mirrored
expansion's is a disjunction with no ceiling, because expanding costs one bandwidth-bound kernel and then does exactly
the vendor's work (`cublas.cc:378-381`). Both predicates check `BATCHLAS_EXPAND_ROUTE` **before** their window
(`expansion_budget.hh:95-101`, delegated from `cublas.cc:365-367`), so a pin overrides the measurement — which is
exactly the seam the `sytrd_blocked` half-guard bug below fell through.

**An open A/B, not a settled window.** `her2k_gemm_preferred` was swept over *square* rank-k shapes, but the
`sytrd_blocked` panel loop issues narrow ones — `k = ib = nb ∈ {16,24,32}` against `n2` up to 480 — where the GEMM is
near bandwidth-bound and the fold adds an `n2²·batch` write plus read the two direct GEMMs never pay. The halved
arithmetic may not survive that, and the call site says so in place (`sytrd_blocked.cc:803-809`). `complex<double>` is
deliberately excluded from that route for the same reason: it would reach the same fast path, but its scratch is 16
bytes per element and none of it has been measured (`:811-814`). Guessing is how the 7.8x double inversion below got
written down in the first place.

## Negative results

Built, measured, rejected. These cost as much to establish as the wins.

1. **The `split-tu` WP1 design** — split each level-3 TU into portable and CUDA halves, transcribing the gate
   thresholds into `RouteTable::preferred`. Killed by a *confirmed silent route change*: the live thresholds are
   **gate-only** (`syrk_cuda_custom`'s Auto arm takes `syrk_triangular_tiles` unconditionally once the gram test
   fails, with no second preference check), so a transcribed `>= 3` rule rejects the tile route for **129 ≤ n ≤ 383 at
   every batch**, sending n = 256 to a route that writes both triangles. Two judges found it independently; it scored
   3/4/6 against the shipped `retarget-only` design's 9/7/8.
2. **Making the sideways vendor fallback the public entry point.** Every `*_vendor_cuda_raw` site is reached *after* a
   gate that already returned true, so a public call from there re-enters the same gate with the same environment and
   views: unbounded recursion, reachable with a forced fused pin on a CPU queue. The shipped fix is a
   dedicated seam (`level3_vendor_fallback.hh:5-27`).
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
   bound where real syrk is bandwidth bound. `herk` keeps its route; the conjugating path stays reachable as
   `BATCHLAS_SYRK_ROUTE=gram` so it stays measurable and tested.
6. **`syr2k` for the `sytrd_blocked` trailing update, in `double`.** 7.7x and 7.4x slower at n2=256/batch 1024, 1.9x
   slower at n2=512/batch 512; 1.55x *faster* only at n2=2048/batch 32. Double wins only where the batch is small
   enough that per-item launch cost amortises — the opposite of the regime that matters. The route stays CUDA + float.
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
   non-float tile route at all) and too wide in **backend** (the facade gate is guarded on `Backend::CUDA`). It took a
   scalar parameter instead (`:37-64`).
9. **A compile-time coverage gate.** `resolve_route` is an inline function template, so every TU instantiates its own
   weak copy, and ELF resolves the executable's weak symbols ahead of a shared library's. A test compiled without the
   macro interposed its uninstrumented copy over the library's instrumented one; the run produced a coverage file with
   a correct header and **zero `reached` rows** (`coverage.hh:27-49`). The gate is now a runtime bool in exactly one
   TU, and `cmake/BatchLASOptions.cmake:109` records that the option was deliberately never added.

## Correctness findings

Wrong answers found, how they hid, and what guards them now.

* **`ormqr`'s buffer size and call disagreed by 108x.** `cta`, `two_stage` and `jacobi` all parsed but matched no
  branch, so `ormqr_dispatch` ran on the vendor while `ormqr_buffer_size` returned the *blocked* size — 2560 bytes
  against the 276480 the call then demanded, so sizing a workspace with the public API and passing it to the public
  call threw deterministically on every GPU type. Structurally prevented now: an op and its `*_buffer_size` query
  call the same `choose()` (flat selection, rule R5).
* **`{Vendor, FusedDevice}` satisfies `is_vendor` but is not "the plain vendor call".** The level-3 dispatchers'
  `request == Vendor` tests meant `cublasSsyrk` specifically; rendering them as `is_vendor()` makes a forced cuBLASDx
  request answer yes to "did the caller ask for the vendor?". The level-3 pin words kept `vendor` and `cublasdx`
  apart (`Level3Pin`, deleted with cuBLASDx in the level-3 flat-selection wave).
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
  the public declarations would have passed `alpha` where `side` was expected on **every** backend. Bodies are lifted
  verbatim instead (`level3_vendor_fallback.hh:36-42`).
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
  is wider than 128, and every pre-existing case in `tests/sytrd_blocked_tests.cc` was n ≤ 128 — so the syr2k route
  had **no test coverage at all**. `SytrdBlockedTest.TrailingUpdateRoutesAgree` (n=320, nb=32) was added and checked
  for teeth: forcing `alpha = -0.5` fails it at worst eigenvalue error 2.777. Its backward-error bound alone (`4 n eps
  ||A||` = 3.2e-3) is **~1000x looser** than either route's actual error and would pass almost anything; the
  load-bearing assertion is the relative one, syr2k within `4 × (GEMM route error) + 8 eps ||A||`.
* **A guard that modelled half its predicate.** `sytrd_blocked`'s her2k guard replicated only the size ceiling, so
  under `BATCHLAS_EXPAND_ROUTE=loop` the call site concluded her2k would take its batched-GEMM route while
  `her2k_gemm_preferred` returned false and sent it to a per-batch loop — one sequential launch per batch member, for
  every panel with n2 > 128. Both halves now live together in `expansion_budget.hh:85-101`.

## The coverage instrument

Two tables, answering different questions (`src/select/coverage.hh`). **static** (`linked`) iterates the route predicates
with no kernel run — exact, instant, no GPU needed — and answers *"is the kernel in the build"*, the planning
question. **dynamic** (`reached`) counts `(op, scalar, backend, shape_class)` and records the chosen route plus
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
2. ~~`BATCHLAS_SYR2K_ROUTE=native` throws a cuBLASDx message it did not ask for~~: fixed by the level-3 pin words
   (flat selection phase 5). `native` now takes the tile kernel, and
   `SyrkCudaCustomTest.AutoAndNativeRoutesLeaveTheOtherHalfUntouched` poisons the other triangle under it.
3. ~~The four level-3 ops still have no tables~~: they do (level-3 flat-selection wave), transcribed from the old
   windows, untimed; measuring them is the phase-4 retune. hemm, herk and her2k still have none.
4. ~~`symm` has no `expansion_fits()` ceiling~~: symm's `expand` family checks it in `can_run`.
5. ~~Heterogeneous `symm` is unmeasured and untested~~: measured on `ff340fc6` (a heterogeneous B or C made the
   expansion's gemm throw; a heterogeneous A ran at the storage order); `expand` now refuses any heterogeneous operand
   (`HeterogeneousBatchHasNoNativeRoute`).
6. ~~MathDx-present boxes are untestable here~~: cuBLASDx is deleted.
7. ~~Level-3 non-float is still cuBLAS-only~~: double syrk `gram`, double symm `expand` and every trmm family run
   vendor-free. **`syr2k` still has no non-float native route** — `syr2k_triangular_tiles` is float only.
8. ~~The static coverage table's `trsm` row is hardcoded `false`~~: it reads `true` (WP3).
9. **A coverage row cannot confirm that a particular shape ran**: rows are keyed on a power-of-two `shape_class`,
   first-writer-wins, so the m/n/k/batch columns can report a *different* call's shape. Prove a shape with a break
   that is red only for it.
10. **`symm_benchmark`, `syrk_benchmark` and `syr2k_benchmark` abort before printing anything** — a SYCL scheduler
    assertion (`adjustNDRangePerKernel: NDR.LocalSize[0] == 0`) on the host backend at tiny shapes, attributed by
    revert-and-rebuild as pre-existing and not WP1's. WP1 S2 needed a standalone harness.
11. **`Backend::INTEL` is hard-wired FALSE and oneMKL cannot be tested here**; WP0 only removed the dead branch that
    produced undefined references. Separately, `syev.hh`'s measured-grid guard still reads `s.backend ==
    Backend::CUDA`, left alone deliberately: there it is measurement *provenance*, not wiring, so rewriting it as
    `level3_tile_route_available` would assert something false.
12. **Unverified windows, in one place:** the 257 ≤ n ≤ 383 band admitted by `syrk_prefer_triangular_tiles`; the
    batch-3 and `128 < n < 256` bands refused by `expansion_preferred`; and `syrk_prefer_cuda_custom_heuristic`'s
    `tiled_work >= 8` and 2:1 aspect ratio, none of which has a bracketing grid in these sources.

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

# Dispatch and coverage

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · level-3 tables transcribed 2026-10-06; WP0/WP1 measurements before 2026-09-30

The shared dispatch rules for every op, and the level-3 four (`symm`, `syrk`, `syr2k`, `trmm`): vendor availability,
candidate families, `can_run` terms, pin spellings, what the tuned tables rank, and the measurements behind the
table edges. How ops choose in general: @ref design_flat_selection and @ref selection. Per-op ranking:
@ref selection_tables.

## The vendor-availability gate

`src/select/vendor.hh` checks availability per library, not per device family. On NVIDIA, `getrf`/`getri` come from
cuBLAS and `potrf`/`syev` from cuSOLVER; on AMD all come from rocSOLVER (`vendor.hh:23-27`):

```cpp
template <Backend B>  // gemm gemv trsm trmm symm syrk syr2k hemm herk her2k
inline constexpr bool level3_vendor_available =
    B == Backend::CUDA   ? bool(BATCHLAS_HAS_CUBLAS)  :
    B == Backend::ROCM   ? bool(BATCHLAS_HAS_ROCBLAS) :
    B == Backend::NETLIB ? kHasNetlib : false;
```

- `factorization_`, `solver_` and `sparse_` are siblings; `kHasNetlib = BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS`.
- Each `OpSpec` names its library (`select::Lib::level3` for the level-3 four). Selection sets
  `select::Device::has_vendor`, which every `vendor` family's `can_run` reads. The `vendor` launch compiles the
  vendor call only under `if constexpr (select::has_library<B>(spec.vendor))`, so an absent library is not compiled.
- With nothing runnable, `select::throw_no_vendor_route<T>` records a coverage miss and throws
  `batchlas::NoRouteError` (`include/batchlas/no_route.hh`).
- `select::level3_tile_route_available<B, T>` (`src/select/vendor.hh:94-96`) is the "kernel is linked" predicate for
  `ormqr_blocked.cc`, `ortho.cc`, `sytrd_blocked.cc` and the static coverage table. It keeps its pre-flat-selection
  value on purpose: widening it would move ortho's and ormqr's vendor-free routes.

## Level-3 route arms {#level-3-route-arms}

`src/ops/<op>/choice.hh` lists each op's families, and `src/ops/<op>/<op>.cc` holds `key_of`, `can_run`, the
launch and the workspace. `select::run` takes the first runnable entry of the nearest row of
`tuned/<op>.<dtype>.<device>.txt`. Every family is fieldless. Derived legs (gram's tile width, the triangular
kernels' aligned/predicated leg, `trmm_row_tile`) stay derived. hemm, herk and her2k still choose by hand in
`cublas.cc` (vendor builds only) and have no `BATCHLAS_<OP>_ROUTE`.

| op | candidates, tie-break order | keys | last resort |
|---|---|---|---|
| `symm` | `expand` (mirror A's triangle into scratch, then the public `gemm`), `vendor`; float and double | `form:exact m:log n:log batch:log` | `expand, vendor` |
| `syrk` | float `gram, triangular, vendor`; double `gram, vendor` | `form:exact trans:exact n:log:2 k:log batch:log` | `triangular, gram, vendor` |
| `syr2k` | float `triangular, vendor`; double `vendor` | `n:log:2 k:log batch:log` | `triangular, vendor` |
| `trmm` | `triangular` (`Side::Left` tile kernel), `expand` (expand the triangle, then the public `gemm`), `vendor` (the `cublas?trmm` loop); all four types | `side:exact order:log:2 q:log batch:log` | `expand, triangular, vendor` |

`form` is `sq` when `2*min >= max` over C's `m, n` (symm) or over `n, k` (syrk), else `tall` or `wide`. The
`expand` families' inner GEMMs go through the public `gemm` and take its table's choice.

**`can_run` is correctness only.** Every native family needs `B == Backend::CUDA`, a GPU queue, a homogeneous
batch (tile kernels index `base + batch * stride`), extents >= 1 and `batch <= 65535`. The batch is grid z (y for
gram); 65536 throws `Number of work-groups exceed limit`, confirmed by launch.

| op | family | further terms | where |
|---|---|---|---|
| `symm` | `expand` | `max_wg >= 256` (8 x 32 group); `expansion_fits(q, k, batch, bytes)` (int-linearised grid, `BATCHLAS_EXPAND_MAX_BYTES` or a quarter of memory) | `src/ops/symm/symm.cc:44-56` |
| `syrk` | `gram` | `syrk_gram_supported` (square C, `n <= 128`; past it the kernel answers wrongly rather than throwing); `max_wg >= gram_threads(n)`; SLM tile fits | `src/ops/syrk/syrk.cc:57-60` |
| `syrk` | `triangular` | float only; `max_wg >= 256`; `T(T+1)/2 <= 65535` tiles, `T = ceil(n/128)` (n = 46208 runs, 46209 throws) | `src/ops/syrk/syrk.cc:61-64` |
| `syr2k` | `triangular` | float only; `max_wg >= 256`; `transA != ConjTrans`; the same tile ceiling | `src/ops/syr2k/syr2k.cc:52-56` |
| `trmm` | `triangular` | `Side::Left`; `ceil(m/tile_m) * ceil(q/128) <= 65535` tiles | `src/ops/trmm/trmm.cc:59-62` |
| `trmm` | `expand` | `max_wg >= 256`; `expansion_fits` | `src/ops/trmm/trmm.cc:63-68` |
| all four | `vendor` | the level-3 library is linked, **and a homogeneous batch** (every vendor loop runs each item at the top-level extents, a wrong answer for a heterogeneous batch) | `symm.cc:57`, `syrk.cc:65`, `syr2k.cc:57`, `trmm.cc:69` |

syrk is real-only, so `ConjTrans` reaches its kernels as `Trans`. It is not a `can_run` term; the exact `trans` key
keeps real `ConjTrans` on the vendor.

**What the tables rank.** All level-3 tables are **transcribed, untimed** (`source=transcribed:ff340fc6`, entries
`<spelling> -`). They are identical for sm_89 and sm_120, and ship for float and double only for the three real-only
ops (`tuned/README.md`). Each row is the old vendor-present Auto choice, then every other structurally runnable
candidate, with `vendor` last unless it was the Auto choice. The off-grid gate agreed with the old choice on 100.00%
of 2500-3000 random points per (op, dtype, device) (flat-kernel-selection.md §12).

| op | rows that rank a native family first (float) | double |
|---|---|---|
| `symm` | `form=sq` rows with `batch >= 4` or `max(m, n) >= 256` rank `expand` first (`m=128 n=255` `vendor` at batch 3, `expand` at 4; `m=128 n=256` `expand` from batch 1). Every reachable `tall`/`wide` row ranks `vendor` first. | `vendor` first everywhere |
| `syrk` | `n <= 128`: `gram`. Above: `triangular` on every `form=sq` row; on `tall`/`wide` rows only where `n >= 257`, `k >= 8` and `batch * T(T+1)/2 >= 160` (`tall n=512 k=64` `vendor` to batch 15, `triangular` from 16). `trans=C` rows `vendor` first. | `gram` to n = 128, `vendor` above |
| `syr2k` | `batch >= 2`: `triangular`; `batch = 1`: `vendor` | `vendor` (only candidate) |
| `trmm` | `side=L`: `triangular`; `side=R`: `expand` (all four types) | same |

### The level-3 pin words {#the-level-3-pin-words}

`BATCHLAS_<OP>_ROUTE` for these four ops is parsed by `src/select` like every other op's: `auto`, `native`,
`vendor`, or a family spelling, case-folded and trimmed. `native` takes the row's first runnable non-vendor entry.
`vendor` is the vendor; with nothing of its class runnable, either class word falls back to Auto with a warning, so
`vendor` in a vendor-free build means Auto.

A spelling that does not parse, is not a compiled candidate for the dtype, or whose `can_run` refuses the call throws
`std::invalid_argument` ("cannot run this shape") before anything is written. Tests:
`<Op>Candidates.UnknownPinsThrow`, `<Op>Candidates.CanRunFalsePinsThrow` and `<Op>Candidates.ClassWordsAndSpellings`
in `tests/{symm,syrk,syr2k,trmm}_candidates_tests.cc`. The value comes from the settings snapshot; call
`detail::reload_settings()` after a raw `setenv`.

| op | spellings | notes |
|---|---|---|
| `symm` | `expand`, `vendor` | `native` = `expand`. `expand` throws on a heterogeneous operand or when `expansion_fits` fails (e.g. `BATCHLAS_EXPAND_MAX_BYTES=0`) |
| `syrk` | `gram`, `triangular`, `vendor` | `triangular` on double throws (not a double candidate). `gram` at n > 128 throws |
| `syr2k` | `triangular`, `vendor` | `triangular` on double, or with real `ConjTrans`, throws |
| `trmm` | `triangular`, `expand`, `vendor` | `triangular` with `Side::Right` throws. `vendor` is the `cublas?trmm` loop; the old `vendor` (expand + gemm) is now `expand` |

Every other word throws: another level-3 op's spelling, `cublasdx`, the legacy `tiles`/`narrow`/`gemm`/`custom`/
`dx`/`fused` words, `diag_full_gemm`, and the `origin:algorithm` forms `native:auto` and `vendor:auto`
(`SyrkCudaCustomTest.RemovedRouteWordsThrow`, `SymmCudaCustomTest.RemovedRouteWordsThrow`,
`TrmmCandidates.UnknownPinsThrow`). Pins apply and are checked for every dtype.

Two neighbouring variables are not pins of these four ops. `herk` reaches syrk's conjugating Gram kernel only when
`BATCHLAS_SYRK_ROUTE` is exactly `gram` (`ops::syrk::herk_gram_pinned()` in `src/ops/syrk/choice.hh`).
`BATCHLAS_EXPAND_ROUTE=expand|loop` is read by hemm, herk and her2k only. `src/backends/route_common.hh` holds only
`ceil_div` and `is_gpu_queue`.

## Measured boundaries {#dispatch-measured-boundaries}

These measurements are the evidence behind the level-3 tables. Each boundary is a row edge in
`tuned/<op>.float.sm_89.txt`. The windows are route-era decisions that have not been re-timed; the phase-4 retune
would replace the transcribed rows with timed ones. Gate sources quoted here are at `ff340fc6`
(`git show ff340fc6:src/backends/<file>`).

All figures: RTX 4090 / sm_89, CUDA 13.2, `RelWithDebInfo`, one dedicated GPU via `experiments/gpu_guard.sh`, at a
saturating batch (batch 1 is not a design target).

### Expansion crossover

The old rule: expand when `batch >= 4 || max_dim >= 256` (`kExpandMinBatch`/`kExpandMinDim` in
`src/backends/triangular_expand.hh`). In code it survives only in `expansion_preferred`, which decides **hemm** and
reads `BATCHLAS_EXPAND_ROUTE` first. For **symm** the rule is the `form=sq` rows of `tuned/symm.float.sm_89.txt`.

Measured against a per-batch loop over the vendor's triangular primitive (float symm n 16..2048 x batch 1..512;
complex64 hemm n 16..512 x batch 1..16), expansion wins **1.2x to 72x** everywhere except **batch <= 2 with n <= 128**,
where it loses by **up to 2.5x**.

> **Note:** the shipped constants are 4 and 256, the complement of a wider region than the loss region quoted in
> the exploration notes (2 and 128). Batch 3, and 128 < n < 256 at batch <= 3, is refused expansion with no
> bracketing measurement. The symm table carries the same refusal (`form=sq m=255 n=255` ranks `vendor` first at
> batch 3, `expand` at 4).

`trmm` never consulted this rule. `cublas?trmm` has a flat ~110 µs floor, so expansion beat it in every cell measured,
batch 1 included. Its table ranks `expand` first on every `side=R` row, and `triangular` (then `expand`) on every
`side=L` row.

### `syrk` tile boundaries {#syrk-tile-boundaries}

The old triangular gate (`syrk_prefer_triangular_tiles` at `ff340fc6`, `kTriangularTile = 128`,
`kTriangularTileK = 8`): at least 3 tiles a side (`n >= 257`), `k >= 8`, and `batch * T(T+1)/2 >= 160`, with
`T = ceil(n/128)`. The table edges that carry it:

- `n=256` rows rank `vendor` first at every batch.
- `n=257` rows rank `triangular` first from batch 27 (6 tiles).
- `k=7` rows `vendor`; `k=8` rows `triangular` (batch 1024, n = 512).
- `n=512 k=64` rows switch at batch 16 (10 tiles, 160 blocks).

| boundary | winner side | bracketing non-winner |
|---|---|---|
| tile grid >= 3 a side | 1.45x at n=512 batch 512; 1.63x at n=1024 batch 64; 1.71x at n=2048 batch 16 | n=256: **0.84x–1.22x**, depending on where its grid fell against a wave boundary |
| batch x tile count >= 160 | won from 168 blocks up | **1.14x slower at 144 blocks, 1.25x slower at 136** |

> **Warning:** `n >= 257` is the edge the rows carry, but the supporting measurement says "from n = 384 up every
> saturated shape won" (`experiments/GEMM_TO_LEVEL3_SURVEY.md`). The band **257 <= n <= 383** ranks `triangular`
> first with no bracketing cell. Every shape there takes the predicated path: the unpredicated fast path needs
> `n % 128 == 0`, `k % 8 == 0` and 4-element alignment on both operands (`src/backends/syrk_triangular_tiles.hh:36-44`),
> and the sweep never sampled the predicated path. On `tall`/`wide` rows, n = 256 (`tiles_per_side == 2`) ranks
> `vendor` first, a recorded 0.89x pre-existing loss on the tile route.

The Gram gate was `C.rows() <= kGramMaxTile` (128, `src/backends/syrk_gram_tiles.hh:28`). It is now a `can_run`
term and the row edge: every row with `n <= 128` ranks `gram` first, float and double. Float, against the GEMM
spelling (`experiments/TRMM_SYRK_BATCHED_KERNELS.md`):

| reduction m | n | batch | gemm (ms) | syrk before (ms) | syrk now (ms) | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.334 | 33.588 | 0.0780 | 4.29x | 431x |
| 512 | 64 | 1024 | 0.345 | 31.911 | 0.1875 | 1.84x | 170x |
| 1024 | 128 | 512 | 0.409 | 30.296 | 0.4146 | **0.99x** | 73x |
| 2048 | 128 | 256 | 0.404 | 29.370 | 0.3820 | 1.06x | 77x |

n = 128 is the bracketing cell: parity, near the ceiling (reading A once costs 298 µs against the GEMM's 409 µs, so
the prize there was 1.37x). At n = 32 the kernel reads 71 MB in 78 µs (933 GB/s, the memory roofline). In `double`
the win grows with n, because FP64 at 1/64 rate is compute bound: 1.08x, 1.78x and 2.10x at m/n/batch
256/32/2048, 512/64/1024 and 1024/128/512, against 138x, 58x and 17x for the host loop.

The old third disjunct (`syrk_prefer_cuda_custom_heuristic` at `ff340fc6`, a cuBLASDx heuristic) required
`n >= 16`, `min_dim * 2 >= max_dim` over `(n, k)`, and `tiled_work = batch x ceil(n/32)^2 x ceil(k/32) >= 8`. At
batch >= 8 the last term always holds, so the disjunct reduces to `n >= 16 && min_dim * 2 >= max_dim`. Its crossover
has no bracketing grid, so it is unverified; today it is the `form=sq` edge (every `form=sq` row above n = 128 ranks
`triangular` first).

### `syr2k` batch boundary

The old gate was `A.batch_size() >= 2` (`syr2k_prefer_triangular_tiles` at `ff340fc6`). The table keeps it: every
row with `batch >= 2` ranks `triangular` first, every `batch=1` row ranks `vendor` first, at every n and k. Measured
in float over n 8..3072 x k 4..2048 x batch 1..1024:

* **From batch 2**, the kernel won every shape: 1.06x at n=3072, 1.12x at n=1024, 1.3–1.4x through the middle, up
  to 226x where the whole cost is the launch.
* **Batch 1** does not sort by n: the vendor wins 1.18–1.60x below n=1280 and 1.16x at n=3072; the kernel wins
  1.02–1.71x between; the vendor wins 4–10x on a deep k with a small n. Batch 1 keeps the vendor.

Neither n, k nor the tile count entered the gate; the table keys on them because the transcription grid spans them.
Double has no native family and ranks `vendor` alone.

### `trmm`: no threshold

The table has no size threshold: every `side=L` row ranks `triangular` first at every order, `q` and batch, in all
four types. The first router gated the tile kernel to `m <= 64 || m >= 512`, comparing against gemm instead of the
vendor. That sent m = 128..256 to the expansion and cost **up to 1.29x** on those shapes.

Float, tile against vendor, at saturating batch:

| m | nC | batch | tile (ms) | vendor (ms) |
|---|---|---|---|---|
| 128 | 512 | 1024 | 0.698 | 0.784 |
| 128 | 1024 | 512 | 0.686 | 0.687 |
| 256 | 256 | 512 | 0.536 | 0.692 |
| 256 | 1024 | 256 | **0.915** | **0.855** |

The last row is a 7% loss and the bracketing cell. It is not worth a special case that would have to be re-tuned
whenever either route changes.

Against the GEMM spelling, `trmm` wins everywhere in `double` (1.29x–2.02x) and `complex<double>` (1.05x–1.95x), and
at 8 of 10 float shapes. It **loses below m = 512 in `complex<float>`** (0.69x–0.93x): a register-file ceiling, since
cuBLAS's cgemm runs near FP32 FMA peak on these shapes while a complex accumulator costs twice the registers. The
cfloat table ranks `triangular` first on every `side=L` row; `triangular` vs `expand` for cfloat below m = 512 is
untimed.

### herk and her2k crossovers

hemm, herk and her2k choose by hand in the cuBLAS-gated TU. The predicates below are the same expansion decision as
above, with different constants.

| predicate | shipped condition | evidence | bracketing non-winner |
|---|---|---|---|
| `herk_gemm_preferred` (`src/backends/cublas.cc`) | `batch >= 4 && n <= 768` | complex64, n 32..1024 x batch 1..256: 1.6x–72x for batch >= 4 at n <= 512 | a wash at n = 640..768; **0.82x–0.93x from n = 896 up**; batch <= 2 a wash or loss at every n |
| `her2k_gemm_preferred` (`src/expansion_budget.hh`) | `batch >= 2 \|\| n >= 128` | 1.4x–128x everywhere else | batch 1 at n <= 64: **0.74x at n = 32, 0.89x at n = 64** |

herk's rule is a conjunction with a large-n ceiling, because its GEMM computes both triangles and keeps one. The
her2k rule is a disjunction with no ceiling. Both check `BATCHLAS_EXPAND_ROUTE` before their window
(`expansion_route_pin` in `src/expansion_budget.hh`, read by `rankk_route_pin` in `cublas.cc`), and both also need
`expansion_fits`, else they take the vendor loop.

> **Note:** `her2k_gemm_preferred` was swept over square rank-k shapes. The `sytrd_blocked` panel issues narrow ones
> (`k = ib = nb` in {16, 24, 32}, n2 up to 480), where the fold adds an `n2^2 x batch` write and read that the two
> direct GEMMs never pay. Those cells are unmeasured. `complex<double>` is excluded from this route for the same
> reason: its scratch is 16 bytes per element and unmeasured (`src/extensions/sytrd_blocked.cc:777-782`; the her2k
> call and its fit check are at `:830-845`).

## Negative results {#dispatch-negative-results}

Built, measured, rejected.

| Alternative | Result | Verdict |
|---|---|---|
| `split-tu` WP1 design: split each level-3 TU into portable and CUDA halves, transcribing the gate thresholds into `RouteTable::preferred` | Silent route change. The live syrk gate was gate-only, so a transcribed `>= 3` rule rejected the tile route for 129 <= n <= 383 at every batch; n = 256 went to a route that wrote both triangles. Scored 3/4/6 against 9/7/8 for the shipped design | Rejected. Flat selection instead transcribes the whole old gate per grid cell (`--fidelity` against `git show ff340fc6:`), adds the `form` key, and was accepted at 100.00% off-grid agreement ([flat-kernel-selection.md](../design/flat-kernel-selection.md), §12) |
| Public entry point as the sideways vendor fallback | Each `*_vendor_cuda_raw` site ran after a gate that had already returned true, so a public re-entry recursed without bound (reachable with a forced fused pin on a CPU queue) | Rejected. Each `vendor` family calls `backend::<op>_vendor<B, T>` directly |
| Vendor-independence stub layer `src/dispatch/absent/*.cc` | Restated all 26 vendor signatures; the two real bugs of the work were signature divergence between copies | Declined. The `if constexpr` at the call site replaces it |
| `syrk`/`herk` for `ortho`'s Gram matrix, before a Gram kernel existed | 73x–96x slower at m 256..2048, k 32..128, batch 256..2048 (k < 384 failed both disjuncts; one `cublasSsyrk` per batch member) | Reversed once the Gram-tile kernel existed ([syrk tile boundaries](#syrk-tile-boundaries)) |
| `trmm` for the WY block factor, before a tile kernel existed | Lost at every shape (0.195 to 0.238 ms at ib=32, nC=256, batch 2048; 0.779 to 1.152 ms at ib=256). The tile kernel then won: float 1.006x–1.046x, double 1.004x–1.016x; `complex<float>` 0.944x–0.995x; `complex<double>` 0.958x–1.010x at ib=16; netlib float/double 0.336x/0.379x at n=128, ib=16 | Shipped for float and double only; complex takes the GEMM. Split is per type, not per precision. A 16-row tile took `complex<double>` ib=16 from 0.958x to 0.996x and `double` to 1.013x–1.036x; it is kept for `double` |
| Complex Gram tiles (`herk`) | 0.217 vs 0.206 ms at n=32, batch 2048; 2.08 vs 1.57 ms at n=128, batch 512. A complex multiply is four real ones, so herk is compute bound where real syrk is bandwidth bound | Rejected. The conjugating path stays reachable only with `BATCHLAS_SYRK_ROUTE=gram` (`ops::syrk::herk_gram_pinned()`), so it stays testable |
| `syr2k` for the `sytrd_blocked` trailing update, in `double` | 7.7x and 7.4x slower at n2=256, batch 1024; 1.9x slower at n2=512, batch 512; 1.55x faster only at n2=2048, batch 32 | Rejected for `double`. Float ships: 1.25x–1.66x on the update, 1.07x–1.27x end to end. The float path drops the upper-triangle symmetrize pass: nothing in the `sytrd_blocked` pipeline reads A's upper triangle, so the pass cost more than half the win |
| `syr2k` at n = 1024 after WP1 S2, reported 10.9% faster | Repeating the step gave a 5.65–6.40 ms spread: noise | Not a win. Repeat flattering results as carefully as alarming ones |
| `route_compiled.hh`'s prediction that one flag "becomes true for every backend" | Too wide in type (only float moved) and in backend (the facade gate was guarded on `Backend::CUDA`) | Replaced by the scalar `select::level3_tile_route_available<B, T>` |
| Compile-time coverage gate | `resolve_route` was an inline template; weak-symbol interposition let an uninstrumented copy replace the library's, giving a file with a correct header and zero `reached` rows | Runtime bool in exactly one TU (`src/select/coverage.cc`). `cmake/BatchLASOptions.cmake:141` records that no build option exists |

## Correctness findings {#dispatch-correctness-findings}

Wrong answers found, how they hid, and what guards them now.

* **`ormqr` buffer size and call disagreed by 108x.** A forced `cta`, `two_stage` or `jacobi` matched no branch, so
  the call ran on the vendor while the size query returned the blocked size: 2560 bytes against the 276480 the call
  demanded. Fixed by construction: an op and its `*_buffer_size` run the same `choose()`.
* **`vendor` must be one fieldless family.** Route-era `is_vendor()` tests let a forced cuBLASDx request answer yes
  to "did the caller ask for the vendor?". cuBLASDx is deleted; `vendor` is the only family that calls the vendor
  library.
* **The order-walk fallback inverted GEMM's default.** Taking the first merely supported route picked Native, moving
  an 8x8x8 batch-1 GEMM from vendor to native. gemm has no order walk now; its last resort (`direct`, then `vendor`)
  applies only when no table entry can run.
* **Two ROCm defects invisible to the CUDA build.** `scripts/rocm_syntax_check.sh` caught a `trsm` instantiation in
  the old parameter order and four orphaned macro-continuation lines. Its gate is "exactly one expected error" (a
  `get_native<ext_oneapi_hip>` overload this CUDA-only DPC++ lacks); any other diagnostic is a real defect.
* **Signature divergence between vendor TUs.** `trsm`'s vendor form takes `alpha` last; the public form takes it
  third. Generating facade bodies from the public declarations would have passed `alpha` where `side` was expected
  on every backend. Each `vendor` family now spells its `backend::<op>_vendor<B, T>` call by hand, so the hazard is
  one call site per op.
* **An instantiation binds as hard as a definition.** `syev` and `ormqr` were defined in headers, but their
  instantiations lived in `cusolver.cc`/`cublas.cc`, so they vanished from builds without those libraries.
  `scripts/facade_symbol_check.sh` checks symbols with Itanium mangling, because `nm -C` fails to demangle
  concept-constrained templates.
* **The SYRK 128-wide tile wrote nothing to some elements.** Thread-tile granularity clashed with the triangle split,
  and the bug failed only at n > 64. `SyrkTest.NarrowShapesMatchGemmReference` sweeps n in {24, 32, 48, 64, 96, 128}
  x trans x uplo at k = 200.
* **A `herk` test that could not fail.** Checking the unreferenced triangle and the `uplo` agreement misses
  conjugating the wrong operand, which returns a still-Hermitian `conj(C)`. `MatchesGemmReference` catches it.
* **An unexercised branch.** The `sytrd_blocked` trailing update runs only for trailing blocks wider than 128, and no
  case covered it. `SytrdBlockedTest.TrailingUpdateRoutesAgree` (n=320, nb=32) fails at worst eigenvalue error 2.777
  when `alpha = -0.5` is forced. Its backward-error bound (`4 n eps ||A||`) is about 1000x looser than either route's
  error, so the load-bearing check is the relative one.
* **A guard that modelled half its predicate.** `sytrd_blocked`'s her2k guard copied only the size ceiling. Both
  halves now live in `her2k_takes_gemm_route` (`src/expansion_budget.hh`), which the call site asks
  (`src/extensions/sytrd_blocked.cc:838`).

### The ormqr chooser's forced pin {#dispatch-the-ormqr-chooser-that-forced-past-supports}

The route-era `choose_ormqr_provider` returned a forced provider without checking it against
`ormqr_supports_blocked`. Two defects followed: a forced `blocked` ran on complex `Trans` and on non-GPU queues, and
the buffer size and the call could disagree (see the correctness finding above). Flat selection prevents both: a pin
that fails `can_run` throws, and `ormqr_buffer_size` runs the same `select::pick` as the call.

`ormqr` has no measured window. Its tables rank `blocked`, then `vendor`, in every row where `blocked` can run, and
`vendor` alone on the complex `trans=T` rows, where `blocked`'s `can_run` refuses (`src/ops/ormqr/choice.hh:14`).

## Buffer-size queries {#dispatch-buffer-size-queries-and-the-route-they-size}

Every `<op>_buffer_size` runs the same `select::pick` as its op and returns exactly the chosen family's need, through
the op's `workspace()` visit in `src/ops/<op>/<op>.cc`. A nested op adds its children's sizes by calling their public
sizing functions. Query and call agree because they run the same choice, not because the query over-allocates.

1. **The vendor path sizes itself from the vendor query.** `potrf_vendor` in `src/backends/cusolver.cc` calls
   `backend::potrf_vendor_buffer_size`. An unqualified `potrf_buffer_size` resolved to the facade query, which returns
   the max of native and vendor sizes and so hands cuSOLVER the wrong workspace size.
2. **Sizing is pure.** `band_reduction.cc` and `sytrd_sy2sb.cc` size `geqrf` once against a dummy view and use
   `geqrf_buffer_size_bound` (the maximum over every candidate the device can run, `src/ops/geqrf/geqrf.cc`). No
   `geqrf` sizer may dereference `A.data_ptr()` or `tau.data()`, which are `nullptr` there. `getri_buffer_size` runs
   under `BumpAllocator::measuring()`, so everything reachable from it must be pure with respect to the workspace.

## Coverage instrument {#dispatch-the-coverage-instrument}

The instrument (`src/select/coverage.hh`) has two tables and a `miss` row for calls nothing could serve.

- **static** (`linked`): per op and backend, the vendor gate and whether a native kernel is linked. No kernel runs.
  Answers "is the kernel in the build".
- **dynamic** (`reached`): counts `(op, scalar, backend, shape_class)` and records the chosen family spelling and
  `native_route_existed` / `native_route_supported`. Answers "did a call get there".

Linked is not reachable: a symbol being present is never evidence that it runs. `native_route_supported` is a
tri-state (`1` yes, `0` no, `-1` the call site could not tell). Every op records its row from `select::TraceScope`,
with the choice spelling as the algorithm. `uplo`, `side`, `diag` and `transA` are part of the level-3 key.

`scripts/route_diff.sh capture|compare` is the only tool that sees vendor-to-vendor route changes. It treats a capture
with zero `reached` rows as a hard error. `scripts/coverage_merge.sh` merges the per-PID shards of a `ctest` run.

### Instrument defects {#instrument-defects}

Five defects each made the instrument look healthy while it reported almost nothing:

1. The gate-declined half was unrecorded, so a shape moving off a native kernel was invisible.
2. `uplo`, `side` and `diag` were missing from the key, so calls differing only in `uplo` shared one row.
3. `emit()` opened the file with `"w"`, so each of 53 test binaries truncated the last.
4. The compile-time gate and weak-symbol interposition (see [negative results](#dispatch-negative-results)).
5. `route_diff.sh compare` applied no `backend != AUTO` filter, so pure-layer test shapes made a clean 65-decision move
   look like 240 lines of churn.

## Vendor-free baseline

`cmake -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` gives a CUDA device with every
CUDA math library at 0. It configures, compiles, links, loads and runs. It is not green: the failures are missing
kernels (`NoRouteError`), not dispatch faults.

| milestone | `ctest -LE slow` | note |
|---|---|---|
| WP0 | 20 / 53 | failures are `NoRouteError`, not crashes or link errors |
| WP1 | 24 / 53 | `gemm_tests` 48 to 167 of 184 |
| WP2 correctness track | 25 / 53 | `gemm_tests` 184 / 184 |
| WP8 (latest recorded) | 35 / 57 | the failing **set** is the reviewable artefact, not the count |

Later, the level-3 wave added vendor-free routes for float syr2k at batch 1, double symm and syrk `n <= 128`, and
every trmm dtype on both sides (flat-kernel-selection.md §12).

The four level-3 TUs live in `BACKEND_COMMON_SOURCES` (`src/backends/CMakeLists.txt:136-141`). The CMake gate cannot
be relaxed by deleting `if(BATCHLAS_HAS_CUBLAS)`, because `BACKEND_CUDA_SOURCES` feeds an object library that is not
created when no CUDA math library is present.

## Open debts

1. **`syr2k` has no non-float native route.** `syr2k_triangular_tiles` is float only.
2. **hemm, herk and her2k have no tables and no `BATCHLAS_<OP>_ROUTE`.** Vendor-free, they throw `NoRouteError`.
3. **The level-3 tables are transcribed and untimed.** The phase-4 retune measures them.
4. **Unverified windows**, each a transcribed row edge with no bracketing grid: the 257 <= n <= 383 band that
   `tuned/syrk.float.*` ranks `triangular` first; the batch-3 and 128 < n < 256 `form=sq` cells that
   `tuned/symm.float.*` ranks `vendor` first (and the same constants in hemm's `expansion_preferred`); and the old
   squareish 2:1 aspect ratio (the `form=sq` edge of the syrk table).
5. **A coverage row cannot confirm that a particular shape ran.** Rows are keyed on a power-of-two `shape_class` and
   are first-writer-wins, so the m/n/k/batch columns can describe a different call. Prove a shape with a break that is
   red only for it.
6. **`Backend::INTEL` is hard-wired FALSE**, so oneMKL cannot be tested here.
7. **`symm_benchmark`, `syrk_benchmark` and `syr2k_benchmark` abort before printing** at tiny shapes on the host
   backend (`adjustNDRangePerKernel: NDR.LocalSize[0] == 0`). Pre-existing; not a dispatch regression.

## Raw evidence

Raw data is kept at the git tag `perf-evidence/vendor-independence`: `git show <tag>:<path>`.

| topic | path |
|---|---|
| Gram/tile kernel design, all dtype tables, the trmm routing mistake, the 16-row tile | `experiments/TRMM_SYRK_BATCHED_KERNELS.md` |
| Which level-3 op can replace a GEMM in `src/extensions`, and the three rejections | `experiments/GEMM_TO_LEVEL3_SURVEY.md` |
| SYRK triangular-route crossover sweep (n x batch) | `experiments/syrk_sweep.sh` |
| SYRK crossover with k free | `experiments/syrk_kskew.sh` |
| SYR2K triangular-route crossover sweep (n x k x batch) | `experiments/syr2k_sweep.sh` |
| herk/her2k GEMM-route vs per-batch-loop crossover | `experiments/herk_crossover.sh` |
| the dedicated-GPU harness every sweep ran under | `experiments/gpu_guard.sh` |

Superseded root documents at the same tag: `WP0_DISPATCH_SPEC.md`, `WP1_LEVEL3_SPEC.md`, `VENDOR_FREE_BASELINE.md`
and `VENDOR_INDEPENDENCE_PLAN.md`.

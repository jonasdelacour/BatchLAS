# LU: getrf, getrs, getri, gesv {#perf_lu}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · measured 2026-09-10 to 2026-10-23

Routing rules and the measurements behind them for native `getrf` (`tiny`, `cta`, `blocked`), `getrs` (`cta`, `blocked`), `getri` (`blocked`), `gesv` (`tiny`, `blocked`) and their shared row-interchange kernels. Machine: RTX 4090 (128 SMs, 97,280 B local memory per work-group, ~1008 GB/s). Every timed row was checked against a host oracle with its resolved route read back. Older evidence uses retired spellings (`native:cta`, `vendor:auto`); they are not valid pins.

## Selection

* **Families** (`src/ops/<op>/choice.hh`, tie-break order): getrf `tiny`, `cta`, `blocked`, `vendor`; getrs `cta` (fused narrow-RHS), `blocked` (composition), `vendor`; getri `blocked`, `vendor`; gesv `tiny`, `blocked`. None has fields.
* **Correctness** is one `can_run` per op in `src/ops/<op>/<op>.cc`, the only gate; it never carries a speed term. Every family refuses a batch mask.
* **Speed** is the first runnable entry of the nearest row of `tuned/<op>.<device>.txt`. Keys: getrf and getri `n:log:3 batch:log` (getrf's `n` is `A.rows()`); getrs `n:log:2 nrhs:log batch:log`; gesv `n:log:3 nrhs:log`. The sm_89 tables are `transcribed` at `424a45bc` (the old routers' order per cell, **untimed**). The sm_120 tables are deep-measured, with 45 / 94 / 790 `transcribed` rows left in getrf / getri / getrs ([tuned/README.md](../../tuned/README.md)).
* **Sub-op seams**: the blocked drivers call the public `gemm` and `trsm` (@ref selection_tables); an empty seam throws.
* **Pins**: `BATCHLAS_{GETRF,GETRS,GETRI,GESV}_ROUTE` take `auto`, `native`, `vendor` or a family spelling. A spelling that does not parse, is not compiled, or is refused by `can_run` throws. `native` takes the best runnable non-vendor entry. `vendor` with no runnable vendor family (vendor-free builds; always for gesv) warns once and runs Auto.
* **Derived, not chosen**: the panel leaf (`BATCHLAS_GETRF_LEAF`), the interchange spellings (`BATCHLAS_GETRF_LASWP`, `BATCHLAS_GETRF_RIGHT_LASWP`, `BATCHLAS_GETRS_LASWP`), `nb`, the blocked driver's batch parts, the tiny bucket and launch bound. Launchers derive them; they are not families, keys or `can_run` terms.

### LU: families and their limits

The common native term is `B != Backend::NETLIB && d.is_gpu && d.has_sg32`, a homogeneous batch, `n, batch >= 1`, and a square A (getrs: a conforming pair; getrf: any m x n, see [non-square](#getrf-non-square-m-x-n)).

| op | family | driver | `can_run` beyond the common term |
|---|---|---|---|
| `getrf` | `tiny` | `getrf_tiny.cc`, register-resident | square, `n <= getrf_tiny_max_n<T>()` (32; cdouble 16), `max_wg >= 64` |
| `getrf` | `cta` | `getrf_cta.cc`, local-memory ?GETF2 | square: `n <= getrf_cta_max_n_for_slm<T>(budget)`; m x n: `getrf_cta_fits_rect`; both occupancy-scaled |
| `getrf` | `blocked` | `getrf_blocked.cc`, right-looking, register panel leaf | `getrf_blocked_available<T>()`, `getrf_cta_max_n_for_slm<T>(budget, 1) >= 1`; any m x n |
| `getrf` | `vendor` | cuBLAS, rocSOLVER, LAPACKE | `d.has_vendor`, homogeneous; on CUDA square only (cuBLAS `getrfBatched`) |
| `getrs` | `cta` | `getrs_fused.cc` | `nrhs <= kGetrsFusedMaxRhs` (8), `n * nrhs <= getrs_fused_max_rhs_elems<T>(budget)`, `max_wg >= 32` |
| `getrs` | `blocked` | `getrs_native.cc`, permutation then two public `trsm` | `getrs_blocked_available<T>()` |
| `getrs` | `vendor` | cuBLAS, rocSOLVER, LAPACKE | `d.has_vendor`, homogeneous A and B |
| `getri` | `blocked` | `getri_blocked.cc`, P into C, two public `trsm`, no workspace | `getri_blocked_available<T>()` |
| `getri` | `vendor` | cuBLAS, rocSOLVER, LAPACKE | `factorization_vendor_available<B>`, homogeneous |
| `gesv` | `tiny` | `gesv_tiny.cc`, fused factor and solve | `n <= gesv_tiny_max_n<T>()` (32; cdouble 16), `nrhs <= 4`, `max_wg >= 64` |
| `gesv` | `blocked` | public `getrf` then public `getrs` | homogeneous A and B |

Capacities are asked of the runtime local-memory budget. On this box `can_run(cta)` admits **77 / 54 / 54 / 38** (float / double / cfloat / cdouble), occupancy-scaled from 155 / 109 / 109 / 77 ([occupancy rule](#lu-the-occupancy-rule)). `getrf` uses `nb = 32`.

### LU: what the tables rank first

sm_89 rows, pinned by the `*TranscribedTable.*` cases and `GetrfCandidates.AutoReadsTheShippedTablesOnTheRealDevice`:

| op | native first | vendor first (native order behind it) |
|---|---|---|
| getrf float | `tiny` at n = 5..32; `blocked` at n >= 256 | n <= 4 (`vendor tiny cta blocked`); 33..255 (`vendor cta blocked`) |
| getrf cfloat | `tiny` at 5..7, 9..24; `blocked` at n >= 512, and 256..511 at batch >= 256 | n <= 4, 8, 25..32; 33..255; 256..511 below batch 256 |
| getrf double | none | `vendor cta tiny blocked` n <= 32; `vendor blocked cta` n >= 33 |
| getrf cdouble | none | `vendor cta tiny blocked` n <= 16; `vendor cta blocked` above |
| getrs | `cta` at n >= 32 for nrhs <= 2 (every type) and float nrhs 3..4; `blocked` for float nrhs >= 64, double nrhs >= 128, at batch >= 128 | everything else, `vendor cta blocked` |
| getri | `blocked` at float n >= 128, cfloat n >= 256 (every batch row) | double, cdouble, and smaller orders |
| gesv | `tiny` at float n <= 32, cfloat n <= 16 (`tiny blocked`) | no vendor; `blocked` elsewhere and for fp64 |

A tiny-first gesv row with nrhs > 4 runs `blocked` (`can_run(tiny)` refuses it).

### LU: the order among native families

A vendor-free build or a `native` pin takes the first runnable native entry. **getrf**: double ranks `blocked` before `cta` from n = 33; every other type ranks `cta` first wherever `blocked` is not. `blocked_ms / cta_ms` at batch 4096-8192:

| type | n=64 | n=76 | n=96 | n=128 |
|---|---:|---:|---:|---:|
| float | 1.74 | 1.48 | 1.49 | 1.13 |
| cfloat | 1.39 | 1.59 | 1.30 | -- |
| cdouble | 1.37 | 1.09 | -- | -- |
| **double** | **0.98** | **0.85** | **0.77** | -- |

At double n <= 32, `nb = n` and the blocked driver is one CTA-leaf panel (1.8126 vs 1.8113 ms), so CTA first costs nothing.

**getrs**: fused `cta` precedes `blocked` outside clause C; it leads at all 51 cells inside its capability (worst 1.11x). The crossover is nrhs = 16 (double 0.55x), outside `can_run(cta)`. **Raising `kGetrsFusedMaxRhs` requires the getrs rows at nrhs >= 16 to rank `blocked` before `cta` in the same change.**

## Measured boundaries {#lu-measured-boundaries}

Ratio is `vendor_med / native_med` (> 1: native wins) against `cublas?getr{f,s,i}Batched`; acceptance gate zero losses and no cell below 1.15 (tiny windows: 1.11). cuBLAS does not saturate at n >= 1024 (`cublasZgetrfBatched` n=2048: 2587.7 ms at batch 4, 2594.1 at 32), so large-n ratios are quoted at fixed batch. cdouble `getrf` already runs at 90-91% of FP64 peak at n = 512-1024.

### `getrf` window evidence

`experiments/wp8_getri/lu_c1.csv` (11 reps, idle box): **15 cells, geomean 1.997, min 1.2626**. Float n=256 reads 1.254-1.675 over batch 128-1024; n=2048 b128 3.091.

| clause | bracketing non-winner |
|---|---|
| float `n >= 256` | `n >= 128`: 0.825 / 0.773 / 0.872 at batch 256 / 512 / 1024 |
| cfloat `n >= 512` | -- (batch-free; cells at n >= 512: 1.528-2.754) |
| cfloat `256 <= n < 512 && batch >= 256` | batch 128: 0.922, batch 64: 0.993; batch 192 is the first winner (1.135). n = 192: 0.86-1.10 (`benchmarks/results/p4_cfloat256_batch_ladder.csv`) |
| double, cdouble: none | double `n >= 512` admits 0.7486; best cells anywhere 1.067 / 1.012 |

The cfloat 256 band exists because of the register leaf; the local-memory leaf read 0.885 there.

### `getri` window evidence

`lu_c1.csv` and `lu_p1.csv`, worse of two passes: **30 cells, geomean 4.181, min 1.2028**, batch 1 to 16,384. Brackets: float n=64 loses at 0.850-0.857 (batch 8192-16384); cfloat n=128 0.707-0.711 (batch 512); double has no clean band (n=2048 b128 1.08); cdouble n=512 loses 0.954. No batch floor: at batch 1-32 native beats cuBLAS's per-item loop by 1.7-28x for every type.

### `getrs` fused window evidence

Clauses A and B, from `experiments/wp6_perf/bench/` (461 pooled cells, relsd <= 10%):

| clause | cells | geomean | min | losses |
|---|---:|---:|---:|---:|
| A: `nrhs <= 2`, every type | 286 | 2.261 | 1.116 | 0 |
| B: float, `nrhs = 3..4` | 36 | 1.611 | 1.133 | 0 |

Rejected widenings: float `nrhs <= 8` (worst 0.686), `nrhs <= 4` every type (worst 0.577, cdouble n=32), the whole capability (worst 0.294). The grid's smallest order was 32; see the floor below.

### `getrs` order floor evidence

Below n = 32, clauses A and B lose at every type, worst **0.234** (cdouble n=4 nrhs=1 batch 32768), deepening with batch: one work-group per few-dozen-flop solve. Worst rung per order, nrhs = 1:

| T | n=4 | n=8 | n=16 | n=17 | n=24 | n=32 |
|---|---:|---:|---:|---:|---:|---:|
| float | 0.71 | 1.12 | 1.04 | 0.76 | 0.95 | 2.29 |
| cfloat | 0.59 | 0.69 | 3.76 | 1.38 | 1.27 | 1.40 |
| double | 0.52 | 0.98 | 0.99 | 0.87 | 3.66 | 3.77 |
| cdouble | 0.23 | 0.41 | 2.00 | 2.02 | 1.93 | 2.13 |

32 is the first order where all four types clear and stay clear; below it the band is non-monotone. Source: `benchmarks/results/factor_baseline_getrs_*.csv`. This is the worst rung, not `small-n-baseline.md`'s top-batch statistic.

### `getrs` composition window evidence

Clause C (float nrhs >= 64, double nrhs >= 128, batch >= 128), `experiments/wp8_getrs/` re-measured in `lu_c1.csv`: **37 cells, geomean 2.60, min 1.2858**.

| rejected clause | refuting cell | ratio |
|---|---|---:|
| float `nrhs >= 32` | n=64 nrhs=32 b=4096 | 0.9069 |
| double `nrhs >= 64` | n=64 nrhs=64 b=2048 | 0.9984 |
| cfloat `nrhs >= 128` | n=64 nrhs=128 b=1024 | 0.9944 |
| cdouble `nrhs >= 64` | n=128, 12 losses of 26 cells | 0.69-0.83 |

The batch floor of 128 is conservative (float nrhs=128 wins 3.87-5.93x at batch 32-64); below 32 only contaminated readings exist.

## The `laswp` gather

The row interchange is the family's largest lever: a timing-only break priced it at 74% of float `getrf` n=512 b512 and 44% at n=2048 b32 (22% cdouble). `ipiv` is ordered transpositions, the selected rows are scattered, and the walk moves one 32 B sector per 4 B element.

### `getrf` deferred left gather

`lu_laswp_deferred_left_launch` (`lu_laswp.hh`), unconditional: deferring the left pass out of the block loop makes the staged rows equal the transpositions, so no gate is needed. The three `BATCHLAS_GETRF_LASWP` spellings are bit-identical (`LuTest.LeftInterchangeSpellingsAgreeBitForBit`). Against the in-loop walk, 58 blocked cells at batch >= 128: geomean 1.207x, min 1.018x. A tall A (m > n) takes the deferred walk instead (see [non-square](#getrf-non-square-m-x-n)).

### `getrs` collapsed permutation

`GetrsPermGatherKernel` (`getrs_native.cc`), default at `nrhs >= kGetrsPermGatherMinNrhs = 16`: one work-group per item stages a column tile and the index array, then writes back in place; no workspace. Walk / gather: nrhs 8 geomean 1.0011, **nrhs 16 1.1182** (min 1.0004), nrhs 128 1.5728. `getri` needs no permutation kernel: each work-item traces its row through the interchange list.

### The fused narrow-RHS `getrs`

`getrs_fused.cc`: one work-group per matrix, one launch for permutation and both triangular solves. Against the composition 8.26x at nrhs=1 down to 2.67x at 8, worst 1.419 over 107 cells. Shipped choices: resident diagonal block (0.65 vs 1.01 ms streaming), `nb = 16` below n=1024 and 32 above (n=2048: 1.55 / 1.38 / **1.28** ms at nb 8 / 16 / 32), work-group ~n/2 clamped to [64, 1024] and capped per (type, body, width) by `resident::sm89_max_work_group`.

## LU: L2 batch chunking {#lu-l2-batch-chunking}

The `blocked` driver hands the trailing matrix between its per-panel kernels through L2; once `batch x m n x sizeof(T)` exceeds L2 those hand-offs go to DRAM. `getrf_blocked_chunk` (`getrf_blocked.cc`) therefore factors the batch in parts, each a complete factorisation with pivot (`b0 * min(m, n)`) and `info` offsets.

* **Part size**: `tuning::blocked_chunk_items`: `BLOCKED_L2_CHUNK_PERCENT` = **75**% of `global_mem_cache_size` in items, rounded up to a multiple of `max_compute_units` (at least 1x). Below two chunks: one pass; otherwise parts of one chunk, the last taking the remainder. At n >= 256 the round-up sets the chunk (sm_120 float: n=64 6204, n=128 1692, n=256 564, n=512 188 items).
* **Never split** at `n <= nb`. **Workspace** is one part's. **Overrides**: `BATCHLAS_TUNE_L2_CHUNK_PERCENT`, `BATCHLAS_TUNE_L2_CHUNK_ITEMS`.

Evidence, sm_120 (RTX PRO 6000 Max-Q, 128 MiB L2, 2026-10-21), float pinned blocked, cold and warm agree: geomean **0.932** over 31 cells (n 32-512), worst 1.004; n=128 b32768 459.6 -> 383.0 ns (0.833). Percent 75 vs 100: getrf 0.901 vs 0.909, potrf 0.930 vs 0.949. sm_89 (72 MB L2, 2026-10-23): float n=128 b16384 0.787, n=256 b2048 0.785; 150% worse in every cell. Two rules the sweep forced: equal parts below one chunk lose (n=128 b2048 +17%), and a part one item past a CU multiple pays for the next (potrf n=512 parts of 391: +6%, of 376: -10%).

## LU: negative results

| Alternative | Result | Verdict |
|---|---|---|
| Fuse the blocked driver's 5P-4 launches | 16 launches at n=128 = 0.12% of the call. | Rejected. |
| `getrs` wide-nrhs window from one saturating batch | Advantage falls with batch; best candidate min 1.0436. | Rejected. |
| Defer the interchange, keep the per-column walk | geomean 1.055x, three losing cells; 15x less parallelism at n=512. | Rejected. |
| Packed sub-group `getrf` (one sub-group per matrix) | float 1.13-1.38 at n=16..32, 0.79-0.31 at 48..128; fp64 loses everywhere. | Refuted. |
| Division-free (row, column) split in the rank-1 update | float 0.936x, cdouble 0.976x. | Reverted. |
| `getrf_leaf_wg` raised to 1024 | geomean 0.974x, worst 0.814x. | Reverted. |
| `sycl::reduce_over_group` for the pivot search | 1.5-4.7x slower for fp64; causes the 48 KB hole. | Rejected. |
| `getri` zero-fill as `range<3>` | 0.874x. | Reverted. |
| Complex `Tiled16` GEMM for trailing updates | Complex already reaches the register GEMM. | Refuted. |
| One 64-bit shuffle per complex element | Identical to two 32-bit (NVPTX has no 64-bit SHFL). | No gain. |

Traps: a harness that shrinks the batch cannot ask the routing question; the two 4090s are not independent (a concurrent sweep on the other GPU tripled `getrf` times), so serialise the box; rebuild bench binaries after a table or `can_run` change; read the resolved route per arm; `BATCHLAS_GETRS_ROUTE=native` means the first runnable non-vendor entry.

## The shape of the GETRF device code

`getf2_panel_device` (`getrf_cta_device.hh`) is ?GETF2 over a `Tile` with `at(r, c)`, instantiated on a `local_accessor` (CTA tier) and a global pointer (blocked leaf). Load-bearing spellings: the pivot search is a sub-group butterfly plus a scan over `kLuRedSlots = 32` slots, never `reduce_over_group`; the metric is LAPACK's `cabs1`; `info` is exact-zero, 1-based, global, first-failure-wins, with no epsilon floor; barriers B1..B4 sit at the top of the kernel-uniform `k` loop; `ld` is padded odd so a row exchange does not hit one bank.

## LU: correctness findings

* The `info` zero-fill raced the panel on an out-of-order queue; fixed with `if (!ctx.in_order()) ctx.wait();` (`LuTest.InfoFillIsOrderedAheadOfThePanelOnAnOutOfOrderQueue`).
* Native kernels write packed 1-based int32 pivots; NETLIB writes int64. Every native `can_run` carries `B != Backend::NETLIB`, else NETLIB on a GPU queue returned a wrong inverse with `info == 0`.
* cuBLAS pivots complex on the modulus, LAPACK and this library on `cabs1`; elementwise native-vs-vendor pivot comparison is wrong for complex (`PivotSelectionUsesCabs1AndNotTheModulus`).
* `info` is structural: non-zero exactly when `|U(i,i)|` is a binary zero.
* `getrs_fused_max_rhs_elems` must not floor-divide: `getrs_hole_padded` is not monotone.
* The fused getrs register cap fired at float n=2048 nrhs=8 Trans (wg 1024 x 68 registers); `FusedGetrsLaunchHoleAt48KiB` covers it.

### Blind guards and what made them blind

1. A diagonally dominant matrix makes every pivot test vacuous: row-permute each item randomly.
2. An inverse test on a self-inverse instance is vacuous: use an n-cycle.
3. A probe that never asserts passed a 1.2e-01 residual: every criterion carries a `Tol<T>`.
4. A capability helper that never set the fused capacities made every getrs routing assertion hold.
5. `max |L| <= 1` is wrong for complex: `cabs1` allows sqrt(2).
6. `break.py` anchors must match exactly once.

The cabs1-to-modulus break is green on random matrices; it needs a purpose-built probe.

## LU: the 48 KB launch hole

A `PAD=` knob: **49,024 B pass / 49,152 B fail / 49,280 B pass**, deterministic at every work-group width, inside the band `(47104, 49664]`. Cause: `reduce_over_group` allocates local memory the accessor accounting cannot see. The shipped kernels use no group collective; the pad in `getrf_cta.cc`, `lu_laswp.hh` and `getrs_native.cc` is defensive.

## Open debts

* The sm_120 `blocked` rows of getrf and potrf predate L2 chunking; retune float and cfloat.
* The sm_89 tables are untimed transcriptions; the getrf blocked window has no measured batch term below 128, and getri at batch 1-64 is unmeasured for fp64.
* `getrs` clause C: batch floor 128 gives up wins at 32-64; unmeasured above n = 1024; 84 winning A/B cells go to the vendor (largest 3.944x, double n=1024 nrhs=4 b256).
* The collapsed getrs gather is parallel over batch only; the fused getrs permutation is 8% of the call at n=2048.
* The right-hand `getrf` interchange is the per-column walk above the gate in [the right-hand gather](#the-right-hand-gather).
* `can_run(cta)` asks the occupancy-scaled ceiling, so float 96-129, cfloat 64-96 and cdouble 48-65 cannot reach CTA although CTA beats blocked there (1.39-1.49x); undoing it needs a grid between the two ceilings. The committed `factor_baseline_getrf_*.csv` predate this.
* Float `n = 65` takes CTA in a vendor-free build, where blocked is 1.62x faster (n = 64: CTA by 1.13); the grid has no cell between 64 and 96.
* Float getrf at 33..255 is a batch question (n=40: 0.68 at batch 32, 1.22 at 1024); the rows rank `vendor` first at every batch.
* Tall non-square `getrf` that fits the CTA tile takes `cta`, 1.9-4.6x slower than blocked at batch 1024; needs a `form` key and a retune.
* cfloat gesv at n <= 4 reverses past batch 65536 (n=4 nrhs=4: 0.85 at 131072); the gesv table has no batch key.
* Register-leaf double n=32 loses 0.89-0.93 to the local-memory leaf (Auto never routes there); G-packing for the register leaf is owed.
* `nb = 32` is untuned; `getrf_panel_factorize` re-queries device properties per block step; `MatrixView::data_ptrs(ctx)` re-runs `init_data_ptr_array` on every call; `getrf_panel_reg.cc` duplicates `resident::sm89_max_work_group`'s gate.
* `Backend::NETLIB` on a GPU queue is gated but not exercised end to end.

## Raw evidence

Older `experiments/` paths live at the tag `perf-evidence/vendor-independence` (`git show perf-evidence/vendor-independence:<path>`): `wp6_lu/` (baseline, tier sweep, laswp cost, test breaks), `wp6_perf/` (getrs clauses A/B), `wp8_getrf/` (deferred gather), `wp8_getrs/` (permutation gather, clause C), `wp8_getri/` (`lu_c1.csv`, the clean re-measure the windows are scored from). Later rows are in `benchmarks/results/` (`p4_*`, `p2_gesv_*`, `p8_getrf_tiny_window.csv`), produced by `benchmarks/factor_bench.cc`.

> **Warning:** plain `blocked` takes the default leaf, `reg`. A/B the leaves as `blocked/leaf=slm,blocked/leaf=reg`, or the register leaf is measured twice at 1.000.

## LU: the occupancy rule

`getrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` and `getrf_cta_fits` divide the budget by an occupancy target (default 4, `resident::kMinBlocksPerSm`) before testing the footprint: 77 / 54 / 54 / 38 against 155 / 109 / 109 / 77 at target 1, both pinned in `tests/resident_capacity_tests.cc`. The walk is shared with potrf in `src/util/resident_capacity.hh`.

### One spelling per ceiling

* `getrf_cta_max_n_for_slm` takes the runtime budget; 0 means the tier is absent. `kGetrfReferenceSlmBudget = 97280` serves only convenience overloads; the generated `device_limits.hh` value (49,152) is wrong here and is on no decision path ([potrf budget](potrf.md#the-slm-budget-and-the-fit-ceilings)).
* `getrf_cta_fits` is admission, occupancy-scaled; `getrf_leaf_fits` is residency, at the whole budget.
* `getrf_tiny_max_n<T>()` is the one spelling of the tiny ceiling (compile-time, {4, 8, 16, 32} ladder).
* `getrf_tiny_buffer_size` is not zero: a short caller `info` draws pool scratch.
* `getrf_cta_debug_launch<T>(ctx, m, n)` packs G (matrices per work-group) low and work-group width high, 0 when not resident. `getrf_blocked_debug_params<T>(ctx, n)` packs nb low and the leading panel's leaf high (1 = local, 2 = global).

### LU: the panel leaf is not the tier ceiling

`getrf_leaf_fits` is not occupancy-scaled: at float n=512, nb=32 the 65,664 B panel is resident at 97,280 B but not at 24,320 B, and the n >= 256 wins run through it. The launcher's `internal_error` asserts advertised implies resident.

### Packed resident leaves

`getf2_panel_device` is templated on `LuScope`. Under `SubGroup` all barriers are sub-group scope and `getrf_leaf_launch` packs one matrix per sub-group, for `m, n <= 32` only; under `WorkGroup`, `G > 1` would race, so the launcher derives both together. After/before 0.52-0.80 in every probed cell (double n=16 b32768 1.49 -> 0.78 ms, vendor 0.41). The win is barrier scope and matrices in flight, not registers.

## LU: the tiny tier

`getrf_tiny.cc`: one matrix per `SubGroupPartition<N>`, N in {4, 8, 16, 32}; lane `r` owns row `r` of `D rA[NC]`; two sub-groups per 64-wide work-group. Pad rows and columns carry the identity, so the unrolled body needs no lane guard. Pivoting is a lazy relabel (`rowid`); metric `cabs1`; `ipiv` 1-based global; elimination continues past a zero pivot. Zero barriers of work-group scope; the 48 KB hole is unreachable.

**Window** (`benchmarks/results/p8_getrf_tiny_window.csv` and later passes, batch 8192-131072, gate 1.11, `t_vendor / t_tiny`):

| edge | in | bracketing out |
|---|---|---|
| float low | n = 5: 1.46 | n = 4: 0.888 at batch 131072 |
| float high | n = 32: 1.29 (tier ceiling; 33 refused) | structural |
| cfloat low | n = 5: 1.15, n = 9: 1.99 | n = 4: 0.833 at 65536; n = 8: 1.086 at 131072 |
| cfloat high | n = 24: 1.16 | n = 28: 0.99, n = 32: 0.96 |

fp64 gets no window (double n=4 0.72, cdouble 0.68; 1:64 FP64 part).

### The register probe, and the unroll that decides it

`scripts/register_probe.sh out.log '' batchlas_extensions_cta`. Worst `kWorstRegsPerThread = 143` (cfloat N=32), static-asserted; every entry frame 0, spill 0. Two spellings silently move `rA` to the stack with zero spill: `if (j >= n) break;` inside `#pragma unroll` (use `continue`), and `for (k = 0; k < N; ++k) { if (k <= j) continue; }` (start at `k = j + 1`). Stack frame, not spill, is the residency gate.

### Pad rows and the argmax, corrected

The pad-row mask in the pivot candidate set is not load-bearing; the tie-break toward the lowest `rowid` is, since `rowid == j` is always live. The argmax carries a (magnitude, key) pair, exact for double. The NaN map is required, or lanes disagree about the winner.

### The pivot key encoding

`tiny_key(order_field, lane)` in `tiny_device.hh` is the only encoding: the ordering field (high bits) breaks ties, lowest wins; the lane (low `kTinyKeyLaneBits = 8` bits) is only the broadcast source. Ordering fields must be distinct within a partition (non-candidates seed at `rowid + N`).

### Why the rank-1 update starts at `k = j + 1`

It is a compile-time bound once `j` is unrolled (see [the register probe](#the-register-probe-and-the-unroll-that-decides-it)). The store guard is `lane < n`. The tier writes `info` once and never reads it, so it cannot be a blocked panel leaf as is.

### Armed breaks

Red: `rowid > j` -> `>=`; dropping the store's pad guard (red only at `ld = n+5`); reversing the argmax comparison (`ipiv` red, residual green); `tiny_partition_id` -> group id; `cabs1` -> modulus (`TinyPivotsMatchLapacke` only). Green: dropping `rowid < n` from the candidate mask (the tie-break guards it); `continue` -> `break` (only the probe's frame column sees it).

### The work-group A/B

wg = 64 vs 128 over 30 cells: 26 within 2%, 2 each way. wg = 64 ships: shared constant, half the launch tail, 7.2x register-gate slack. The tier is bound by outstanding memory requests, not occupancy.

### The pivot-margin gate on elementwise comparisons

Device FMA and host arithmetic can pick different rows among candidates equal within 1 ulp (double n=10: `ipiv[6] = 7` vs LAPACKE's 9). `host_getf2_with_margins` records `cabs1(runner-up) / cabs1(winner)` per step, and elementwise pivot assertions stop at the first step within `kTinyAmbiguous` of 1.

### The host dgetrf oracle is broken on this box

`LAPACKE_dgetrf` is wrong for n >= 10 here (residual 0.42 at n=10) because this box's OpenBLAS dgemm is bad; sgetrf is fine. Tests use a BLAS-free triple-loop `getf2` as the primary oracle and LAPACKE only where it passes its own residual check.

### Why the tiny-vs-CTA element bound is relative

`TinyPaddingIsInertAgainstTheCtaRoute` compares pivots exactly and elements within `4 n` ulp of the value held; an absolute bound scaled by the diagonal would license the error a padding defect produces.

### R7: the device link, all three tiny tiers landed

`batchlas_extensions_cta` built in 167.4 s against 117-125 s before the tiny tiers (idle box). ptxas: `getrf_tiny` 19.2 s (10.5%), `geqrf_tiny` 19.4 s (10.6%), `potrf_tiny` 11.2 s (6.1%); together 27.2%, over the 15% budget. A fourth tier inherits that.

### The register probe at wg = 64

Register counts are unchanged from wg = 128; worst 143 x 64 = 9,152.

### D3: why cdouble stops at N=16

`rA[32]` of `complex<double>` is 128 registers before the working set (N=16 already uses 138), for a predicted 1.16x. Reopening it means changing `tiny_cap<T>()`.

## The register panel leaf

`getrf_panel_reg.cc` and `getrf_panel_reg_device.hh`: one work-group per `m x 32` panel, work-item `tid` owns row `tid` of `D rA[32]`, lazy relabel, pivot row published through local memory; a drop-in for `getrf_panel_factorize`. **Default leaf** (`BATCHLAS_GETRF_LEAF=slm` selects the local-memory panel); work-group `roundup(m, 32)`, height cap 512 (cdouble 256).

### Two barriers per column

Against four in `getf2_panel_device`. Removing either breaks a WAR one column later; dropping B2 is red from m = 33.

### The register panel leaf: register probe

Device link: float 64, double 105, cfloat 99, cdouble 176 registers, frame 0, spill 0. The device code needs `[[gnu::always_inline]]`; plain `inline` left an ABI call and a 200 B frame with zero spill.

### The register cap that binds is per sub-partition

sm_89's register file is four sub-partitions of 16,384, warps dealt round-robin, so the fit is

```
ceil(warps / 4) * 32 * ((regs + 7) & ~7) <= 16,384
```

i.e. `threads <= 128 * floor(512 / regs_alloc)`, not `regs x wg <= 65,536` (that allowed 288 cdouble lanes and the launch failed). The one spelling is `resident::sm89_max_work_group(regs)` (`src/util/resident_capacity.hh`). Size against the device-link count plus margin: a probe of 148 rounds to 170 and allows 384 lanes, but ptxas allocates 176, where 384 aborts. `RegPanelCapacityIsOneSpelling` launches at the cap.

### LU: the register leaf A/B

P4 grid (2026-09-11, 156 paired cells, `benchmarks/results/p4_leaf_getrf_*.csv`), `slm/reg` at the nominal `SAT_LADDER` batch (n 32/33/64: 8192, 128: 4096, 256: 2048, 512: 512, 1024: 512):

| n | float | cfloat | double | cdouble |
|---:|---:|---:|---:|---:|
| 32 | 2.13 | 1.48 | **0.93** | 1.82 |
| 64 | 1.31 | 1.16 | 1.47 | 1.34 |
| 256 | 1.11 | 1.08 | 1.29 | 1.14 |
| 1024 | 1.01 | 1.01 | 1.01 | -- |

Faster at 153 of 156 cells; the three losses are double n=32. The ratio is flat in batch and decays with n as the leaf becomes a minority of the call.

### Armed breaks (P4)

| Break (`getrf_panel_reg_device.hh`) | Red cell |
|---|---|
| drop `rowid >= j` from the candidates | `RegPanelAgreesWithTheLocalMemoryLeaf` |
| `rowid = p` for both relabel arms | same, residual explodes |
| drop barrier B2 | same, from m = 33 |
| drop `piv_base` | `RegPanelIpivAndInfoAreGlobalAtANonZeroPivBase` |
| `info_local = 0` unconditionally | `RegPanelPlantedZeroColumnIsGlobalAndFirstFailureWins` |
| store to `a[tid + k*ld]` | residual at every pivoting cell |
| restore `regs x wg <= 65536` | cdouble launch fails |

## The fused gesv tier

`gesv_tiny.cc`: one launch for `A X = B` at n <= 32 (cdouble 16), nrhs <= 4. `getrf_tiny`'s elimination carries `D rB[NR]` with forward substitution fused and back substitution in the same kernel; B is permuted for free by the lazy relabel. No batched vendor gesv exists on any backend, so `GesvChoice` has no vendor family.

Window, `t_blocked / t_tiny` (the arm it replaced), `benchmarks/results/p2_gesv_{float4,cfloat4}.csv` and the column-bucket pass: float clears 1.11 at every n = 4..32 (weakest 1.15, n=17 nrhs=1 b32768); n = 33 is the tier ceiling, the one edge with no measured loser. cfloat n = 16 reads 1.83-2.85; n = 17..32 reads 0.97-1.19 even with the column bucket, so cfloat stops at 16. fp64 has no contiguous winning band (double n=8: 1.55 at nrhs 1, 0.87 at nrhs 4).

### LU: the gesv register bound {#lu-p2-the-register-bound-is-assumed-not-probed}

`kWorstRegsPerThread = 224` in `gesv_tiny.cc` is headroom; the probe (`scripts/register_probe.sh out.log '' batchlas_extensions_gesv`, not the `_cta` library) reads worst 182 (cdouble N=16 NR=4), frame 0. `posv_tiny` is owned by [potrf.md](potrf.md). A non-zero frame shows as a slowdown, never a wrong answer.

## The right-hand gather

`lu_laswp_right_gather_launch` (`lu_laswp.hh`) replaces the in-loop right-hand walk (58% of cfloat n=128 kernel time): one group per (column tile, item), writing only moved rows. Walk / gather:

| n (batch) | float | cfloat | double | cdouble |
|---:|---:|---:|---:|---:|
| 128 (8192) | 1.91 | 1.97 | 1.52 | 1.20 |
| 1024 (128) | 1.21 | 1.03 | 1.01 | 0.98 |
| 2048 (32) | 0.95 | 0.88 | 0.94 | 0.99 |

The gate is per block step on the trailing height `R = n - j0 <= getrf_right_gather_max_rows<T>()`: float 1024, cfloat and double 768, cdouble 384; within 0.5% of the better pure arm at every swept n. `LuTest.RightInterchangeSpellingsAgreeBitForBit`.

## The tiny launch bound

At 143 registers the tiny kernel ran at 25% occupancy, latency-bound. `[[intel::min_work_groups_per_cu(N)]]` is refused on a lambda, so the kernel is the functor `GetrfTinyBody<D, N, Mpw, MinBlocks>`; `getrf_tiny_min_blocks<T, N, NC>()` holds the swept bound per column bucket (32-bit types: 12-20 at N=32, 16 at N=16; fp64 1, unmeasured). Batch 32768, ms at MinBlocks 1 / 16: float n=24 0.363 / 0.277; cfloat n=32 1.300 at 1, **1.051 at 8**, 1.377 at 16. Too tight is the expensive direction (cfloat NC 32 at 20: 1.92 vs 0.86 ms). `gesv_tiny_min_blocks` is the smaller gesv table; posv applies the same lever ([potrf.md](potrf.md)).

## The column bucket and the local-memory pivot row

### The column bucket

`rA[NC]` takes a compile-time column count (`tiny_col_bucket`) while rows stay N lanes, so n=17 no longer pays for 32 register columns: cfloat n=17 0.542 -> 0.233 ms (2.32x), float n=17 1.40x. `getrf` instantiates every even NC from 10 to 32 for 32-bit types; `gesv` and `posv` take {12, 16} and {20, 24, 28, 32}; fp64 keeps NC = N. LLVM declines `#pragma unroll` above size 16 Ki and `rA[]` silently goes to the stack, so the tiny TUs compile with `-mllvm -pragma-unroll-threshold=262144` (`src/CMakeLists.txt`).

### The local-memory broadcast

The pivot lane stores its row as 16-byte vectors to a per-partition local buffer, double-buffered by step parity with one sub-group barrier per step, instead of `N - j - 1` shuffles: float n=32 1.45x, cfloat n=32 1.26x. A single buffer is green everywhere (blind); the pair is kept for the memory model.

### The RHS pad-column cost

nrhs = 2 ran as NR = 4 and paid a complex divide per step via `tiny_select`, slow on zero pad columns. Fixed with partition-uniform branches, loops stopping at `nrhs`, and an NR = 2 bucket: gesv cfloat n=9 nrhs=2 0.453 -> 0.080 ms. Speed only; no functional test sees it.

## getrf non-square (m x n)

An m x n `getrf` is LAPACK `?GETRF`: `min(m, n)` packed 1-based int32 pivots per item at stride `min(m, n)`. cuBLAS is square-only, so CUDA Auto takes the first runnable native.

- **cta**: `?GETF2` runs `min(m, n)` steps over all n columns; admission `getrf_cta_fits_rect<T>(m, n, budget)`.
- **blocked**: `min(m, n)` panels of `m - j0` rows; a wide A's last panel produces `U(:, m:n)`. A tall A takes the deferred left walk, since the gather stages only `n - k0` rows.
- **tiny** is square-only; `getrs` and `getri` remain square.

Non-square rates fall between square 128 and 256 (float b1024: 929-1375 GFLOP/s vs 1465 / 1391; cuBLAS square 256: 332). The key is still `n = A.rows()`, so tall shapes that fit the CTA tile take `cta` and lose 1.9-4.6x at batch 1024 (an open debt). Tests: `GetrfCandidates.NonSquareFactorsWholeMatrixNatively` (16 shapes, every pin, LAPACKE pivots exactly, padded ld, poisoned pad, complex data), `NonSquareSaturatingBatchIsBitIdentical`, `CanRunEqualsLaunch`. Entries must be continuous random values: an integer lattice ties magnitudes and the pivot oracle fails a correct kernel. The deferred walk's pivot-stride break is blind by construction (it runs only where `min(m, n) = n`).

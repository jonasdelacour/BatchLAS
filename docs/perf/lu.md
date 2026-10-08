# LU: getrf, getrs, getri, gesv {#perf_lu}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · measured 2026-09-10 to 2026-09-27

Routing rules and measurements for the LU family: native `getrf` (`tiny`, `cta`, `blocked`), `getrs` (`cta`, `blocked`), `getri` (`blocked`), `gesv` (`tiny`, `blocked`), and the row-interchange kernels they share. Every row of `tuned/{getrf,getrs,getri,gesv}.*.txt` rests on a measurement below.

Machine: 2x RTX 4090 (sm_89, 128 SMs, 97,280 B usable local memory per work-group, ~1008 GB/s DRAM, 1.29 TFLOP/s FP64, ~47 TFLOP/s FP32 GEMM). Every run uses one GPU, the public API, a host oracle on every timed row, the resolved route read off every row, and no timing under `BATCHLAS_KERNEL_TRACE`. Older evidence uses retired spellings (`native:cta`, `vendor:auto`); they are not valid pins today.

## Selection

* **Families** (`src/ops/<op>/choice.hh`, in tie-break order): getrf `tiny`, `cta`, `blocked`, `vendor`; getrs `cta` (fused narrow-RHS), `blocked` (composition), `vendor`; getri `blocked`, `vendor`; gesv `tiny`, `blocked`. None has fields. See [families and their limits](#lu-families-and-their-limits).
* **Correctness** is one `can_run` per op in `src/ops/<op>/<op>.cc`. It is the only gate. A `can_run` term must never be a speed term, or the shape becomes unservable in a vendor-free build.
* **Speed** is the first runnable entry of the nearest row of `tuned/<op>.<dtype>.<device>.txt`. Keys: getrf and getri `n:log:3 batch:log`; getrs `n:log:2 nrhs:log batch:log`; gesv `n:log:3 nrhs:log`. All 32 LU tables are `source=transcribed:424a45bc`: the old routers' order evaluated per grid cell, **untimed**. The sm_120 rows equal the sm_89 rows, so on sm_120 they are hypotheses until retuned. See [what the tables rank first](#lu-what-the-tables-rank-first).
* **Sub-op seams**: the blocked drivers call the public `gemm` and `trsm`, which select from their own tables (@ref selection_tables). An empty seam throws (`getrf_blocked.cc:225-237`).
* **Pins**: `BATCHLAS_GETRF_ROUTE`, `BATCHLAS_GETRS_ROUTE`, `BATCHLAS_GETRI_ROUTE`, `BATCHLAS_GESV_ROUTE` take `auto`, `native`, `vendor` or a family spelling. A spelling that does not parse, is not compiled, or names a family whose `can_run` refuses the shape throws `std::invalid_argument`. `native` takes the best runnable non-vendor entry. `vendor` with no runnable vendor family (vendor-free builds; always for gesv) warns once and runs Auto.
* **Derived, not chosen**: the panel leaf (`BATCHLAS_GETRF_LEAF`), the interchange spellings (`BATCHLAS_GETRF_LASWP`, `BATCHLAS_GETRF_RIGHT_LASWP`, `BATCHLAS_GETRS_LASWP`), the blocking factor `nb`, the tiny bucket and its launch bound. The launchers work these out. They are not families, keys or `can_run` terms.

## What ships

### LU: families and their limits

The common native term is `B != Backend::NETLIB && d.is_gpu && d.has_sg32`, a homogeneous batch, a square A (getrs: a conforming pair with `B.rows == n`), and `n, batch >= 1` (`getrf.cc:43-44`, `getrs.cc:41-43`, `getri.cc:49-50`, `gesv.cc:45`). Beyond it:

| op | family | driver | `can_run` beyond the common term |
|---|---|---|---|
| `getrf` | `tiny` | `getrf_tiny.cc`, register-resident, one matrix per sub-group partition | `n <= getrf_tiny_max_n<T>()` (32; cdouble 16), `max_wg >= 64` (`getrf.cc:48`) |
| `getrf` | `cta` | `getrf_cta.cc`, local-memory ?GETF2 | `n <= getrf_cta_max_n_for_slm<T>(budget)`, occupancy-scaled (`getrf.cc:50`) |
| `getrf` | `blocked` | `getrf_blocked.cc`, right-looking, register panel leaf by default, public `trsm` and `gemm` | `getrf_blocked_available<T>()` and `getrf_cta_max_n_for_slm<T>(budget, 1) >= 1` (`getrf.cc:53-54`) |
| `getrf` | `vendor` | `cublas?getrfBatched`, rocSOLVER, LAPACKE | `d.has_vendor`; the only family for non-square or heterogeneous A |
| `getrs` | `cta` | `getrs_fused.cc`, fused narrow-RHS kernel | `nrhs <= kGetrsFusedMaxRhs` (8), `n * nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)`, `max_wg >= 32` (`getrs.cc:47-50`) |
| `getrs` | `blocked` | `getrs_native.cc`, permutation then two public `trsm` | `getrs_blocked_available<T>()` (`getrs.cc:52`) |
| `getrs` | `vendor` | `cublas?getrsBatched`, rocSOLVER, LAPACKE | `d.has_vendor`; also takes a non-conforming pair |
| `getri` | `blocked` | `getri_blocked.cc`, writes P into C, two public `trsm`, zero workspace | `getri_blocked_available<T>()` (`getri.cc:49-50`) |
| `getri` | `vendor` | `cublas?getriBatched`, rocSOLVER, LAPACKE | `factorization_vendor_available<B>` (`getri.cc:52`) |
| `gesv` | `tiny` | `gesv_tiny.cc`, fused factor and solve | `n <= gesv_tiny_max_n<T>()` (32; cdouble 16), `nrhs <= 4`, `max_wg >= 64` (`gesv.cc:45-47`) |
| `gesv` | `blocked` | public `getrf`, then public `getrs`; each child selects for itself | homogeneous A and B (`gesv.cc:49`) |

The blocked `getrf`'s trailing GEMM and panel TRSM, and both solves of `getrs` and `getri`, go through the public `gemm` and `trsm`, never a kernel-local GEMM.

Capacities are asked of the **runtime** local-memory budget, never of `device_limits.hh`. On this box `can_run(cta)` admits **77 / 54 / 54 / 38** for float / double / cfloat / cdouble (occupancy-scaled; see [the occupancy rule](#lu-the-occupancy-rule)). The unscaled ceilings are 155 / 109 / 109 / 77. `kGetrsFusedMaxRhs = 8` is a build constant. `getrf` uses `nb = 32` for every type.

### LU: what the tables rank first

**getrf** (grid n 1..1024 with both sides of 4|5, 7|8|9, 16|17, 24|25, 32|33, 255|256, 511|512; batch 1..32768 with 255|256):

| dtype | rows ranking a native family first | all other rows (vendor first; native order behind it) |
|---|---|---|
| float | `tiny vendor cta blocked` at n = 5..32; `blocked vendor cta` at n >= 256; every batch | `vendor tiny cta blocked` at n <= 4; `vendor cta blocked` at 33..255 |
| cfloat | `tiny vendor cta blocked` at n = 5..7 and 9..24; `blocked vendor cta` at n >= 512 at every batch, and at n = 256..511 for batch >= 256 | `vendor tiny cta blocked` at n <= 4, 8, 25, 32; `vendor cta blocked` at 33..511 (33..255 at batch >= 256) |
| double | none | `vendor cta tiny blocked` at n <= 32; `vendor blocked cta` at n >= 33 |
| cdouble | none | `vendor cta tiny blocked` at n <= 16; `vendor cta blocked` at n >= 17 |

The blocked rows are the [getrf window](#getrf-window-evidence) plus [the cfloat move to 256](#the-cfloat-window-moves-to-256). The tiny rows are the [tiny getrf window](#the-tiny-getrf-window), widened by [the N=4 bucket](#the-n4-bucket-and-the-cta-band), [the launch bound](#the-tiny-launch-bound) and [the column bucket](#the-column-bucket).

**getrs** (grid n 31|32; nrhs 2|3, 4|5, 63|64, 127|128; batch 127|128):

* Clause A (`nrhs <= 2`, every type) and clause B (float, `nrhs = 3..4`): `cta vendor blocked` at n >= 32.
* Clause C (float `nrhs >= 64`, double `nrhs >= 128`, both at batch >= 128): `blocked vendor cta` at every n.
* Every other row, including narrow nrhs below n = 32: `vendor cta blocked` ([order floor](#getrs-order-floor-evidence)).
* No cfloat or cdouble row ranks `blocked` first.

**getri** (grid n 127|128, 255|256; batch 128..32768, every batch row identical): `blocked vendor` at n >= 128 for float and n >= 256 for cfloat; `vendor blocked` everywhere else, including every n for double and cdouble. A batch below 128 reads the batch-128 row.

**gesv** (grid n 16|17, 32|33; nrhs 4|5; no batch key): `tiny blocked` at n <= 32 for float and n <= 16 for cfloat; `blocked` alone elsewhere and for double and cdouble. A tiny-first row whose nrhs exceeds 4 runs `blocked`, because `can_run(tiny)` refuses it.

The rows are pinned by `GetrfTranscribedTable.RowsHoldTheOldPreferenceOnBothDevices`, `GetrsTranscribedTable.RowsHoldTheOldPreferenceOnBothSidesOfEveryThreshold`, `GetriTranscribedTable.RowsHoldTheOldPreference` and `GesvTranscribedTable.RowsHoldTheOldWindowOnBothDevices`. Auto on the real device is pinned by `GetrfCandidates.AutoReproducesTheOldRouterOnTheRealDevice` and the `AutoReadsTheTranscribedTable(s)` cases.

### LU: the order among native families

A vendor-free build, or a `native` pin, takes the first runnable *native* entry of the row, so the order behind `vendor` is a decision of its own.

**getrf.** The double rows rank `cta` before `blocked` at n <= 32 and `blocked` before `cta` from n = 33. The other types rank `cta` before `blocked` wherever `blocked` is not first. Measured `blocked_ms / cta_ms` (> 1 means CTA is ahead):

| type | n=64 (b8192) | n=76 (b8192) | n=96 (b8192) | n=100 (b4096) | n=128 (b4096) |
|---|---|---|---|---|---|
| float | 1.74 | 1.48 | 1.49 | 1.68 | 1.13 |
| cfloat | 1.39 | 1.59 | 1.30 | 1.33 | -- |
| cdouble | 1.37 | 1.09 | -- | -- | -- |
| **double** | **0.98** | **0.85** | **0.77** | **1.00** | -- |

Double at n = 76 is flat in batch (0.78 to 0.85 from batch 2048 to 16384, every relative sd < 0.2%). At n <= 32 the order returns to CTA for double, and that is not a hedge: `nb = min(32, n) = n`, so the blocked driver runs one panel whose leaf is the CTA device function (1.8126 vs 1.8113 ms at n = 32, batch 8192). Without this order a vendor-free build loses 1.18-1.29x at double n = 76..96. Above the occupancy-scaled ceilings `blocked` answers whatever the row says ([P7](#the-p7-occupancy-change-moved-27-committed-getrf-baseline-cells)).

**getrs.** `cta` (fused) precedes `blocked` in every row outside clause C. The fused family is ahead of the composition at all 51 cells inside its capability (worst 1.11x, float n=2048 nrhs=8). The crossover is at nrhs = 16 (double 0.55x, cfloat 0.58x at n=512), outside `can_run(cta)`. **If `kGetrsFusedMaxRhs` is raised, the getrs rows at nrhs >= 16 must rank `blocked` before `cta` in the same change.** `getri` has one native family.

#### Tiny in the rows

Tiny ranks first where its windows put it. It sits ahead of `cta` for float and cfloat, and behind `cta` for double and cdouble. A `tiny` pin reaches it at any shape `can_run(tiny)` admits.

## The vendor baseline and saturation

Every ratio is against `cublas{S,D,C,Z}getr{f,s,i}Batched`, which is genuinely batched and strong at small n. Saturating batch, vendor build (`experiments/wp6_lu/baseline/grid_norm.csv`, 280 cells):

| n (batch) | float ms / GFLOP/s | double | cfloat | cdouble |
|---:|---|---|---|---|
| 32 (8192) | 0.159 / **1129** | 0.561 / 319 | 0.245 / 732 | 1.488 / 120 |
| 2048 (32) | 519.2 / 353 | 587.9 / 312 | 564.0 / 325 | **2594 / 70.6** |

GFLOP/s is not monotone in n.

**cuBLAS does not saturate at n >= 1024 on any ladder this box can hold.** `cublasZgetrfBatched` at n=2048 takes 2587.7 ms at batch 4 and 2594.1 ms at batch 32 (`baseline/summary_sat.txt`): 8x the work for 0.25% more time. Native saturates at batch 4-32 at n=2048 because it parallelises within the item.

Read at each arm's own best batch (`experiments/wp6_lu/bench/`, 982 rows, max relative sd 7.2%):

| | geo at the grid's batch | geo at each arm's own best batch |
|---|---|---|
| `getrf` (28 cells) | 0.885x | **0.805x** |
| `getri` (28 cells) | 1.463x | **1.284x** |
| n=2048 row (8 cells) | 7.098x | **2.954x** |

The saturation caveat is not uniform. At n=512 the vendor is saturated for all four types. At n=2048 it is unsaturated by 19-50% per doubling, which is why the getri evidence quotes n=2048 at batch 128 and 256 only.

**Roofline.** With each arm at its own best batch, cuBLAS runs cdouble `getrf` at 90-91% of FP64 peak at n=512-1024, so there is no 2x there. FP32 arms sit at 1-10% of peak because of the decomposition ([negative results](#lu-negative-results)).

## Measured boundaries {#lu-measured-boundaries}

Ratio is `vendor_med / native_med`; > 1 means native wins. The closure pass's acceptance rule (GATE-C) is zero losses and zero cells below 1.15 across the admitted set.

### `getrf` window evidence

Scored on `experiments/wp8_getri/lu_c1.csv` (device 1, idle box, 11 reps, median). Shipped set: **15 cells, geomean 1.997, min 1.2626, zero losses.** The 20-cell kernel-stage record on the other device (`experiments/wp8_getrf/after_nv_p{1,2}.csv`) adds batch 256:

| type | n | b128 | b256 | b512 | b1024 |
|---|---:|---:|---:|---:|---:|
| float | 256 | 1.254 | 1.279 | 1.567 | 1.675 |
| float | 512 | 2.350 | 1.988 | 1.737 | 2.183 |
| float | 1024 | 2.773 | 2.186 | 2.237 | -- |
| float | 2048 | 3.091 | -- | -- | -- |
| cfloat | 512 | 1.811 | 1.528 | 1.682 | 1.609 |
| cfloat | 1024 | 2.124 | 1.829 | 2.088 | -- |
| cfloat | 2048 | 2.754 | -- | -- | -- |

Bracketing non-winners:

* float `order >= 128` would admit 0.825 / 0.773 / 0.872 at batch 256 / 512 / 1024.
* cfloat `order >= 256` admits 0.8851 at batch 128 with the local-memory leaf. The register leaf makes 256 clear the gate ([the cfloat window](#the-cfloat-window-moves-to-256)).
* double `order >= 512` admits 0.7486 at batch 1024. Double's best cell anywhere is 1.067; cdouble's is 1.012.

**In the tables:** float ranks `blocked` first from n = 256 at every batch; cfloat from n = 512 at every batch and from 256 at batch >= 256. No double or cdouble row ranks a native family first.

### `getri` window evidence

`experiments/wp8_getri/lu_c1.csv` and `lu_p1.csv`, each ratio at the worse of the two passes. Shipped set: **30 cells, geomean 4.181, min 1.2028, zero losses**, batch 1 to 16,384.

| type | n | measured rungs (batch: ratio) |
|---|---:|---|
| float | 128 | b1 2.169, b256 1.819, b4096 1.262, b16384 **1.203** |
| float | 256 | b128 4.047, b4096 2.641 |
| float | 512 | b1 18.565, b128 7.600, b2048 4.466 |
| cfloat | 256 | b128 1.977, b512 **1.682**, b4096 2.195 |
| cfloat | 512 | b1 24.276, b128 4.138, b1024 3.807 |

Bracketing non-winners for every wider clause:

* float `order >= 64`: n=64 **loses** at 0.8572 / 0.8561 (batch 8192) and 0.8499 / 0.8530 (batch 16384).
* cfloat `order >= 128`: n=128 is 0.7109 / 0.7070 at batch 512.
* double, any order: n=2048 at batch 128 is 1.0848 / 1.0823. Only n=512 is clean, and a clause admitting one order is the leg-predicate defect.
* cdouble, any order: n=512 loses at 0.9542 / 0.9525 (batch 1024); nothing reaches 1.15 at batch >= 128.

**No batch floor.** At batch 1-32 the native driver beats cuBLAS by 1.7-28x for every type, because cuBLAS's batched getri is a per-item loop there. A floor at 128 would forfeit those wins.

**In the tables:** float ranks `blocked vendor` from n = 128; cfloat from n = 256. Double and cdouble read `vendor blocked`. The tables carry a batch key the old window never read; every batch row holds the same ranking.

### `getrs` fused window evidence

Clauses A and B, the rows that rank fused `cta` first, from `experiments/wp6_perf/bench/` (`analyse_window.py`): **461 pooled cells** from seven sweeps, `relsd <= 10%`. The checked-in `bench/window_summary.txt` is a stale 354-cell run.

| clause | cells | geomean | min | losses |
|---|---:|---:|---:|---:|
| A: `nrhs <= 2`, every type | 286 | **2.261** | 1.116 | **0** |
| B: float, `nrhs = 3..4` | 36 | **1.611** | 1.133 | **0** |
| both | **322** | **2.177** | **1.116** | **0** |

On WP6's saturating grid, nrhs=1 moves from 0.256x (0 of 28 wins) to 2.117x (28 of 28). Full batch ladders at n = 32 to 2048 show zero of 322 laddered cells below 1.0 (lowest rung 1.116x), but this grid has no rung below n = 32; see [the order floor](#getrs-order-floor-evidence).

**Why clause B is float-only.** Mid-ladder dips refute the wider candidates: double n=128 at 0.940x (batch 2048); cdouble n=32 at 0.577x. Rejected: float `nrhs <= 8` (3 losses, worst 0.686x), `nrhs <= 4` every type (20 losses, worst 0.577x), the whole capability (55 losses, worst 0.294x). The thinnest margin is cdouble n=32 nrhs=2 at 1.116 (batch 16384); it flattens rather than falls.

**In the tables:** every `tuned/getrs.<dtype>.sm_89.txt` row with nrhs 1-2 and n >= 32 reads `cta vendor blocked`; float rows also at nrhs 3-4.

### `getrs` order floor evidence

Clauses A and B shipped with no order bound, on a grid whose smallest order was 32. Re-measured down to n = 4, they lose in vendor-present traffic at every type, worst **0.234** (cdouble n=4 nrhs=1 batch 32768). The loss deepens with batch: float n=4 nrhs=1 reads 1.50 / 1.07 / 0.71 / 0.52 at batch 8192 / 16384 / 32768 / 65536. This corrects the [fused window evidence](#getrs-fused-window-evidence).

**Mechanism.** The fused kernel gives one work-group to a matrix whose solve is a few dozen flops, so the work-group is the cost. Worst ratio over each order's batch ladder, nrhs = 1 (minimum rung; > 1 means native wins):

| T | n=4 | n=8 | n=16 | n=17 | n=24 | n=32 |
|---|---:|---:|---:|---:|---:|---:|
| float | 0.71 | 1.12 | 1.04 | 0.76 | 0.95 | 2.29 |
| cfloat | 0.59 | 0.69 | 3.76 | 1.38 | 1.27 | 1.40 |
| double | 0.52 | 0.98 | 0.99 | 0.87 | 3.66 | 3.77 |
| cdouble | 0.23 | 0.41 | 2.00 | 2.02 | 1.93 | 2.13 |

This statistic differs from `small-n-baseline.md`, which quotes the top batch of each ladder. Do not compare the two cell by cell.

**32 is the first order where all four types clear the flip gate and stay clear above it.** Below it the band is non-monotone (float passes at 8, fails at 9, 16, 17, 24), so no lower floor is defensible. The floor is a defect fix, not a tuning knob: in the tables it is the 31|32 step, and every row with n <= 31 and nrhs <= 4 reads `vendor cta blocked`. A vendor-free build still takes the fused kernel there. P2's register-resident fused factor-and-solve has no work-group floor and is the route to reclaim this band.

Source: `benchmarks/results/factor_baseline_getrs_{float,cfloat,double,cdouble}.csv`.

### `getrs` composition window evidence

Clause C, the rows that rank the composition (`blocked`) first (float nrhs >= 64 and double nrhs >= 128, both at batch >= 128). From `experiments/wp8_getrs/cl_*.csv` and `gap_*.csv`, re-measured on an idle box in `experiments/wp8_getri/lu_c1.csv`. Union: **37 cells, geomean 2.60, min 1.2858, zero losses.**

Refuting cells for the wider clauses:

| rejected clause | refuting cell | ratio |
|---|---|---:|
| `float nrhs >= 32` | n=64 nrhs=32 b=4096 | 0.9069 |
| `double nrhs >= 64` | n=64 nrhs=64 b=2048 | 0.9984 |
| `cfloat nrhs >= 128` | n=64 nrhs=128 b=1024 | **0.9944** |
| `cdouble nrhs >= 64` | n=128, every rung (0.6928 / 0.7380 / 0.8304) | 12 losses of 26 cells |

cfloat was in this clause until a gap sweep found a mid-ladder dip that the scored cells had missed. No admitted cell is uncovered: 45 were measured on saturated rungs, and 58 are covered by a bound (gather A/B minimum 1.0004 over 80 cells).

**Batch floor of 128 is conservative.** At nrhs=128 the composition still wins at batch 64 and 32 (float 3.87-5.93x at n=64 to 1024). The floor exists because below 32 the only readings come from a contaminated sweep. Clause C has no order floor because it was never measured below order 32.

**In the tables:** float rows read `blocked vendor cta` at nrhs >= 64 and batch >= 128, every n; double at nrhs >= 128. No cfloat or cdouble row ranks `blocked` first.

## The `laswp` gather

The row interchange is the largest lever in the family.

**Cost**, priced by a timing-only break (both interchange passes disabled; answers wrong), vendor-free, saturating batch:

| type | n (batch) | with laswp | without | laswp share |
|---|---|---:|---:|---:|
| float | 512 (512) | 40.44 ms | 10.53 ms | **74%** |
| float | 2048 (32) | 70.86 ms | 39.71 ms | **44%** |
| cdouble | 512 (512) | 232.09 ms | 181.97 ms | 22% |

nsys agrees: `LuLaswpKernel` is 44.2% of the best float `getrf` cell (n=2048, b=32).

**Mechanism.** `ipiv` is a sequence of transpositions applied in order, so parallelism exists only across the columns it touches. The p side is the cost, since the selected rows are scattered over `[j0, n)`. ncu shows **one 32 B sector per 4 B element**.

### `getrf` deferred left gather

`lu_laswp_deferred_left_launch` in `lu_laswp.hh`, unconditional. The gather amortises as `L/R`, transpositions against staged rows. Deferring the left-hand pass out of the block loop makes `L == R` exactly, so the amortisation is 1:1 at every order and **no gate is needed**. A gate on `n` instead of `(n - j0)` inverts the lever.

Correctness: the three spellings (`inloop`, `defer_walk`, `defer_gather`, selected by `BATCHLAS_GETRF_LASWP`) are bit-identical (`LuTest.LeftInterchangeSpellingsAgreeBitForBit`). Against the in-loop arm, batch >= 128, 58 `native:blocked` cells: **geomean 1.207x, min 1.018x, zero cells below 1.00** (float 1.350x, cfloat 1.305x, double 1.138x, cdouble 1.074x). The `cta` rows measure 0.9995x, as a control. Sector traffic falls 7.9x. Against cuBLAS on a 62-cell grid the `getrf` geomean moves 0.839x -> 1.002x; double and cdouble are not closed by this.

### `getrs` collapsed permutation

`GetrsPermGatherKernel` in `getrs_native.cc`, the default at `nrhs >= kGetrsPermGatherMinNrhs = 16` (`getrs_native.hh:47`). One work-group per item stages a column tile and the index array in local memory, then writes `B[i] = tile[idxs[i]]` back to B's own addresses. A/B against the walk (`BATCHLAS_GETRS_LASWP`), interleaved, bit-identical solutions:

| nrhs | cells | geomean | min | max |
|---:|---:|---:|---:|---:|
| 1 | 20 | 0.9993 | 0.9953 | 1.0031 |
| 8 | 20 | 1.0011 | 0.9980 | 1.0256 |
| **16** | 20 | **1.1182** | 1.0004 | 1.2941 |
| 128 | 20 | 1.5728 | 1.0411 | 2.7873 |

Admitted set (nrhs >= 16): 80 cells, geomean **1.3191**, min 1.0004. All 50 cells below 1.00 are at nrhs <= 8. cdouble is marginal at the boundary and is not carved out. The gather needs no workspace (`getrs_blocked_buffer_size` returns 0; `LuTest.GetrsPermGatherBuysNoWorkspace`).

`getri` needs no permutation kernel: each work-item traces its own row backwards through the interchange list in registers, writing F straight into C.

### The fused narrow-RHS `getrs`

`getrs_fused.cc`: one work-group per matrix, one launch: permutation, forward substitution against unit-lower L, back substitution against U. No GEMM, no separate laswp. Against the composition: **8.26x at nrhs=1**, 6.56x at 2, 4.40x at 4, 2.67x at 8; 107 cells, zero losses, worst 1.419.

* **Resident diagonal block, not streaming**: 0.6506 vs 1.0102 ms (float n=512 b=512). Streaming pays a barrier per column.
* **`nb = 16` below n=1024, 32 at and above**: n=2048 b=32 reads nb 8 / 16 / 32 = 1.5513 / 1.3772 / **1.2838** ms.
* **Work-group ~ n/2, clamped to [64, 1024]**, then capped by a per-(type, body, width) register table. A single max-over-everything cap charged float the 86 registers of complex double.
* The `+1` bank-conflict pad is kept for portability (7 of 8 cells within 0.6% either way).

DRAM use at nrhs=1 reaches 82-88% of 1008 GB/s only in the n=256..512 band (float 82%, double 86%, cfloat 88%, cdouble 83% at n=512). Open causes: at large n the CTA count is the batch (32 work-groups on 128 SMs at n=2048); at small n `nb=16` leaves the block solve to 16 lanes.

## LU: negative results

Built or measured, then rejected. Do not re-derive.

| Alternative | Result | Verdict |
|---|---|---|
| Fuse the blocked driver's 5P-4 launches into one | 16 launches at n=128 = 80 us, 0.12% of a 67.1 ms call at batch 8192. The cost is data movement, not launches. | Rejected. |
| `getrs` wide-nrhs composition window from `grid_*.csv` (nrhs=64 1.09x, nrhs=128 1.48x) | One saturating batch per order; on a 464-cell batch ladder the advantage falls with batch, and the best candidate fails GATE-C (min 1.0436). | Rejected. The shipped window is the gather's. |
| Defer the interchange but keep the per-column walk | Measured geomean 1.055x, three cells losing (0.707x-1.315x). The walk has `batch*ib` items instead of `batch*j0`: 15x less parallelism at n=512. | Rejected. |
| Packed sub-group `getrf` (one sub-group per matrix, shuffle argmax) | float 1.38 / 1.25 / 1.13 at n=16/24/32, then 0.79 to 0.31 at n=48 to 128; double and cdouble lose everywhere. | Refuted. |
| Division-free (row, column) split in the rank-1 update | float geomean 0.936x, cdouble 0.976x; worst 0.829x. | Reverted. |
| `getrf_leaf_wg` raised to 1024 | geomean 0.974x; worst 0.814x. The resident tier loses 16-19% at 1024 work-items. | Reverted. |
| `sycl::reduce_over_group` for pivot search | 1.5-4.7x slower for double and cdouble; sole cause of the 48 KB launch hole. | Rejected. Shipped kernel uses a butterfly plus a 32-slot scan. |
| `getri` zero-fill as `range<3>(batch, n, n)` | 0.874x at the largest fill. | Reverted. |
| Complex `Tiled16` GEMM for LU trailing updates | Complex updates reach `GemmRegister64x64K16WideKernel`; `double` never gets a register GEMM on this path. | Refuted for complex; double's deficit belongs to GEMM. |

Partial pivoting costs 1.25-2.65x over the unpivoted bound; the swap alone is 1.00-1.20x, and the argmax search is the cost.

Traps from the same record:

* A harness that shrinks the batch cannot ask the routing question: the CTA-count gate multiplies by batch, and `can_use_64x64_k16_wide_fast_path` reads `stride()`.
* Two RTX 4090s in one chassis are not independent. A sweep on device 1 beside one on device 0 read `getrf float n=256 batch=128` at 3.31-5.51 ms against 1.006 ms alone, with zero foreign processes. Serialise the box.
* A stale bench binary reports a stale route. Rebuild every bench binary after a table or `can_run` change ([agent guide](docs/developer/agent-guide.md)).
* A refused pin used to become the other arm silently. Flat selection throws now. An A/B that does not read the resolved route per arm reports one arm twice and calls it 1.00x.
* `BATCHLAS_GETRS_ROUTE=native` means the first runnable non-vendor entry. To measure the pre-CTA `native`, pin `blocked`.

## The shape of the GETRF device code

`src/extensions/getrf_cta_device.hh` holds the native GETRF panel's device code. `getf2_panel_device` is LAPACK ?GETF2 over a `Tile` providing `at(r, c)`, instantiated on a `local_accessor` (CTA tier) and on a raw global pointer (blocked tier's panel leaf). Four spellings are load-bearing:

* **The pivot search is a sub-group butterfly plus a scan over 32 SLM slots**, never `sycl::reduce_over_group` ([48 KB hole](#lu-the-48-kb-launch-hole)).
* **The pivot metric is LAPACK's `cabs1`** (|Re| + |Im|), not the modulus cuBLAS uses for complex ([correctness findings](#lu-correctness-findings)).
* **`info` is exact-zero, 1-based, global, first-failure-wins**, with no epsilon pivot floor. An epsilon floor would make a singular matrix succeed silently.
* **Barriers B1..B4 sit at the top level of the `k` loop**, whose trip count is kernel-uniform. The launchers add B0 after the tile load and B5 before store-back. A barrier inside a divergent branch hangs.

`kLuRedSlots = 32` is constant, so the capacity query, the fit predicate and the launcher agree on the footprint. The launcher pads `ld` odd: a row exchange walks `wg` items at stride `ld`, and an even `ld` puts them in one local-memory bank.

## LU: correctness findings

* **The `info` zero-fill raced the panel that reads it, in both native `getrf` tiers.** On an out-of-order queue the panel read the caller's pre-call garbage (6,979 of 1,638,400 items for CTA). Fixed with `if (!ctx.in_order()) ctx.wait();`, guarded by `LuTest.InfoFillIsOrderedAheadOfThePanelOnAnOutOfOrderQueue`.
* **Backend gate.** Native kernels write packed 1-based int32 pivots in the caller's int64 span. NETLIB writes genuine int64. Without the `B != Backend::NETLIB` term in every native `can_run`, `Backend::NETLIB` on a GPU queue selected a native arm and returned `||A*C - I||_F / n = 5.32e-01` with `info == 0`.
* **Pre-existing vendor crash, fixed.** `cublas.cc`'s `getrs` at batch 1 passed the raw int64 pivot pointer to `cusolverDnXgetrs`, aborting `getrf` then `getrs` at batch 1. The arm was deleted; `cublas?getrsBatched` is correct at `batchCount = 1`.
* **Complex pivoting: cuBLAS uses the modulus, LAPACK and this library use `cabs1`.** For `(3+0i)` over `(2+2i)`, `cabs1` selects row 0 and the modulus row 1. An elementwise native-vs-vendor pivot comparison is therefore wrong for complex. `PivotSelectionUsesCabs1AndNotTheModulus` pins the library's rule.
* **The exact-zero `info` predicate is not stable across implementations.** The gate is structural: non-zero exactly when `|U(i,i)|` is a true binary zero.
* **Capacity inversion.** `getrs_fused_max_rhs_elems` used floor division, but `getrs_hole_padded` is not monotone. At a cdouble budget of 49,665 B the query admitted a request the pad raised to 49,920 B. Reachable only on a device with 53,761-53,776 B of local memory.
* **The register cap fired.** Float n=2048 nrhs=8 with `transA = Trans` picked wg=1024 against a 68-register kernel: 69,632 registers per work-group against a 65,536 limit. `FusedGetrsLaunchHoleAt48KiB`'s top rung covers it.

### Blind guards and what made them blind

Guards that could not fail, as LU found them:

1. **A diagonally dominant test matrix makes every pivot test vacuous.** Keep the dominance and row-permute each item by a random permutation.
2. **An inverse test on a self-inverse instance is vacuous.** Use an n-cycle and assert `interchange_is_involution()`.
3. **A probe that computes the right number and never asserts on it** passed a 1.2e-01 residual. Every criterion now carries a `Tol<T>` bound.
4. **A capability helper that never set the fused capacities** made every getrs routing assertion hold regardless of the table (78/78).
5. **`max |L| <= 1` is the wrong partial-pivoting oracle for complex**: the `cabs1` bound allows sqrt(2).
6. **A revert that matched an anchor inside a longer line.** `break.py` now requires each anchor to match exactly once.

The pivot-metric break (cabs1 to modulus) is green on the ordinary sweep, since the two rules agree on random matrices; a purpose-built probe is needed.

## LU: the 48 KB launch hole

Measured with a `PAD=` knob that moves only the declared byte count: **49,024 B PASS / 49,152 B FAIL / 49,280 B PASS**, 5/5 deterministic, at every work-group width (32 to 512). An explicit SLM tree at the same byte count launches. Both failure points lie inside the inherited band `(47104, 49664]`. An `n` ladder cannot find the hole.

**Attribution:** the group collective. `reduce_over_group` allocates local memory the `local_accessor` accounting cannot see. The shipped `getrf` CTA kernel uses no group collective and does not reproduce the hole. The pad is defensive, for the day a group algorithm enters the body. The same band and pad target appear in `getrf_cta.cc`, `lu_laswp.hh` and `getrs_native.cc`.

## Open debts

* **`getrf` window has no batch term** (measured at batch 128-1024 only); `GetrfCandidates.AutoReproducesTheOldRouterOnTheRealDevice` asserts `(512, 2)` on an order clause alone.
* **`getri` at batch <= 32 beats cuBLAS by 1.7-28x** for every type, but the batch grid starts at 128. Double and cdouble at batch 1-64 are unmeasured.
* **`getrs` clause-C batch floor of 128 gives up wins** at batch 32 and 64 (`experiments/wp8_getri/gen_floor.py` is the sweep).
* **`getrs` clause C is unmeasured above order 1024**, and 84 winning cells in clauses A and B go to the vendor (largest 3.944x, double n=1024 nrhs=4 batch=256). Recovering them needs a timed retune.
* **The collapsed `getrs` gather is parallel over batch only**: at batch 128 on a 128-SM part, one wave.
* **The right-hand `getrf` interchange is still the per-column walk** outside the gate in [the right-hand gather](#the-right-hand-gather).
* **The fused `getrs` folded permutation is 8.0% of the call at n=2048** (float b=32), the one serial part.
* **Fused tier capped at nrhs 8.** Raising `kGetrsFusedMaxRhs` requires the rows at nrhs >= 16 to rank `blocked` first.
* **`getrf` native order rests on 5 orders and 1-4 batches per type**, and `can_run(cta)` stops below the measured cfloat and cdouble orders ([P7](#the-p7-occupancy-change-moved-27-committed-getrf-baseline-cells)).
* **`nb = 32` is not tuned**: `getrf_leaf_wg` reproduces two widths and extrapolates; the baseline shows an 8.3x spread.
* **`getrf_panel_factorize` re-queries `LOCAL_MEM_SIZE` and `MAX_WORK_GROUP_SIZE` once per block step** (128 queries per n=2048 `getrf`).
* **`MatrixView::data_ptrs(ctx)` re-runs `init_data_ptr_array` unconditionally**, two host drains per panel. Root cause in `matrix.hh`; worked around.
* **`Backend::NETLIB` on a GPU queue is gated but not exercised end to end**, since the fixture's queue is a CPU queue.
* **Float `n = 65` takes CTA in a vendor-free build**, where blocked is 1.62x faster ([which tier serves 33 to 256](#which-native-tier-serves-33-to-256)).
* **`getrf_panel_reg.cc` duplicates the sub-partition gate** of `resident::sm89_max_work_group`. They agree today; collapsing them is owed.
* **Recursive panel not attempted**; no float or cfloat phase split exists; residual tolerances `c*n*eps` with c in [200, 800] are not tightened against a measured distribution.
* **Latent vendor gate** closes by reading (`src/select/vendor.hh:32-35`); no cuSOLVER-absent configure has been built.
* **NETLIB `getri`'s `std::copy(..., n*n, ...)` ignores `ld`.** Pre-existing, not fixed.

## Raw evidence

Raw data is at the tag `perf-evidence/vendor-independence`: `git show perf-evidence/vendor-independence:<path>`. The `experiments/` directories below were still tracked in the working tree when this page was verified.

| topic | path |
| --- | --- |
| vendor baseline, saturation, pivot cost, 48 KB hole, route probe | `experiments/wp6_lu/baseline/` (`grid_norm.csv`, `summary_sat.txt`, `hole.csv`) |
| native vs vendor A/B grid, tier sweep, laswp cost, kernel-side breaks | `experiments/wp6_lu/kernels/` (`grid_{native,vendor}.csv`, `tier.txt`, `laswp_cost.txt`) |
| 982-row measure phase, nsys splits, nrhs crossover | `experiments/wp6_lu/bench/` (`geomeans.txt`, `order_tables.txt`, `nsys_splits.txt`) |
| LU test suite and its breaks | `experiments/wp6_lu/tests/` (`breaks.txt`, `break_*.txt`) |
| clauses A and B: sweeps, window scoring, register-cap A/B | `experiments/wp6_perf/` (`bench/window_summary.txt`, `regcap/`) |
| deferred left-hand gather A/B, ncu sector counts | `experiments/wp8_getrf/` |
| `getrs` permutation gather, boundary sweep, clause C | `experiments/wp8_getrs/` (`ab_summary.txt`, `clause_summary.txt`) |
| clean device-1 re-measure the shipped windows are scored from | `experiments/wp8_getri/` (`lu_c1.csv`, `summary_c1.txt`) |
| campaign narrative | `VENDOR_INDEPENDENCE_PLAN.md` |

The `wp8_*` directories hold the WP6/WP7 performance-closure pass, not sparse `spmm` (`experiments/sparse_spmm/`).

P4's raw rows are in the tree: `benchmarks/results/p4_leaf_getrf_{float,cfloat,double,cdouble}.csv`, `p4_tier_getrf_float.csv`, `p4_e2e_getrf_{float,cfloat}.csv` (`benchmarks/factor_bench.cc` through `benchmarks/gpu_guard.sh 1`).

> **Warning:** plain `blocked` takes the default panel leaf, which is `reg`. `--arms=vendor,blocked,blocked/leaf=reg` measures the register leaf twice and reports a leaf ratio of 1.000. Spell both: `blocked/leaf=slm,blocked/leaf=reg`.

---

## LU: the occupancy rule

**P7, 2026-09-10.** `getrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` and `getrf_cta_fits<T>(n, budget, min_blocks_per_sm)` divide the device budget by an occupancy target (default 4, `resident::kMinBlocksPerSm`) before testing the footprint. At 97,280 B the advertised ceilings are **77 / 54 / 54 / 38** for float / double / cfloat / cdouble, against 155 / 109 / 109 / 77 at target 1. Both scales are pinned in `tests/resident_capacity_tests.cc`. The walk lives in `src/util/resident_capacity.hh`, shared with potrf; the sqrt bound stays here.

### One spelling per ceiling

* `getrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` takes a runtime `local_mem_size` budget. The footprint covers the pivot-search scratch as well as the tile. 0 means the tier is absent from this build.
* `kGetrfReferenceSlmBudget = 97280` (`getrf_cta.cc:34`) serves only the convenience overloads. The generated `device_limits.hh` says 49,152 for any `nvidia_gpu_sm_*` pattern (`cmake/BatchLASDetectSYCL.cmake:45-46`), 2.06x wrong here, so nothing on a decision path reads it. See [potrf.md's budget section](potrf.md#the-slm-budget-and-the-fit-ceilings).
* `getrf_cta_fits` is the tier's admission test, occupancy-scaled. `getrf_leaf_fits` is the residency question, asked at the whole budget ([panel leaf](#lu-the-panel-leaf-is-not-the-tier-ceiling)).
* `getrf_tiny_max_n<T>()` is a compile-time property of the kernel, the {4, 8, 16, 32} ladder ([N=4 bucket](#the-n4-bucket-and-the-cta-band)). It is the one place that ceiling is spelled (`getrf_tiny.cc:328-332`).
* `getrf_tiny_buffer_size` is not zero: a short or empty caller `info` span means "not requested" and draws pool scratch.
* `getrf_cta_debug_launch<T>(ctx, m, n)` packs G (matrices per work-group) in the low 16 bits and the work-group width in the high 16. It answers 0 when the panel is not resident. `G > 1` implies sub-group-scoped barriers; see [packed resident leaves](#packed-resident-leaves).
* `getrf_blocked_debug_params<T>(ctx, n)` packs nb in the low 16 bits and the leading panel's leaf in the high 16 (1 = local, 2 = global).

### The tiny getrf window

**2026-09-14.** The tier shipped pin-only in P1; this grid gave it its first window. Today's rows widen it: float n = 5..32, cfloat n = 5..7 and 9..24.

Grid: `benchmarks/results/p8_getrf_tiny_window.csv` (80 cells x 3 arms), one process per cell, 9 reps. Ratio is `t_vendor / t_tiny`; gate 1.11. Italic cells are below the gate. Rows 12, 20 and 28 are omitted; they lie between the rows shown.

| n | float 8k | 16k | 32k | 65k | cfloat 8k | 16k | 32k | 65k |
|---|---|---|---|---|---|---|---|---|
| 4 | 1.662 | 1.415 | 1.192 | *1.020* | 1.475 | 1.265 | *1.016* | **0.833** |
| 8 | 1.644 | 1.482 | 1.400 | **1.303** | 1.471 | 1.295 | 1.181 | *1.117* |
| 9 | 1.772 | 1.743 | 1.687 | **1.664** | 1.385 | 1.360 | 1.469 | **1.540** |
| 16 | 1.668 | 1.642 | 1.640 | **1.663** | 1.344 | 1.302 | 1.599 | **1.416** |
| 17 | 1.023 | 1.085 | 1.106 | **1.143** | 0.397 | 0.363 | 0.377 | **0.368** |
| 24 | 1.058 | 1.079 | 1.084 | **1.140** | 0.457 | 0.524 | 0.518 | 0.508 |
| 32 | 1.192 | 1.077 | 1.123 | **1.225** | 0.626 | 0.631 | 0.643 | 0.646 |

**The two edges the 65,536 rung could not settle** (both within 0.01 of the gate and falling with batch):

| cell | 65,536 | 131,072 | effect |
|---|---:|---:|---|
| float n = 4 | 1.059 | **0.888** | stays out (a measured loss) |
| cfloat n = 8 | 1.122 | **1.086** | **excluded** |

**Why the floors sit where they do.** The kernel is instantiated on an N-bucket ladder {8, 16, 32}:

* Both types lose at n = 4: the N = 8 register array is half-full, and the vendor's per-batch overhead is amortised while ours is not.
* cfloat's cliff at 17 (1.416 -> 0.368) is the sharpest edge: n = 17 pads into the N = 32 array. float absorbs it. The column bucket later removed that padding cost.
* float's ceiling of 32 is the tier's limit, not a measured loss (`can_run(tiny)` refuses n = 33). A wider register kernel is untested, not refuted.

**Guard arming.** Removing the `tiny_max_n < 1` gate and the Blocked anti-overlap line stayed green, because the ceiling test already refuses those orders. Both were kept as defence in depth, not counted as coverage.

**What moved end to end.** `auto` read 1.000-1.007 against the vendor at these orders: Auto was taking cuSOLVER everywhere. fp64 gets no tiny-first row; on this part fp64 runs at 1/64 the fp32 rate, so the ratio measures a crippled unit.

### LU: the panel leaf is not the tier ceiling

`getrf_leaf_fits` is deliberately not occupancy-scaled, and the blocked driver asks it at the whole budget. The residency question is "resident tile or global stream". At float n=512, nb=32 the panel is 65,664 B: resident at 97,280 B, not at 24,320 B, and getrf's 1.25-2.35x wins at n >= 256 run through it. The launcher's `internal_error` asserts that advertised implies resident.

### Packed resident leaves

`getf2_panel_device` is templated on `LuScope`. Under `SubGroup` every barrier is sub-group scope, the cross-team argmax scan and its B1 disappear, and `getrf_leaf_launch` places `pack_matrices_per_wg` matrices per work-group, one per sub-group. Offered only for `m <= 32 && n <= 32`. Under `WorkGroup` scope, `G > 1` is a race by construction, so the launcher derives both together.

| type | n | batch | vendor ms | before ms | after ms | ratio |
|---|---:|---:|---:|---:|---:|---:|
| float | 8 | 32768 | 0.0319 | 0.1216 | 0.0680 | **0.559** |
| float | 32 | 16384 | 0.3159 | 0.5730 | 0.4596 | **0.802** |
| double | 16 | 32768 | 0.4088 | 1.4945 | 0.7790 | **0.521** |
| cfloat | 16 | 32768 | 0.2280 | 0.3422 | 0.2233 | **0.653** |

Packing is not cheaper per matrix: each matrix gets 32 lanes instead of 64 to 512. The win is barrier scope and matrices in flight per work-group, which is why it is gated to the band where one sub-group is enough. Registers are not the reason (SubGroup instantiations use 33 / 49 / 35 / 62 registers for float / double / cfloat / cdouble, against 49 / 62 / 52 / 46 for WorkGroup, zero spill).

## LU: the tiny tier

`getrf_tiny.cc`: the register-resident kernel for order `n <= 32` (cdouble 16). One matrix per `SubGroupPartition<N>`, N in {8, 16, 32} (the N = 4 bucket and column bucket came later). Lane `r` owns row `r` in a `D rA[N]` register array. `32/N` matrices per sub-group and `tiny_native::kTinySubGroups` (2) per work-group, so 8 / 4 / 2 matrices per 64-wide group. Rows and columns past `n` carry the identity, so the unrolled body needs no lane guard. Pivoting is a lazy relabel (`rowid`), never a row move. The pivot metric is `cabs1`; `ipiv` is 1-based and global; elimination continues finitely past a zero pivot (MAGMA's `update = 0`).

Zero local memory and zero barriers of any scope: `ptxas` reports `used 0 barriers, 0 bytes smem` for all 22 entry functions. The (47104, 49664] launch hole is structurally unreachable here.

### The register probe, and the unroll that decides it

`scripts/register_probe.sh out.log '' batchlas_extensions_cta`: 22 entry functions (4 types x 3 N, less cdouble N=32, each also `_with_offset`).

| type | N=8 | N=16 | N=32 |
|---|---|---|---|
| float | 48 | 63 | 96 |
| double | 64 | 90 | 138 |
| cfloat | 62 | 86 | 143 |
| cdouble | 94 | 138 | not instantiated |

Every cell: stack frame 0, spill stores 0, spill loads 0, smem 0, barriers 0. Worst `regs x work_group_size` = 143 x 64 = 9,152 against 65,536. `kWorstRegsPerThread` is 143 and static-asserted.

Two spellings decide whether `rA` stays in registers. Neither is diagnosed as an error:

1. `if (j >= n) break;` inside `#pragma unroll` makes the trip count data-dependent. The unroll is declined, `rA` is dynamically indexed, and ptxas moves it to the stack with zero spill. Use `continue`; the guard is kernel-uniform.
2. `for (k = 0; k < N; ++k) { if (k <= j) continue; ... }` asks for N(N+1)/2 copies and folds half away. Start the loop at the bound: `for (k = j + 1; k < N; ++k)`.

Stack frame is the wrong gate for a spill hunt and the right gate for a residency claim: a register-resident design with zero spill and a frame equal to `N * sizeof(D)` is not register-resident.

### Pad rows and the argmax, corrected

Masking padded rows out of the pivot candidate set was thought load-bearing. Removing the mask changes nothing. The guard is the tie-break direction: the argmax orders ties toward the lowest `rowid`, and `rowid == j` is always live, so a pad row never wins a tie. Reversing the comparison elects a pad row and yields `ipiv[j] > n`. The mask is kept as intent.

The argmax carries a (magnitude, key) pair, not one packed `uint64`. Packing is exact only for 32-bit magnitudes, so the pair costs no more and is exact for double.

The NaN map is required. `ov > NaN` and `ov == NaN` are both false, so an unmapped NaN survives every butterfly round and lanes disagree about the winning lane.

### The pivot key encoding

`tiny_device.hh`'s `tiny_key(order_field, lane)` is the only key encoding. The ordering field, in the high bits, decides ties (lowest wins) and makes pad rows unable to win a tie. The winner's lane, in the low `kTinyKeyLaneBits = 8` bits, is carried only as a broadcast source. Every ordering field must be distinct within a partition: `getrf_tiny.cc` seeds non-candidates at `rowid + N`, below every candidate.

### Why the rank-1 update starts at `k = j + 1`

`for (k = j + 1; k < N; ++k)` is a compile-time bound once `j` is unrolled. The `continue` form asks the unroller for copies it then folds away (see [the register probe](#the-register-probe-and-the-unroll-that-decides-it)). The store guard is `lane < n`, equivalent to `rowid < n` and loop-invariant. The tier writes `info` once and never reads it, so it needs no zero pre-fill, and cannot serve as a blocked-driver panel leaf until one is added.

### Armed breaks

Each applied, built, observed red, and restored.

| break | observed |
|---|---|
| `act = (rowid > j)` -> `rowid >= j` | pivot row scales itself; residual red at every n >= 1 |
| drop `if (k >= n) continue` from the store | pad written; the pad assertion is the only one that fires at `ld = n+5` |
| `ok < key` -> `ok > key` in `tiny_argmax_pair` | `ipiv[0] = 3` where 1 is required; residual stayed green |
| `tiny_partition_id` -> `get_group_linear_id()` | 4 sub-groups alias 4 matrices; item 4 untouched |
| drop `rowid < n` from the candidate mask | **passes**: the tie-break direction is the guard |
| `lu_cabs1` -> modulus, in shared `getrf_cta_device.hh` | `TinyPivotsMatchLapacke` red on complex; tiny-vs-CTA cases stayed green, since both tiers read the same helper |
| `if (j >= n) continue` -> `break`, then read the probe | zero spill, so the headline gate stays green; 14 of 22 functions grow a frame of `N * sizeof(D)` |

Breaks (a), (d), (h) and (i) were re-armed after the work-group moved to 64.

### The work-group A/B

30 cells (4 types x n in {4, 8, 16, 32} x batch {4096, 16384}), 7 reps, arms interleaved: wg=64 faster by > 2% in 2 cells, slower in 2, within 2% in 26. **Decision: wg = 64**, because the tie goes to the shared constant, it halves the launch-tail waste, and it doubles register-gate slack (143 x 64 = 9,152 against 65,536, 7.2x, where wg=128 gave 3.6x). Occupancy arithmetic predicted 8-17% wins that did not appear: at these sizes the tier is bound by outstanding memory requests, not resident warps. **Occupancy is not this tier's figure of merit.**

### First timing: the tiny tier against cuBLAS and the CTA tier

First timing, wg = 64, every row correctness-checked. The tiny tier beat cuBLAS at float and cfloat n <= 16 at batch 16384 (1.28-1.65x), and lost at cfloat n=32 (0.64x). Double and cdouble lost ground as batch grew, because their FP64 units hit the compute roof while 32-bit types are still memory-bound. The kill criterion (float n=32 below 1.0x of cuBLAS at batch 16384) did not fire: 1.051x. Tiny beat the CTA tier it replaced in 14 of 15 cells, by 1.2-3.6x.

For small n the roof is sector-rounded, not `n^2`: at n=4 the N=8 bucket reads a 16-byte column from a 32-byte sector, so a ratio against `4 n^2 sizeof(T) batch / 950 GB/s` reads as failure against a ceiling it cannot reach.

### The pivot-margin gate on elementwise comparisons

Comparing a pivot sequence elementwise with another implementation is well posed only where the argmax is decided by more than rounding. Device FMA contraction and host arithmetic can pick different rows of equal-within-1-ULP candidates, and both factorisations are correct. Measured before the guard: double n=10, item 0, `ipiv[6] = 7` against LAPACKE's 9, at a margin of about 1 - 1e-16.

`host_getf2_with_margins` records `margin[k] = cabs1(runner-up) / cabs1(winner)` at each step. Every elementwise pivot assertion stops at the first step with margin within `kTinyAmbiguous` of 1. `TinyPivotsMatchLapacke` is the only case whose oracle shares no code with the kernel. It runs on random matrices, not dominant-permuted ones.

### The host dgetrf oracle is broken on this box

`LAPACKE_dgetrf` returns a wrong factorisation for every n >= 10 on this machine: residual `||PA - LU||_F / ||A||_F` is 4.20e-01 at n=10 and 1.10 at n=16, measured in a standalone C program. sgetrf is correct at every order. The cause is this box's known-bad OpenBLAS double GEMM kernel (`dgetrf2` issues dgemm from n=10). The test suite therefore uses a BLAS-free triple-loop `getf2` as the primary oracle and cross-checks LAPACKE only where LAPACKE passes its own residual check.

### Why the tiny-vs-CTA element bound is relative

`TinyPaddingIsInertAgainstTheCtaRoute` compares against the CTA tier, which has no compile-time N and so cannot share a padding defect. The pivot sequence is compared exactly. The element bound is relative: factors may differ by a few ulp of the value they hold, times n steps. The allowance is `4 * n` ulp per element. An absolute bound scaled by the diagonal `4n` would license thousands of ulp where a padding defect shows. On this box the tiers agree exactly at every order, type and element.

### R7: the device link, all three tiny tiers landed

This is the one site for the tiny tier's R7 figure. `docs/perf/potrf.md` and `docs/perf/qr.md` point here.

`time cmake --build build/presets/dev-tests --target batchlas_extensions_cta --parallel 4` after touching `getrf_tiny.cc`: **167.4 s**, against a 117-125 s baseline: +34% to +43% for the library with all three tiers.

| tier | entry functions | ptxas | share of library |
|---|---:|---:|---:|
| `getrf_tiny` | 22 | 19.2 s | **10.5%** |
| `geqrf_tiny` | 22 | 19.4 s | 10.6% |
| `potrf_tiny` | 22 | 11.2 s | 6.1% |
| all three | 66 | 49.8 s | 27.2% |

getrf_tiny is inside the R7 budget of 15%. The three together are not: that is a property of the package, and whoever adds a fourth tier inherits it. N is what keeps `rA[N]` in registers; collapsing buckets to a runtime n costs the tier. Take the figure on an idle machine: a repeat under load measured 206.3 s.

### The register probe at wg = 64

Register counts match the wg=128 table. All 22 entries: frame 0, spill 0, 0 bytes smem, 0 barriers, worst 143 x 64 = 9,152.

### D3: why cdouble stops at N=16

`rA[32]` for `Cx<double>` is 128 registers before the working set; N=16 is already 138. The compute-to-DRAM ratio predicts cdouble N=32 at 1.16x against the vendor, not worth the spill risk. Reopening it means changing `tiny_cap<T>()`.

## The register panel leaf

**P4, 2026-09-11.** `src/extensions/getrf_panel_reg.cc` and `getrf_panel_reg_device.hh`: one work-group per `m x ncols` panel. Work-item `tid` owns physical row `tid` of a compile-time `D rA[NB]`, NB = 32, pivoting by lazy relabel and publishing the pivot row through NB words of local memory. It is a drop-in for `getrf_panel_factorize`: same global 1-based `ipiv`, same read-modify-written `info`, same `cabs1`.

**The register leaf is the default.** It is faster at 153 of 156 paired cells, by 1.01x to 2.13x. The exceptions are double n=32 at its three rungs. `BATCHLAS_GETRF_LEAF=slm` selects the local-memory panel. It moved cfloat's `getrf` floor from 512 to 256 at batch >= 256. The leaf is a derived spelling inside `blocked`, not a family. Its local-memory footprint is `NB*sizeof(D) + 32*(sizeof(R)+4)`, under a kilobyte for every type. The work-group is `roundup(m, 32)`, one row per work-item.

### Two barriers per column

The leaf executes two work-group barriers per column, against four (B1..B4) in `getf2_panel_device`. The third is removable: the argmax slots are written before B1 and read after it, the next write is at step j+1 with B2 of step j between; the published pivot row is written before B2 and read after it, with B1 of step j+1 between. Removing either breaks a WAR one column later. Dropping B2 is red from m = 33.

### The register panel leaf: register probe

`clang++ -fsycl -fsycl-device-only -S`, `llc -mcpu=sm_89`, `ptxas -arch=sm_89 -O3 -v`.

| type | registers | stack frame | spill st / ld | `regs x wg` at the cap |
|---|---|---|---|---|
| float | 64 | **0** | 0 / 0 | 64 x 512 = 32,768 |
| double | 104 | **0** | 0 / 0 | 104 x 512 = 53,248 |
| cfloat | 96 | **0** | 0 / 0 | 96 x 512 = 49,152 |
| cdouble | 176 | **0** | 0 / 0 | 176 x 288 = 50,688 |

This pipeline skips `sycl-post-link`. The device link differs by 0-3%: float 64 / 105 / 99 / 176 for float / double / cfloat / cdouble. A first version that used `inline` left the inliner declining the unroll, and every entry showed a 200-byte frame with zero spill; the accessor triple passed by value through an ABI call. `[[gnu::always_inline]]` removed the call and the frame together. A non-zero frame must be explained, not assumed to be `rA`.

### The register cap that binds is per sub-partition

The first version derived the height cap from `regs x wg <= 65,536`. At cdouble that allowed a 288-lane work-group, and the launch failed with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.

On sm_89 the register file is four sub-partitions of 16,384 registers, and a block's warps are dealt round-robin over them. What has to fit is

```
ceil(warps / 4) * 32 * ((regs + 7) & ~7) <= 16,384
```

equivalently `threads <= 128 * floor(512 / regs_alloc)`. The `& ~7` is ptxas's allocation granularity. The rules coincide only when the work-group is a multiple of 128 wide. The site now calls `resident::sm89_max_work_group(regs)` (`src/util/resident_capacity.hh:90-103`, used by `getrs_fused.cc:99`). Any site still dividing into 65,536 uses the wrong spelling. `getrs_fused.cc` had four cells on or past the partition limit (for example double NoTrans nrhs 5..8, exactly at it).

Device-link register counts (1,082 entry functions, 0 with spill):

| type | standalone probe | device link | budget (15% + granularity) | warps/partition | height cap |
|---|---:|---:|---:|---:|---:|
| float | 64 | **64** | 80 | 6 | 512 (`kPanelRegMaxRows`) |
| double | 104 | **105** | 120 | 4 | 512 |
| cfloat | 96 | **99** | 112 | 4 | 512 |
| cdouble | 176 | **176** | 208 | 2 | **256** |

The height cap is `min(512, 128 * floor(512 / regs_budget))`, rounded to a sub-group. One step is 128 lanes wide, so rounding matters: a probe of 148 registers carries to 170 and would allow 384 lanes, but ptxas allocates 176, where 384 lanes aborts.

**Guard that was missing.** `RegPanelCapacityIsOneSpelling` compared two arithmetic spellings and never launched at the cap. It now factorises an `m = cap` panel and asserts the launch does not throw.

### LU: the register leaf A/B

**2026-09-11, P4 integration grid.** GPU 1, one process per cell, three arms interleaved: `vendor`, `blocked` (local-memory leaf) and `blocked/leaf=reg`. 156 paired cells, zero flagged bad (worst rel_sd 0.0206). Raw: `benchmarks/results/p4_leaf_getrf_{float,cfloat,double,cdouble}.csv`.

`slm/reg` is the leaf ratio; > 1 means the register leaf is faster. Rungs are half / nominal / double of `SAT_LADDER`.

| n | rungs | float slm/reg | cfloat slm/reg | double slm/reg | cdouble slm/reg |
|---|---|---|---|---|---|
| 32 | 4096/8192/16384 | 1.78 / 2.13 / 2.03 | 1.95 / 1.48 / 1.48 | **0.90 / 0.93 / 0.89** | 1.79 / 1.82 / 1.84 |
| 33 | 4096/8192/16384 | 1.63 / 1.70 / 1.68 | 1.24 / 1.26 / 1.28 | 1.70 / 1.71 / 1.72 | 1.28 / 1.28 / 1.29 |
| 64 | 4096/8192/16384 | 1.37 / 1.31 / 1.33 | 1.18 / 1.16 / 1.17 | 1.50 / 1.47 / 1.47 | 1.35 / 1.34 / 1.34 |
| 128 | 2048/4096/8192 | 1.36 / 1.31 / 1.31 | 1.14 / 1.14 / 1.14 | 1.62 / 1.61 / 1.61 | 1.38 / 1.37 / 1.37 |
| 256 | 1024/2048/4096 | 1.13 / 1.11 / 1.11 | 1.08 / 1.08 / 1.08 | 1.30 / 1.29 / 1.29 | 1.14 / 1.14 / -- |
| 512 | 256/512/1024 | 1.08 / 1.07 / 1.06 | 1.06 / 1.08 / 1.07 | 1.08 / 1.08 / 1.08 | 1.02 / 1.02 / -- |
| 1024 | 256/512 | 1.01 / 1.01 | 1.01 / -- | 1.01 / -- | -- |

The one losing cell is double n=32: 0.892 / 0.932 / 0.900 at batch 4096 / 8192 / 16384. The register leaf runs one 32-row panel per 32-lane work-group, capped at 50% occupancy, while the local-memory leaf packs G matrices per work-group. Auto never reaches this cell (double rows read `vendor cta tiny blocked` at n <= 32), so the default flip does not ship it. G-packing for the register leaf is owed.

The ratio decays with n: the leaf is the whole call at n=32, about half at n=128, a minority at n >= 384. Only the leaf ratio is flat in batch, so no window below 256 is written from this grid. Not established: m > n panels, batch < 256 except the probes below, and ncu or nsys counters for the 1.3-2.1x.

### Which native tier serves 33 to 256

The leaf A/B says the register leaf is the better leaf, not that blocked is the better tier. The same harness ran `vendor | cta | blocked/leaf=reg` at nominal batch, with a coverage readback per arm (`benchmarks/results/p4_tier_getrf_float.csv`; route-era spellings in the readback column).

| n | batch | vendor ms | cta ms | blocked+reg ms | best native |
|---:|---:|---:|---:|---:|---|
| 33 | 8192 | 0.666 | 0.610 | **0.585** | blocked, by 1.04 |
| 48 | 8192 | 1.301 | 1.016 | **0.962** | blocked, by 1.06 |
| 64 | 8192 | 2.647 | **1.628** | 1.840 | **CTA, by 1.13** |
| 65 | 4096 | 1.558 | 1.856 | **1.144** | blocked, by 1.62 |
| 96 | 4096 | 3.196 | 3.191 | **2.968** | blocked |
| 128 | 4096 | 5.785 | 5.778 | **5.543** | blocked |
| 256 | 2048 | 31.557 | 23.966 | **21.639** | blocked |

At n >= 96 the cta pin is refused (P7 dropped float's CTA ceiling to 77), so that column is a second vendor arm there.

Two findings, neither fixed:

1. **float n = 65 takes the wrong native tier in a vendor-free build.** Float rows at 33..255 read `vendor cta blocked`, so CTA runs wherever it fits, up to 77. CTA at 65 is 1.62x slower than blocked. The repair is a float row at 65 ranking blocked first, bracketed by the 64 cell; the grid steps 64 -> 96, so it cannot say that today.
2. **The float window below 256 is a batch question.** At the nominal batch, vendor / blocked+reg clears 1.11 at n = 33..65 (1.15-1.69). At small batch it does not: float n=40 reads 0.71 at batch 2 and 0.68 at batch 32 against the vendor, 1.22 at batch 1024. **Recorded, not shipped**: float rows at 33..255 rank `vendor` first at every batch.

### The cfloat window moves to 256

The cfloat getrf window gained the band 256..511 at batch >= 256. `>= 512` is unchanged and batch-free. The clause is `order >= 512 || (order >= 256 && batch >= 256)`.

**(1) The register leaf makes 256 clear the gate.** The 512 floor was set by a cell at 0.885 that used the local-memory leaf.

**(2) The band loses at batch 64..128.** Batch ladder at n=256, interleaved, 9 reps, rel_sd <= 0.005 (`benchmarks/results/p4_cfloat256_batch_ladder.csv`):

| batch | vendor ms | slm leaf ms | reg leaf ms | vendor/reg | slm/reg |
|---:|---:|---:|---:|---:|---:|
| 32 | 1.186 | 1.242 | 1.098 | 1.081 | 1.131 |
| 64 | 1.192 | 1.337 | 1.200 | **0.993** | 1.114 |
| 128 | 1.431 | 1.656 | 1.552 | **0.922** | 1.067 |
| 192 | 2.656 | 2.663 | 2.340 | 1.135 | 1.138 |
| 256 | 3.860 | 3.288 | 3.003 | 1.285 | 1.095 |
| 4096 | 78.000 | 73.705 | 68.073 | 1.146 | 1.083 |

The cuBLAS fused kernel is flat from batch 32 to 64 while the blocked driver's four kernels per step are not; that is the dip. The bracketing non-winner on the batch axis is 128 (0.922); 192 is the first winner. On the order axis, n = 192 is the bracketing non-winner (1.10 / 0.92 / 0.86 down its ladder); n = 257 (1.34-1.53) says 256 is not a knife edge.

**Measurement trap.** The first probe read the two leaves as identical (3.036 vs 3.016 ms). It ran with a plain `blocked` arm after the default flip, which is now `reg`, so it compared the register leaf with itself.

**End to end, automatic route** (`benchmarks/results/p4_e2e_getrf_cfloat.csv`, `p4_e2e_getrf_float.csv`), before the batch term was added: cfloat n=256 at batch 1024 / 2048 / 4096 reads 1.295 / 1.191 / 1.148; cfloat n=512 b512 reads 1.805; float n=512 b512 reads 1.893 (all `native:blocked`). Controls at n=192 read 1.002-1.004.

Not established: float's 256 floor at batch 64..128 with the register leaf, and cfloat n >= 512 at small batch.

### Armed breaks (P4)

Eight breaks were armed and observed red, including: dropping the `rowid >= j` candidate mask (an eliminated row wins), dropping barrier B2 (red from float m = 33), and restoring the `regs x wg <= 65536` cap (the cdouble launch fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES`). Publishing the pivot row before the relabel stayed green, since it is a no-op refactor.

### The P7 occupancy change moved 27 committed `getrf` baseline cells

Arithmetic, no GPU: the advertised ceilings are 77 / 54 / 54 / 38 at target 4, and 155 / 109 / 109 / 77 at target 1. The committed P0 baselines (`benchmarks/results/factor_baseline_getrf_*.csv`, taken before P7) record `native:cta` up to the old ceilings:

| type | last order recorded as `native:cta` | advertised ceiling now | orders whose native arm moved |
|---|---:|---:|---|
| float | 129 | 77 | 96, 128, 129 |
| double | 32 (blocked before cta from 33) | 54 | none |
| cfloat | 96 | 54 | 64, 65, 96 |
| cdouble | 65 | 38 | 48, 64, 65 |

Three orders x three batches x three types = 27 paired cells no longer resolve `native` to the route the CSV records. Blocked is slower than CTA at each (`blocked_ms / cta_ms` of 1.49 at float n=96 b8192, up to 1.39 at cfloat n=64).

**Not fixed.** `can_run(cta)` asks the occupancy-scaled ceiling, so no row can send these orders to CTA. Undoing it needs a CTA-against-blocked grid between the two ceilings and a decision on whether the occupancy target belongs in a correctness gate (R3). The baseline CSVs must be re-taken before any P4 A/B is scored against them.

## The fused gesv tier

`src/extensions/gesv_tiny.cc`: one launch for `A X = B` at n <= 32 (cdouble 16) and nrhs <= 4. `getrf_tiny`'s elimination carries the RHS in `D rB[NR]` alongside `D rA[N]`, with forward substitution fused into elimination and back substitution in the same kernel. The `blocked` family is the composition: public `getrf` then public `getrs`, each selecting from its own table (`gesv.cc:60-66`). `BATCHLAS_GESV_ROUTE=tiny` reaches the kernel at any shape `can_run(tiny)` admits.

* **No batched vendor gesv exists on any backend.** cuBLAS, cuSOLVER and rocSOLVER ship none, so `GesvChoice` has no vendor family and a `vendor` pin runs Auto.
* **B is permuted for free.** Lane `lane` loads row `lane` of B, and its `rowid` moves with A's under the lazy relabel. No `laswp` is needed.

### P2: the measured gesv window

The flip was decided on a measured, interleaved A/B against the arm Auto took before. The arms: `tiny` (one fused kernel), `blocked` (`getrf`, `getrs` under Auto: the incumbent), `vendor` (vendor getrf and getrs), `composed` (tiny getrf, cta getrs) and `native` (the shipped walk). The flip is gated on `tiny` against **`blocked`**, the arm it replaces.

**Float: `t_blocked / t_tiny`** (ld = n + 1, 7 reps, `benchmarks/results/p2_gesv_float4.csv`). Every cell of n = 4..32, nrhs 1 and 4, batch 8192..32768 clears the 1.11 gate. The weakest is n=17 nrhs=1 at batch 32768 (1.15), which rises to 1.30 at 65536.

**Cfloat: the cliff at n = 17** (`benchmarks/results/p2_gesv_cfloat4.csv`):

| n | nrhs | b=8192 | b=16384 | b=32768 |
|---:|---:|---:|---:|---:|
| 16 | 1 | 2.68 | 2.85 | 2.52 |
| 16 | 4 | 1.93 | 1.83 | 1.98 |
| **17** | **1** | **0.84** | **0.66** | **0.68** |
| **17** | **4** | **0.50** | **0.45** | **0.51** |
| 24 | 1 | 0.76 | 0.79 | 0.83 |
| 32 | 4 | 0.60 | 0.72 | 0.76 |

n = 17 is the bracketing non-winner, at 0.45-0.84x. It is the first order served by the `N = 32` kernel, and for complex, `rA[32] + rB[4] + rX[4]` costs more occupancy than the second launch saves. Float wins 1.15-1.55x across 17..32, so the cfloat verdict does not transfer to float.

**Top edge is structural.** For float the window ends at 32, the tier's ceiling. The composed family answers at n = 33. This is stated, not measured, and it is the one window edge with no measured non-winner beside it.

**Cfloat reversal at large batch (open risk).** Pushed past batch 16384, two cfloat shapes reverse: n=2 nrhs=4 reads 0.92 at batch 131072, and n=4 falls to **1.08** (nrhs 1) and **0.85** (nrhs 4) at 131072 (1.39 and 1.10 at 65536). The n = 4 reversal reproduces across processes. The window was not narrowed, because the gesv table has no batch key. **If a caller runs cfloat gesv at n <= 4 with batch >= 65536, re-measure this cell first.**

**Double and cdouble get no window.** Their ratios swing across n and nrhs with no contiguous winning band (double n=8 reads 1.55 at nrhs 1 and 0.87 at nrhs 4). Both stay on the composed arm.

**Route readback after the flip** (`benchmarks/results/p2_route_readback.txt`): float n=32 -> tiny, n=33 -> blocked; cfloat n=16 -> tiny, n=17 -> blocked; double and cdouble n=16 -> blocked. The kill criterion did not fire (float n=32 nrhs=1 reads 1.19-1.25 against the native arm).

**Not established.** `linalg::solve` was not rewired, so no end-to-end caller A/B exists for gesv. No nrhs = 2 or 3 cell was timed; both run NR = 4 and are correctness-tested.

### LU: P2: the register bound is assumed, not probed

`gesv_tiny.cc` carries `kWorstRegsPerThread = 224` and `posv_tiny.cc` carries 256, each in a static assert against the 65,536-register per-block file at `kTinyWgSize = 64`. Neither was measured; they are getrf_tiny's 143 and potrf_tiny's 176 plus margin. The real gate is the probe's stack-frame column: `scripts/register_probe.sh` on `batchlas_extensions_cta` must report frame 0 and spill 0 for every `GesvTinyKernel` and `PosvTinyKernel` before either tier is timed. A non-zero frame shows up as a slowdown, never as a wrong answer.

## The N=4 bucket and the CTA band

**2026-09-26.** Three changes, measured at batch 131072 with gate 1.11:

1. **`getrf_tiny` gains an N = 4 bucket**: eight matrices per sub-group instead of four half-empty N = 8 partitions. At batch 32768, `t_vendor / t_tiny` moves float n=4 1.22 -> 1.45 and cfloat 1.04 -> 1.40. fp64 still loses (double 0.72, cdouble 0.68) and gets no tiny-first row.
2. **Native order**: the vendor-free walk takes Tiny at every n <= 8 for float and cfloat.
3. **Vs-vendor windows.** Decisive cells (`vendor/tiny`, `vendor/cta`):

| cell | vendor/tiny | vendor/cta | ships |
|---|---:|---:|---|
| float 5 | 1.46 | 0.37 | tiny |
| float 7 | 1.32 | 0.36 | tiny |
| float 17 | 1.14 | **1.31** | CTA |
| float 22 | 1.15 | **1.19** | CTA |
| float 24 | **1.18** | 1.11 | tiny |
| float 32 | **1.29** | 0.73 | tiny |
| cfloat 5 | 1.15 | -- | tiny |
| cfloat 7 | 1.24 | -- | tiny |
| cfloat 8 | 1.10 | -- | vendor (under the gate) |

float n = 23 sits between 22 (CTA) and 24 (tiny) and was assigned to tiny unmeasured. The float CTA band at 17..22 was retired the next day ([tiny launch bound](#the-tiny-launch-bound)).

**Still losing** after this pass: cfloat 17..32 and cfloat 64 / 128 on Blocked. [The column bucket](#the-column-bucket) later moved cfloat 17..24 to tiny.

## The right-hand gather

**2026-09-26.** nsys on cfloat n=128, batch 4096, native blocked getrf: **58% of kernel time was `LuLaswpKernel`**, the in-loop right-hand interchange, one sector per element. `lu_laswp_right_gather_launch` (`lu_laswp.hh`) gathers instead: one group per (column tile, item), parallel over columns as well as batch, writing back only rows the permutation moves.

Walk / gather ratio (`BATCHLAS_GETRF_RIGHT_LASWP`):

| n (batch) | float | cfloat | double | cdouble |
|---:|---:|---:|---:|---:|
| 64 (16384) | 1.45 | 1.38 | 1.20 | 1.17 |
| 128 (8192) | 1.91 | 1.97 | 1.52 | 1.20 |
| 512 (512) | 1.80 | 1.35 | 1.19 | 1.04 |
| 1024 (128) | 1.21 | 1.03 | 1.01 | 0.98 |
| 2048 (32) | 0.95 | 0.88 | 0.94 | 0.99 |

The gate is per block step on the trailing height `R = n - j0`, swept at n = 1024 and 2048. Thresholds: float **1024**, cfloat and double **768**, cdouble **384** (`getrf_right_gather_max_rows`). At every swept n the gated arm is within 0.5% of the better pure arm or ahead of both. `LuTest.RightInterchangeSpellingsAgreeBitForBit` asserts the spellings are bit-identical.

Native vs cuSOLVER at benchviz batches, before and after the gather: getrf float n=256 1.47 -> **4.26**; getrf cfloat n=128 0.56 -> **1.05**; gesv cfloat n=128 0.61 -> **1.10**. cfloat 17..32 took Blocked in the vendor-free walk; superseded by [the column bucket](#the-column-bucket).

## The tiny launch bound

**2026-09-27.** ncu on tiny getrf, cfloat n=32, batch 32768: 143 registers per thread, 25% theoretical occupancy, latency-bound for want of warps. `[[intel::min_work_groups_per_cu(N)]]` caps registers for occupancy, but DPC++ refuses it on a lambda, so the kernel is now the functor `GetrfTinyBody<D, N, Mpw, MinBlocks>`.

Tiny arm, batch 32768, ms (MinBlocks 1 / 8 / 12 / 16):

| cell | 1 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|
| float 24 | 0.363 | 0.375 | 0.334 | **0.277** |
| float 32 | 0.557 | 0.560 | 0.501 | **0.469** |
| cfloat 24 | 0.938 | 0.781 | **0.654** | 0.739 |
| cfloat 32 | 1.300 | **1.051** | 1.182 | 1.377 |

`getrf_tiny_min_blocks`: float N=32 -> 16, cfloat N=32 -> 8, cfloat N=16 -> 16, everything else 1 (fp64 unmeasured). At batch 131072, tiny beats CTA at every float order 17..32 (tiny / CTA vendor ratio 1.47 / 0.73 at n=32), so the float CTA band is retired and the float tiny window is 5..32. cfloat tiny still loses at 17..32 and stays out.

`gesv_tiny` takes the same functor and a smaller table (`gesv_tiny_min_blocks`): float N=32 and cfloat N=16 at 16 blocks, 1.07-1.09x.

## The column bucket and the local-memory pivot row

**2026-09-27, second pass.** Three levers on the tiny kernels, measured at batch 32768 with interleaved arms, rel_sd <= 0.05.

### The column bucket

The lane bucket N also sized the register row, so n=17 paid for 32 columns of registers even though `continue` skipped the work. `rA[NC]` now takes a compile-time column count (`tiny_col_bucket`), with rows still N lanes. Best launch bound, ms, NC = N against NC = bucket:

| cell | NC = N | NC = bucket | gain |
|---|---:|---:|---:|
| cfloat n=17 | 0.542 | 0.233 (NC 20) | 2.32x |
| cfloat n=24 | 0.766 | 0.519 (NC 24) | 1.48x |
| float n=17 | 0.180 | 0.129 (NC 20) | 1.40x |

`getrf` instantiates every even NC from 10 to 32 for the 32-bit types. `gesv` and `posv` take {12, 16} and {20, 24, 28, 32}. fp64 keeps NC = N.

**Unroll threshold.** LLVM declines `#pragma unroll` above a size of 16 Ki. Above it, `rA[]` is dynamically indexed and ptxas puts it on the stack: 256 B frame, zero spill, green tests. The three tiny TUs compile with `-mllvm -pragma-unroll-threshold=262144` (`src/CMakeLists.txt`). A remark (`-Rpass-missed=loop-unroll`) is the only trace.

### The local-memory broadcast

The rank-1 update needs the pivot row on every lane: `N - j - 1` shuffles, two per complex element. The pivot lane now stores its row as 16-byte vectors into a per-partition local buffer, one sub-group barrier, and every lane reads it back as a broadcast. Double-buffered by step parity, so one barrier per step orders each store after every read two steps back. 512 B of local memory per work-group at N = 32.

| cell | shuffles | local row | gain |
|---|---:|---:|---:|
| cfloat n=17 | 0.233 | 0.193 | 1.21x |
| cfloat n=32 | 1.064 | 0.844 | 1.26x |
| float n=32 | 0.453 | 0.312 | 1.45x |

The launch bound was re-swept per (type, NC) (`getrf_tiny_min_blocks`): cfloat 12 at NC 18/20/26/28, 16 at 22/24, 8 at 30/32; float 16-20, flat within 3%. Too tight is the expensive direction: cfloat NC 32 at 20 blocks is 1.92 ms against 0.86 at 8. Still latency-bound after this change (the argmax butterfly's serial chain).

### The RHS pad-column cost

gesv and posv ran nrhs = 2 as NR = 4 with two zero columns and measured slower than nrhs = 4 (gesv cfloat n=9 took 0.082 / 0.451 / 0.357 / 0.268 ms at nrhs 1 / 2 / 3 / 4). Two defects: the solve steps evaluated `tiny_select(use_mul, mul, div)`, paying the complex divide on every step, and a zero pad column sends that divide down its slow path. The fix is a branch on partition-uniform flags, loops stopping at `nrhs`, and an NR = 2 bucket. gesv cfloat n=9 nrhs=2 moves 0.453 -> 0.080 ms. Speed only; no functional test sees it.

### Negative results of this pass

* One 64-bit shuffle per complex element is identical to two 32-bit shuffles (NVPTX has no 64-bit SHFL).
* Two lanes per row for cfloat N=32: not built. It needs a work-group barrier per step; the kernel is latency-bound, not register-bound.
* Improving CTA for cfloat 17..32: not attempted. The tiny tier beats it by 1.4-1.9x there.

### End to end after the column bucket

Batch 32768, `t_vendor / t_arm`, arms interleaved. The `native` arm is the vendor-free walk.

| getrf cfloat | 9 | 12 | 16 | 17 | 20 | 24 | 28 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | 1.99 | 2.00 | 1.79 | 1.41 | 1.30 | 1.16 | 0.99 | 0.96 |
| best other native | 0.91 | 0.85 | 1.01 | 1.04 | 0.93 | 0.81 | 0.72 | 0.82 |

The cfloat tiny window is therefore 5..7 and 9..24; 25..32 ties the vendor and stays out. Before this pass the cfloat walk measured 1.03 / 0.92 / 0.81 / 0.74 / 0.82 at 17 / 20 / 24 / 28 / 32. In the tables: `tiny vendor cta blocked` at n = 5..7 and 9..24, and `vendor tiny cta blocked` at n <= 4, 8, 25, 32.

Float getrf tiny at 9..32 reads 1.92-2.68 at batch 131072 after this pass (was 1.30-1.57 at 17..32).

gesv cfloat with `tiny` against the composition reads 0.97-1.19x at n = 17..32, under the 1.11 gate on most cells, so the gesv window stays cfloat <= 16. Float gesv: tiny 1.97-4.99x against the vendor at every cell, 1.5-2.3x over the composition.

**Still losing:** getrf cfloat 25..32 against cuBLAS (0.96-1.03x). The vendor-free order takes Tiny there.

### Armed breaks (column bucket pass)

Run against `getrf_tests`, `gesv_tests` and `posv_tests` on GPU 1, then reverted. The route-layer breaks are table rows checked by `GetrfTranscribedTable.RowsHoldTheOldPreferenceOnBothDevices`. **Single buffer instead of the parity pair is green everywhere**: a blind guard, kept for the memory model. Other breaks went red in `T1`-`T11` (tiny), `G1` (gesv) and `P1`, `P7` (posv).

The RHS pad-column fix is a speed change only; no functional test can see it.

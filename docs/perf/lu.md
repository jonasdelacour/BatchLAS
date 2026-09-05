# LU: getrf, getrs, getri, the laswp gather, and the four routed windows (WP6 and the WP6/WP7 closure pass)

Two `getrf` tiers, two `getrs` tiers, one `getri` driver, and the row-interchange kernel all three share. This page records what routes, why each boundary sits where it does, what was built and measured worse, and what is still owed.

Measurement context, unless stated otherwise: 2x RTX 4090 (sm_89, 128 SMs, 97,280 B usable local memory per work-group, ~1008 GB/s DRAM, 1.29 TFLOP/s FP64, ~47 TFLOP/s measured FP32 GEMM), one GPU pinned per run, public API, in-process host oracle on every timed row, resolved route read off every row, nothing timed under `BATCHLAS_KERNEL_TRACE`.

**Naming.** `experiments/wp8_getrf/`, `wp8_getri/` and `wp8_getrs/` are a misnomer: they are the **WP6/WP7 performance-closure pass**, not WP8 (WP8 is sparse `spmm`, in `experiments/sparse_spmm/`). This page calls that work "the closure pass" throughout.

## What ships

### Route arms

| op | route order (`kGet??Order`) | native arms |
|---|---|---|
| `getrf` | `{Native,CTA}`, `{Native,Blocked}`, `{Vendor,Auto}` | CTA-resident leaf (`getrf_cta.cc`) and right-looking blocked driver (`getrf_blocked.cc`), sharing ONE `?GETF2` device body (`getrf_cta_device.hh`) |
| `getrs` | `{Native,CTA}`, `{Native,Blocked}`, `{Vendor,Auto}` | CTA = the **fused narrow-RHS kernel** (`getrs_fused.cc`); Blocked = the **composition** (`getrs_native.cc`: permutation + two routed `trsm`) |
| `getri` | `{Native,Blocked}`, `{Vendor,Auto}` | one arm (`getri_blocked.cc`): write `P` straight into `C`, then two routed `trsm`. Zero workspace. |

The blocked `getrf`'s trailing GEMM and panel TRSM, and both solves of `getrs`/`getri`, go through the ROUTER as injected `std::function`s -- never by calling `sycl_gemm::gemm_custom` from a kernel TU. Proved live rather than merely present: the same blocked `getrf` at n=256 runs `GemmRegister128x128Kernel` vendor-free and `ampere_sgemm_128x128_nn` in the vendor build.

Capacities are asked of the **runtime** local-memory budget, never of `device_limits.hh` (whose hardcoded 49,152 is 2.06x wrong here). On this box `GetrfShape::cta_max_n` measures **155 / 109 / 109 / 77** for float/double/cfloat/cdouble; `kGetrsFusedMaxRhs = 8` (a build fact, not a device one); `getrf` block width `nb = 32` for every type.

### The shipped `preferred()` windows

Four windows ship. Each is native-vs-**vendor**. `supports()` carries no speed term anywhere: a forced route bypasses `preferred()` but never `supports()`, so a speed gate there would make a pinned route fall through to cuBLAS and pass green over a kernel nothing executed.

`include/batchlas/blas/dispatch/route_getrf.hh:67-74`:

```cpp
static bool preferred(Route r, const GetrfShape& s) {
    if (!is_native(r)) return false;
    if (r.algo != Algorithm::Blocked) return false;      // CTA loses: 0.825/0.773/0.872 at float n=128
    if constexpr (std::is_same_v<T, float>)               return s.order() >= 256;
    if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 512;
    return false;   // double and cdouble earn nothing at any order
}
```

`include/batchlas/blas/dispatch/route_getri.hh:65-72`:

```cpp
static bool preferred(Route r, const GetriShape& s) {
    if (!is_native(r)) return false;
    if (r.algo != Algorithm::Blocked) return false;      // the only native arm
    if constexpr (std::is_same_v<T, float>)               return s.order() >= 128;
    if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 256;
    return false;   // double and cdouble earn nothing
}
```

`include/batchlas/blas/dispatch/route_getrs.hh:79-98` -- three clauses across two arms:

```cpp
if (r.algo == Algorithm::Blocked) {                      // the COMPOSITION -- clause C
    if (s.batch < 128) return false;
    if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 64;
    if constexpr (std::is_same_v<T, double>) return s.nrhs() >= 128;
    return false;                                        // cfloat, cdouble: nothing at any width
}
if (r.algo != Algorithm::CTA) return false;              // the FUSED tier
if (s.nrhs() <= 2) return true;                          // clause A, every type
if constexpr (std::is_same_v<T, float>) { if (s.nrhs() <= 4) return true; }  // clause B
return false;
```

**Correction to the exploration notes.** `experiments/wp6_lu/README.md`, `bench/README.md` and `kernels/README.md` all state that `preferred()` is false everywhere for all three ops. That was the WP6 merge state. The shipped predicates are the four windows above; **the code wins**. All three route headers say so in the WP8-ROUTING-PASS blocks preceding the predicates (`route_getrf.hh:67`, `route_getri.hh:65`, `route_getrs.hh:79`).

The stale sentence survives in more shipped sources than the exploration notes, and the list is longer than the earlier draft of this page gave: `getrf_cta.cc:5-6` and `:154`, `getrf_blocked.cc:38` and `:236`, `getrs_native.cc:2-3`, `getrs_native.hh:3-4`, `getri_native.hh:3-4`, `getri_blocked.cc:140`, and the two shape builders `src/backends/getrs_route.hh:95` and `getri_route.hh:50`. Every one of them says some form of "`preferred()` is false / all-false, so nothing routes here". None of them is true any more for `getrf`, `getrs` or `getri`. Read the predicate, not the prose above it.

**What clauses A and B actually moved, captured rather than reasoned about** (`route_diff.sh`, before/after with `preferred()` the only difference, `ctest -LE slow` both sides, `wp6_perf/README.md`): in the cuBLAS-present build, **27 decisions** moved `vendor:auto -> native:cta` -- float x18, double x3, cfloat x3, cdouble x3 -- and **`getrs` was the only op touched**, over 3600 decisions in 4012 rows. The vendor-free build moved 14, all at `Backend::AUTO`. A window that changes nothing is the failure mode this instrument exists to catch. [The `getrf`, `getri` and clause-C flips landed in the later closure pass; no equivalent `route_diff.sh` capture for them was found under `experiments/` -- unverified.]

### `native_tier_preferred()`

The native-vs-native tie-break, consulted **only** in the vendor-free walk, so declaring it moves nothing in a vendor-present build. That is exactly why it is the right instrument and `preferred()` is not: `preferred()` runs above that walk regardless of `vendor_available`, so a window written to fix the tier choice would also drag vendor-present traffic onto that tier.

`route_getrf.hh:78` -- `cta_max_order = 32` for `double`, `1 << 30` for the other three. Measured `blocked_ms / cta_ms` (>1 = CTA ahead), both arms **pinned** with every pin verified from the resolved route (four rows whose `cta` pin fell through above the capacity ceiling are excluded), from `experiments/wp6_lu/kernels/tier.txt`:

| type | n=64 (b8192) | n=76 (b8192) | n=96 (b8192) | n=100 (b4096) | n=128 (b4096) |
|---|---|---|---|---|---|
| float | 1.74 | 1.48 | 1.49 | 1.68 | 1.13 |
| cfloat | 1.39 | 1.59 | 1.30 | 1.33 | -- |
| cdouble | 1.37 | 1.09 | -- | -- | -- |
| **double** | **0.98** | **0.85** | **0.77** | **1.00** | -- |

`double` re-run across four batches at its worst order (n=76): 0.78 / 0.84 / 0.85 / 0.85 at batch 2048 / 4096 / 8192 / 16384 -- one-directional, flat in batch, every relative sd < 0.2%. `n <= 32` returns to CTA for `double` too, and that is not a hedge: there `nb = min(32, n) = n`, so the blocked driver runs one panel whose leaf **is** the CTA device function (1.8126 vs 1.8113 ms at n=32, batch 8192 -- the same code, one launch instead of three). Not declaring this hook would cost 1.18-1.29x at double n=76..96 in the build this campaign exists for.

`route_getrs.hh:102` -- CTA (fused) always preferred over Blocked. No crossover to encode: the fused tier is ahead of the composition at **every** cell inside its own capability (51 cells, worst 1.11x at float n=2048 nrhs=8). The column where it would turn is nrhs=16 (double 0.55x, cfloat 0.58x at n=512), and that is outside `supports()` by `kGetrsFusedMaxRhs`. **If that constant is raised, this predicate must gain a window in the same change.** `getri` declares none -- one native arm, no native-vs-native question.

## The vendor baseline and saturation

Every ratio here is against `cublas{S,D,C,Z}getr{f,s,i}Batched`, which -- unlike WP5's `orgqr` -- is genuinely batched and genuinely strong at small `n`. Vendor build, saturating batch (`experiments/wp6_lu/baseline/grid_norm.csv`, 280 cells, none discarded, none flagged BAD):

| n (batch) | float ms / GFLOP/s | double | cfloat | cdouble |
|---:|---|---|---|---|
| 32 (8192) | 0.159 / **1129** | 0.561 / 319 | 0.245 / 732 | 1.488 / 120 |
| 128 (4096) | 5.770 / 992 | 8.803 / 651 | 6.988 / 820 | 24.66 / 232 |
| 512 (512) | 49.73 / 921 | 59.53 / 770 | 76.82 / 596 | 163.0 / 281 |
| 2048 (32) | 519.2 / 353 | 587.9 / 312 | 564.0 / 325 | **2594 / 70.6** |

GFLOP/s is **not monotone in n** -- the shape of a routine with a small-n special case and no large-n blocking.

**cuBLAS does not saturate at n >= 1024 on any ladder this box can hold**, and the effect is large enough to invert conclusions. `cublasZgetrfBatched` at n=2048 takes 2575.9 / 2585.8 / 2587.7 / 2589.2 / 2591.0 / 2594.1 / 2659.3 ms at batch 1 / 2 / 4 / 8 / 16 / 32 / 64 (`baseline/summary_sat.txt`) -- **8x the work for 0.25% more time** from batch 4 to 32, and 64x the work for 3.2% from batch 1 to 64. (An earlier draft of this page quoted a batch-128 rung at 2801 ms; that ladder stops at batch 64 and no such point exists in the record.) `getrf` float n=1024 is 48.9 ms at batch 1 and 170.3 ms at batch 256. Native is the opposite: it saturates at batch 4-32 at n=2048, because it parallelises *within* the item.

Read at the A/B grid's batch schedule versus at each arm's own best batch (`experiments/wp6_lu/bench/`, 982 timed rows, 491 per arm, 0 discarded, max relative sd 7.2%):

| | geo at the grid's batch | geo at each arm's own best batch |
|---|---|---|
| `getrf` (28 cells) | 0.885x | **0.805x** |
| `getri` (28 cells) | 1.463x | **1.284x** |
| the n=2048 row (8 cells) | 7.098x | **2.954x** |

Individual collapses: `getrf` float n=2048 7.31 -> **2.33x**; `getri` float n=2048 33.84 -> **8.09x**; `getrf` cdouble n=2048 3.74 -> **1.05x** (and cuBLAS is *still* unsaturated at batch 128, the largest 24 GB holds, so 1.053x is an upper bound, not a measurement).

**The saturation caveat is not uniform, and the WP6-era blanket form ("every n >= 512 ratio is against an unsaturated vendor") is wrong in both directions.** `route_getri.hh` corrects it: at n=512 the vendor IS saturated for all four types (us/item moves under 0.2% over the last doubling); at n=2048 it is unsaturated for all four by 19-50% per doubling. That is why the shipped `getri` window quotes n=2048 at batch 128 and 256 and never at the batch-32 grid schedule, where float n=2048 reads 33.9x and means nothing.

**The roofline says which half is closed.** With each arm at its own best batch, cuBLAS runs cdouble `getrf` at 90% and 91% of this card's FP64 peak at n=512-1024 -- there is no 2x to find. For FP32 *both* arms sit at 1-10% of peak, and the cause is a decomposition, not a slow kernel (see [negative-results](#negative-results) item 1).

## Measured boundaries

Ratio is always `vendor_med / native_med`; > 1 means native wins. The closure pass's acceptance rule (GATE-C) is zero losses **and** zero cells below 1.15 across the admitted set.

### `getrf` window evidence

**Two sources, and they are not the same grid -- an earlier draft of this page ran them together.** The clause is *scored* on `experiments/wp8_getri/lu_c1.csv` (device 1, nothing else on the box, the two arms are two BUILDS run back to back on each cell, 11 reps, median, host oracle per row, foreign compute-process count 0 before and after every cell). Shipped set there: **15 cells, geomean 1.997, min 1.2626, zero losses, zero cells below 1.15** (`summary_c1.txt`: float `order>=256` 9 cells geo 2.009; cfloat `order>=512` 6 cells geo 1.978):

| type | n | b128 | b512 | b1024 |
|---|---:|---:|---:|---:|
| float | 256 | 1.2626 | 1.5658 | 1.5750 |
| float | 512 | 2.3455 | 1.7400 | 2.1783 |
| float | 1024 | 2.7782 | 2.2309 | -- |
| float | 2048 | 3.1115 | -- | -- |
| cfloat | 512 | 1.7748 | 1.6718 | 1.6065 |
| cfloat | 1024 | 2.1348 | 2.1035 | -- |
| cfloat | 2048 | 2.8017 | -- | -- |

The wider 20-cell grid that the route header transcribes -- the one with a `b256` column -- is the *kernel* stage's own record on the OTHER device (`experiments/wp8_getrf/after_nv_p{1,2}.csv` against `base_v_p{1,2}.csv`, `route_getrf.hh:67-74`). It is the second source, not the first:

| type | n | b128 | b256 | b512 | b1024 |
|---|---:|---:|---:|---:|---:|
| float | 256 | 1.254 | 1.279 | 1.567 | 1.675 |
| float | 512 | 2.350 | 1.988 | 1.737 | 2.183 |
| float | 1024 | 2.773 | 2.186 | 2.237 | -- |
| float | 2048 | 3.091 | -- | -- | -- |
| cfloat | 512 | 1.811 | 1.528 | 1.682 | 1.609 |
| cfloat | 1024 | 2.124 | 1.829 | 2.088 | -- |
| cfloat | 2048 | 2.754 | -- | -- | -- |

**Both boundaries are bracketed from below by measured non-winners**, which is why the two thresholds differ by one grid step. `float order >= 128` would steal the `native:cta` rows at **0.825 / 0.773 / 0.872** (batch 256 / 512 / 1024; the clean re-measure reads 1.0037 at b128 and 0.7757 at b512). `cfloat order >= 256` admits **0.8851** at batch 128, with 1.188 at batch 1024 still under the bar. `double order >= 512` admits **0.7486** at batch 1024. Double's best cell anywhere is 1.067 and cdouble's is 1.012 on the 20-cell grid (1.0816 and 1.0165 on the clean 15-cell re-measure) -- neither wide type earns a window at any order in either record.

Reproduction across sources rather than across repeats: 26 cells are common to this clean run and the kernel pass's own device-0 record (a different device, session and binary); median spread 1.0053, worst 1.0311, none above 1.10.

### `getri` window evidence

`experiments/wp8_getri/lu_c1.csv` and `lu_p1.csv`, every ratio quoted at the **worse** of the two passes. Shipped set: **30 cells, geomean 4.181, min 1.2028, zero losses, zero cells below 1.15** (`summary_p1c1.txt`), batch spanning 1 to 16,384.

| type | n | measured rungs (batch: ratio) |
|---|---:|---|
| float | 128 | b1 2.169, b2 2.163, b32 2.089, b256 1.819, b1024 1.556, b4096 1.262, b16384 **1.203** |
| float | 256 | b128 4.047, b4096 2.641 |
| float | 512 | b1 18.565, b32 15.103, b128 7.600, b1024 3.991, b2048 4.466 |
| float | 1024 | b128 9.033, b512 5.068, b1024 5.910 |
| float | 2048 | b128 9.331, b256 7.095 |
| cfloat | 256 | b128 1.977, b512 **1.682**, b2048 2.050, b4096 2.195 |
| cfloat | 512 | b1 24.276, b32 8.861, b128 4.138, b1024 3.807 |
| cfloat | 1024 | b128 5.970, b512 5.475 |
| cfloat | 2048 | b128 8.600 |

Bracketing non-winners for every wider clause:

* `float order >= 64`: n=64 **LOSES** at 0.8572 / 0.8561 (batch 8192) and 0.8499 / 0.8530 (batch 16384).
* `cfloat order >= 128`: n=128 is **0.7109 / 0.7070** at batch 512 -- an outright loss in the *middle* of its own ladder (2.63 at batch 1-4, 1.92 at 32, 0.968 at 2048). The WP6-era note's "crossover n ~ 128 for float and cfloat" read that cell at one batch and at no other.
* `double`, any order: refuted by the same cell for every threshold -- n=2048 batch=128 at **1.0848 / 1.0823**. n=1024 is 1.1602 / 1.1551, 0.5% above the bar and falling with the last rung lowest; n=256 is 1.124-1.129. Only n=512 is clean (1.297-1.312 over four batches), and a clause admitting exactly one order is the leg-predicate defect this campaign has already found twice.
* `cdouble`, any order: n=512 **LOSES** at 0.9542 / 0.9525 (batch 1024); n=1024 tops out at 1.135. Nothing at or above 1.15 anywhere at batch >= 128.

**No batch floor, and that is measured rather than assumed.** At batch 1-32 the native driver beats cuBLAS by 1.7x-28x for *every* type at every order measured, because cuBLAS's batched `getri` is a per-item loop there. A floor at 128 -- predicted as necessary -- would have given those away for nothing.

### `getrs` fused window evidence

Clauses A and B, the fused tier, from `experiments/wp6_perf/bench/` (`analyse_window.py`): **461 pooled cells** from seven sweeps, deduplicated on `(type, n, nrhs, batch)`, both arms' routes read from the printed route column on every row, `relsd <= 10%`. Re-running the analyser reproduces every figure below. Note that the *checked-in* `bench/window_summary.txt` is a stale 354-cell run from before the last two sweeps landed and scores C2/C3 at 187/215 cells; the README's table and this one are the current pool.

| clause | cells | geomean | min | losses |
|---|---:|---:|---:|---:|
| A -- `nrhs <= 2`, every type | 286 | **2.261** | 1.116 | **0** |
| ... `nrhs = 1` | 142 | 2.290 | 1.242 | 0 |
| ... `nrhs = 2` | 144 | 2.232 | 1.116 | 0 |
| B -- `float`, `nrhs = 3..4` | 36 | **1.611** | 1.133 | **0** |
| both | **322** | **2.177** | **1.116** | **0** |

On WP6's own saturating grid the reversal is complete: nrhs=1 goes from 0.256x (0 wins of 28) to **2.117x (28 of 28)**, nrhs=2 from 0.331x to 2.173x. WP6's worst cell in the whole LU family -- `getrs` cdouble n=512 nrhs=1 at **0.083x** -- is now **1.765x**.

**Flat in batch, and that took four passes.** Full ladders at n = 32, 64, 128, 256, 512, 1024, 2048, all four types, at nrhs=1, nrhs=2 and float nrhs=4: zero of 322 laddered in-window cells crosses 1.0 anywhere, lowest rung 1.116x. The n=32 and n=256 ladders (`flat4`) exist **only because a review caught their absence** -- before them there was no order-32 ladder in the directory at any width, so the small-n end of clause A, including its own stated minimum, rested on a single saturating batch point.

**Why clause B is float-only** -- the refuting mid-ladder dips: `double n=128` 0.940x at batch 2048 (1.363x at 256, 1.111x at 8192); `cfloat n=1024` 0.976x at batch 16; `cdouble n=128` 0.980x at batch 1024; `cdouble n=1024` 0.987x at batch 16; and `cdouble n=32` 0.577x outright. A dip in the *middle* of a ladder cannot be closed by any boundary in `n` or in batch. That killed candidates C4 (float `nrhs <= 8`, 3 losses, worst 0.686x), C6 (`nrhs <= 4` every type, 20 losses, worst 0.577x), C7 (the whole capability, 55 losses, worst 0.294x) and C8/C9 (`nrhs <= 4, n >= 128` plus a batch or order bound, 4 losses each at 0.940-0.987x) -- C8/C9 were the leading proposal until the third flatness pass measured the interior orders.

The **thinnest margin in the window** is cdouble n=32 nrhs=2, whose ladder runs 1.257 / 1.162 / 1.132 / 1.120 / **1.116** at batch 1024 -> 16384. It declines and then flattens rather than falling, so it is a flat win by the rule; it is the only cell of 322 under 1.12x and the first place a re-measurement on another box should look.

### `getrs` composition window evidence

Clause C, the composition, from `experiments/wp8_getrs/cl_*.csv` + `gap_*.csv` scored into `clause_summary.txt`, re-measured on an idle box in `experiments/wp8_getri/lu_c1.csv`. Union: **37 cells, geomean 2.60, min 1.2858, zero losses, zero cells below 1.15** (float `nrhs >= 64`: 22 cells, geomean 3.138, min 1.7695; double `nrhs >= 128`: 15 cells, geomean 1.979, min 1.2858). The earlier 45-cell reading of the same clause is geomean 2.467, min 1.2791.

Refuting cell for every wider clause, so none is rediscovered:

| rejected clause | refuting cell | ratio |
|---|---|---:|
| `float nrhs >= 32` | n=64 nrhs=32 b=4096 | 0.9069 |
| `double nrhs >= 64` | n=64 nrhs=64 b=2048 | 0.9984 |
| `cfloat nrhs >= 128` | n=64 nrhs=128 b=1024 | **0.9944** |
| `cdouble nrhs >= 128` | n=128 nrhs=128 b=1024 | 0.9238 (2 losses, 8 more between 1.00 and 1.15) |
| `cdouble nrhs >= 64` | n=128, every rung of the ladder (0.6928 / 0.7380 / 0.8304) | 12 losses of the clause's 26 cells |

**cfloat was in this clause until the coverage bound's own gap was measured**, and that is the methodological result of the pass. `cfloat nrhs >= 128` scored 15 cells, geomean 1.974, min 1.482, zero losses on the directly measured rungs -- a clean PASS. The coverage bound then named five admitted cfloat cells it could not cover; measuring them produced 0.9944 at n=64 nrhs=128 batch=1024, with **1.2901 at batch 512 and 1.4824 at batch 2048 on either side of it**. A mid-ladder dip, invisible to every candidate scored before the gap sweep existed.

**cdouble's losses cluster on the TYPE, not mid-ladder** (its n=128 nrhs=64 ladder loses at batch 1024, 2048 *and* 4096), which is why a per-type predicate can exclude them and why the clause is per-type rather than a single scalar. What is left losing for cdouble is the trsm/GEMM arm, not the permutation: the gather is worth only 1.04-1.26x there against 1.12-2.79x for float, exactly as the `(32 + sizeof(T)) / sizeof(T)` sector inflation predicts.

**The batch floor of 128 is a conservative policy choice and is known to be one.** At nrhs=128 the composition still WINS at batch 64 and 32 (float 5.93 / 5.96 / 5.60 / 4.71 / 3.87 at n=64/128/256/512/1024; double 4.31 / 4.05 / 3.56), so the floor gives up measured wins rather than excluding measured losses. It is there because below 32 the only readings come from a contaminated sweep (0.055x-0.33x at batch 1-2), and because nrhs=64 -- the other half of the clause -- has no low-batch ladder at all.

Coverage of the admitted set, stated exactly: 45 cells measured directly on three saturated rungs of each of five orders, two passes each side. A further 58 admitted cells at other rungs are covered by a **bound**, not a measurement: the vendor arm did not move in this pass, and the gather's own A/B has minimum 1.0004 over 80 cells with zero cells below 1.00, so `post_ratio >= walk_ratio` at every admitted cell -- and all 58 already clear 1.15 on the walk ladder (min 1.1933, geomean 2.1616). Zero admitted cells are uncovered by measurement or bound.

## The `laswp` gather

The row interchange is the single largest lever in the family, and the closure pass took it twice with the same mechanism.

**The cost, priced by a timing-only break** (both interchange passes disabled; answers wrong by construction), vendor-free, saturating batch:

| type | n (batch) | with laswp | without | laswp share |
|---|---|---:|---:|---:|
| float | 512 (512) | 40.44 ms | 10.53 ms | **74%** |
| float | 2048 (32) | 70.86 ms | 39.71 ms | **44%** |
| cdouble | 512 (512) | 232.09 ms | 181.97 ms | 22% |
| cdouble | 2048 (32) | 694.05 ms | 645.02 ms | 7% |

Confirmed independently by nsys: `LuLaswpKernel` is 44.2% of the best `getrf` cell (float n=2048 b=32) and **63.2%** at n=1024 -- two methods, same answer. Without it those two float cells run 4445 and 4652 GFLOP/s against cuBLAS's 918 and 354, so WP6's 0.886x `getrf` geomean was very largely this one kernel's number.

**The mechanism.** `ipiv` is a *sequence* of transpositions applied in order, so the only parallelism is over the columns it is applied to, and column-major puts consecutive columns `ld` apart. The k side is free (rows `j0..j0+ib-1` are consecutive, so one line per column serves all `ib` steps); the **p side is the whole cost** -- the `ib` selected rows are scattered over `[j0, n)`. `ncu` settled the unit: **one 32 B sector per 4 B element**, load and store (245,760 element touches -> 249,600 load sectors, 1.016 per touch), not one 128 B line. The line model would have implied 1,130 GB/s, above this device's DRAM peak, i.e. impossible on its face.

### `getrf` deferred left gather

`lu_laswp.hh`'s `lu_laswp_deferred_left_launch`, unconditional and shipped.

The gather amortises as `L/R` -- L transpositions against R staged rows -- so applied in place it only pays below a crossover, and the crossover is on `(n - j0)`, **not** on `n` and not on batch. Under LAPACK's schedule the right-hand pass has `L = ib = 32` against `R = n - j0`, so it pays only below `n - j0 ~ 288` (float) / 160 (double, cfloat) / 96 (cdouble). **Deferring the left-hand pass out of the block loop makes `L == R` exactly** -- the deferred list runs to the end of the matrix -- so the amortisation is 1:1 at every order and **no gate is needed at all**. A lever with no gate cannot carry the leg-predicate defect that has bitten this campaign twice. Writing that gate on `n` instead of on `(n - j0)` inverts it.

The correctness argument is a one-line composition identity: under LAPACK's schedule column block `r` receives the lists of panels `r+1..P-1` in order, which concatenated is exactly `[j0_{r+1}, n)` in increasing `k`, and no driver step ever reads a column below its own `j0` afterwards. Bit-for-bit the same composition; all three spellings (`inloop`, `defer_walk`, `defer_gather`, selectable through `BATCHLAS_GETRF_LASWP`) are asserted **bit-identical** by `LuTest.LeftInterchangeSpellingsAgreeBitForBit`.

Measured against the arm it replaces, interleaved inside one process, 11 reps, median, two passes, vendor-free, batch >= 128, 58 `native:blocked` cells: **geomean 1.207x, min 1.018x, zero cells below 1.00**, cross-pass median spread 1.0011 / worst 1.033. By type: float 1.350x, cfloat 1.305x, double 1.138x, cdouble 1.074x. The `native:cta` rows measure **0.9995x** -- the anti-vacuity check that the change cannot reach them. Sector traffic over the same composition: 249,600 against 1,966,080 for the walk, **7.9x fewer** and within 1.6% of the 245,760 that perfect coalescing predicts.

Against cuBLAS on the same 62-cell grid (batch 128-1024, order 128-2048, all four types) the `getrf` geomean moves **0.839x -> 1.002x**, 20 wins -> 28: float 1.273 -> 1.594, cfloat 0.974 -> 1.271, double 0.629 -> 0.716, cdouble 0.610 -> 0.659. double and cdouble are **not** closed by this and will not be closed by anything short of a register-resident fused panel.

### `getrs` collapsed permutation

`getrs_native.cc`'s `GetrsPermGatherKernel`, default at `nrhs >= kGetrsPermGatherMinNrhs = 16` (`getrs_native.hh:46`). The same collapse, in **local** memory: one work-group per item stages a column tile plus the index array in SLM, reads B coalesced, and writes `B[i] = tile[idxs[i]]` back to B's own addresses. A/B against the walk, interleaved rep by rep inside one process via `BATCHLAS_GETRS_LASWP`, two passes with the worse quoted, both arms' solutions asserted bit-identical on every row:

| nrhs | cells | geomean | min | max |
|---:|---:|---:|---:|---:|
| 1 | 20 | 0.9993 | 0.9953 | 1.0031 |
| 8 | 20 | 1.0011 | 0.9980 | 1.0256 |
| **16** | 20 | **1.1182** | 1.0004 | 1.2941 |
| 32 | 20 | 1.2148 | 1.0141 | 1.4482 |
| 128 | 20 | 1.5728 | 1.0411 | 2.7873 |

Admitted set (nrhs >= 16): 80 cells, geomean **1.3191**, min 1.0004, zero cells below 1.00 -- float 1.5799, cfloat 1.4766, double 1.2061, cdouble 1.0761. **The boundary is transcribed from a CSV, not inferred from an inequality**: the rungs at 2, 8 and 24 are a separate sweep (`ab_bnd_p{1,2}.csv`) run for exactly that reason, because the main grid samples 4 and then 16 and so brackets the boundary without measuring either rung it separates. Every one of the 50 cells that measured below 1.00 across both sweeps is at nrhs <= 8, and none is below 0.995. One row was refused for relsd > 0.10 (float n=64 nrhs=2 batch=8192) and is named rather than dropped. **cdouble is the marginal type at the boundary and is recorded as such rather than carved out** (1.0004 at n=512 nrhs=16): a per-type boundary would add a decision surface with no measured payoff, since cdouble at nrhs=32 is only 1.01-1.09 either way.

**The budgeted cost of this gather did not exist.** Both the plan and the source header priced it at "an out-of-place RHS plus an `int32[n]` per item" -- 67,371,008 B at n=2048 nrhs=64 batch=32 -- and then reasoned at length about the facade billing that to every nrhs=1 call. Two things are wrong with that pricing, and the source header names both. (i) **The figure was read at the wrong `nrhs`.** nrhs=64 is the width at which the gather *wins*, so it never carried the decision; at nrhs=1 -- the only width the library actually issues -- the same buffer is `n*batch*sizeof(T)` = **262,144 B, 257x smaller**, and the argument against buying it was never the memory but that at nrhs=1 the gather buys nothing measurable (0.9993 above; the loss there is in the two triangular solves). (ii) The cost was in any case an artefact of the *prototype* (`experiments/wp6_lu/baseline/lubench.cpp:162-192`, `perm_build` + `gather_rows`), which gathered into a separate global buffer and never copied back, so its 1.55x also omitted a full extra pass. Permuting in local memory removes both buffers and the copy-back: `getrs_blocked_buffer_size` still returns 0 at every shape and every width, and `LuTest.GetrsPermGatherBuysNoWorkspace` asserts it.

`getri` needs no permutation kernel at all: it writes `F` straight into `C`, each work-item tracing its own row backwards through the interchange list in registers, instead of writing `I` and permuting. Same store count, one kernel, zero workspace -- and that one choice is what moves the `getri` composition's geomean from 0.97x to 1.60x in the baseline survey.

### The fused narrow-RHS `getrs`

`getrs_fused.cc` -- one work-group per matrix, one launch: permutation + forward substitution against unit-lower L + back substitution against U, no GEMM and no separate laswp. It exists because the composed tier is 0.32x of cuBLAS at nrhs=1 and the loss is structural: `trsm`'s blocked driver amortises a panel over many columns, and one column gives it nothing to amortise. nsys on the composed arm (float n=512 nrhs=1 batch=512, one public call) shows 39.7% of the time in `GemmTiledGeneralKernel<float,16>` -- matrix-*vector* products run through a tile-16 GEMM kernel -- plus 26.4% in the separate `LuLaswpKernel`. The matrix is not resident and cannot be (n=512 float is 1 MB per item); what is resident is the RHS block and one `nb x nb` diagonal block.

Against the composition: **8.26x at nrhs=1**, 6.56x at 2, 4.40x at 4, 2.67x at 8; 107 pooled cells, zero losses, worst 1.419.

Design choices, each with its measured margin:

* **Resident diagonal block, not pure streaming**: 0.6506 vs 1.0102 ms (float n=512 b=512). Streaming pays a work-group barrier per *column*; the blocked form pays one per *block* and runs the `nb`-step recurrence inside one sub-group with shuffles. The streaming arm is a reverted variant with its number.
* **`nb = 16` below n=1024, 32 at and above**: n=2048 b=32 reads nb 8 / 16 / 32 = 1.5513 / 1.3772 / **1.2838** ms.
* **Work-group ~ n/2 clamped to [64, 1024]**, then capped by a per-`(type, body, width)` register table. The earlier max-over-everything cap charged `GetrsFusedNKernel<float,8>` (48 registers) the 86 of `GetrsFusedTKernel<complex<double>,8>`, capping wg at 672 instead of 1024. The repair is worth 1.027-1.062x at nrhs=8 -- and `double n=1344` measures **exactly 1.000 at every batch**, because there the kernel is bound by the dependent recurrence and not by thread count. A measured negative inside a kept change, and why it is scoped small. A second null in the same A/B: `float n=2048 nrhs=1` is **1.000 at every batch** too. And one cell was **discarded rather than reported**: `float n=2048 nrhs=1 batch=4` re-ran at 0.854 / 1.012 / 1.071 ms, a 25% cross-pass spread on 4 work-groups over 128 SMs -- not reportable in either direction.
* The `+1` bank-conflict pad on the block's leading dimension is **kept for portability, not for measurement**: eight cells timed on the transposed path (the only one whose recurrence can care), seven within 0.6% and split both ways, and the eighth (float n=512 nrhs=1) is 2.17% *against* the spelling that was kept.

DRAM fraction of 1008 GB/s at nrhs=1, per cell:

| n | float | double | cfloat | cdouble |
|---:|---:|---:|---:|---:|
| 32 | 72% | 38% | 95% | 24% |
| 512 | **82%** | **86%** | **88%** | **83%** |
| 2048 | 41% | 50% | 60% | 41% |

The original claim "82% of DRAM peak, the ceiling is reached" holds only in the n=256..512 band. Two named mechanisms, both open work rather than a ceiling: at large n the CTA count **is** the batch (32 work-groups on 128 SMs at n=2048); at small n `nb=16` leaves the block solve to 16 lanes of one sub-group.

## Negative results

Everything here was built or measured and then rejected. Re-deriving any of it is wasted work.

1. **"Four kernels per block step versus cuBLAS's one fused kernel" is not a launch-count problem.** The blocked arm launches `5P-4` kernels -- 16 at n=128, nsys-confirmed launch for launch -- which at 5 us is 80 us against a 67.1 ms call at batch 8192: **0.12% of the call**, and ~0.2% of the native-minus-vendor gap *there*. The number that matters is the worst one, and the shipped header (`route_getrf.hh:67-74`) records it: **8.7% of the gap at the smallest saturating batch**, falling monotonically with batch from there. Even at its worst the launches are not the gap. (`VENDOR_INDEPENDENCE_PLAN.md:1813` quotes only the 0.2%; that is the batch-8192 reading, not the bound.) And the fused arm already exists and already loses: float `n <= 155` resolves `native:cta`, one kernel with no laswp, and it measures 0.77-1.00x of cuBLAS. The decomposition costs **data movement, not launches**. No fused blocked LU was attempted, correctly.
2. **`getrs`'s recorded wide-`nrhs` window for the composition does not exist.** "nrhs=64 geomean 1.09x, nrhs=128 geomean 1.48x, 9 and 4 losses of 28" came from `grid_*.csv`, which carries exactly one saturating batch per order and no ladder on the batch axis at any width >= 16. Built properly -- 464 paired cells over 7 batches -- the composition's advantage falls **monotonically with batch**, because below saturation neither arm is measuring its own speed (at float n=128 nrhs=128 the composition costs 9.96 / 2.80 / 1.90 / 1.76 us per item at batch 32 / 128 / 256 / 512 and cuBLAS 38.2 / 10.2 / 5.59 / 3.31). Read at saturation, the walk's best candidate (float nrhs >= 128, 11 cells, geomean 1.761, zero losses) has **minimum 1.0436 and fails GATE-C**. The window that shipped is the *gather's*, not the walk's.
3. **A pure re-schedule refutes its own prediction.** Deferring the interchange while *keeping* the per-column walk moves byte-for-byte identical traffic (the column-visit sums are the same arithmetic series read from either end) and was predicted at 1.00x "by construction". Measured over 11 cells: geomean 1.055x with **three cells losing**, spread 0.707x-1.315x (float n=512 batch 128 is 0.707x, float n=1024 batch 128 is 0.916x, float n=512 batch 1024 is 1.281x). The mechanism is the **work-item count**: `batch*ib` items walking `n-j0` steps instead of `batch*j0` items walking `ib` is 15x less parallelism at n=512. That is also why the gather wins more than its 7.9x traffic saving alone predicts -- it puts the parallelism back too.
4. **The packed sub-group `getrf` arm, prototyped and refuted.** One sub-group per matrix, shuffle argmax, no work-group barriers. `pivman_ms / pivsg_ms`: float 1.38 / 1.25 / 1.13 at n=16/24/32, then 0.79 / 0.64 / 0.39 / 0.31 at n=48/64/96/128; double and cdouble lose at every order. It wins only for 32-bit types at `n <= 32` -- **and changes no routing decision even there**, because the shipped native `getrf` is 0.551x of cuBLAS at float n=32 and 0.584x at n=16, so the best cell in the table lands at **0.81x of cuBLAS**. Also established, closing the obvious next idea: the shipped small-n `getrf` is *already* one fused kernel (nsys shows a single `GetrfPanelResidentKernel<float>` and nothing else), so there is no launch-count win available at small n.
5. **Four WP6 micro-optimisations, each built and swept, three reverted.**
   * A division-free power-of-two (row, column) split replacing the rank-1 update's two runtime integer divisions: **geomean 0.936x for float `getrf`, 0.976x for cdouble, worst cells 0.829-0.833x** (float n=128: 3.938 -> 4.746 ms at batch 2048). The description of the defect was exact; the fix lost because at the resident tier `mm ~ wg`, so the split gives every work-item an inner loop of trip count **one**. Reverted.
   * `getrf_leaf_wg` raised from 512 to the device maximum 1024, re-swept over all 156 saturation cells: **geomean 0.974x**, cdouble 0.939x, float 0.960x, worst 0.814-0.837x (float n=256 batch 256: 2.046 -> 2.514 ms; cdouble n=1024 batch 16: 64.4 -> 77.9 ms). Right about the one shape it aimed at (the blocked driver's global panel leaf at n=2048 improves 0.6-1.2%) and wrong about the knob, because the same function serves the resident tier, where 1024 work-items on an order-256 tile lose 16-19%. Reverted.
   * `getri`'s `range<2>(batch, n^2)` zero-fill replaced by `range<3>(batch, n, n)`: **1.000 geomean** over the 78-cell sweep, 0.874x at the largest fill. Reverted.
   * `getri`'s hardcoded `wg = 256` derived from `n` instead: **0.9999 geomean**, spread 0.982-1.023 = noise. Kept for portability, with the comment at the site saying it is *not* a performance claim.
6. **`sycl::reduce_over_group` is the wrong pivot-search tool on three independent counts.** It is 1.5-4.7x *slower* than an explicit tree for double and cdouble (double n=16: 7.07x the unpivoted bound against the tree's 2.00x) and a wash for float/cfloat (0.87-1.25x); its per-work-item scratch -- `wg*(sizeof(real)+sizeof(int))`, 2040 B at wg 256 for float, 3060 B for cdouble -- moves the blocks-per-SM cliff **down by two orders of n** (n=110 instead of n=112, costing 1.73x exactly there) and cost a launch outright at cdouble n=78, taking the request from 98,608 B to 101,668 B past this device's 101,376 B cap; and it is the sole attributable cause of the 48 KB launch hole. The shipped kernel uses a sub-group XOR butterfly plus a scan over one slot per sub-group, allocated at a constant 32 slots -- **384 B for cdouble instead of 3,060 B, and independent of the work-group width**, which is what lets the capacity query, the fit predicate and the launcher agree without any of them knowing the others' `wg`.
7. **Partial pivoting costs 1.25-2.65x over the unpivoted lower bound, it is almost entirely the SEARCH and not the swap, and it gets cheaper as n grows.** Swap alone is 1.00-1.20x; the argmax alone is 1.25-2.2x. 2.65x at n=16, 1.27x at n=152, and flat in batch (float n=64 wg 256: 1.85 / 1.72 / 1.53 / 1.57 / 1.53 / 1.52 / 1.52 / 1.52 from batch 128 to 16384). Any effort spent making the row *exchange* clever is spent on the 3% end of the problem.
8. **The complex-`Tiled16` prediction is refuted for LU, and the deficit is `double`'s.** Both complex trailing updates reach `GemmRegister64x64K16WideKernel`, because `nb = 32` exactly meets the wide-scalar `min_dim >= 32` gate. `double` is the type with **no register GEMM on this path at any problem size** -- `Tiled16` at all 13 measured shapes -- and structurally so: the CTA-count relaxation is `if constexpr (is_std_complex_v<T>)`, complex only, and the only other wide-scalar door needs `min_dim >= 256`, which `k = nb` can never satisfy. The deficit is bounded (`double` at 1.01-1.08x of `Tiled16`, itself ~92% of the FP64 ceiling) but there is no LU-local fix; it needs a transposed/predicated wide-scalar kernel and belongs to GEMM.
9. **A harness that shrinks the batch cannot ask the routing question.** The first `routeq_lu.cpp` used batch=1 parents and reported `Tiled16` for every complex trailing update -- exactly the answer the brief predicted, and wrong. The CTA-count gate multiplies by `A.batch_size()`, and `can_use_64x64_k16_wide_fast_path` also reads `stride()`.
10. **Two RTX 4090s in one chassis are not two independent machines.** A sweep on device 1 running alongside a sweep on device 0 read `getrf float n=256 batch=128` at 3.31-5.51 ms against 1.006 ms alone. Both cards correctly reported zero foreign processes (`nvidia-smi --query-compute-apps` is *per device*) and `rel_sd` on the contaminated rows was 0.0004-0.017, so **neither instrument can see it**. Same NUMA node, same CPU affinity mask, one UVM driver, managed memory. It is cell-specific and intermittent: `getri` and `gemv` were unaffected (long, device-resident timed regions) while `getrf` and `getrs` were wrong by up to 5x. One contaminated reading looked exactly like a mid-ladder loss inside the admitted set and **caused the float `getrs` boundary to be narrowed from `nrhs >= 64` to `>= 128`, giving up 15 cells at 1.77x-4.07x**, before a re-measure on an idle box reverted the narrowing (that cell reads 0.8859 contaminated and 1.9563 alone). A whole `getri` pass (`lu_p2.csv`) was discarded for the same cause: 26 of its 55 comparable cells are 1.2x-5.8x slower on **both** arms at once with `foreign == 0` and rel_sd as low as 0.0012. **Serialise the box**; note also that device 0 drives the display here, which independently depresses an L2-resident vendor arm by up to 1.8x.
11. **A stale bench binary reports a stale route and it looks exactly like a failed flip.** The first unpinned run after clauses A/B landed reported `vendor:auto` on all 63 in-window cells. `lubench6.cpp` includes `src/backends/getrs_route.hh` and resolves the *printed* route in its own translation unit, while dispatch happens inside the `.so`, so rebuilding the `.so` alone leaves the harness printing the old table's answer. **Any `preferred()` change requires rebuilding every bench binary before its route column can be believed.**
12. **A pin that is refused is not a pin, and it silently becomes the other arm.** The `wp6_perf` sweeps drop rows on exactly this: 21 of `flat`'s 180 cells are `cta PIN-FELL-THROUGH to native:blocked` (every `n=2048` cell at `nrhs=8` for double/cfloat, and `nrhs=4` and `8` for cdouble -- the fused capacity cannot hold them), and the `getrf` tier sweep excludes four rows for the same reason. Any A/B that does not read the resolved route back per arm reports the *same* arm twice and calls it 1.00x. It is the same instrument failure as item 11, one layer down.
13. **`BATCHLAS_GETRS_ROUTE=native` changed meaning** when CTA joined `kGetrsOrder` ahead of Blocked: a bare origin resolves to the first supported route of that origin, which is now the fused tier. Any baseline recorded with a bare `native` pin -- `experiments/wp6_lu/bench/run_cells.sh:37` and `kernels/run_grid.sh:39` export one value into all three LU variables at once -- is measuring a different `getrs` today than when it was recorded. Pin `native:blocked` to mean what `native` used to mean.

## Correctness findings

* **The `info` zero-fill raced the panel that reads it, in BOTH native `getrf` tiers.** `getf2_panel_device` *reads* `info[b]` to keep first-failure-wins across panels, so the fill is a read-after-write dependence, not a pure output. On an out-of-order queue (the public API) the panel read the caller's pre-call garbage and wrote it back: **6,979 of 1,638,400 items on the CTA tier and 3,743 of 983,040 on the blocked tier returned the caller's own `-12345`**. Fixed with the `if (!ctx.in_order()) ctx.wait();` guard every other dependent boundary in the family already carried; re-measured 0 wrong of 1,638,400 and 0 of 983,040. Guarded by `LuTest.InfoFillIsOrderedAheadOfThePanelOnAnOutOfOrderQueue`; deleting the guard from both tiers turns it RED (4,682 of 1,638,400 CTA items, 4,370 of 491,520 blocked items). **The first version of that test stayed green with both guards deleted**, because a 300 MB host copy serialised the queue and closed the window it was testing -- [unverified: the ordinal is this page's own, not the sources'. `tests/potrf_tests.cc:641-908` is recorded as the repository's *fifth*, and `getrs_forward` below as the "sixth-plus"; no source numbers this one] the seventh blind guard in this repository, and the second written in the same change as the fix it guards.
* **`supports()` never gated on `s.backend`, so `Backend::NETLIB` on a GPU queue could select the native arm.** The native kernels write and read **packed 1-based int32** in the caller's `int64` pivot span (matching cuBLAS and rocSOLVER); netlib writes and reads **genuine int64**. Measured before the gate: `||A*C - I||_F / n = 5.32e-01` with `info == 0`, against 5.15e-07 when both arms agree -- silent, no throw, no flag, and invisible to the suite because its NETLIB rows run on a CPU queue. Now one predicate in each of the three tables, enumerated by the disagreeing backend rather than by an allow-list (so a new GPU backend that packs int32 needs no edit), plus a `RouteLuPivotFormat` test with a **backend axis** -- the axis the route tests did not have. Deleting the predicate from all three tables fires 5 assertions.
* **Pre-existing vendor crash, fixed.** `cublas.cc`'s `getrs` had a `batch_size <= 1` arm calling `cusolverDnXgetrs` -- a different library, the 64-bit non-batched API -- handed the raw `int64` pivot pointer, while every `getrf` in the tree writes packed int32. `getrf` then `getrs` at batch 1, the exact sequence `linalg::solve` performs, aborted with `CUDA_ERROR_ILLEGAL_ADDRESS` (exit 134). No batched test could reach it, because they all use `batch >= 2`; it was found by a pivot-contract survey, not by a test. The arm was deleted -- `cublas?getrsBatched` is correct at `batchCount = 1` and reads the format actually written. Now `||A*X - B||/||B|| = 1.20e-07` at batch 1.
* **cuBLAS pivots complex on the MODULUS; LAPACK, netlib and this kernel pivot on `cabs1`.** On a matrix with `(3+0i)` in row 0 and `(2+2i)` in row 1 of column 0, `cabs1` reads 3 vs 4 and the modulus reads 3 vs 2.828 -- the two rules select different rows. Both native tiers return `ipiv[0] = 2`, matching host LAPACKE; `cublas?getrfBatched` returns 1. Substituting the modulus into `lu_cabs1` reproduces cuBLAS's answer exactly, which identifies the cause rather than merely observing a difference. **Consequence: an elementwise native-vs-vendor pivot comparison is a wrong test and will go red on complex.** `PivotSelectionUsesCabs1AndNotTheModulus` pins the rule this library implements. Mixing arms is still safe, because `getrs`/`getri` consume `ipiv` together with the factor the same `getrf` produced.
* **The exact-zero `info` predicate is not stable across implementations.** On a singular probe, cuBLAS itself mismatches the host oracle at cdouble (`|U66| = 2.93e-18 -> info 0` against the host's 6), and the host mismatches at cfloat. "device info == host info" cannot be a test gate; the gate used is structural -- non-zero exactly when `|U(i,i)|` is a true binary zero, the failed item stays finite, non-singular items report 0.
* **A capacity-inversion defect no launch test could find.** `getrs_fused_max_rhs_elems` answered a budget with a plain floor division, but `getrs_hole_padded` is **not monotone**: for a budget a few bytes above the hole's upper edge the division rounds the implied request back *down into* the band, where it is raised to 49,920 B and no longer fits. At a cdouble budget of 49,665 B the query admitted 2,048 elements whose raw request is exactly 49,664 B, which the pad then raises to 49,920 -- a capacity whose launch the runtime would refuse. The window is `sizeof(T)` bytes wide and sits at a per-type budget; reaching any of them needs a device with **53,761-53,776 B** of local memory, so none is reachable on this box, and the `cap_inversion` break nonetheless goes red for all four types. A byte-by-byte sweep of a **pure function** found it; no launch test ever could.
* **The register cap is not defensive; it fired.** float, n=2048, nrhs=8, `transA = Trans` picks wg=1024 against a 68-register kernel = 69,632 registers per work-group against a 65,536 limit, and the launch **aborts**; the NoTrans arm of the very same call ran green first at 48 registers. A review asserted that no test reaches a shape where the cap can bite. `FusedGetrsLaunchHoleAt48KiB`'s top rung is n=1428 at nrhs=8 with `transA = Trans`, and removing the cap turns it RED as a hard abort. The review had looked only at the two tests with "Width" and "Boundaries" in their names.

The full break records are at the bottom of `tests/getrf_tests.cc`: fourteen WP6 breaks (all red), sixteen fused-`getrs` breaks (fifteen red), five window breaks, two repair-pass breaks. Each was applied to the source, the `.so` rebuilt, and the whole binary re-run.

**Count the rows, not the prose** -- the fused-`getrs` record says so in as many words, because an earlier version of its own summary sentence said "fourteen ... thirteen of the fourteen" over a sixteen-row table. The two properties no break can reach are named rather than averaged away: `cap_band` (the hole band dropped from the capacity query) and `B5` (the `+1` bank-conflict pad, recorded in `getrs_fused.cc` and not in the test file). Two of the five window breaks are findings in the same way: **W3** (the composition also made preferred) turns *nothing* red in `getrf_tests` and RED x2 in `route_vocabulary_tests`, because CTA is first in `kGetrsOrder` and `automatic()` returns the first supported-and-preferred route, so only a direct assertion on `preferred()` can see it; and **W1** (clause A switched off) correctly leaves *float* green, because float `nrhs = 1` is still inside clause B.

### Blind guards and what made them blind

This repository has a recurring class of guards that cannot fail. LU produced six more.

1. **A diagonally dominant test matrix makes every pivot test vacuous.** On `A = rand + n*I` partial pivoting selects the diagonal at every step, `ipiv` is the identity, and the entire pivot path -- the vendor's, the probe's, and the composition's `laswp` -- is unexercised. `BREAK=piv` and `BREAK=laswp` both turned **nothing** red, residuals bit-identical at 2.446e-07 / 1.055e-15. The fix is one line of *setup*, not of assertion: keep the dominance (it is what makes the residual measure the kernel) and then **row-permute each item by a per-item random permutation**. Both breaks go red immediately, to 1.903 and 1.989. An anti-vacuity assertion (`ntpiv`, the count of non-diagonal pivots on item 0, flagged BAD at zero) was added alongside -- necessary and not sufficient, since it says nothing about whether the probe *uses* the pivots, which is what `BREAK=piv` is for.
2. **A test of an inverse operation is vacuous on any self-inverse instance, and self-inverse instances are exactly the tidy ones an author reaches for.** The fixture permuted rows by a **reversal**, which is its own inverse, so `F = F^-1` and the transposed `getrs` arm -- whose whole content is "the same list walked backwards" -- returns the identical answer walked forwards. Three direction tests (getrs Trans, getrs ConjTrans, getri's backward trace) were unfalsifiable on every scalar type while reading as the file's strongest. Fixed with a **cyclic shift** (an n-cycle) plus `interchange_is_involution()` asserted at every direction-sensitive use.
3. **A probe that computes the right number and does not assert on it.** The first kernel harness gated `ok` on `isfinite()` alone; the `laswp_left` break drove the `getrf` residual to 1.2e-01 and the row still printed `ok`, `FAILS=0`. Every criterion now carries a `Tol<T>` bound.
4. **`route_vocabulary_tests`' `getrs_shape()` helper never set the fused capacities**, so `supports({Native, CTA})` was false on every shape in the pure suite and **every getrs routing assertion in it held regardless of the table** -- 78/78 through the window flip *and* through its inverse. Worse than uncovered: two assertions asserted the *opposite* of live behaviour. Both rewritten around the window, both sides of it; break V1 (re-zeroing the capacities) turns three tests red, which is the proof the repair is load-bearing.
5. **`max |L| <= 1` is the wrong partial-pivoting oracle for complex.** LAPACK selects on `cabs1` and `cabs1(z) <= sqrt(2)|z|`, so a correct `zgetrf` returns `|L|` up to sqrt(2) -- measured at 1.051 on the first random cfloat matrix. The metric-aware form is strictly stronger and turns the *ordinary* complex sweeps red, where the earlier oracle needed an adversarial probe matrix.
6. **A revert that patched the wrong line.** The `getrs_reverse` break's 8-space anchor was a substring of the 12-space line, so the revert left **both** permutation walks inverted in the tree. It was caught only because the next break's run showed `getrs` failing for float and double -- types that break could not touch. `break.py` now requires every anchor to match **exactly once**, and break runs capture full output so rows a break should not have moved can be read.

Two more results are findings rather than confirmations. `pivot_metric` (cabs1 -> modulus) **turned nothing red on the ordinary sweep** against a `|L| <= 1` oracle: on a random matrix the two selection rules agree at every step, so the elementwise pivot comparison -- the strongest oracle in that harness -- was blind to the metric, and a purpose-built probe matrix was needed. Under the metric-aware oracle it now turns the ordinary complex sweeps red, and it correctly turns nothing red for float and double, where the two functions coincide. And `short_final` (drop the short final panel) is red at n=33 and n=100 and correctly **green at n=64 and n=96** -- it discriminates exactly the short final panel, which is why the order sweep straddles `nb`.

### The 48 KB launch hole

Re-measured from scratch with a `PAD=` knob holding kernel, shape and work-group fixed and moving only the declared byte count, one process per point: **49,024 B PASS / 49,152 B FAIL / 49,280 B PASS**, 5/5 deterministic across five separate processes, at every work-group width (32/64/128/256/512). The control -- the identical shape with an explicit SLM tree at the identical byte count -- launches. The band is wider for wide scalars: the collective also fails at 48,896 B for double and cdouble. Both failure points lie inside the inherited band `(47104, 49664]`, checked rather than assumed. An `n` ladder finds nothing, because the hole is specific byte counts and an `n` ladder steps over them.

**Attributed:** it is the group collective, not the tile. `reduce_over_group` allocates local memory the `local_accessor` accounting cannot see.

The shipped `getrf` CTA kernel uses no group collective (only `permute_group_by_xor`, a register shuffle) and **the hole does not reproduce for it**: removing the pad turns the *arithmetic* layer of the test red for every in-band row and every type, and leaves the *launch* layer green -- the 49,152 B resident launch succeeds without the pad. Both layers are `EXPECT`, not `ASSERT`, precisely so the first cannot mask the second. The pad is defensive, kept for the day a group algorithm enters the body (the condition WP4 wrote down and WP5 walked into anyway); the arithmetic assertion is the guard with teeth. The same band and pad target are repeated verbatim in `getrf_cta.cc`, `lu_laswp.hh` and `getrs_native.cc`, so the three cannot drift.

## Open debts

* **`getrf`'s window has no batch term and was measured only at batch 128-1024.** Below 128 nothing was measured after the gather landed. `getrf_tests`' `RouteTableAndTheVendorFreeFallback` nonetheless asserts `native:blocked` at n=512 **batch=2** for float and cfloat -- a batch the perf grids never measured. The window is applied there on the strength of an order clause alone.
* **`getri` at `batch <= 32` beats cuBLAS by 1.7x-28x for *every* type**, double and cdouble included, because the vendor's batched `getri` is a per-item loop there. **Unrouted.** A batch clause has to be bracketed at every `(type, order)` it admits, and low batch was measured only at orders 128 and 512. Missing cells, named rather than fitted away: double and cdouble at orders 32, 64, 256, 1024 and 2048, batch 1-64.
* **`getrs`'s clause-C batch floor of 128 gives up measured wins** (at nrhs=128 the composition wins at batch 32 and 64: float 3.87x-5.96x, double 3.56x-4.31x). It is conservative on purpose -- below 32 the only readings come from the contaminated sweep. Moving it down is one cheap sweep (`experiments/wp8_getri/gen_floor.py`).
* **`getrs double`'s minimum sits at the largest order clause C measures.** That is n=**1024**, nrhs=128, batch=512: 1.2791 on the 45-cell reading and 1.2858 on the clean pass. n=2048 and above are unmeasured for this clause and are the one place a future order could fall under the bar. (An earlier draft of this page put this cell at "1.274x at n=2048"; no clause-C cell at n=2048 exists.) The *related* risk -- "that cell is the last rung at its order and the batch ladder is falling" -- is **closed**, not open: the clean pass measured batch 1024 at the same cell and got **1.3070**, so the ladder turns back up.
* **84 measured winning cells are handed to the vendor by clauses A and B**, the largest at 3.944x (double n=1024 nrhs=4 batch=256), then 3.144x, 3.097x, 2.880x, 2.745x. They are given up because the clause that would capture them dips below 1.0 elsewhere on its own ladder. Recovering them needs a per-`(type, order)` predicate measured at more orders, or a kernel fix for the dip -- real work, not a constant.
* **`getrs`'s collapsed gather is parallel over batch only** -- `nd_range<1>(batch*wg, wg)` with `wg = 256`, i.e. exactly `batch` work-groups: one wave at the clause's own floor of 128 on a 128-SM part, 32,768 work-items on a part that holds 196,608. This is the campaign's signature defect sitting in the arm the closure pass shipped. It is an **unclaimed lever rather than a defect** -- the gather never loses to the walk it replaces (min 1.0004 over 80 cells) -- and `getrf`'s gather does not have it (`nblk*batch` groups, 896 at n=256 batch=128). Not attempted, because the ladder that justifies clause C was measured against this geometry.
* **The right-hand `getrf` interchange is untouched.** Only the left-hand pass was deferred and gathered; the remaining walk is still `range<2>(batch, ncols)`, which at nrhs=1 degenerates to `batch` work-items (32 at n=2048 batch=32) each walking `n` dependent swaps. The conditional right-hand gather was declined with its arithmetic -- a one-order lever behind a per-block-step runtime gate, worth ~1.29x at n=256, ~1.07x at n=512 and ~1.00x at n >= 1024 (figures from `VENDOR_INDEPENDENCE_PLAN.md`'s closure-pass section; no per-cell grid for them was found under `experiments/`).
* **The fused `getrs` kernel's folded permutation is 8.0% of the call at n=2048** (float b=32, 1.2802 -> 1.1776 ms with the walk removed; 3.5% at float n=512 b=512, 2.4% at cdouble n=2048). It is the one fully serial part: `if (tid < nrhs)` over `n` dependent local-memory swaps, so at nrhs=1 one work-item of up to 1024 does `n` round-trips while the rest wait at a barrier. Named as the next lever, not fixed.
* **The serial SLM index walk in the composition's gather has two incompatible estimates and no profile.** The header prices it at ~3% at n=1024 batch=128; an independent estimate from dependent-swap latency puts it nearer 10%. Neither figure comes from a profile, and both should before anyone spends effort on either.
* **`nrhs > 8` for the fused tier.** The kernel is instantiated to `kGetrsFusedMaxRhs = 8`. Raising it requires `native_tier_preferred` to gain a window in the same change -- at nrhs=16 the composition is already ahead for double (0.55x) and cfloat (0.58x) at n=512.
* **The `getrf` tier window rests on 5 orders and 1-4 batches per type, and the band between each type's last measured order and its capacity ceiling -- float 129-155, cfloat 101-109, cdouble 77 -- is EXTRAPOLATED onto CTA.** cdouble's advantage is visibly collapsing (1.37x at n=64 -> 1.09x at n=76 against a ceiling of 77) and is where a re-measurement would find a crossover first.
* **`nb = 32` is not tuned.** It satisfies the structural constraints (a multiple of 16; never below 32 for complex, or the trailing GEMM loses the wide-scalar kernel -- geqrf measured 1.72-2.30x lost at nb=24) and nothing more. `getrf_leaf_wg` reproduces the two measured best widths (256 at n=64, 512 at n=128) and extrapolates; the baseline measured an 8.3x spread across widths (float n=128 batch 4096, unpivoted: 39.72 ms at wg=32 vs 4.77 ms at wg=512), so a real sweep is owed.
* **`getrf_panel_factorize` re-queries `LOCAL_MEM_SIZE` and `MAX_WORK_GROUP_SIZE` once per block step** -- 128 device queries per n=2048 `getrf`. Real, unmeasured, and fixing it changes an exported signature.
* **`MatrixView::data_ptrs(ctx)` re-runs `init_data_ptr_array` unconditionally**, a submit plus a blocking `.wait()`, so a vendor-routed `trsm`/`gemm` inside the blocked driver costs two host drains per panel. Root cause is in `matrix.hh`; a known open bug the campaign works around rather than fixes.
* **`Backend::NETLIB` on a GPU queue is gated but not exercised end to end** by `getrf_tests` -- that fixture skips every NETLIB row because its queue is a CPU queue. The gate is guarded by a synthetic route test and a standalone probe, not by a device test.
* **`native_tier_preferred` for `getrf` is covered synthetically** in `route_vocabulary_tests.cc`, not against the real device; `getrf_tests` asserts only that the real builder reports non-zero `cta_max_n` and `blocked_available`. The tier split is *visible* in a coverage capture but not asserted there.
* **Two of the fourteen WP6 breaks are red by crash** (`short_final` SIGSEGV exit 139, `piv_stride_nb` SIGABRT exit 134), so they do not demonstrate *which* assertion would have caught them. Break runs must be filtered one scalar type at a time, or the three types after the abort report nothing.
* **Residual tolerances are `c*n*eps` with `c` in [200, 800]**, not tightened against a measured error distribution. No break in the record was caught by a tolerance -- every one was caught by an equality or a structural assertion.
* **A latent vendor gate defect**, recorded not fixed: `cublas.cc`'s `getrs` sits in a TU gated on `BATCHLAS_HAS_CUBLAS`, so a cuBLAS-present / cuSOLVER-absent configure claims a vendor it cannot link. The fix belongs in `vendor_available.hh`.
* **NETLIB `getri`'s `std::copy(..., n*n, ...)` ignores `ld`** -- pre-existing, not fixed.

## Raw evidence

Raw data is preserved at the tag `perf-evidence/vendor-independence`. Retrieve any path with `git show perf-evidence/vendor-independence:<path>`. (At the time this page was verified the `experiments/` directories below were still present and tracked in the working tree; the tag is what survives their removal.)

| topic | path |
|---|---|
| vendor baseline, saturation ladders, pivot cost, the 48 KB hole, the composition prototypes, the route probe | `experiments/wp6_lu/baseline/` -- `README.md`, `grid_norm.csv`, `summary.txt`, `sat.csv`, `summary_sat.txt`, `pivot.csv`, `summary_pivot.txt`, `hole.csv`, `hole2.csv`, `wg.csv`, `routeq_lu_{v,nv}.csv`, `lubench.cpp`, `pivotcost.cpp` |
| the native kernels' A/B grid, the tier sweep, the laswp cost, the cross-build split, the kernel-side breaks | `experiments/wp6_lu/kernels/` -- `README.md`, `grid_{native,vendor}.csv`, `summary.txt`, `tier.txt`, `run_tier.sh`, `crossbuild.txt`, `laswp_cost.txt`, `luverify.cpp`, `break.py`, `break_*.txt` |
| the 982-row measure phase: saturation, both readings, nsys splits, the nrhs crossover, the order/batch separation | `experiments/wp6_lu/bench/` -- `README.md`, `sat_*`, `tail_*`, `getrs_*`, `order32_*`, `order1024_*`, `geomeans.txt`, `order_tables.txt`, `nsys_splits.txt`, `lubench6.cpp`, `run_cells.sh` |
| the LU test suite and its 14 + 2 breaks | `experiments/wp6_lu/tests/` -- `README.md`, `break.py`, `run_break.sh`, `breaks.txt`, `break_*.txt` |
| the fused-getrs prototype, its `nb`/`wg` tuning, the `+1` pad A/B, the no-permutation break, the fused break tooling | `experiments/wp6_getrs/` -- `proto/grid_nv.csv`, `proto/grid_big.csv`, `proto/noperm.csv`, `proto/tune.sh`, `pad_ab.sh`, `public_*.csv`, `tests/` |
| clauses A and B: seven sweeps, four flatness passes, the window scoring, the register-cap A/B, the refuted sub-group `getrf` | `experiments/wp6_perf/` -- `README.md`, `bench/`, `bench/grid_cta.csv`, `bench/window_summary.txt`, `bench/default_summary.txt`, `bench/shipped_summary.txt`, `regcap/` (`before_p{1,2,3}.csv`, `after_p{1,2,3}.csv`, `analyse.py` -- there is no `regcap/README.md`; the write-up is §7 of `wp6_perf/README.md`), `proto/ab.csv` |
| the deferred left-hand `getrf` gather: A/B, ncu sector counts, the clean re-measure | `experiments/wp8_getrf/` -- `after_nv_p{1,2}.csv`, `base_v_p{1,2}.csv`, `ab_{ctl,gather}_p{1,2}.csv`, `ncu_float512.txt` |
| the `getrs` permutation-gather A/B and its boundary sweep, the wide-`nrhs` ladder, clause C and its gap sweep | `experiments/wp8_getrs/` -- `ab_p{1,2}.csv`, `ab_summary.txt`, `ab_bnd_p{1,2}.csv`, `lad_*.csv`, `hi_*.csv`, `cl_*.csv`, `gap_*.csv`, `clause_summary.txt` |
| the clean device-1 re-measure that all three shipped windows are scored from | `experiments/wp8_getri/` -- `lu_c1.csv`, `lu_p1.csv`, `lu_p2.csv` (discarded), `summary_c1.txt`, `summary_p1c1.txt`, `pair_cells.sh`, `analyse.py`, `gen_floor.py` |
| the campaign narrative: "WP6 has landed" and "The WP6/WP7 performance-closure pass" | `VENDOR_INDEPENDENCE_PLAN.md` |

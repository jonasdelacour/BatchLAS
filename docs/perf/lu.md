# LU: getrf, getrs, getri, the laswp gather, and the four routed windows (WP6 and the WP6/WP7 closure pass)

Two `getrf` tiers, two `getrs` tiers, one `getri` driver, and the row-interchange kernel all three share. This page records what routes, why each boundary sits where it does, what was built and measured worse, and what is still owed.

Measurement context, unless stated otherwise: 2x RTX 4090 (sm_89, 128 SMs, 97,280 B usable local memory per work-group, ~1008 GB/s DRAM, 1.29 TFLOP/s FP64, ~47 TFLOP/s measured FP32 GEMM), one GPU pinned per run, public API, in-process host oracle on every timed row, resolved route read off every row, nothing timed under `BATCHLAS_KERNEL_TRACE`.

**Naming.** `experiments/wp8_getrf/`, `wp8_getri/` and `wp8_getrs/` are a misnomer: they are the **WP6/WP7 performance-closure pass**, not WP8 (WP8 is sparse `spmm`, in `experiments/sparse_spmm/`). This page calls that work "the closure pass" throughout.

## What ships

### Route arms

| op | route order (`kGet??Order`) | native arms |
|---|---|---|
| `getrf` | `{Native,Tiny}`, `{Native,CTA}`, `{Native,Blocked}`, `{Vendor,Auto}` | register-resident sub-group tier (`getrf_tiny.cc`, pin-only -- see "The tiny tier"), CTA-resident leaf (`getrf_cta.cc`) and right-looking blocked driver (`getrf_blocked.cc`), the last two sharing ONE `?GETF2` device body (`getrf_cta_device.hh`) |
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

`include/batchlas/blas/dispatch/route_getrs.hh:82-102` -- three clauses across two arms:

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

**Correction to the exploration notes.** `experiments/wp6_lu/README.md`, `bench/README.md` and `kernels/README.md` all state that `preferred()` is false everywhere for all three ops. That was the WP6 merge state. The shipped predicates are the four windows above; **the code wins**. All three route headers say so in the WP8-ROUTING-PASS blocks preceding the predicates (`route_getrf.hh:67`, `route_getri.hh:65`, `route_getrs.hh:76`).

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

`route_getrs.hh:107` -- CTA (fused) always preferred over Blocked. No crossover to encode: the fused tier is ahead of the composition at **every** cell inside its own capability (51 cells, worst 1.11x at float n=2048 nrhs=8). The column where it would turn is nrhs=16 (double 0.55x, cfloat 0.58x at n=512), and that is outside `supports()` by `kGetrsFusedMaxRhs`. **If that constant is raised, this predicate must gain a window in the same change.** `getri` declares none -- one native arm, no native-vs-native question.


#### Why Tiny is absent from the window

`preferred()` names only the Blocked arm. Tiny is left out deliberately: R8b -- a non-empty
`preferred()` pre-empts `native_tier_preferred` (`docs/design/small-n-factorization-plan.md`)
-- means the FIRST native arm answering true there takes every shape it can hold, whatever
the tier hook says, so a Tiny clause written before the grid exists would move
vendor-**present** traffic onto an untimed tier. Its window is set from the measured grid in
a separate change.

In the tier hook the Tiny arm is spelled out rather than left to `default:`, which returns
**true**. Without the explicit `case Algorithm::Tiny: return false;` the vendor-free walk and
a bare `BATCHLAS_GETRF_ROUTE=native` pin would both hand Tiny every order <= 32 the day the
kernel landed -- unmeasured, and invisible to a vendor-present build. The explicit `false`
still leaves the tier reachable by an explicit `{Native, Tiny}` pin, which `route_resolve.hh`
honours regardless of this hook. This arm and `preferred()` flip TOGETHER in the measured
change.

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

**The cfloat floor moved 512 -> 256 in P4**, and the paragraph below is the record of why it was 512 before that: the 0.885 cell it names is the LOCAL-MEMORY leaf, and the register leaf is what makes 256 clear the gate ([the cfloat window moves to 256](#the-cfloat-window-moves-to-256)). float's 256 floor is unchanged.

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

### `getrs` order floor evidence

Clause A (`nrhs <= 2`, every type) and clause B (`float`, `nrhs = 3..4`) shipped with **no order
bound at all**, on a grid whose smallest order was 32. Re-measured down to n = 4 they lose in
**live routed traffic in a vendor-present build** at every type, worst **0.234** (cdouble n=4,
nrhs=1, batch 32768), and the loss deepens with batch rather than washing out: float n=4 nrhs=1
reads **1.50 / 1.07 / 0.71 / 0.52** at batch 8192 / 16384 / 32768 / 65536.

That is a correction to [`getrs` fused window evidence](#getrs-fused-window-evidence) above,
which records clause A as "286 cells, geomean 2.261, **min 1.116, zero losses**" and clause B as
"min 1.133, zero losses". Both readings score the same predicate; the earlier grid had no rung
below order 32.

**The mechanism.** The fused kernel gives one work-group to a matrix whose whole solve is a few
dozen flops, so the work-group **is** the cost. `cublas?getrsBatched` keeps its per-item work
inside one kernel and pays no such floor.

Worst ratio over the batch ladder, `nrhs = 1`, by order (`vendor_med / native_med`, > 1 means
native wins; the **minimum** rung of each order's ladder, not its top rung):

| T | n=4 | n=8 | n=16 | n=17 | n=24 | n=32 | n=48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| float | 0.71 | 1.12 | 1.04 | 0.76 | 0.95 | 2.29 | 1.94 |
| cfloat | 0.59 | 0.69 | 3.76 | 1.38 | 1.27 | 1.40 | 1.60 |
| double | 0.52 | 0.98 | 0.99 | 0.87 | 3.66 | 3.77 | 3.74 |
| cdouble | 0.23 | 0.41 | 2.00 | 2.02 | 1.93 | 2.13 | 2.46 |

**This is a different statistic from the one `small-n-baseline.md` §`getrs` quotes**, and the two
must not be read against each other cell by cell: that page quotes the **top** batch of each
ladder — the most saturated reading taken — where this table takes the worst rung anywhere on
the ladder. Its float n=24 cell reads 1.11; the same ladder's worst rung is 0.95.

**32 is the first order where all four types clear the flip gate AND stay clear above it.** The
band below it is non-monotone — float passes at 8, fails at 9, 16, 17 and 24 — so no lower floor
is defensible. Clause B is measured on the same grid and takes the same floor: float nrhs=4 reads
**0.45 / 0.46** at n = 4 / 8 and **1.07 / 1.10** at 17 / 24, clearing only from 32 (**1.30**).

The floor is therefore a **defect fix rather than a tuning knob**: it excludes measured losses in
traffic the router was already sending to the fused tier, not marginal wins.

**P2 of the small-n plan is what reclaims this band**: a fused factor-and-solve kernel that holds
the matrix in registers has no work-group floor to pay.

Source: `benchmarks/results/factor_baseline_getrs_{float,cfloat,double,cdouble}.csv`, ratio
`vendor_median / native_median`, minimum over each order's batch ladder. All 28 cells of the table
were re-derived from those CSVs and reproduce exactly.

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

**Clause C carries no order floor, and that is not an oversight.** Its axis is `nrhs`, plus its own batch floor of 128; it was never measured below order 32 in the first place, so there is no bracketing non-winner below that order to bound it with. The fused tier's order floor is a separate clause with its own grid — see [`getrs` order floor evidence](#getrs-order-floor-evidence).

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

## The shape of the GETRF device code

`src/extensions/getrf_cta_device.hh` holds all of the native GETRF panel's device code.
`getf2_panel_device` is LAPACK ?GETF2 -- unblocked right-looking LU with partial pivoting --
over a `Tile` supplying `at(r, c)`, instantiated twice: on a `local_accessor` for the
resident CTA tier, and on a raw global pointer for the blocked tier's panel leaf. Four
spellings in it are load-bearing, and each is a wrong answer or a launch failure if changed:

* **The pivot search is a sub-group butterfly plus a scan over 32 SLM slots**, never
  `sycl::reduce_over_group`. The collective fails to launch, deterministically, near 48 KB
  of local memory -- see [the 48 KB launch hole](#the-48-kb-launch-hole).
* **The pivot metric is LAPACK's `cabs1`** (|Re| + |Im|), not the modulus cuBLAS uses for
  complex, so a pivot test must take the HOST as oracle -- see
  [correctness findings](#correctness-findings).
* **`info` is exact-zero, 1-based, global and first-failure-wins**, with no epsilon pivot
  floor. An epsilon floor would make a singular matrix succeed silently.
* **Barriers B1..B4 sit at the top level of the `k` loop**, whose trip count is
  kernel-uniform; the launchers add B0 after the tile load and B5 before store-back. A
  barrier moved inside a divergent branch hangs rather than slows.

`kLuRedSlots = 32` is CONSTANT rather than a function of the work-group width, so the
capacity query, the fit predicate and the launcher agree on the SLM footprint. `LuRawTile`
addresses its tile through a raw pointer because G matrices share one `local_accessor` and
each needs its own base; the launcher pads `ld` ODD, because a row exchange walks `wg`
work-items at stride `ld` and an even `ld` puts them all in one local-memory bank.

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
* **P4's register panel leaf is the DEFAULT panel leaf**, measured over 156 paired cells (1.01-2.13x; the only loss is double `n = 32`, at all three of its rungs), and it moved cfloat's `getrf` floor from 512 to 256 at batch >= 256. `BATCHLAS_GETRF_LEAF=slm` still selects the older local-memory panel. [The A/B](#the-register-leaf-ab), [the window](#the-cfloat-window-moves-to-256). Two things that grid found and did NOT fix: float `n = 65` takes CTA in a vendor-free build where blocked is 1.62x faster, and the float window below 256 is a batch question, not an order one ([which native tier serves 33 to 256](#which-native-tier-serves-33-to-256)).
* **The two later steps of P4 are not attempted.** The recursive panel (outer `nb = 128` split into 32-wide register leaves, so the trailing GEMM's `k` is 128 rather than 32) and the right-hand interchange gather are both untouched, and `nb` is still 32 for every type. The recursive step is the one that would actually move the trailing GEMM into `Tiled128x128RegisterK8` territory; the leaf swap alone does not.
* **The P4 leaf is now measured against the vendor at 150 paired cells** ([the register leaf A/B](#the-register-leaf-ab)), but the PHASE SPLIT behind it is still the **pre-gather, double-only** profile (`nsys_splits.txt`, `lose_getrf_double_128`) that says "48.5% of a double n = 128 call is the panel". No float or cfloat `getrf` phase split exists at any order, so the A/B says the leaf is 1.3-2.1x faster without saying which phase paid. The nsys split is still owed.

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

P4's own raw rows are **in the tree**, not at the tag: `benchmarks/results/p4_leaf_getrf_{float,cfloat,double,cdouble}.csv` (the 450-row leaf A/B), `p4_tier_getrf_float.csv` (vendor / CTA / blocked+reg with a coverage readback per arm) and `p4_e2e_getrf_{float,cfloat}.csv` (the automatic route against the vendor). Every one was produced by `benchmarks/factor_bench.cc` through `benchmarks/gpu_guard.sh 1`.

**One trap the default flip creates for anyone re-running these.** An arm named plain `blocked` takes whatever leaf the DEFAULT is, and the default is now `reg` -- so `--arms=vendor,blocked,blocked/leaf=reg` measures the register leaf TWICE and reports a leaf ratio of 1.000. Spell both: `blocked/leaf=slm,blocked/leaf=reg`. The tables above were taken before the flip, when plain `blocked` still meant `slm`.

---

## The occupancy rule

**P7, 2026-09-10.** `getrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` and
`getrf_cta_fits<T>(n, budget, min_blocks_per_sm)` divide the device budget by an occupancy
target (default 4, `resident::kMinBlocksPerSm`) before testing the footprint. At this box's
97,280 B budget the advertised ceilings become **77 / 54 / 54 / 38** for
float / double / cfloat / cdouble, against 155 / 109 / 109 / 77 at target 1. Both scales
are pinned budget-parameterised in `tests/resident_capacity_tests.cc`.

The walk itself now lives in `src/util/resident_capacity.hh`, shared with potrf; the sqrt
bound stays here because it is getrf-specific, and the `break` is unchanged.

### One spelling per ceiling

The header's declarations carry the contracts; this is what they mean.

* `getrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` takes a **runtime**
  `local_mem_size` budget, not `device_limits.hh`'s build-time constant, and the footprint
  it walks must cover the **pivot-search scratch** as well as the tile. 0 means the tier is
  absent from this build.
* `kGetrfReferenceSlmBudget = 97280` (`getrf_cta.cc:35`) exists **only** for the
  convenience overloads `getrf_cta_max_n<T>()` / `getrf_cta_max_n_for_slm<T>(budget)`, and
  it is this box's figure: `local_mem_size` reports **101,376 B**, less the standard
  **4,096 B** reserve. The generated `device_limits.hh` says **49,152** for any
  `nvidia_gpu_sm_*` pattern with no device query at all
  (`cmake/BatchLASDetectSYCL.cmake:45-46`) — **2.06x wrong here**, which is why nothing on
  a real decision path reads it. Same number, same reason, as
  [potrf.md's budget section](potrf.md#the-slm-budget-and-the-fit-ceilings) and
  `geqrf_cta.cc`'s `kGeqrfReferenceSlmBudget`.
* `getrf_cta_fits` is the tier's admission test and is occupancy-scaled by default;
  `getrf_leaf_fits` is the residency question and is asked at the whole budget. See [the
  panel leaf is not the tier ceiling](#the-panel-leaf-is-not-the-tier-ceiling).
* `getrf_tiny_max_n<T>()` is a compile-time property of the **kernel** — the {8, 16, 32}
  template ladder — not of the device: the tier holds no local memory at all, so no budget
  enters and there is no walk. 0 would spell "absent from this build"; it never is. It is
  the ONE place that ceiling is spelled, called by the capacity query, the route builder,
  the dispatch entry point and the tests alike.
* `getrf_tiny_buffer_size` is **not** zero: the kernel needs no algorithmic workspace, but
  a short or empty caller `info` span means "not requested" and draws pool scratch, as
  every tier's sizing does.
* `getrf_cta_debug_launch<T>(ctx, m, n)` packs **G (matrices per work-group) in the low 16
  bits and the work-group width in the high 16**, and answers 0 when the panel is not
  resident. The pairing `G > 1` ⟺ sub-group-scoped barriers is derived inside the launcher,
  so this hook is the only way to observe it from outside — see [packed resident
  leaves](#packed-resident-leaves).
* `getrf_blocked_debug_params<T>(ctx, n)` packs **nb in the low 16 bits and the leading
  panel's leaf in the high 16** (1 = local, 2 = global); 0 when the driver is absent.

### The tiny getrf window

**2026-09-14.** The register-resident tier shipped with P1 and was left unrouted for want
of a saturation grid. It has one now, and it routes: **float `8 <= n <= 32`**, **cfloat
`9 <= n <= 16`**. This is the band the scoreboard showed us losing 0.21-0.74x in, and the
losing arm there was the blocked/CTA one -- the tier that wins was simply never asked.

Grid: `benchmarks/results/p8_getrf_tiny_window.csv` (240 rows, 80 cells x 3 arms) and
`p8_getrf_tiny_edges.csv`. One process per cell, arms interleaved in that process, 9 reps,
warm-up interleaved in the timed loop's arm order, host residual on items 0 and batch-1.
**No cell was discarded**: worst `rel_sd` in the whole grid is under 0.10 and nothing came
back `bad`. Ratio is `t_vendor / t_tiny`, so above 1 the tier wins; the gate is 1.11.

| n | float 8k | 16k | 32k | 65k | cfloat 8k | 16k | 32k | 65k |
|---|---|---|---|---|---|---|---|---|
| 4 | 1.662 | 1.415 | 1.192 | *1.020* | 1.475 | 1.265 | *1.016* | **0.833** |
| 8 | 1.644 | 1.482 | 1.400 | **1.303** | 1.471 | 1.295 | 1.181 | *1.117* |
| 9 | 1.772 | 1.743 | 1.687 | **1.664** | 1.385 | 1.360 | 1.469 | **1.540** |
| 12 | 1.743 | 1.707 | 1.708 | **1.735** | 1.346 | 1.325 | 1.436 | **1.941** |
| 16 | 1.668 | 1.642 | 1.640 | **1.663** | 1.344 | 1.302 | 1.599 | **1.416** |
| 17 | 1.023 | 1.085 | 1.106 | **1.143** | 0.397 | 0.363 | 0.377 | **0.368** |
| 20 | 1.043 | 1.123 | 1.125 | **1.141** | 0.429 | 0.421 | 0.421 | 0.409 |
| 24 | 1.058 | 1.079 | 1.084 | **1.140** | 0.457 | 0.524 | 0.518 | 0.508 |
| 28 | 1.101 | 1.097 | 1.094 | **1.183** | 0.584 | 0.579 | 0.570 | 0.547 |
| 32 | 1.192 | 1.077 | 1.123 | **1.225** | 0.626 | 0.631 | 0.643 | 0.646 |

#### The two edges the 65,536 rung could not settle

Two cells sat within 0.01 of the gate *and were falling with batch*, which is the direction
that kills a window. Both were taken one doubling deeper, and one of them changed the
answer:

| cell | 65,536 | 131,072 | effect on the window |
|---|---|---|---|
| float n = 4 | 1.059 | **0.888** | stays out; the lower bracket is a measured LOSS |
| float n = 8 | 1.306 | 1.256 | stays in |
| cfloat n = 8 | 1.122 | **1.086** | **falls below the gate -- excluded** |
| cfloat n = 9 | 1.618 | 1.993 | stays in, and is still rising |

Taking cfloat n = 8 on the 65,536 reading alone would have shipped a cell that does not
clear its own gate one rung deeper. It is the second time in this campaign that an
unsaturated rung nearly bought a window (compare the geqrf leaf, where the unsaturated cell
was misleading in the *other* direction).

#### Why the floors sit where they do, and why cfloat's ceiling is 16

The kernel is instantiated on an **N-bucket ladder, N in {8, 16, 32}**, and the window is
that ladder read off a graph:

* **Both types lose at n = 4.** n = 4 half-fills the N = 8 register array while cuBLAS
  `getrfBatched` is at its most efficient -- the work is small enough that the vendor's
  per-batch overhead is amortised and ours is not. The tier only pays once its bucket is
  full.
* **cfloat's cliff at 17 is the sharpest edge in this table: 1.416 -> 0.368.** n = 17 pads
  into the N = 32 array, so a complex item carries roughly twice the register traffic it
  needs. float absorbs that (1.143 at n = 17, still a win); complex does not. cfloat's
  window is therefore *exactly the N = 16 instantiation*.
* **float's ceiling of 32 is the TIER's, not a measured loss.** `supports()` refuses n = 33
  because the kernel is square-only to 32, so there is no bracketing non-winner above the
  window -- a wider register kernel is **untested, not refuted**. That is an open cell, not
  a closed one.
* **float n = 5..7 are unmeasured and deliberately excluded** by the floor of 8 rather than
  admitted on a guess. They sit between a measured loss (4) and a measured win (8).

#### Arming the window test: three breaks red, two unfalsifiable

`RouteGetrf.TheMeasuredTinyWindowAndNothingElse` was armed in five directions. Three went
red as predicted; **two could not, and the reason is worth recording rather than glossing**:

| planted break | observed | why |
|---|---|---|
| float floor 8 -> 1, admitting the 0.888x cell at n = 4 | **red** | |
| cfloat ceiling 16 -> 32, admitting the 0.368x cliff at 17 | **red** | |
| cfloat floor 9 -> 8, admitting the cell that falls to 1.086x | **red** | |
| remove the `tiny_max_n < 1` tier-absent gate | *stayed green* | the ceiling test `order > tiny_max_n` already refuses every order >= 1, so two guards defend one property |
| remove the anti-overlap line from the Blocked arm | *stayed green* | Tiny (<= 32) and Blocked (>= 256) are disjoint by construction, so the line changes no answer today |

Both unfalsifiable lines were **kept and relabelled** rather than deleted: they are defence
in depth against a future widening of either window, and the code now says so instead of
implying a test covers them. This is the same shape of finding P6 reported from its own
arming -- a guard that cannot fail is not automatically a guard that should go, but it must
never be counted as coverage.

#### What moved, end to end

The `auto` arm was measured alongside every cell above and read **1.000-1.007 against the
vendor throughout**, which is the flip's before-picture: `Auto` was taking cuSOLVER at every
one of these orders. fp64 is not routed here and was not re-gridded: on this part fp64 runs
at 1/64 the fp32 rate, so the ratio measures a crippled unit rather than a kernel.

### The panel leaf is not the tier ceiling

`getrf_leaf_fits` is deliberately **not** occupancy-scaled, and `getrf_panel_factorize`
and `getrf_blocked.cc` keep asking it at the whole budget. The residency question is
"resident tile or global-memory stream", and a blocked panel that stops being resident is
a large-n regression rather than an occupancy win: at float n = 512, nb = 32 the panel is
513 x 32 x 4 = 65,664 B, resident at 97,280 and *not* at 24,320, and getrf's 1.25-2.35x
wins at n >= 256 all run through it. One predicate serving both questions is what made
this a trap; there are now two, and the launcher's `internal_error` still asserts they
agree in the direction that matters (advertised implies resident).

### Packed resident leaves

`getf2_panel_device` is templated on `LuScope`. Under `SubGroup` every phase barrier is a
sub-group barrier, the cross-team argmax scan and its B1 disappear (the butterfly already
spans every lane that scanned the column), and `getrf_leaf_launch` places
`pack_matrices_per_wg` matrices in one work-group, one per sub-group. Offered only while
`m <= 32 && n <= 32`; above that the panel wants `getrf_leaf_wg`'s wider group and its
work-group barriers. `G > 1` under `WorkGroup` scope is a race by construction -- a wrong
answer, not a launch failure -- so the launcher derives the two together and
`getrf_cta_debug_launch` is the only way to see the pairing from outside.

Native arm, large batch, against `benchmarks/results/factor_baseline_getrf_*.csv`:

| type | n | batch | vendor | before | after | ratio |
|---|---|---|---|---|---|---|
| float | 8 | 32768 | 0.0319 | 0.1216 | 0.0680 | **0.559** |
| float | 16 | 32768 | 0.1218 | 0.2605 | 0.1636 | **0.628** |
| float | 32 | 16384 | 0.3159 | 0.5730 | 0.4596 | **0.802** |
| double | 16 | 32768 | 0.4088 | 1.4945 | 0.7790 | **0.521** |
| cfloat | 16 | 32768 | 0.2280 | 0.3422 | 0.2233 | **0.653** |

1.25-1.92x, and it closes rather than crosses the vendor gap (float n = 8 goes from 3.8x
behind cuBLAS to 2.1x). The register probe shows the SubGroup instantiations at
33/49/35/62 registers for float/double/cfloat/cdouble against 49/62/52/46 for the
WorkGroup ones, zero spill on every entry function.

### Negative result: registers are not why packing wins

The packed path is not cheaper per matrix -- it gives each matrix 32 lanes where the
unpacked launcher gave it 64 to 512. The win is the barrier scope and the number of
matrices in flight per work-group, which is why it is gated on the band where one
sub-group is enough work rather than applied wherever the tile fits.

## The tiny tier

`Algorithm::Tiny` (`src/extensions/getrf_tiny.cc`), the register-resident `getrf` arm
for order `n <= 32`. It lands **pin-only**: `preferred()` is false for it and
`native_tier_preferred()` carries an explicit `case Algorithm::Tiny: return false;`, so
`automatic()`, the vendor-free walk and a bare `native` pin all resolve exactly as they
did before. Only `BATCHLAS_GETRF_ROUTE=tiny` reaches it. The window comes from the
measured grid in a separate PR.

Shape: one matrix per `SubGroupPartition<N>`, `N in {8, 16, 32}` from the compile-time
ladder (`n <= 8 -> 8`, `<= 16 -> 16`, `<= 32 -> 32`), lane `r` owning row `r` in a
`D rA[N]` register array; `32/N` matrices per sub-group and `tiny_native::kTinySubGroups`
(2) per work-group, so 8 / 4 / 2 matrices per work-group of 64 -- see "The work-group
A/B" below for why 2 and not 4. Rows and columns past `n`
carry the identity, so the padded tile is `[[A, 0], [0, I]]` and the unrolled body needs
no lane guard. Partial pivoting is a **lazy relabel** (`rowid`), never a row move; the
pivot metric is LAPACK's `cabs1`; `ipiv` is 1-based and global; the elimination
continues finitely past a zero pivot (MAGMA's `update = 0`).

Zero local memory and **zero barriers of any scope** -- `ptxas` reports `used 0
barriers, 0 bytes smem` for all 22 entry functions -- so the `(47104, 49664]` launch
hole is structurally unreachable here and none of the defensive 49,920 padding the CTA
tiers carry is present.

### The register probe, and the unroll that decides it

`scripts/register_probe.sh out.log '' batchlas_extensions_cta`, 11 instantiations
(4 types x 3 N, less cdouble N=32), both the `<name>` and `<name>_with_offset` variants:

| type | N=8 | N=16 | N=32 |
|---|---|---|---|
| float | 48 | 63 | 96 |
| double | 64 | 90 | 138 |
| cfloat | 62 | 86 | 143 |
| cdouble | 94 | 138 | not instantiated |

Every cell: **stack frame 0, spill stores 0, spill loads 0, smem 0, barriers 0.** Worst
`regs x work_group_size` = 143 x 64 = **9,152** against the 65,536 hard limit, i.e. 7.2x
of slack. `kWorstRegsPerThread` in the TU is set to 143 and static_asserts on it. (The
table was first taken at `wg = 128`; the work-group A/B below moved it to 64 and did not
move a single register count.)

**Two spellings decide whether the array is in registers at all, and neither is
diagnosed as an error.**

1. `if (j >= n) break;` inside `#pragma unroll` makes the trip count data-dependent.
   This toolchain then declines the unroll (`-Wpass-failed`: "loop not unrolled"), `rA`
   becomes dynamically indexed, and `ptxas` relocates it to the stack **with zero
   spill**, so the probe's own gate stays green. Measured with `break`: float N=32 came
   back at 128 B stack frame -- exactly 32 floats -- and 40 registers; double N=16 128 B
   / 48 registers; cdouble N=16 256 B / 56 registers. N=8 was unaffected. Replacing
   `break` with `continue` -- the guard is kernel-uniform either way, because the entry
   point rejects a heterogeneous batch -- put every one of those cells back in
   registers.
2. Writing the rank-1 update as `for (k = 0; k < N; ++k) { if (k <= j) continue; ... }`
   asks the unroller to build `N(N+1)/2` copies and then fold half of them away. At the
   widest complex cell it gave up part way: cfloat N=32 alone kept a 256 B frame (32
   `Cx<float>`) at 69 registers while every other cell was clean. Starting the loop at
   the compile-time bound, `for (k = j + 1; k < N; ++k)`, fixed that cell (143
   registers, frame 0) and changed no other.

The moral for the sibling tiers: **stack frame is the wrong gate for a spill hunt but
the right gate for a residency claim.** A register-resident design that reports zero
spill and a frame equal to `N * sizeof(D)` is not register-resident.

### Pad rows and the argmax, corrected

The design this tier was built to claimed that masking padded rows out of the pivot
candidate set (`rowid < n`) was load-bearing, because a pad row carries the identity
and so reads exactly 0 in a live column -- which is not *smaller* than every live
candidate when that column is entirely zero. **Armed and measured: removing the mask
changes nothing.** The property is guarded by the tie-break DIRECTION instead: the
argmax key orders ties towards the lowest `rowid`, and `rowid == j` is always a live
candidate because `j < n`, so a pad row can never win a tie it is only ever tied in.
Reversing the tie-break (`ok < key` -> `ok > key`) is what elects a pad row, and on a
padded order that yields `ipiv[j] > n`. The mask is kept as intent, not as the guard.

**Why the argmax carries a (magnitude, key) PAIR and not one packed `uint64`.** Packing
the magnitude and the rowid into a single 64-bit word so the butterfly can compare one
value is exact only for a 32-bit magnitude, and on a 32-bit shuffle unit it still costs
**two 32-bit shuffles per round** -- it saves the compare, not the traffic. The pair form
is therefore no more expensive and stays exact for `double`. The XOR mask is a power of
two below N and so is uniform across the whole 32-lane sub-group, which is what
`sg_compat.hh`'s non-NVPTX fallback requires.

The NaN map is real and is not shared with the CTA tier's reasoning.
`getrf_cta_device.hh` is safe from NaN for free -- it seeds its running best at `R(-1)`
and updates through `v > bv`, so a NaN never enters. The tiny argmax seeds every lane
from its OWN magnitude, and `ov > NaN` and `ov == NaN` are both false, so an unmapped
NaN survives every butterfly round and the lanes end up disagreeing about the winning
LANE: different broadcast sources, a wrong factor, no crash. Measured with the map
removed: float n=4 item 0 returned `ipiv[1] = 3` where the CTA oracle returns 2; cfloat
n=4 item 0 returned `ipiv[3] = 9`, which is not even inside `[4, 4]`.

### Armed breaks

Each was applied, built, observed red, and restored.

| break | expected | observed |
|---|---|---|
| (a) `act = (rowid > j)` -> `rowid >= j` | pivot row scales itself; residual red | float n=1 b=0, `\|\|PA-LU\|\|/\|\|A\|\|` = 1.25 vs tol 2.38e-05, red at every larger n |
| (b) drop `if (k >= n) continue` from the store | pad written | float `n=1, ld=6, stride=17`, element 6 -- item 0's column 1, the first column past the matrix -- differs from `alloc()`'s poison by 9750. The residual goes red too (float n=2, **0.708** against tol **4.77e-05**, every item) but only at `ld = n`, where item b's pad columns land inside item b+1; at `ld = n+5` they do not, and then the pad assertion is the ONLY one that fires |
| (c) `ok < key` -> `ok > key` in `tiny_argmax_pair` | exact `cabs1` tie goes to the wrong row | float n=4, `ipiv[0] = 3` where 1 is required, every item; residual, `worst_pivot_ratio` and the planted-zero-column case all stayed GREEN |
| (d) `tiny_partition_id` -> `part.get_group_linear_id()` | 4 sub-groups alias 4 matrices | float n=1 batch 19, item 4 untouched: `ipiv[0] = 195935983` (0x0BADBEEF), `info` still -12345 |
| (e) drop `rowid < n` from the candidate mask | pad row wins, `ipiv > n` | **PASS** -- see above; the guard is the tie-break direction, not the mask |
| (f) drop the `mag == mag` NaN map | pivots diverge from the CTA oracle | float n=4 `ipiv[1]` 3 vs 2; cfloat n=4 `ipiv[3] = 9`, outside `[4, 4]` |
| (g) add `sycl::group_barrier(it.get_group())` | source check red | red, quoting the inserted line; no rebuild needed |
| (g') add a 1-element `sycl::local_accessor` | source check red | red on the `local_accessor` token |
| (h) `lu_cabs1` -> the modulus, in the SHARED `getrf_cta_device.hh` | `TinyPivotsMatchLapacke` red on the complex types; the tiny-vs-CTA cases GREEN, because both tiers read that helper | exactly that. cfloat n=2 b=1 `ipiv[0] = 1` where the host oracle requires 2, at an argmax margin of 0.987; cdouble likewise. `TinyPaddingIsInertAgainstTheCtaRoute`, `TinyArgmaxIgnoresANaNCandidate` and `TinyBreaksAnExactCabs1Tie` stayed green on ALL FOUR types. `TinyFactorisesAndPivotsExactlyAtEveryOrder` also went red, via `expect_piv` |
| (i) `if (j >= n) continue` -> `break`, then read the PROBE | the array leaves registers with ZERO spill, so the probe's own headline gate stays green and only the stack-frame column moves | 28 `-Wpass-failed` "loop not unrolled" warnings; probe summary still reported "entry functions with non-zero spill (THIS IS THE GATE): **0**"; and **14 of 22 functions grew a stack frame of exactly `N * sizeof(D)`** -- float N=16 64 B, float N=32 128 B, double N=16 128 B, double N=32 256 B, cfloat N=16 128 B, cfloat N=32 256 B, cdouble N=16 256 B, every N=8 cell unaffected. Registers collapsed with it (float N=32 96 -> 64, cfloat N=32 143 -> 66, cdouble N=16 138 -> 68) |

(a), (d), (h) and (i) were re-armed and re-observed after the work-group moved to 64;
(d) in particular can only fail at more than one partition per work-group, so it is the
one the work-group change could have invalidated. It did not: float n=1 batch 19 item 4
still came back completely untouched, `ipiv[0] = 195935983` (0x0BADBEEF) and `info` still
-12345, identical to the wg=128 observation.

### The work-group A/B

The tier shipped its first draft at a work-group of **4 sub-groups (128)**, justified in
the source comment by "a wider group divides the launch tail further". **That reason is
backwards.** The launch tail wastes up to `kMpw - 1` partitions and `kMpw` is `wg / N`,
so a wider work-group makes the tail COARSER: at `N = 8`, `wg = 128` wastes up to 15
partitions where `wg = 64` wastes at most 7. getrf was also the odd tier out --
`potrf_tiny.cc` and `geqrf_tiny.cc` both take the shared `tiny_native::kTinyWgSize`
(2 x 32 = 64).

Measured A/B, one binary rebuilt between the two runs, 30 cells each
(4 types x `n in {4, 8, 16, 32}` x `batch in {4096, 16384}`, cdouble n=32 excluded by
D3), 7 reps, arms `vendor,cta,tiny` interleaved in one process under
`benchmarks/gpu_guard.sh 1`. Raw: `benchmarks/results/p1_tiny_getrf_wg{128,64}.csv`.

| | cells |
|---|---|
| wg=64 faster by > 2% | 2 |
| wg=64 slower by > 2% | 2 |
| within 2% | 26 |

Both apparent losses were re-run and did NOT reproduce: cfloat n=32 b4096 came back at
0.1555-0.1565 ms against the grid's one-off 0.1840 (wg=128 measured 0.1549), and cfloat
n=8 b4096 sits in a 0.0114-0.0119 band that both grids fall inside. The one cell that
moves reproducibly is **cdouble n=16 batch 16384: 0.7038 ms at wg=128 against
0.6693-0.6701 ms across three independent wg=64 runs, a stable 5% win** -- the widest
cell the tier holds for that type.

**Decision: wg = 64.** Not because 5% in one cell is decisive, but because the A/B says
the work-group is not a performance knob at all here, and the tie then goes to the value
that (a) is the shared constant the other two tiny tiers already use, (b) halves the
launch-tail waste, and (c) doubles the register-gate slack (143 x 64 = 9,152 against
65,536, 7.2x, where wg=128 gave 3.6x).

**A prediction that did NOT survive contact.** The occupancy arithmetic the change was
argued from -- registers/SM divided by the block's demand, giving blocks/SM and hence
"matrices resident per SM" -- predicted wg=64 would win 8-17% in five cells (float n=8,
double n=32, cfloat n=16, cfloat n=32, cdouble n=16). Four of those five measured as a
wash. Register counts are identical between the two work-groups in all 22 entry
functions, so the model's inputs were right and its conclusion was still wrong: at these
sizes the tier is bound by outstanding memory requests, and the number of resident warps
stops mattering long before the occupancy table says it should. **Occupancy is not this
tier's figure of merit; do not tune it against one.**

### First timing: the tiny tier against cuBLAS and the CTA tier

Ratios are `vendor_ms / tiny_ms` and `cta_ms / tiny_ms`; **> 1 means the tiny tier
wins**. wg = 64, 7 reps, three arms interleaved in one process, `rel_sd` under the 10%
gate on every row, and every row's untimed correctness re-run reported `ok` with zero
pivot mismatches against the harness's host `xgetrf`. Raw:
`benchmarks/results/p1_tiny_getrf_wg64.csv`.

| type | n | batch 4096 vs vendor | batch 16384 vs vendor | b16384 vs CTA |
|---|---|---|---|---|
| float | 4 | **1.76** | **1.43** | 2.65 |
| float | 8 | **1.74** | **1.49** | 2.66 |
| float | 16 | **1.70** | **1.65** | 2.13 |
| float | 32 | **1.24** | **1.05** | 1.54 |
| cfloat | 4 | **1.60** | **1.28** | 2.57 |
| cfloat | 8 | **1.52** | **1.30** | 2.66 |
| cfloat | 16 | **1.34** | **1.31** | 1.58 |
| cfloat | 32 | 0.71 | 0.64 | 1.22 |
| double | 4 | **1.29** | 0.69 | 3.55 |
| double | 8 | **1.17** | 0.86 | 3.49 |
| double | 16 | 0.80 | 0.71 | 1.46 |
| double | 32 | 0.71 | 0.65 | 0.88 |
| cdouble | 4 | 0.95 | 0.52 | 3.55 |
| cdouble | 8 | 0.96 | 0.80 | 3.17 |
| cdouble | 16 | **1.23** | **1.41** | 1.33 |

**The plan's kill criterion is met.** "If float n=32 tiny is below 1.0x of cuBLAS at
batch 16384, stop and record the number": it measured **1.051x** (0.29912 ms against
0.31445). It is the narrowest win in the float column and it is a win.

**The tier beats the CTA tier it would replace in 14 of 15 cells**, by 1.2-3.6x, the one
exception being double n=32 at 0.88. That is the comparison the design was really
making, and it is not close.

Against the VENDOR the picture splits exactly as predicted, and the two predictions the
grid existed to falsify both survived:

* **Predicted wins -- float and cfloat at n <= 16.** Measured 1.28-1.76x. Confirmed.
* **Predicted loss -- the widest complex cell, cfloat n = 32.** Measured 0.64-0.71x.
  Confirmed.

Three results the design did NOT predict:

1. **float n = 32 wins** (1.05-1.24x), where the residency table predicted a loss on the
   same reasoning that correctly predicted cfloat n=32's.
2. **cdouble n = 16 wins** (1.23-1.41x) while cdouble n = 4 and 8 LOSE (0.52-0.96x) --
   the opposite of the monotone-in-n shape every other type shows.
3. **double and cdouble lose ground as the batch grows** (double n=4: 1.29x at b4096,
   0.69x at b16384), where float and cfloat only soften. The FP64 rate is 1/64 here, so
   these two types reach their compute roof while the 32-bit types are still on the
   memory roof, and a larger batch just buys the vendor more of what it is better at.

Result (2) is worth a sentence because it is where the padding cost is visible in
isolation: at `n = 4` the `N = 8` bucket does 4x the useful arithmetic and reads a 16-byte
column out of a 32-byte sector, i.e. 2x structural over-fetch that no mapping in this
family removes. **The roof for a small `n` is SECTOR-ROUNDED, not `n^2`**, and a
ratio-to-roof computed from `4 n^2 sizeof(T) batch / 950 GB/s` will read `n = 4` as a
failure against a ceiling it cannot reach.

`preferred()` is still all-false and this PR does not change it: the window belongs in
the change that also runs the padded orders (`n = 9, 17, 24`) and the intermediate
batches, which this grid does not cover.

### The pivot-margin gate on elementwise comparisons

Comparing a pivot SEQUENCE elementwise against any other implementation is only well posed
where the argmax is decided by more than rounding. After a few rank-1 updates two
candidates in unstructured data can sit within 1 ULP of each other, and then the device
(which contracts `a - b*c` into an FMA) and the host (which does not) legitimately pick
different rows — both correct LU factorisations, both with correct residuals. **Measured
before this guard existed: double n = 10, item 0, `ipiv[6] = 7` against LAPACKE's 9, at a
margin of ~1 - 1e-16.**

`host_getf2_with_margins` therefore records, at each step,
`margin[k] = cabs1(runner-up) / cabs1(winner)`, in [0, 1], and every elementwise pivot
assertion stops at the first step whose margin is within `kTinyAmbiguous` of 1 — because
every step after a divergence is a factorisation of a different matrix.

`TinyPivotsMatchLapacke` is the only case in `tests/getrf_tests.cc` whose oracle shares no
line of code with the kernel under test. `expect_piv` is the permutation
`make_dominant_permuted` built, so the winner at each step is forced by construction and
the data is never ambiguous; and the tiny-vs-CTA cases compare against a tier that
`#include`s the same `getrf_cta_device.hh` and calls the same `lu_cabs1`, so a defect in
the shared pivot METRIC moves both tiers together and leaves every one of them green —
which is exactly what break (h) above measured. It runs on `make_random` deliberately: on
a dominant-permuted matrix the argmax is decided by a gap of orders of magnitude, so a
merely WRONG metric still picks the right row.

It carries TWO oracles. `host_getf2_with_margins` is a BLAS-free triple loop, correct on
any machine, and it supplies the margins. LAPACKE is the reference implementation, and is
cross-checked against the triple loop — but only on cells where LAPACKE passes a residual
check on its OWN factor, for the reason the next section records. The gate is therefore:
the DEVICE is always asserted against the triple loop; LAPACKE is additionally asserted
against it wherever LAPACKE can be trusted. The case counts the cells it had to drop and
fails if LAPACKE was untrustworthy EVERYWHERE, because a silent fallback would let the
LAPACKE arm quietly stop testing anything.

### Why the tiny-vs-CTA element bound is relative

`TinyPaddingIsInertAgainstTheCtaRoute` factorises an order inside a wider bucket and
compares against the CTA tier, which is the oracle for the transition period because it
carries no compile-time `N` at all and so cannot share a padding defect.

It is NOT bit-exact: the two bodies are separate translation units and nothing forbids the
compiler from contracting one multiply-subtract and not the other. The PIVOT SEQUENCE is
compared exactly, because that is integer bookkeeping and a padding defect shows there
first.

The element bound is RELATIVE, and tight. Contraction is the only licensed difference: one
fused multiply-add per elimination step, so the two factors may part by a few ulp of the
value they hold, times the n steps that value passed through — never by a fixed slack
measured against the largest entry in the matrix. The fixture's diagonal is `4n` while its
off-diagonal entries are O(1), so an ABSOLUTE bound scaled by `4n` licenses thousands of
ulp on exactly the entries where a padding defect shows up; scaling by the reference
element instead keeps the bound at a few ulp everywhere. The allowance is `4 * n` ulp per
element — the factor of four is margin for a toolchain that contracts differently in the
two translation units, and it is margin and nothing else: **on this box the two tiers agree
EXACTLY at every order, type and element of the sweep.**

### The host dgetrf oracle is broken on this box

**`LAPACKE_dgetrf` on this machine returns a wrong factorisation for every `n >= 10`.**
Measured in a 40-line standalone C program with no BatchLAS code linked at all
(`gcc -O0 lap_check.c -llapacke -llapack -lblas`), random matrices, residual
`||PA - LU||_F / ||A||_F` computed from LAPACKE's own returned factor and interchange
list:

| n | dgetrf residual | sgetrf residual |
|---|---|---|
| 2..9 | 0 .. 1.4e-16 | 6e-09 .. 5e-08 |
| 10 | **4.20e-01** | 5.50e-08 |
| 11 | **8.29e-01** | 6.78e-08 |
| 12..15 | **4.3e-01 .. 4.9e-01** | 5.3e-08 .. 7.3e-08 |
| 16 | **1.10e+00** | 7.26e-08 |

`n = 10` is where LAPACK's recursive `dgetrf2` starts issuing a `dgemm`, and `sgetrf` is
correct at every order -- the signature of this box's known-bad OpenBLAS double GEMM
kernel, which the memory ledger records under "Broken OpenBLAS dgemm on this machine".
Nothing in BatchLAS is involved.

Consequence for the test suite: `TinyPivotsMatchLapackeOnUnstructuredData` cannot use
LAPACKE as an unconditional oracle. It runs a BLAS-free triple-loop `getf2` as the
primary oracle (correct on any machine, and it also supplies the argmax MARGIN, without
which an elementwise pivot comparison is not well posed at all), and cross-checks
LAPACKE against it only on cells where LAPACKE passes a residual check on its own factor.
The case counts the cells it dropped and FAILS if LAPACKE was untrustworthy everywhere,
so the arm cannot silently stop testing. Observed on this box: the `double`
instantiation drops **161 of 224 cells**, which is exactly the `n >= 10` cells at
`batch = 7` (`(32 - 9) * 7 = 161`); float, cfloat and cdouble drop none.

### R7: the device link, all three tiny tiers landed

**This is the ONE site for the tiny tier's R7 figure**; `docs/perf/potrf.md` and
`docs/perf/qr.md` point here rather than carrying a third and fourth copy. The number is
a property of `batchlas_extensions_cta` as a whole, not of any one tier.

`docs/perf/potrf.md` recorded 151 s with `geqrf_tiny` landing concurrently and
`getrf_tiny` in flight, called it not attributable, and asked for a re-measure once all
three had landed. They have.

`time cmake --build build/presets/dev-tests --target batchlas_extensions_cta --parallel 4`
after touching `getrf_tiny.cc`: **167.4 s** (2m47.4) against the 117-125 s recorded
baseline, i.e. **+34% to +43% for the LIBRARY** with all three tiers present.

Attribution, from the probe log's per-function `Compile time` fields (982 entry functions,
182.9 s of `ptxas` in total):

| tier | entry functions | ptxas | share of library total |
|---|---|---|---|
| `getrf_tiny` | 22 | 19.2 s | **10.5%** |
| `geqrf_tiny` | 22 | 19.4 s | 10.6% |
| `potrf_tiny` | 22 | 11.2 s | 6.1% |
| all three | 66 | 49.8 s | 27.2% |

**getrf_tiny is inside R7's 15% budget at 10.5%. The three tiers TOGETHER are not**, at
27.2% of `ptxas` and a +34-43% wall-clock delta on the library. That is a fact about the
package as a whole, not about this tier, and it is recorded here rather than silently
split three ways: whoever adds a fourth tier to `batchlas_extensions_cta` inherits it.

**The written justification R7 requires, since the budget is exceeded.** The 66 entry
functions are not redundancy that can be traded away: they are 3 ops x 3 compile-time
`N` buckets x 4 types, less the capped cdouble N=32 cell, each also emitted as
`_with_offset`. `N` is what makes `rA[N]` a register array rather than a dynamically
indexed one, which is the whole tier -- collapsing the buckets to a runtime `n` moves
`rA[]` to the stack and costs the win outright (the unroll finding above). The type list
is the library's public contract. The `_with_offset` pair is the batched/strided split
every kernel in this library already carries. The only lever that does not destroy the
tier is the bucket count, and P1 already ships the minimum that covers `n <= 32` on a
32-lane sub-group (`tiny_n_is_legal`: 8, 16, 32). The cost is accepted knowingly and
paid once, at build time, for a tier that is 2-4x on the shapes it serves.

**Re-measure hygiene.** The figure is contention-sensitive: a repeat of the same
touch-one-TU build while another compile job held the box came in at **206.3 s**. Take
this number on an idle machine or not at all.

### The register probe at wg = 64

`scripts/register_probe.sh out.log '' batchlas_extensions_cta`. 22 entry functions
(4 types x 3 N, less cdouble N=32, each also as `_with_offset`). **Register counts are
identical to the wg=128 table above** -- the work-group does not enter them.

Every one of the 22: **stack frame 0, spill stores 0, spill loads 0, 0 bytes smem,
0 barriers.** Worst `regs x work_group_size` = 143 x 64 = **9,152** against the 65,536
hard limit, 7.2x of slack.

### D3: why cdouble stops at N=16

`rA[32]` for `Cx<double>` is 128 registers before any working set, and the probe puts
the N=16 cell at 138 registers already. With G-packing the compute-to-DRAM ratio is
`r = N*q / (43.1*s)` (`q` = clocks per warp-instruction of the update: 0.25 float, 1.5
cfloat, 16 double, 96 cdouble; `s` = bytes per element), which for cdouble at N=32 is
4.45 -- predicted 565 us x 4.45 = 2,514 us against a measured vendor 2,917 us, i.e.
1.16x. Not worth the spill risk. Re-opening the decision is the one `tiny_cap<T>()`
constant.

### The pivot key encoding

`tiny_device.hh`'s `tiny_key(order_field, lane)` is the tier's ONLY pivot-key encoding,
and both halves are load-bearing:

* the **ordering field**, in the high bits, decides ties — lowest wins, `I?AMAX`'s order —
  and is what makes a pad row unable to win a tie it is only ever tied in;
* the **winner's lane**, in the low `kTinyKeyLaneBits = 8` bits, is carried only as a
  broadcast source for `tiny_bcast`.

The caller must keep every ordering field DISTINCT within the partition, or the lane bits
start deciding a tie they know nothing about. `getrf_tiny.cc` does that by seeding a
non-candidate at `rowid + N` rather than at `rowid`, which sorts every non-candidate
strictly below every candidate and so lets even an all-NaN live column elect a live row.
Reversing the comparison is break (c) above; dropping the `mag == mag` NaN map is break
(f). `tiny_argmax_pair` itself owes nothing to the caller except that seed: `ov > a` and
`ov == a` are both false when either operand is NaN, so a lane seeded with NaN keeps it
through every round and the partition ends with lanes disagreeing about the winner — a
silent wrong answer, not a crash.

### Why the rank-1 update starts at `k = j + 1`

`for (k = j + 1; k < N; ++k)` is a compile-time lower bound once `j` is unrolled. The
equivalent-looking `for (k = 0; k < N; ++k) { if (k <= j) continue; ... }` asks the
unroller to build `N(N+1)/2` copies and then fold half of them away, and at the widest
complex cell it gives up part way and leaves `rA` indexed dynamically — the measured cell
and its 256 B frame are item 2 of [the register probe, and the unroll that decides
it](#the-register-probe-and-the-unroll-that-decides-it).

The store loop's guard is `lane < n` and not `rowid < n`: a lane indexed at or above `n`
keeps a `rowid` at or above `n` for the whole loop, so the two are equivalent and the lane
form is loop-invariant. And the tier writes `info` once per item and never reads it, so —
unlike the CTA tier, whose `getf2_panel_device` READS `info` for first-failure-wins across
panels — it needs no zero pre-fill, and cannot serve as a blocked-driver panel leaf until
one is added.

---

## The register panel leaf

**P4, 2026-09-11.** `src/extensions/getrf_panel_reg.cc` plus
`getrf_panel_reg_device.hh`: one work-group per `m x ncols` panel, work-item `tid`
owning physical row `tid` of a compile-time `D rA[NB]`, `NB = 32`, pivoting by lazy
relabel and publishing the pivot row through `NB` words of local memory. It is a
**drop-in for `getrf_panel_factorize`**, not a tier — same global 1-based `ipiv`, same
read-modify-written `info`, same `cabs1` pivot metric — so the blocked driver may take
either and the two must agree on `ipiv` **exactly**.

**Since the P4 integration grid the register leaf is the DEFAULT**, and
`BATCHLAS_GETRF_LEAF=slm` is how the older local-memory panel is asked for. It is faster
at 153 of the 156 paired cells measured, by 1.01x to 2.13x, and the three exceptions are
one cell (double `n = 32`) at its three batch rungs. It also moved one shipped route:
cfloat's `getrf` floor from 512 down to 256 at batch >= 256. The grid, the discards and the one
loss are in [the register leaf A/B](#the-register-leaf-ab); the tier question the grid
raised but did not close is in [which native tier serves 33 to 256](#which-native-tier-serves-33-to-256);
the window is in [the cfloat window moves to 256](#the-cfloat-window-moves-to-256).

`native_tier_preferred` is untouched.

### What this replaces, and why it should be faster

Today's leaf (`getrf_panel_resident_launch`) stages the panel into local memory and runs
`getf2_panel_device` there, so **every** element of **every** rank-1 update is a local
read-modify-write. At the blocked driver's own shape — `mp x 32` with `mp` up to `n` —
the update is `sum_k (mp-k-1)(31-k)` local RMWs per panel. The register leaf does the
same arithmetic entirely in registers: memory is touched twice, once at the load and
once at the store, and the only local traffic per column is one work-item publishing at
most 32 words.

Two second-order consequences, both of which the measurement must confirm rather than
assume:

* **Local-memory footprint collapses.** The resident leaf asks for
  `(m|1) * ncols * sizeof(D) + 256` B — 16,768 B for float `m = 128, ncols = 32`, and it
  stops fitting at all above a height the budget sets, at which point today's driver
  falls to the *global* leaf. The register leaf asks for `NB*sizeof(D) + 32*(sizeof(R)+4)`
  — **under a kilobyte for every type**, independent of `m`. It cannot reach the 48 KB
  launch hole and is not padded for it.
* **The work-group is exactly the panel height**, `roundup(m, 32)`, one row per
  work-item. Today's `getrf_leaf_wg` picks "about four columns' worth of rows, clamped to
  [64, 512]" — 512 work-items for a 128-row panel, i.e. four work-items per row with
  three of them idle through the argmax.

### Two barriers per column

Per column `j` the kernel executes **two** work-group barriers, against
`getf2_panel_device`'s four (B1..B4). The plan's design sketch asked for three and MAGMA's
`getf2_fused` has four.

The third is removable because the two shared arrays are separated by *each other's*
barrier, one step apart:

* the argmax slots `rval`/`ridx` are written before B1 of step `j` and read after it; the
  next write is at step `j+1`, and **B2 of step `j`** lies between;
* the published pivot row `sx` is written before B2 of step `j` and read after it; the
  next write is at step `j+1`, and **B1 of step `j+1`** lies between.

Removing either barrier therefore breaks a WAR one column later, not a RAW in the same
column. The armed break for B2 was predicted to need the **tall** cells; measured, it is
red from `m = 33` -- the shortest panel that spans two sub-groups is already enough.
evidence: [armed breaks (P4)](#armed-breaks-p4)

### The register panel leaf: register probe

`clang++ -fsycl -fsycl-device-only -S` to nvptx64 IR, `llc -mcpu=sm_89`, then
`ptxas -arch=sm_89 -O3 -v`. Both the `<name>` and `<name>_with_offset` entries; the
counts are identical.

| type | registers | stack frame | spill st / ld | `regs x wg` at the cap |
|---|---|---|---|---|
| float | 64 | **0** | 0 / 0 | 64 x 512 = 32,768 |
| double | 104 | **0** | 0 / 0 | 104 x 512 = 53,248 |
| cfloat | 96 | **0** | 0 / 0 | 96 x 512 = 49,152 |
| cdouble | 176 | **0** | 0 / 0 | 176 x 288 = 50,688 |

Stack frame is the gate for a residency claim and the spill columns are the ones that
mislead — [the register probe, and the unroll that decides
it](#the-register-probe-and-the-unroll-that-decides-it) is why. **This pipeline is not
the shipped device link**, which is where `scripts/register_probe.sh` reads its numbers;
it skips `sycl-post-link`. It was calibrated against the tiny tier through the identical
steps and read float `N=32` at 96 (published 96), double at 136 (published 138) and
cfloat at 128 (published **143**) — so it was expected to read up to **11.7% low**, and
that is the headroom factor `getrf_panel_reg.cc` applies before deriving each type's
height cap. **The integration pass ran the real probe** and it reads 0-3% low on this
kernel, not 11.7%: 64 / 105 / 99 / 176 against the standalone 64 / 104 / 96 / 176. The cap
this table's last column computes was, separately, the wrong cap:
[the register cap that binds is per sub-partition](#the-register-cap-that-binds-is-per-sub-partition).

**A measured negative, recorded because it cost an hour and reads like success.** The
first version had `getf2_panel_reg_device` as a plain `inline` function. Unrolled over
`NB = 32` it is large enough that the inliner declines it, the kernel entry then passes
the three `local_accessor`s **by value** through an ABI call, and every entry reported a
**200-byte stack frame with zero spill** — the exact signature this page already
documents as a dead register-resident tier. It was not one: the array was in registers
and the depot was the accessor triple. `[[gnu::always_inline]]` removed the call and the
frame together. The lesson stands anyway: a non-zero frame must be *explained*, not
assumed to be `rA`.

### The register cap that binds is per sub-partition

**A defect this leaf shipped with, found by running the tests it shipped with.** The
first version derived each type's height cap from `regs x work-group <= 65,536`, the
per-block register file that `register_probe.sh`, `getrs_fused.cc`, `trsm_native.cc` and
five `*_tiny.cc` files all name as *the* gate. At cdouble that arithmetic allows a
288-lane work-group (208 budgeted registers x 288 = 59,904), the predicate advertised
`max_m = 288`, and the launch failed outright:

```
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (701)  ->  UR_RESULT_ERROR_OUT_OF_RESOURCES
getrf_tests: RegPanelAgreesWithTheLocalMemoryLeaf, cdouble, m=257 ncols=1
```

**On sm_89 the register file is owned by four sub-partitions of 16,384 registers each**
and a block's warps are dealt round-robin over them, so what has to fit is

```
ceil(warps / 4) * 32 * ((regs + 7) & ~7) <= 16,384
```

— the `& ~7` because ptxas allocates in granules of 8 and the count the gate divides into
is the ALLOCATED one. Equivalently `threads <= 128 * floor(512 / regs_alloc)`. It
reproduces the observation exactly: cdouble at 176 registers budgets to 208, allows 2
warps per partition = **256 threads**, and 256 passed where 288 aborted. The SYCL
runtime's own check does not see it either — it refuses a launch with the per-block
message (`131 registers ... 512 work-items`, seen when an armed break pushed float to
131), so a kernel can clear the runtime's gate and still be refused by the driver.

**The discriminator is the warp count, not the register count.** An earlier draft of this
section said the sub-partition rule "is strictly tighter than the per-block rule whenever
`regs > 128`". That is wrong, and wrong in the dangerous direction: it clears sites it
should flag. The two rules coincide **exactly when `warps % 4 == 0`**, i.e. when the
work-group size is a multiple of 128 — there the `ceil` is an identity and
`(threads / 4) * regs <= 16,384` is `threads * regs <= 65,536` verbatim. At every other
work-group width the `ceil` rounds up and the sub-partition rule is strictly tighter,
*at any register count*. Worked counter-example at a count far below 128: `r = 76` gives
`128 * floor(512 / 76) = 768` against `floor(65,536 / 76) = 862`.

That counter-example is not hypothetical; it was a live site until 2026-09-15.
`src/extensions/getrs_fused.cc` computed its work-group cap as `(65536 / regs) & ~31` —
the per-block spelling, rounded to a sub-group, hence a cap that is a multiple of 32 but
need not be a multiple of 128, which is exactly the case where the two rules diverge. By
the sub-partition arithmetic (probed counts from the table at `getrs_fused.cc:62-77`,
allocation rounded to 8), that spelling handed out widths with no margin left — two of
them sitting exactly on the partition limit, and one already past it:

| cell | probed / allocated | old cap | warps | busiest partition | verdict |
|---|---:|---:|---:|---:|---|
| double, `NoTrans`, `nrhs` 5..8 | 61 / 64 | 928 | 29 | `8 * 32 * 64 = 16,384` | **exactly at** the limit |
| cdouble, `Trans`, `nrhs` 3..4 | 58 / 64 | 992 | 31 | `8 * 32 * 64 = 16,384` | **exactly at** the limit |
| float, `Trans`, `nrhs` 5..8 | 68 / 72 | 832 | 26 | `7 * 32 * 72 = 16,128` | fits, 1.6% spare |
| cdouble, `Trans`, `nrhs` 5..8 | 86 / 88 | 672 | 21 | `6 * 32 * 88 = 16,896` | **over by 512** |

A drift inside `kGetrsFusedRegMargin` (8) on either of the first two turns a legal launch
into `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, and the last one is over the partition before
any drift at all. **The whole table is arithmetic on the shipped constants, not a launch
anyone has run** — the only runtime abort this section records is the cdouble 288-lane
one above, at a different site. A reviewer applying the deleted `regs > 128` test would
have cleared every row of it without computing anything, since the largest count in the
table is 86.

The site now calls the shared helper, `resident::sm89_max_work_group(regs)`
(`src/extensions/getrs_fused.cc:99`), which is this rule in one place:
`src/util/resident_capacity.hh:90-103` gives both the ceiling and `sm89_fits(regs, wg)`,
the predicate form, so a `static_assert` can read as prose. Any site still dividing into
65,536 is using the spelling that produced the four rows above.

Device-link register counts for the shipped kernel, from `scripts/register_probe.sh
<log> GetrfPanelRegKernel batchlas_extensions_cta` (1,082 entry functions, **0 with
spill**, 327 s link), against the standalone `ptxas -v` pipeline the implementation used:

| type | standalone probe | device link | budget (15% + granularity) | warps/partition | height cap |
|---|---:|---:|---:|---:|---:|
| float | 64 | **64** | 80 | 6 | 512 (capped by `kPanelRegMaxRows`) |
| double | 104 | **105** | 120 | 4 | 512 (capped) |
| cfloat | 96 | **99** | 112 | 4 | 512 (capped) |
| cdouble | 176 | **176** | 208 | 2 | **256** |

The standalone pipeline read 0-3% low here, not the 11.7% its tiny-tier calibration
predicted; the 15% headroom constant stands, and it is what keeps cdouble's budget (208)
on the same side of the partition boundary as its true count (176).

**The guard that was missing.** `RegPanelCapacityIsOneSpelling` compared one arithmetic
spelling of the cap with another and never launched at it, so it was green through the
whole defect. It now factorises an `m = cap` panel and asserts the launch does not throw
— see [armed breaks](#armed-breaks-p4).

### Two plan corrections this probe carries

1. **cdouble at `NB = 32` is present, not impossible.** The plan priced `rA[32]` of
   `Cx<double>` at "128 registers for `rA` alone, so cdouble may need `NB = 16`", and
   `NB = 16` would have forced the blocked driver's `nb` to 16 for that type — which
   collides with the `min_dim >= 32` gate on the wide-scalar complex GEMM. Measured: 176
   registers, no spill, no frame. The type ships at `NB = 32` with a height cap of **256**
   (an earlier draft of this line said 288 — that is the value that aborted; `panel_reg_max_m<cdouble>()`
   is 256, as the table above and `getrf_panel_reg.cc:71-87` both give).
2. **Every type is register-cheaper than the tiny tier's `rA[32]` cell** (64/104/96/176
   against 96/138/143/not-instantiated), because the pivot row travels through local
   memory instead of `NB` sub-group broadcasts. That is what buys heights of 512 for
   float, double and cfloat rather than the plan's 512 / 512 / 384 / 256 guess.

The height caps are `min(512, 128 * floor(512 / regs_budget))` rounded down to a
sub-group — the sub-partition spelling above, which is what `panel_reg_wg_ceiling`
implements (`getrf_panel_reg.cc:71-74`). They are **not** `floor(65536 / regs_budget)`;
that per-block form is the one this section's defect shipped with. `regs_budget` is the
measured count plus 15% headroom rounded **up to ptxas's allocation granularity of 8**.
Under the corrected rule that rounding changes the cap only when it carries the budget
across a `512/k` boundary, and **not one shipped type is changed by it**: double
119 -> 120, cfloat 110 -> 112 and cdouble 202 -> 208 keep their quotient exactly (4, 4 and
2), and float 73 -> 80 drops from 7 to 6 but both are over `kPanelRegMaxRows`, so its cap
is 512 either way. The four caps in the table above are what the *unrounded* budgets would
have given too.
It is still not decoration, because one step is 128 lanes wide: a probe of **148**
registers carries to `148 * 115 / 100 = 170` with headroom, and `128 * floor(512 / 170) =
3 * 128 = 384`; ptxas allocates **176**, and `128 * floor(512 / 176) = 2 * 128 = 256`.
Launch the 384-lane panel on the unrounded figure and a sub-partition holds 3 warps of the
allocated 176 — `3 * 32 * 176 = 16,896` against 16,384, a launch abort — where the
unrounded 170 would have fitted at 16,320. (This line previously illustrated the point with
a 320-lane cap divided into 65,536; 320 is not a multiple of 128, so the corrected rule
cannot produce it, and that example went with the per-block spelling.)

### The register leaf A/B

**2026-09-11, the P4 integration grid.** GPU 1 through `benchmarks/gpu_guard.sh`, no run
exited 5, `benchmarks/factor_bench.cc`, one process per cell, **three arms interleaved in
that one process**: `vendor` (cuBLAS `getrfBatched`), `blocked` (the native blocked driver
on today's local-memory leaf) and `blocked/leaf=reg` (the same driver, register leaf).
7 reps, medians, 1.5 s of warm-up per arm inside the timed loop's arm order, host residual
on items 0 and batch-1 of every timed row. 14 orders x up to 3 batch rungs x 4 types =
**156 paired cells, 468 timed rows, ZERO flagged `bad`** (no row exceeded rel_sd 0.10;
the worst was 0.0206). Raw: `benchmarks/results/p4_leaf_getrf_{float,cfloat,double,cdouble}.csv`.

The leaf axis is not a route, so it could not be pinned like one: `factor_bench` gained an
arm-name suffix `<pin>/leaf=<x>` that scopes `BATCHLAS_GETRF_LEAF` to that arm's reps.
Ratioing two processes through a shared vendor arm was the alternative, and it is exactly
the thing this harness exists to avoid.

`slm/reg` is the leaf ratio (>1 = the register leaf is faster). Rungs are half / nominal /
double of `SAT_LADDER`.

| n | rungs | float slm/reg | cfloat slm/reg | double slm/reg | cdouble slm/reg |
|---|---|---|---|---|---|
| 32 | 4096/8192/16384 | 1.78 / 2.13 / 2.03 | 1.95 / 1.48 / 1.48 | **0.90 / 0.93 / 0.89** | 1.79 / 1.82 / 1.84 |
| 33 | 4096/8192/16384 | 1.63 / 1.70 / 1.68 | 1.24 / 1.26 / 1.28 | 1.70 / 1.71 / 1.72 | 1.28 / 1.28 / 1.29 |
| 48 | 4096/8192/16384 | 1.47 / 1.43 / 1.42 | 1.15 / 1.15 / 1.16 | 1.55 / 1.54 / 1.52 | 1.21 / 1.21 / 1.20 |
| 64 | 4096/8192/16384 | 1.37 / 1.31 / 1.33 | 1.18 / 1.16 / 1.17 | 1.50 / 1.47 / 1.47 | 1.35 / 1.34 / 1.34 |
| 65 | 2048/4096/8192 | 1.80 / 1.78 / 1.73 | 1.31 / 1.31 / 1.31 | 2.10 / 2.12 / 2.13 | 1.47 / 1.46 / 1.46 |
| 96 | 2048/4096/8192 | 1.49 / 1.33 / 1.31 | 1.19 / 1.17 / 1.16 | 1.75 / 1.71 / 1.70 | 1.44 / 1.42 / 1.42 |
| 128 | 2048/4096/8192 | 1.36 / 1.31 / 1.31 | 1.14 / 1.14 / 1.14 | 1.62 / 1.61 / 1.61 | 1.38 / 1.37 / 1.37 |
| 129 | 1024/2048/4096 | 1.48 / 1.32 / 1.30 | 1.17 / 1.14 / 1.14 | 1.71 / 1.63 / 1.65 | 1.30 / 1.29 / 1.29 |
| 192 | 1024/2048/4096 | 1.21 / 1.16 / 1.15 | 1.10 / 1.08 / 1.08 | 1.44 / 1.40 / 1.40 | 1.23 / 1.23 / -- |
| 256 | 1024/2048/4096 | 1.13 / 1.11 / 1.11 | 1.08 / 1.08 / 1.08 | 1.30 / 1.29 / 1.29 | 1.14 / 1.14 / -- |
| 257 | 256/512/1024 | 1.17 / 1.16 / 1.12 | 1.06 / 1.07 / 1.07 | 1.33 / 1.29 / 1.27 | 1.10 / 1.10 / 1.10 |
| 384 | 256/512/1024 | 1.10 / 1.08 / 1.07 | 1.05 / 1.06 / 1.06 | 1.17 / 1.15 / 1.15 | 1.05 / 1.05 / -- |
| 512 | 256/512/1024 | 1.08 / 1.07 / 1.06 | 1.06 / 1.08 / 1.07 | 1.08 / 1.08 / 1.08 | 1.02 / 1.02 / -- |
| 1024 | 256/512 | 1.01 / 1.01 | 1.01 / -- | 1.01 / -- | -- |

**The one losing cell -- three rungs of it -- is the packing the plan asked for and the
kernel does not do.** double `n = 32` reads **0.892 / 0.932 / 0.900** at batch
4096 / 8192 / 16384: the register leaf runs one 32-row panel in a
32-lane work-group, i.e. one sub-group per matrix, which caps at 24 blocks = 24 warps =
50% occupancy, while today's leaf packs `G` matrices into one work-group at `L = 32`.
float, cfloat and cdouble win that cell anyway (1.48-2.13) because they are bound by the
local-memory traffic the register leaf removes; double at `n = 32` is not. The blocked
driver is **only reachable at order 32 by an explicit pin** -- `native_tier_preferred`
sends double `<= 32` to CTA and the vendor serves the rest -- so the default flip does
not ship this cell to anyone. The G-packing remains owed.

**Why the ratio decays with `n`.** The leaf is the whole call at `n = 32`, roughly half of
it at `n = 128`, and a minority at `n >= 384` where the trailing GEMM and the interchange
dominate; at `n = 1024` most panels are taller than the 512-row register cap and fall back
to the local-memory leaf per panel, so that arm is a MIXTURE and reads 1.01. That is the
`n = 1024` row's only meaning: it is the bracketing cell above the cap, not a regression.

**Saturation (R8a).** Only the leaf ratio is flat: it moves less than 5% across its rungs
at 48 of 55 (type, n) ladders, so it is a property of the kernel and not of the
occupancy point. The `vendor/reg` ratio is **not** flat and has no fixed point in batch at
several orders -- float `n = 96` reads 1.65 / 1.08 / 0.98 and float `n = 129` reads
1.77 / 1.04 / 0.91 down the ladder -- which is why no window below 256 is written from
this grid. Where a rung is missing above, the cell was refused for memory (`cdouble`
`n >= 192` at the double rung, `n = 1024` at everything but the half rung); those are
skips, not discards, and are named in the grid log.

**What this does not establish.** (1) Nothing at `m > n`: no tall panel was measured,
though a tall panel is the leaf's whole point and the driver hands it one at every block
step. (2) Nothing about the recursive panel or the right-hand gather -- neither exists.
(3) Nothing at batch < 256 except the two probes in
[which native tier serves 33 to 256](#which-native-tier-serves-33-to-256). (4) No `ncu`
or `nsys` profile, so the 1.3-2.1x is attributed to the local-memory traffic the design
removes by argument, not by counter.

### Which native tier serves 33 to 256

The leaf A/B says the register leaf is the better *leaf*. It does not say the blocked
driver is the better *tier*, and P4's plan asked for the routed window to come down from
256 to 64 -- so the same interleaved harness ran `vendor | cta | blocked/leaf=reg` at the
nominal rung, with a dispatch-coverage readback per arm in its own process
(`benchmarks/results/p4_tier_getrf_float.csv`). The readback earns its place twice over:

| n | batch | vendor ms | cta ms | blocked+reg ms | cta's `resolved_route` | best native |
|---|---:|---:|---:|---:|---|---|
| 33 | 8192 | 0.666 | 0.610 | **0.585** | `native:cta` | blocked, by 1.04 |
| 48 | 8192 | 1.301 | 1.016 | **0.962** | `native:cta` | blocked, by 1.06 |
| 64 | 8192 | 2.647 | **1.628** | 1.840 | `native:cta` | **CTA, by 1.13** |
| 65 | 4096 | 1.558 | 1.856 | **1.144** | `native:cta` | blocked, by 1.62 |
| 96 | 4096 | 3.196 | 3.191 | **2.968** | **`vendor:auto`** | blocked |
| 128 | 4096 | 5.785 | 5.778 | **5.543** | **`vendor:auto`** | blocked |
| 192 | 2048 | 12.620 | 12.497 | **10.037** | **`vendor:auto`** | blocked |
| 256 | 2048 | 31.557 | 23.966 | **21.639** | **`native:blocked`** | blocked |

At `n >= 96` the `cta` pin is REFUSED -- P7 dropped float's advertised CTA ceiling to 77 --
and `automatic()` hands those rows back to the vendor, so that column is a second vendor
arm there. It agrees with the real one to within 0.2%, which is a free control on the
whole harness. At `n = 256` the same pin lands on `native:blocked` instead -- with the
default leaf, which at the time of this grid was still `slm` -- so that row's two native
columns, 23.97 and 21.64, are the leaf ratio (1.108) one more time.

**Two findings, neither of them fixed here.**

1. **float `n = 65` takes the wrong native tier in a vendor-free build.**
   `native_tier_preferred` gives float `cta_max_order = 1 << 30`, so the vendor-free walk
   picks CTA wherever CTA is supported -- and CTA at 65 is **1.62x slower** than the
   blocked driver on the register leaf, and slower than the *vendor* too. The boundary is
   non-monotonic (`n = 64` is CTA's, by 1.13), so the honest repair is
   `cta_max_order = 64` for float, bracketed by the 64 cell itself. It is one line and it
   is not in this package: it changes the vendor-free route and wants its own ladder at
   65..77.
2. **The float window below 256 is a BATCH question, not an order question, and this grid
   was taken at large batch only.** At the nominal rung `vendor / blocked+reg` clears 1.11
   at `n = 33..65` (1.15-1.69). At small batch it does not: float `n = 40`, the order
   `inverse_tests` uses, reads

   | batch | vendor ms | blocked+reg ms | ratio |
   |---:|---:|---:|---:|
   | 2 | 0.0528 | 0.0749 | **0.71** |
   | 32 | 0.0571 | 0.0836 | **0.68** |
   | 256 | 0.0594 | 0.0956 | DISCARDED, rel_sd 0.134 |
   | 1024 | 0.1587 | 0.1306 | 1.22 |

   so a window with no batch term would ship a 1.4x regression to every small-batch caller
   at a 33..65 order. A batch term is available -- the cfloat clause below now carries one
   -- but it needs its own crossover ladder, and one order is not a ladder. **Recorded,
   not shipped.**

### The cfloat window moves to 256

`RouteTable<Op::getrf, complex<float>>::preferred()` gains the band **256..511 at
batch >= 256**; `>= 512` is unchanged and stays batch-free. Two things had to be true and
only one of them was known when the section above was drafted.

**(1) The register leaf is what makes 256 clear the gate.** The floor was 512 because the
clause it was written against admitted cfloat `n = 256` at **0.885**; that cell is the
local-memory leaf.

**(2) The band loses at batch 64..128, and the first probe of it was wrong.** The batch
ladder at `n = 256`, all three arms interleaved in one process, 9 reps, every cell
`rel_sd <= 0.005` (`benchmarks/results/p4_cfloat256_batch_ladder.csv`):

| batch | vendor ms | slm leaf ms | reg leaf ms | vendor/reg | slm/reg |
|---:|---:|---:|---:|---:|---:|
| 32 | 1.186 | 1.242 | 1.098 | 1.081 | 1.131 |
| 64 | 1.192 | 1.337 | 1.200 | **0.993** | 1.114 |
| 128 | 1.431 | 1.656 | 1.552 | **0.922** | 1.067 |
| 192 | 2.656 | 2.663 | 2.340 | 1.135 | 1.138 |
| 256 | 3.860 | 3.288 | 3.003 | 1.285 | 1.095 |
| 512 | 10.461 | 7.397 | 6.754 | 1.549 | 1.095 |
| 1024 | 20.202 | 17.126 | 15.758 | 1.282 | 1.087 |
| 2048 | 39.640 | 36.089 | 33.365 | 1.188 | 1.082 |
| 4096 | 78.000 | 73.705 | 68.073 | 1.146 | 1.083 |

The 4096 row is the grid's; every other row is the ladder's, and the 1024 and 2048 rows
reproduce the grid (20.459 / 17.133 / 15.799 and 39.831 / 36.131 / 33.412) to within 1.1%
across a default flip and three rebuilds. **Two cells were DISCARDED and re-run**: the
first readings of batch 1024 and 2048 were taken while a full `ctest` run had GPU 1, and
came back 12% slow on ALL THREE arms (22.811 / 19.155 / 17.479). `gpu_guard.sh` would have
said so -- its foreign-process check is exactly this case -- but the call had its stderr
redirected to /dev/null, so the warning was swallowed. The ratios were barely moved
(1.305 against 1.282), which is what makes contamination like this quotable if nobody
checks.

cuBLAS's fused kernel is **flat** from batch 32 to 64 (1.186 -> 1.192) and nearly flat to
128, while the blocked driver's four kernels per block step are not; that is the whole
dip. It closes by batch 192 (1.135) and the gate is cleared from 256 up. So the clause is
`order >= 512 || (order >= 256 && batch >= 256)`, with 128 (0.922, a LOSS) as the
bracketing non-winner on the batch axis and 192 as the first winner — 256 is the
conservative side of a crossover that sits between them. `route_getrs.hh` (`batch < 128`)
and `route_gemm.hh` (`batch < 64`) are the precedent for a batch term in `preferred()`.

**A measurement trap this cost an hour to.** The first probe of batch 128/256 read the two
leaves as *identical* (3.036 vs 3.016 ms) and was quoted here as a win. It was taken after
the default flip with `--arms=vendor,blocked,blocked/leaf=reg`, and a plain `blocked` arm
takes the DEFAULT leaf — which is now `reg`. It was the register leaf against itself. The
ladder above re-measures with both leaves spelled out, and the b1024 row reproduces the
pre-flip grid (20.459 / 17.133 / 15.799) to within 1%.

**The bracketing non-winner below on the ORDER axis is `n = 192`**: 1.10 / 0.92 / 0.86 down
its ladder, under the bar at the unsaturated rung and a loss at the other two. `n = 257`
(1.34-1.53) says 256 is not a knife edge, and `n = 384` and `512` (1.50-1.80) are the band
that was already routed.

**End-to-end, automatic route, nothing pinned** (`benchmarks/results/p4_e2e_getrf_cfloat.csv`
and `p4_e2e_getrf_float.csv`), with a coverage readback per cell confirming which arm
`automatic()` took. These were run before the batch term was added, so every batch here is
>= 512 and inside the final clause:

| type | n | batch | vendor ms | auto ms | auto's route | ratio |
|---|---:|---:|---:|---:|---|---:|
| cfloat | 192 | 2048 | 15.442 | 15.406 | `vendor:auto` | 1.002 (control: the same arm twice) |
| cfloat | 256 | 1024 | 20.379 | 15.734 | `native:blocked` | 1.295 |
| cfloat | 256 | 2048 | 39.777 | 33.395 | `native:blocked` | 1.191 |
| cfloat | 256 | 4096 | 78.120 | 68.042 | `native:blocked` | 1.148 |
| cfloat | 257 | 512 | 11.853 | 7.796 | `native:blocked` | 1.520 |
| cfloat | 512 | 512 | 76.995 | 42.661 | `native:blocked` | 1.805 |
| float | 192 | 2048 | 12.613 | 12.561 | `vendor:auto` | 1.004 (control) |
| float | 256 | 2048 | 31.711 | 21.535 | `native:blocked` | 1.472 |
| float | 512 | 512 | 49.558 | 26.185 | `native:blocked` | 1.893 |

The float rows are the control the leaf flip needs: that window did not move, but its arm
did, and it improved — 1.327 -> 1.472 at `n = 256` b2048 and 1.773 -> 1.893 at `n = 512`
b512 against the same vendor.

**What is NOT established.** float's own 256 floor was never re-probed at batch 64..128
with the register leaf; the old record has it at 1.2626 at b128 with the local-memory leaf,
and the leaf only made that arm faster, so the direction is safe — but it is an inference,
not a measurement. cfloat `n >= 512` was not re-probed at small batch either.

**R8b.** `preferred()` still answers for exactly one native tier (`if (r.algo !=
Algorithm::Blocked) return false;`), so the pre-emption defect this campaign shipped once
cannot recur through this clause.

### Armed breaks (P4)

Seven were predicted by the implementation agent for the leaf kernel and three more were
planted for the guards added at integration. Each was applied, built, observed, restored.

| break | expected | observed |
|---|---|---|
| (a) drop `rowid >= j` from the argmax candidate mask | an eliminated row wins; `RegPanelAgreesWithTheLocalMemoryLeaf` reports a differing `ipiv` | exactly that: cdouble m=33 ncols=31, `ipiv[11] = 11` where the local-memory leaf chose 18, and the 1-based range assertion fires with it |
| (b) relabel `rowid = j` in both arms | two items claim logical row `j`; residual explodes | red, 15 cases: residual 6.58 against tol 1.47e-12, pivot-ratio oracle 6.55 > 1 |
| (c) publish the pivot row BEFORE the relabel | non-deterministic residual | **GREEN -- the prediction was wrong.** The publisher is the SAME physical item in both orders and `act` is computed after the relabel either way, so moving the publish up is a no-op refactor |
| (c') publish AFTER the relabel but select on the pre-swap identity `rowid == p` | (what break (c) was reaching for) | red: pivot ratio 1.56e17, and `info = 6` where 0 is required |
| (d) drop barrier B2 | the scale reads an unfinished `sx`; red at the TALL cells first (m >= 256) | red, 16 cases -- but the prediction's *direction* was wrong: it is red from the SHORTEST two-sub-group panel, float m=33 ncols=1. Any `m > 32` puts publisher and reader in different sub-groups |
| (e) write `piv_item[j] = p + 1`, dropping `piv_base` | `RegPanelIpivAndInfoAreGlobalAtANonZeroPivBase` red, `piv_base == 0` cells green | exactly that, and `RegPanelAgreesWithTheLocalMemoryLeaf` (all `piv_base = 0`) stayed green |
| (f) `info_local = 0` instead of reading `*info_item` | `RegPanelPlantedZeroColumn...` red on the second planted column only | exactly that: "the leaf overwrote a non-zero info, so first-failure-wins does not hold across the blocked driver's panels" |
| (g) store to `a[tid + k*ld]` instead of `a[rowid + k*ld]` | the lazy swap never lands; residual red everywhere that pivots | red on 4 tests x 4 types. It also moved float's register count 64 -> 131 and turned the new launch-at-the-cap assertion red with the RUNTIME's per-block message -- an unplanned second reading of that guard |
| (h) height cap back to `regs x wg <= 65536` | the cdouble cap returns to 288 and `RegPanelCapacityIsOneSpelling` must fail AT THE LAUNCH | red: `UR_RESULT_ERROR_OUT_OF_RESOURCES`, "the panel at the ADVERTISED cap m=288 ... does not launch". `RegPanelAgreesWithTheLocalMemoryLeaf` red too -- this is the defect exactly as it shipped |
| (i) panel-leaf default back to `Slm` | the unset-environment block of `BlockedDriverTakesTheRegisterLeafUnderTheKnob` red | red on all four types: `getrf_blocked_debug_leaf` = 1 where 3 is required, at n = 64, 128 and 256 |
| (j) cfloat window floor back to 512 | the new 256 / 192 bracket red for cfloat only | red: `RouteTableAndTheVendorFreeFallback` cfloat only in `getrf_tests`, and 1 of 118 in `route_vocabulary_tests` ("cfloat order 256", "cfloat order 257") |
| (k) drop the `batch >= 256` term from the cfloat clause | the batch side of both brackets red, the order side green | exactly that: `getrf_tests` red on cfloat only ("cfloat n=256 batch=2 is OUTSIDE it"), 179 of 180 still passing, and `route_vocabulary_tests` red on `PreferredIsTheMeasuredOrderWindowPerTypeAndBlockedOnly` at cfloat orders 256/257/511 for batches 1, 2 and 128 -- and silent at batch 8192, which is the side that would have passed vacuously |

### The P7 occupancy change moved 27 committed `getrf` baseline cells

**Confirmed by arithmetic, no GPU.** Recomputing `getrf_slm_bytes` /
`resident::resident_max_n` by hand reproduces the advertised ceilings **77 / 54 / 54 / 38**
(float/double/cfloat/cdouble) at this box's 97,280 B budget and **155 / 109 / 109 / 77** at
`min_blocks_per_sm = 1`. The committed P0 baselines
(`benchmarks/results/factor_baseline_getrf_*.csv`, taken **before** P7) record
`resolved_route` = `native:cta` up to the *pre-P7* ceilings:

| type | last order recorded as `native:cta` | advertised ceiling today | orders whose native arm moved |
|---|---|---|---|
| float | 129 | 77 | 96, 128, 129 |
| double | 32 (`cta_max_order` caps it) | 54 | none |
| cfloat | 96 | 54 | 64, 65, 96 |
| cdouble | 65 | 38 | 48, 64, 65 |

Three orders x three batches x three types = **27 paired cells** whose `native` arm no
longer resolves as the CSV's `resolved_route` column records, under a `native` pin or a
vendor-free build. And [the tier table](#native_tier_preferred) says blocked is *slower*
than CTA at every one of them — `blocked_ms / cta_ms` of 1.49 (float n=96 b8192), 1.13
(float n=128 b4096), 1.39 / 1.30 (cfloat n=64 / 96), 1.37 (cdouble n=64). `potrf`
recorded its equivalent cost explicitly; this page had no equivalent entry and now does.

This is **not** fixed here — the fix is a routing change and belongs with a measured
grid — but it is the band P4 is measured in, so the baseline CSVs must be **re-taken**
before any P4 A/B is scored against them. Scoring P4 against a stale `native` column would
credit or debit the register leaf for a tier flip that happened in P7.

## The fused gesv tier

`src/extensions/gesv_tiny.cc`, one launch for `A X = B` at order `n <= 32` and
`nrhs <= 4`: `getrf_tiny.cc`'s elimination with the RHS carried in `D rB[NR]`
alongside `D rA[N]`, forward substitution fused into the elimination loop, back
substitution in the same kernel. **It is not routed.** `preferred()` is all-false
and `native_tier_preferred` answers false for `Tiny`, so `Auto` takes the composed
`getrf; getrs` arm at every shape; the tier is reachable only through
`BATCHLAS_GESV_ROUTE=native:tiny` or the dispatch entry point directly.

Two structural facts separate this op from every other one in this file.

**There is no batched vendor `gesv` on any backend.** Neither cuBLAS, cuSOLVER nor
rocSOLVER ships one, so `route_gesv.hh`'s order array carries no vendor entry and
`resolve_gesv_route` passes `vendor_available=false` unconditionally. The
consequence for the tier hook is the one R8b warns about, inverted: because every
walk is a vendor-free walk, `native_tier_preferred` is consulted on every call, so
answering `true` for `Tiny` there would *ship* it. That is why it answers false.

**B is permuted for free.** Lane `lane` loads row `lane` of B and its `rowid` moves
with A's under the lazy relabel, so the lane holding row `r` of `P A` also holds row
`r` of `P b`. There is no `laswp` in this tier and none is needed. The solution
vector is *not* permuted — `x_i` is the i-th unknown — so the RHS store writes at
`rowid`, exactly as the factor store does.

### P2: the window this tier expects

**Arithmetic on committed CSVs, not a measurement.** No GPU time was spent on P2.
The bound below is
`(t_vendor_getrf + t_vendor_getrs) / t_tiny_getrf`, an *upper* bound on what the
fused kernel can be worth against the vendor composition, because the fused kernel
does strictly more work than `getrf_tiny` alone: it also loads B, runs two
substitutions, stores X, and holds `2 * NR` more registers per lane.

Sources: `benchmarks/results/p1_tiny_getrf_wg64.csv` (its own `vendor` and `tiny`
arms, interleaved in one process) for the getrf halves, and
`benchmarks/results/factor_baseline_getrs_<T>.csv` (`vendor` arm, `nrhs = 1`) for
the getrs half. **This splices two sweeps**, so it is an order-of-magnitude bound
and not a ratio: the two sweeps' vendor `getrf` medians agree to within 2% at every
shared cell, which is the only reason the splice is defensible at all.

| T | n = 4 | n = 8 | n = 16 | n = 32 |
|---|---:|---:|---:|---:|
| float | 3.97 | 5.88 | 2.87 | **1.79** |
| cfloat | 3.79 | 3.38 | 4.86 | **0.99** |
| double | 2.00 | 2.77 | 1.33 | 1.53 |
| cdouble | 1.40 | 1.84 | 3.01 | — (capped at 16) |

Batch 16384, `nrhs = 1`. At batch 4096 the two `n = 32` gating cells read 2.43
(float) and 1.16 (cfloat).

**Three things follow, and the first kills a target.**

1. **P2's headline target — ">= 3x vs `getrf + getrs` vendor at n <= 32, nrhs = 1,
   float and cfloat" — is arithmetically unreachable at n = 32 for both gating
   types, and at n = 16 for float.** The ceiling is 1.79 (float) and 0.99 (cfloat).
   No kernel can clear 3x there, whatever it does. The target must be re-scoped to
   `n <= 16`, where the ceiling is 2.87-5.88 (float) and 3.38-4.86 (cfloat).
2. **cfloat n = 32 is a hard kill, with no measurement needed.** `getrf_tiny` alone
   already costs *more* than the vendor's `getrf + getrs` together there (bound
   0.99), and it loses 0.64-0.71x against cuBLAS `getrf` on its own
   ([first timing](#first-timing-the-tiny-tier-against-cublas-and-the-cta-tier)).
   Any P2 window must **exclude cfloat at n = 32**; that cell is the bracketing
   non-winner R8 asks for at the upper edge, and it is already in hand.
3. The plan's stated kill criterion — "< 1.15x against the two-launch native arm at
   n = 32, nrhs = 1" — points at the one cell where fusion is worth least. The band
   it should gate is `n <= 16`.

**The window to propose to the measurement phase**, therefore, is:
`float` and `cfloat` at `order <= 16`, `nrhs <= 4`, batch at saturation — with
`n = 32` measured as the bracketing non-winner at both types and expected to fail
the 1.11 flip gate. `double` and `cdouble` are measured and reported, not gated
(R10); note that `cdouble n = 16` reads 3.01 here and is the one FP64 cell that
looks like a real win, consistent with the same anomaly the P1 grid found.

**What this bound does NOT settle.** It compares against the *vendor composition*.
The other arm P2 owes a number against is the two-launch *native* one
(`getrf_tiny` then fused `getrs`), and that comparison is now differently shaped
than the plan assumed: the fused `getrs` tier has carried an
[order floor of 32](#getrs-order-floor-evidence) since P0, so below that order the
two-launch native arm is `getrf_tiny` + the composed `getrs`, not
`getrf_tiny` + fused `getrs`. The "what fusion itself buys" arm has to be built
before it can be read.

### P2: the measured gesv window

**This supersedes the arithmetic bound above as the routing evidence.** The bound
was an upper bound on `tiny` vs the *vendor composition*; the flip was decided on a
measured, interleaved A/B against the arm `Auto` actually takes today.

**What shipped.** `route_gesv.hh::tiny_window_max_n()` = 32 for `float`, 16 for
`std::complex<float>`, 0 (no window) for `double` and `cdouble`. The predicate lives
in `native_tier_preferred`, **not** in `preferred()`: `resolve_gesv_route` always
passes `vendor_available = false`, so `automatic()` never reaches `preferred()` for
this op and the vendor-free walk (`supports && native_tier_preferred`, first hit
wins) is the shipping path. `preferred()` stays all-false and `gesv_tests.cc`
asserts that permanently. The fit is resolved before the hook is consulted, which is
the property `best_native_tier` exists to give an op whose window sits in
`preferred()`; composing it here would add nothing.

**The three arms, and why there are three.** `gesv` has no vendor entry in its order
array, so "the vendor arm" is not a routing option — it is the composed `Blocked`
arm with its legs pinned. `benchmarks/factor_bench.cc::composed_pins` makes the
three explicit and `run_solve_grid.sh` interleaves them in one process:

| arm | `BATCHLAS_GESV_ROUTE` | legs |
|---|---|---|
| `tiny` | `tiny` | — (one fused kernel) |
| `blocked` | `blocked` | `getrf`, `getrs` under `Auto` — **the incumbent, what a user gets today** |
| `vendor` | `blocked` | `getrf`=vendor, `getrs`=vendor |
| `composed` | `blocked` | `getrf`=tiny, `getrs`=cta (the two-launch native arm; named `native` until 2026-09-26) |
| `native` | `native` | `getrf`=native, `getrs`=native -- the SHIPPED native walk, what benchviz plots |

The flip is gated on `tiny` vs **`blocked`**, because that is the arm it replaces.

#### gesv float — `t_blocked / t_tiny`, ld = n + 1, 7 reps, interleaved

Raw: `benchmarks/results/p2_gesv_float4.csv`. `!` marks a cell where some arm's
`rel_sd` exceeded 0.10 and the cell was re-run (below).

| n | nrhs | b=8192 | b=16384 | b=32768 |
|---:|---:|---:|---:|---:|
| 4 | 1 | 3.55 | 3.02! | 2.23 |
| 4 | 4 | 3.20 | 2.49 | 1.95 |
| 8 | 1 | 3.16 | 2.81 | 2.65 |
| 8 | 4 | 2.63 | 2.19 | 1.97 |
| 9 | 1 | 2.73 | 2.36 | 2.18 |
| 9 | 4 | 3.93 | 3.82 | 3.79 |
| 16 | 1 | 2.35 | 2.19 | 2.07 |
| 16 | 4 | 3.63 | 3.65 | 3.65 |
| 17 | 1 | 1.25 | 1.26 | **1.15** |
| 17 | 4 | 1.62 | 1.52 | 1.51 |
| 24 | 1 | 1.29 | 1.29 | 1.27 |
| 24 | 4 | 1.69 | 1.53 | 1.63 |
| 32 | 1 | 1.41 | 1.27 | 1.31 |
| 32 | 4 | 1.55 | 1.36 | 1.52 |

Every cell clears the 1.11 flip gate. The weakest, n = 17 nrhs = 1 at batch 32768
(1.15), was extended one doubling further and **rises** to 1.30 at 65536
(`p2_satladder_gesv_float_n17.csv`), so the window is not gated on a falling edge.

#### gesv cfloat — the cliff at n = 17

Raw: `benchmarks/results/p2_gesv_cfloat4.csv`.

| n | nrhs | b=8192 | b=16384 | b=32768 |
|---:|---:|---:|---:|---:|
| 4 | 1 | 2.88 | 2.13 | 1.71! |
| 4 | 4 | 2.16 | 1.54 | 1.23 |
| 8 | 1 | 2.52 | 2.12 | 1.76 |
| 8 | 4 | 1.85 | 1.48 | 1.29 |
| 9 | 1 | 1.73 | 1.59 | 1.47 |
| 9 | 4 | 1.95 | 1.68 | 1.57 |
| 16 | 1 | 2.68 | 2.85 | 2.52 |
| 16 | 4 | 1.93 | 1.83 | 1.98 |
| **17** | **1** | **0.84** | **0.66** | **0.68** |
| **17** | **4** | **0.50** | **0.45** | **0.51** |
| 24 | 1 | 0.76 | 0.79 | 0.83 |
| 24 | 4 | 0.48 | 0.56 | 0.63 |
| 32 | 1 | 0.82 | 0.90 | 0.93 |
| 32 | 4 | 0.60 | 0.72 | 0.76 |

**n = 17 is the bracketing non-winner R8 asks for, measured at both `nrhs` and all
three batches, 0.45-0.84x.** The mechanism is the instantiation ladder, not the
order: n = 17 is the first order served by the `N = 32` kernel, and for a complex
scalar `rA[32] + rB[4] + rX[4]` costs more occupancy than the second launch it
saves. n = 16 (the last `N = 16` order) wins 1.83-2.85x immediately below it.

The cheap-kill probe predicted exactly this for cfloat n = 32 (bound 0.99); measured
0.90-0.93. Its conclusion — "re-scope to n <= 16" — was right for cfloat and **wrong
for float**, which wins 1.15-1.55x over the whole 17..32 band.

#### The bracket at the top edge is structural, not measured

For `float` the window's upper edge is 32, which is also the tier's instantiation
ceiling, so there is no reachable cell above it to measure: at n = 33 `supports()` is
false and the composed arm answers. The route readback below records that, and
`gesv_tests.cc::AutoTakesTheMeasuredWindow` pins it. **This is stated rather than
measured, and it is the one window edge in P2 with no measured non-winner beside it.**

#### The cfloat reversal at batch 131072 — a NAMED open risk

R8a. Pushed four doublings past the `SAT_LADDER` nominal (16384), two cfloat shapes
inside the shipped window **reverse**:

| n | nrhs | 8192 | 32768 | 65536 | 131072 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | — | — | — | 1.35 |
| 2 | 4 | — | — | — | **0.92** |
| 4 | 1 | 2.88 | 1.71 | 1.39 | **1.08** |
| 4 | 4 | 2.16 | 1.23 | 1.10 | **0.85** |
| 6 | 4 | — | — | — | 1.48 |
| 7 | 4 | — | — | — | 1.99 |
| 8 | 1 | 2.52 | 1.76 | 1.68 | 2.10 |
| 8 | 4 | 1.85 | 1.29 | 1.33 | 2.34 |

Raw: `p2_satladder_gesv_cfloat_n4.csv`, `p2_satladder_gesv_cfloat_bigbatch.csv`,
`p2_gesv_cfloat_loweredge.csv`. The n = 4 reversal was measured twice in separate
processes (0.91 and 0.85) and reproduces.

**The window was NOT narrowed for it, deliberately.** The losers are n = 2 and n = 4
at `nrhs = 4` while n = 1, 6, 7 and 8 all win at the same batch — the reversal is
**non-monotonic in n, so it is not an order boundary** and no predicate this route
table can express would capture it without overfitting to this grid. Against that,
the window wins 1.15-2.88x over every cell of the 8192-32768 band that
`docs/perf/lu.md`'s `SAT_LADDER` names for this order. The judgement recorded here is
that two shapes at 0.85-0.92x, eight times the nominal batch, do not pay for giving
up the rest. **If a caller runs cfloat gesv at n <= 4 with batch >= 65536, this is
the cell to re-measure first.**

#### P2: double and cdouble

gesv double and cdouble get no window.

Raw: `p2_gesv_double.csv`, `p2_gesv_cdouble.csv`, `nrhs` 1 and 4, batches 16384 and
32768, `t_blocked / t_tiny`:

| n | double nrhs=1 | double nrhs=4 | cdouble nrhs=1 | cdouble nrhs=4 |
|---:|---:|---:|---:|---:|
| 4 | 1.06 / 0.78 | 0.53 / 0.37 | 0.56 / 0.47 | 0.31 / 0.27 |
| 8 | 1.55 / 1.38 | 0.87 / 0.76 | 0.88 / 0.82 | 0.63 / 0.64 |
| 16 | 0.90 / 0.92 | 0.87 / 0.86 | 1.71 / 1.77 | 1.15 / 1.16 |
| 32 | 0.66 / 0.64 | 0.68 / 0.68 | — (capped at 16) | — |

Neither type has a contiguous winning band. `double` wins only at n = 8 nrhs = 1;
`cdouble` only at n = 16, which is its instantiation cap, so the "window" would be a
single order with no measured neighbour above it inside the tier. **Both are left on
the composed arm.** This is the section D3 anticipated and the R10 posture: measured,
reported, not routed.

#### Route readback after the flip

`benchmarks/results/p2_route_readback.txt`, from the dispatch coverage instrument
under `--arms=auto` (a `reached` row, not a `linked` one):

```
gesv   float    n=32  -> native:tiny      gesv   float    n=33  -> native:blocked
gesv   cfloat   n=16  -> native:tiny      gesv   cfloat   n=17  -> native:blocked
gesv   double   n=16  -> native:blocked   gesv   cdouble  n=16  -> native:blocked
```

#### The kill criterion did not fire

The plan's kill is "fused beats the two-launch native arm by < 1.15x at n = 32,
nrhs = 1". Measured against the `native` arm (`getrf`=tiny, `getrs`=cta), float
n = 32 nrhs = 1 reads **1.25 / 1.19 / 1.20** at batches 8192 / 16384 / 32768. The
kernel ships. (cfloat at n = 32 reads 1.20-1.29 against that arm but 0.90-0.93
against the incumbent, which is why the *routing* decision cannot be taken against
the native arm alone.)

#### Cells discarded for rel_sd, by name, and their re-runs

R8 discards any cell whose `rel_sd` exceeds 0.10. Four did, across ~340 rows:

| file | cell | arm | rel_sd | re-run at 15 reps |
|---|---|---|---:|---|
| `p2_gesv_float4.csv` | n=4 nrhs=1 b=16384 | blocked | 0.120 | 2.86, `p2_rerun_gesv_f_4_1_16384.csv` |
| `p2_gesv_cfloat4.csv` | n=4 nrhs=1 b=32768 | blocked | 0.128 | 1.64, `p2_rerun_gesv_cf.csv` |
| `p2_gesv_cfloat4.csv` | n=16 nrhs=4 b=8192 | vendor | 0.224 | 1.94, `p2_rerun_gesv_cf.csv` |
| `p2_posv_float.csv` | n=4 nrhs=4 b=8192 | tiny | 0.171 | 9.07, `p2_rerun_posv_f_4_4_8192.csv` |
| `p2_posv_float.csv` | n=16 nrhs=4 b=16384 | tiny | 0.207 | 6.58, `p2_rerun_posv_f_16_4_16384.csv` |

Every re-run came back inside the gate and agreed in direction with its neighbours.
No conclusion here rests on a discarded cell.

#### What P2's integration could NOT establish

- **The register probe was not run.** `kWorstRegsPerThread` in `gesv_tiny.cc` and
  `posv_tiny.cc` is still the assumed number the section below describes. The tier
  ships on timing and correctness; the frame-0/spill-0 gate is still owed.
- **`linalg::solve` was not rewired.** The plan's framing was "fused factor+solve
  behind `linalg::solve`", but P2 shipped `linalg::solve_spd` (to `posv`) and left
  `linalg::solve` on `getrf; getrs`. **The end-to-end caller A/B the plan asks for
  therefore does not exist for `gesv`**: the public `gesv` op gets the window, its
  in-tree callers do not.
- **No `nrhs = 2` or `3` cell was timed.** Both run the `NR = 4` instantiation and
  are correctness-tested, but the grid measured `nrhs` 1 and 4 only.
- **Saturation is partial.** Several cells were still moving more than 5% on the
  last doubling; every ratio quoted above names its batch, per R8a.

### P2: the register bound is assumed, not probed

`gesv_tiny.cc` carries `kWorstRegsPerThread = 224` and `posv_tiny.cc` carries 256,
each in a `static_assert` against the 65,536-register per-block file at
`kTinyWgSize = 64`. **Neither number was measured.** They are `getrf_tiny`'s 143 and
`potrf_tiny`'s 176 plus generous room for the `2 * NR` scalars this tier adds, chosen
so the assert cannot fire spuriously while the real gate — the probe's *stack frame*
column — is still owed. `scripts/register_probe.sh` on `batchlas_extensions_cta` must
report frame 0 and spill 0 for every `GesvTinyKernel` and `PosvTinyKernel`
instantiation before either tier is timed; a non-zero frame means an unrolled loop
was declined and the arrays left the register file, which shows up as a slowdown and
never as a wrong answer.

Instantiation count, for R7: 2 kernels x 3 N x 2 NR x 4 types = 48, less the two
`cdouble` N = 32 arms per kernel = **44**. The device-link delta of
`batchlas_extensions_cta` is likewise owed and not taken.

# QR: geqrf, orgqr, ormqr, and the third routing predicate (WP5)

Native SYCL `geqrf` (two tiers) and `orgqr` (one tier), the `ormqr` they are built on, and the routing that selects between them.

All timings: GPU 1 of a 2x RTX 4090 box (sm_89, 128 SMs), `CUDA_VISIBLE_DEVICES=1`, `WARM_S=1.5`, medians of interleaved A/B, cells with relative sd > 10% discarded, nothing timed under `BATCHLAS_KERNEL_TRACE`. "Vendor-free" always means the **build** (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`), never an env var inside a build that still links cuSOLVER: `route_resolve.hh:76-80` falls through to `automatic()` when a forced route is unsupported and `automatic()` returns `{Vendor, Auto}` (:129), so a forced-route A/B inside one build can silently be vendor-vs-vendor. `local_mem_size` here is 101,376 B; every capacity below derives from that minus the standard 4,096 B reserve, i.e. a 97,280 B budget. The generated `device_limits.hh`'s 49,152 is hardcoded for any `nvidia_gpu_sm_*` architecture with no device query at all (`cmake/BatchLASDetectSYCL.cmake:44-45`) and is 2.06x wrong on this box; nothing in this family reads it — `geqrf_cta.cc:33`'s `kGeqrfReferenceSlmBudget = 97280` answers only the "at this repository's reference budget" convenience overloads, and every real decision reads the device through `geqrf_route.hh:51-53`.

## What ships

### Route arms

| op | arms, in `order` sequence | `preferred()` |
|---|---|---|
| `geqrf` | `{Native, Tiny}`, `{Native, CTA}`, `{Native, Blocked}`, `{Vendor, Auto}` (`route_geqrf.hh:kGeqrfOrder`) | **`Tiny` is false, and `native_tier_preferred(Tiny)` is false too** — see [the tiny tier](#the-tiny-tier-wp6--p1-square-n--32-in-registers). Otherwise native above a per-type **order floor** — `float` 64, `cfloat` 48, `double` 96, `cdouble` 256 — plus a **tall-panel clause**, `rows >= 128 && cols >= 32 && rows >= tall_aspect*cols`, where `tall_aspect` is **4 for the 32-bit types and 8 for the 64-bit ones** (`route_geqrf.hh:preferred`) -- the split is load-bearing, see [tall panels cross over earlier](#tall-panels-cross-over-earlier) |
| `orgqr` | `{Native, Blocked}`, `{Vendor, Auto}` (`route_orgqr.hh:21-24`) | native at `rows <= 512 && cols <= 512`, every type (`route_orgqr.hh:preferred`) |
| `ormqr` | `{Native, Blocked}`, `{Vendor, Auto}` (`route_ormqr.hh:45-48`) | `is_native(r) && supports(r, s)` (`route_ormqr.hh:77-79`) |

`geqrf` and `orgqr` no longer ship route-neutral. A vendor-present build now takes the native arm inside the windows above; outside them — `geqrf` below its floor and off the tall clause, `orgqr` above n = 512 — it still takes cuSOLVER, and the kernels are then reachable only from a vendor-free build (`route_resolve.hh:38-49`), from `BATCHLAS_GEQRF_ROUTE` / `BATCHLAS_ORGQR_ROUTE`, or from the direct entry points `geqrf_cta_dispatch` / `geqrf_blocked_dispatch` / `orgqr_blocked_dispatch`. The windows are the cells that clear the repository's flip gate on the n = 4..512 grid, bracketed on both sides; the grid, including every excluded cell, is [`small-n-baseline.md`](small-n-baseline.md#geqrf). The 3.24x (`geqrf`) and 7.85x (`orgqr`) geomeans below span the whole grid and so are **still not** what the default build realises — only the in-window part of them is.

`ormqr` is the exception: `preferred()` is native-first, so a supported blocked `ormqr` runs natively in every build. That predates WP5 (no shape ever sent a supported blocked `ormqr` to the vendor) and is why `orgqr`'s native arm — an identity fill plus a routed `ormqr` — works at all.

`supports()` for both new tables is correctness-only. Gates that matter: `m >= n` for both `geqrf` native arms (handed a wide view the trailing update walks past the bottom of the panel, `route_geqrf.hh:46-48`); `n <= m` for `orgqr` (Q's columns live in C^m, `route_orgqr.hh:40-41`); GPU-only for both; and heterogeneous batch for both, refused because nothing in this tree gets heterogeneous-batch QR right (netlib included — `netlib_lapack.cc:1430-1443` hoists m and n out of its loop, and its `orgqr` at :1472-1477 hoists m, n and k).

**The sub-group gate is `geqrf`'s alone, and `orgqr` deliberately has none.** `geqrf` tests sub-group size 32 **enumerated** from `sycl::info::device::sub_group_sizes` (`route_geqrf.hh:25-27`, `queue-impl.cc:339-345`), never inferred from `get_property(MAX_SUB_GROUP_SIZE)` — that returns `sub_group_sizes()[0]`, so the weak test refuses a `{8,16,32}` device and accepts a `{64}` one, a launch abort for a kernel carrying `[[sycl::reqd_sub_group_size(32)]]`. `orgqr`'s shape builder sets no `has_sg32` and no SLM capacity at all (`orgqr_route.hh:76-84`), because `ormqr_blocked` carries no `reqd_sub_group_size` and holds nothing resident: a sub-group field there would be a decorative input. Do not read the `geqrf` gate list as covering both tables.

### CTA capacity

`geqrf`'s CTA tier holds the whole `m x n` panel in a `local_accessor`, so its ceiling is an **area**, `m*n <= cta_max_elems` in int64 (`route_geqrf.hh:59-61`). `cta_max_m` is also tested but is not independently binding with the shipped layout — there is one tile and no per-row resident array, so the largest admissible m at n = 1 *is* the area bound (`geqrf_cta.cc:171-310`). It is kept as a separate number because it is what moves if a per-row array (a staged `v`, a norm cache) is ever added. On this box: float 24,320 elems (square n = 155), double and cfloat 12,160 (n = 110), cdouble 6,080 (n = 77).

Where the budget itself comes from — this box's 101,376 B `local_mem_size`, the 4,096 B reserve, and why `geqrf_cta.cc:33`'s `kGeqrfReferenceSlmBudget = 97280` serves the convenience overloads only while `device_limits.hh`'s hardcoded 49,152 is 2.06x wrong here — is in the preamble at the top of this page.

Above the area bound the blocked driver serves the shape. Its panel leaf is the same device body (`geqrf_cta_device.hh`) instantiated against a global pointer instead of a `local_accessor`, chosen per panel by the same `geqrf_cta_fits` predicate the route table's capacity uses — so the ceiling the table advertises and the allocation the launcher makes cannot disagree.

### The third predicate

`preferred()` cannot express "which of two **native** tiers". It is consulted by the loop above the vendor-free walk and runs regardless of `vendor_available`, so a window written to fix the vendor-free tier choice also moves vendor-**present** traffic, including where cuSOLVER beats both natives. WP5 added an optional third predicate, `RouteTable::native_tier_preferred`, detected with a `requires` expression and defaulting to `true` so every table that does not declare it keeps its old answer (`route_resolve.hh:18-25`). It is consulted **only** on the vendor-free walk, which is now two passes (`route_resolve.hh:38-49`); `gemm`, `trsm`, `potrf` and `gesvd` were untouched by construction at WP5 and the route diff confirmed it. Since then WP6 has declared the hook for `getrf` (`route_getrf.hh:78`), `getrs` and — as of P7 — `potrf` ([potrf.md#native_tier_preferred](potrf.md#native_tier_preferred)); the debt recorded below is closed.

The shipped `geqrf` window, verbatim (`route_geqrf.hh:79-103`):

```c++
static bool native_tier_preferred(Route r, const GeqrfShape& s) {
    if (!is_native(r)) return true;
    const int64_t cta_max_cols = [] () -> int64_t {
        if constexpr (std::is_same_v<T, float>)       return 96;      // 1.294 at 96, 0.821 at 112
        else if constexpr (std::is_same_v<T, double>) return 48;      // 1.049 at 32, 0.983 at 48 (a tie), 0.922 at 64
        else                                          return 1 << 30;
    }();
    switch (r.algo) {
        case Algorithm::CTA:     return s.cols() <= cta_max_cols;
        case Algorithm::Blocked: return s.cols() >  cta_max_cols;
        default:                 return true;
    }
}
```

Both complex types get `1 << 30`, i.e. CTA wherever the area gate admits it. `orgqr` has one native arm and declares no such hook.

**The notes and the code disagree, and the code wins.** `experiments/wp5_qr/README.md` §3a and `VENDOR_INDEPENDENCE_PLAN.md` record the crossover as "blocked ahead from n ~= 104 (float) and n ~= 48 (double)". The shipped predicate is **n <= 96 float, n <= 48 double**: it keeps the last measured CTA-ahead cell on CTA rather than interpolating, and at double n = 48 it resolves an in-resolution tie (0.983, blocked ahead 1.7%) **in CTA's favour** — the opposite direction from the note. The reason is not timing: CTA's workspace is **zero** (the tile is local memory, `tau` is the caller's span) while the blocked driver allocates `m*nb*batch` of V plus T plus WY scratch. cfloat n = 96 (0.8%) is the same call.

### Block widths

Neither driver inherits `tuning::ormqr_block_size_for_n`. Both compute their own, keyed on the **type** and clamped to `k = min(m,n)`: **16 for double, 32 otherwise** — `geqrf_blocked.cc:41-181` (`geqrf_nb_for_type` plus the `geqrf_blocked_nb` clamp), `orgqr_blocked.cc:36-107`. Evidence in [block-width-evidence](#block-width-evidence).

## Measured boundaries

### CTA vs blocked crossover

`tier_summary.txt`, vendor-free, `BATCHLAS_GEQRF_ROUTE=cta` against `=blocked`, both arms pinned and every pin verified to have taken. **20 of 44 (type, n) cells had the `cta` pin silently resolve to `native:blocked`** (`m*n > cta_max_elems`) and are excluded; tabulating them would have produced a CTA/Blocked table in which both arms are the same code. Ratio is `blocked_ms / cta_ms`, so **> 1 means CTA ahead**. Batch 8192 at n <= 64, 4096 at n >= 80.

| type | n=32 | n=48 | n=64 | n=80 | n=96 | n=112 | n=128 |
|---|---|---|---|---|---|---|---|
| float | 1.002 (null) | 2.686 | 2.034 | 2.037 | **1.294** | **0.821** | 0.699 |
| double | **1.049** | **0.983** | 0.922 | 0.772 | 0.731 | pin n/a | pin n/a |
| cfloat | 1.002 (null) | 3.171 | 2.093 | 1.253 | 1.079 | pin n/a | pin n/a |
| cdouble | 0.995 (null) | 2.589 | 1.929 | pin n/a | pin n/a | pin n/a | pin n/a |

n <= nb are NULL CELLS: at `n <= nb` the blocked driver is one panel on the resident leaf, i.e. literally the CTA code, so 1.00x is an identity and not a tie.

**Bracketing.** float is bracketed on both sides (96 → CTA 1.294, 112 → blocked 1.218). double is bracketed by 32 (CTA 1.049) and 64 (blocked 1.085), with 48 a tie resolved to CTA. **The complex boundaries are not bracketed and do not exist**: cfloat's last measured cell is n = 96 at 1.079 against a capacity ceiling of 110, cdouble's is n = 64 at 1.929 against 77. **cfloat 97..110 is extrapolated, not measured** — its margin is collapsing (3.171 → 2.093 → 1.253 → 1.079) and is the first place to look on a re-measure.

Second, independent sweep — the **shipped default** against the other tier forced, same binary, same session, interleaved, three reps, vendor-free, forced arm's resolved route printed and verified on every row. The top block is what the window buys; the bottom block is the non-winner check that it did not overshoot.

| type | n | default | forced other tier | gain |
|---|---|---|---|---|
| float | 112 | 9.15 ms (blocked) | 11.98 ms (cta) | **1.31x** |
| float | 128 | 9.97 ms (blocked) | 15.39 ms (cta) | **1.54x** |
| float | 155 | 16.39 ms (blocked) | 22.73 ms (cta) | **1.39x** |
| double | 64 | 27.42 ms (blocked) | 29.72 ms (cta) | **1.08x** |
| double | 80 | 18.49 ms (blocked) | 24.07 ms (cta) | **1.30x** |
| double | 96 | 23.86 ms (blocked) | 32.75 ms (cta) | **1.37x** |
| double | 110 | 30.55 ms (blocked) | 42.52 ms (cta) | **1.39x** |
| float | 64 | 2.95 ms (cta) | 5.70 ms (blocked) | 1.93x |
| float | 96 | 5.00 ms (cta) | 6.02 ms (blocked) | 1.20x |
| double | 32 | 10.87 ms (cta) | 11.42 ms (blocked) | 1.05x |
| double | 48 | 19.34 ms (cta) | **19.02 ms** (blocked) | 0.98x — tie |
| cfloat | 96 | 10.64 ms (cta) | **10.72 ms** (blocked) | 1.01x — tie |
| cdouble | 64 | 59.18 ms (cta) | 113.79 ms (blocked) | 1.92x |

**Mechanism, and the limit of the window.** `geqrf_cta`'s capacity is a pure byte budget with no blocks-per-SM term, so above ~50 KB the tile forces one work-group per SM (256 of 1536 threads) and the per-reflector barrier chain has nothing to overlap with. The float crossover lands exactly there: n=96 → 36,864 B → 2 blocks/SM, CTA ahead 1.294; n=112 → 50,176 B → 1 block/SM, CTA behind 0.821. That arithmetic is consistent with the cliff but **was not verified with an occupancy counter**. `native_tier_preferred` routes around it; it does not fix it.

### The vendor baseline

cuBLAS `geqrfBatched` saturates at ~380–390 GFLOP/s (float) and ~105–110 (cdouble) **regardless of n** — a small-matrix, one-column-at-a-time routine, latency-bound, whose ceiling does not move. Its wall time is nearly independent of batch at n >= 512: float n=2048 costs 21,361 ms at batch 32 and 23,151 ms at batch 256; float n=1024 costs 1,204 ms at batch 8 and 2,276 ms at batch 256 (32x the work for 1.9x the time). So the **ms column is a valid absolute target** at each stated cell and the **GFLOP/s column at n >= 512 is not a statement about cuBLAS's ceiling**. Do not quote the 181x below as "faster than cuBLAS". Ceiling-to-ceiling at n=1024: native 3564 / 1079 / 1683 / 132 GFLOP/s (float/double/cfloat/cdouble) against ~380–390 / ~200 / ~205 / ~105–110, i.e. **9.2x / 5.4x / 8.2x / 1.2x**.

cuSOLVER `orgqr` is a different kind of thing: **not batched at all** (`cublas.cc:1414-1419` opens an out-of-order sub-queue and calls `cusolverDnXorgqr` once per batch item) and its workspace is `single_ws * batch` (`cublas.cc:1447-1448`) — 1164 MB for float n=64 b=8192 and 4644 MB for cdouble at the same cell, for a problem whose data is 268 MB. Any win over it is "beats the per-item loop", never "beats cuSOLVER".

### `geqrf` order and batch grid

`order.csv`, vendor ms → vendor-free ms (ratio; bold = native ahead). Native wins 25 of 36; geomean 3.24x.

| n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32, 8192 | 0.8 → 1.0 (0.78x) | 2.3 → 10.8 (0.21x) | 1.0 → 1.4 (0.71x) | 5.8 → 17.6 (0.33x) |
| 64, 8192 | 6.3 → 2.9 (**2.14x**) | 15.7 → 29.6 (0.53x) | 13.1 → 5.0 (**2.64x**) | 31.5 → 58.9 (0.54x) |
| 128, 4096 | 30.9 → 15.4 (**2.01x**) | 60.8 → 37.1 (**1.64x**) | 59.2 → 19.5 (**3.04x**) | 115.8 → 224.0 (0.52x) |
| 256, 2048 | 121.4 → 21.6 (**5.62x**) | 228.7 → 72.3 (**3.17x**) | 227.1 → 44.4 (**5.11x**) | 434.9 → 520.6 (0.84x) |
| 512, 512 | 370.9 → 30.5 (**12.18x**) | 685.8 → 101.5 (**6.76x**) | 560.8 → 63.2 (**8.87x**) | 1112.1 → 789.5 (**1.41x**) |
| 1024, 128 | 2112.0 → 51.4 (**41.08x**) | 4290.9 → 169.9 (**25.26x**) | 3428.6 → 108.9 (**31.49x**) | 5993.9 → 1392.2 (**4.31x**) |
| 2048, 32 | 21283.2 → 117.6 (**181.02x**) | 30529.4 → 359.3 (**84.98x**) | 24888.3 → 242.5 (**102.65x**) | 41947.2 → 2815.9 (**14.90x**) |

The batch schedule varies with n (memory-bounded), so an order crossover read off this table is confounded with a batch change; only three order-crossovers in `order.csv` are clean. The clean order statements come from `tier.csv` and the fixed-n blocks of `batch.csv`.

**Batch is the dominant axis for FP64, not order.** At n = 64:

| batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32 | 8.67x | 4.37x | discarded (sd 12.0%) | 5.05x |
| 512 | 5.85x | 1.16x | 3.59x | 1.24x |
| 2048 | 1.79x | **0.46x** | 1.61x | **0.55x** |
| 8192 | 2.12x | **0.53x** | 2.64x | **0.53x** |
| 16384 | 1.74x | **0.53x** | 2.71x | **0.52x** |

Both arms are launch-bound below batch ~512 and linear above it, so the flat 0.53x from batch 2048 to 16384 is the **saturated** ratio and the 4–5x at batch 32 is overhead compared to overhead. **Quote the 0.53x.** Three A/B cells of 184 were discarded, all the vendor arm at tiny batch where its absolute time is ~1.2 ms (`geqrf` float n=64 b=128 sd 13.7%, cfloat n=64 b=128 12.1%, cfloat n=64 b=32 12.0%); every other cell is under 6.1%, and 448 of 456 rows under 2%.

**Tall panels — the shape the library actually asks for.** Both in-tree callers (`band_reduction.cc:595`, `sytrd_sy2sb.cc:504`) pass an `m x r` panel with `r << m`. `tall.csv`, vendor/native, as float / double / cfloat / cdouble: 128x32 b4096 → 1.57 / 0.62 / 3.31 / 0.71; 512x32 b2048 → 2.11 / 1.16 / 2.27 / 1.30; 1024x64 b512 → 4.52 / 3.68 / 2.21 / 0.82; 2048x64 b256 → 7.58 / 6.32 / 3.28 / 1.28; 1024x128 b256 → 10.85 / 8.30 / 7.27 / 1.67. Geomean 3.34x / 2.37x / 2.44x / **0.96x**; native wins 26 of 32, and the margin grows with m at fixed n — the direction the callers move in.

### `orgqr` grid

Native wins 31 of 36 order cells; geomean 7.85x. Ratios are cuSOLVER's per-item loop ÷ native.

| n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32, 8192 | **123.2x** | **114.1x** | **65.1x** | **12.4x** |
| 128, 4096 | **17.1x** | **30.8x** | **10.6x** | **6.0x** |
| 256, 2048 | **9.5x** | **15.3x** | **5.8x** | **3.4x** |
| 512, 512 | **3.8x** | **6.0x** | **2.3x** | **1.7x** |
| 1024, 128 | **1.26x** | **2.44x** | 0.82x | 0.78x |
| 2048, 32 | 0.41x | **1.34x** | 0.31x | 0.46x |

The losses were re-measured on the batch axis before being called losses, because the vendor arm is linear in batch by construction and at b=32 has not yet paid for serialisation (`orgqr_batch.csv`): float n=1024 goes 0.84x / **1.11x** / **1.27x** / **1.33x** at b = 32/64/128/256 — it **flips** at b >= 64 and is still rising (the order sweep's 1.26x reproduces as 1.275x in an independent process). cfloat n=1024 climbs 0.55 → 0.70 → 0.82 → 0.88x and would plausibly cross past b = 512, which does not fit in 24 GB and is therefore **not claimed**. At n=2048 the loss is genuine: float 0.33 → 0.47x over b = 16..128, cfloat 0.26 → 0.35x, cdouble 0.44 → 0.46x. **double never loses at any (n, batch) measured.** Verification that the vendor arm really is the per-item loop: float n=1024 vendor times are 20.4 / 45.4 / 93.9 / 188.5 ms at b = 32/64/128/256, linear to within 4%, against native 24.3 / 40.8 / 73.7 / 141.4.

**The workspace advantage reverses exactly where the speed advantage does**: native is 3.3–6.7x cheaper at small n and large batch (cdouble n=64 b=8192: 4870 MB vendor, 1476 MB native) and 2.7–5.5x more expensive at n >= 1024 (float n=2048 b=32: 103 MB vendor, 562 MB native).

### Where the time goes

nsys `cuda_gpu_kern_sum`, vendor-free, on winning **and** losing cells — a split taken at a winner does not explain a loss. Captures are not committed and must not be quoted as timings (`WARM_S=0.2`, 2 reps); wall times come from `order.csv` and the two agree to ~4% where comparable.

| | float n=1024 (41x win) | cdouble n=1024 (4.3x win) | cdouble n=256 (**0.84x loss**) | double n=64 (**0.53x loss**) |
|---|---|---|---|---|
| transposed GEMM (Tiled16) | **46.6%** | **69.7%** | **51.3%** | — |
| NN GEMM (Register128x128 / 64x64-wide) | 24.3% | 21.4% | 15.6% | — |
| `larft` + `pack_v` | 19.7% | 6.5% | 22.3% | — |
| panel factorisation | 9.3% | 2.5% | 10.6% | **100.0%** |

The transposed panel GEMM `W1 = V^H A22` is the largest single kernel in **all three** blocked cells profiled. `src/sycl/gemm_kernels.cc:464-477` short-circuits every transposed form to `max_dim <= 32 ? Direct : Tiled16` before the register ladder; the TN/NT/TT register forms need `m >= 128 && n >= 32 && k >= 128`, and `m` here *is* the block width, so reaching them costs a block width of 128 — the worst width end to end. Complex cannot reach them at any width for **two** independent reasons, and the first is the load-bearing one: the whole register ladder sits inside `if constexpr (std::is_same_v<T, float>)` (`:471`), and the gate additionally tests `transA == Transpose::Trans` (`:472`) while a complex panel update is `ConjTrans`. **There is no block width at which geqrf's transposed panel gemm reaches a register kernel and is also a good block width.** G3 does reach the good kernel for both float and cdouble, so G3 is not the problem. The `double n=64` column is `native:cta`, which has no trailing update at all: that loss is the panel kernel at FP64 rate (1:64 of FP32 on this card) and no GEMM change can touch it.

**The first `orgqr` capture was wrong, and the way it was caught is the transferable part.** `cuda_gpu_kern_sum` aggregates by kernel *name*, and `qrbench_nv orgqr` builds its factor with an **untimed** `geqrf` call before it times anything — 32 panels of panel + `pack_v` + `larft` + three GEMMs at n=1024. The GEMM kernels carry no tag naming their caller, so `GemmTiledGeneralKernel<float,16,...>` in that capture was the sum of `orgqr`'s applies *and* `geqrf`'s trailing updates. It was caught only because `larft` and `pack_v` **do** separate by tag: a profile of `orgqr` showed `LarftKernelName<GeqrfWyTag,...>` alongside `<OrmqrWyTag,...>`, plus 32 `GeqrfPanel*` launches `orgqr` does not make. `SYNTH=1` (host-fabricated reflectors, `H_i = I - tau v v^H` with `tau = 2/(v^H v)`, so the product is still unitary and the ortho probe still discriminates) removes the `geqrf` call entirely. The contaminated float n=1024 numbers were 33.0% Tiled16 / 13.2% identity fill; the clean ones are 41.9% / 15.2%. The contaminated cdouble capture said 69.2% Tiled16, which happens to be **right** — a profile can be contaminated and still land on the correct headline, and that is not a defence. Everything in the table above is post-`SYNTH`.

One per-kernel outlier is also excluded rather than averaged in: `OrgqrIdentityKernel`'s **first** launch costs 32.0 ms against 3.46 ms thereafter, a unified-memory first-touch page migration and not a kernel cost. The ~17–22% range quoted for n=64 spans the median-corrected and the raw figure.

`orgqr` has a **different** bottleneck at each end of the range: `larft` is 49.0% at float n=64 b=8192, the transposed GEMM 41.9% at float n=1024. A single `preferred()` clause cannot be motivated by one of them. The identity fill and copy-back — the only kernels that exist because `orgqr` is `ormqr`-on-an-identity — cost 8.1 ms of 74 (~11%) at float n=1024 and ~17–22% at n=64.

### Block-width evidence

End-to-end WY apply (`ormqr` on an identity), `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE` forced, median ms, vendor-free, n=1024 batch=64 (`]` = the shipped `ormqr_block_size_for_n` value, `*` = best):

| nb | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 16 | 45.36 | **123.65\*** | 101.53 | 1547.54 |
| 24 | 43.31 | 150.67 | 120.80 | 1818.16 |
| **32** | **36.82\*** | 135.65 | **81.54\*** | **1061.24\*** |
| 48 | 40.09 | 155.28 | 96.67 | 1144.85 |
| 56 `]` | 45.48 | 174.82 | 113.18 | 1333.81 |
| 128 | 83.04 | 303.45 | 296.93 | 1936.75 |

Cost of the shipped ladder here: float 1.24x, double 1.41x, cfloat 1.39x, cdouble 1.26x. At n=256 batch=512 (shipped width 24) the best is 16/16/16/32 and the shipped width costs 1.11x / 1.32x / 1.26x / 1.55x. It is a **type** problem, not merely a build problem: the same ladder in the **vendor** build still costs double 1.32–1.41x and cdouble 1.46–1.47x, while float at n=256 is exactly 1.00x — the one cell it was tuned at. `evaluation/tuning/tune.py:494` takes a single `--type` per run and the `ormqr_blocked` space has no type axis, so the shipped buckets are a CUDA/float optimum applied to all four types.

Three mechanisms, each measured. (1) **Multiple of 16**: G1's `m` *is* the block width and G1 is Tiled16 for every type in a vendor-free build; 24 and 56 lose everywhere **in that build** (in the *vendor* build 24 is the best float and cfloat width at n=256, which is where the shipped ladder's 24 came from). (2) **Never below 32 for complex**: `gemm_kernels.cc:545` gates the complex wide-scalar kernel on `min_dim >= 32` — and on a CTA-count floor, 64 for cfloat / 128 for cdouble — and `min_dim` of G3 *is* the block width, so at nb=24 complex G3 falls to Tiled16 and costs 1.72–2.30x. (3) **Not wider than 32**, see [negative-results](#negative-results). `nb=16` beats `nb=32` for double at both n by 1.10–1.32x while 32 wins for the other three, so a single bucket table keyed only on n cannot express the answer — which is why both drivers key on the type.

### `ormqr` WY `trmm` gate

`wy_trmm_applicable` (`ormqr_blocked.cc:49-60`) chooses the trmm tile kernel over a GEMM for the WY apply; shipped predicate `route_has_tile_kernel && !is_complex<T> && ib <= 64`. Measured on `ormqr_blocked_benchmark`, `Side::Left`, `ConjTrans`, ABBA-ordered against `BATCHLAS_ORMQR_WY=gemm`, batch 256 (128 for cdouble), nb in {16,32,64}, each figure the mean of two runs with the whole sweep repeated at a second measurement window (**the n set is not recorded in the source note — unverified**); ratios are gemm/trmm, so > 1.00 is trmm ahead: float 1.006–1.046x, double 1.004–1.016x, **cfloat 0.944–0.995x**, cdouble 0.958–1.010x. A 16-row tile variant moved every type in the direction the tile-reuse argument predicts (double 1.013–1.036x, cdouble 0.996–1.018x, cfloat 0.946–0.983x) but **did not change the gate**: closing to parity is not a reason to switch a call site, and cfloat stays behind because a complex multiply is four real ones. `ib <= 64` predates this and stays — past it the tile kernel measured 0.83x–0.97x in float. Read that as a local dip and not a monotone cutoff: the same table has the tile kernel back ahead at m >= 512 (1.07–1.32x). No `ormqr` block width reaches there, so the gate costs nothing today. netlib is excluded because OpenBLAS's `?trmm` is weak on a 16x16 triangle against 128 right-hand sides (0.336–1.199x, worst at n=128 ib=16); ROCm because `rocblas_?trmm` is a per-batch vendor loop against a strided-batched GEMM.

## Negative results

**The harnesses that would have lied, and why the measurement uses none of them.** `benchmarks/geqrf_benchmark.cc` counts flops as `2mn^2 + (2/3)n^3` — the **wrong sign** on the second term (LAPACK's `geqrf` is `2mn^2 - (2/3)n^3`) — is registered float/double only with **no complex**, and never checks the answer. `benchmarks/gemm_benchmark` allocates operands at `ld == rows`, which is structurally incapable of seeing the sub-view question this driver's trailing GEMMs pose. Separately, WP4's `phase2.cpp` defined its *own* `Blocked<T>` class instead of timing the shipped code: it was 2x slower and contradicted the real numbers by a factor of two. Everything on this page times the **public API** with the workspace the facade's own `*_buffer_size` asked for, and checks the residual in the same process — five apparent wins entered the WP4 record because a racing kernel was fast and wrong.

**"WP5 will be decided by the panel factorisation" is refuted for the shipped blocked driver.** The claim came with a 63.2x-headroom estimate computed against a **routed** trailing update; the shipped driver's largest GEMM lands on Tiled16. Measured, the panel is 9.3% (float n=1024), 2.5% (cdouble n=1024) and 10.6% at the losing cdouble n=256 cell, against ~71% for the trailing update. The one place the panel *is* the whole cost is the CTA tier — 100.0% one kernel — which is exactly where FP64 loses.

**Tuning the block width on the trailing GEMMs alone is the wrong instrument.** Measuring the G1+G3 pair in isolation says wider is always better and shows a float cliff at nb=128 — effective throughput 6896 → 18,906 GFLOP/s, because `m >= 128` finally admits `Tiled128x32RegisterK32TN`. **End to end that cliff does not survive**: nb=128 is the worst width tested in both builds, 83.0 ms against 36.8 ms at nb=32, because the panel and `larft` costs a per-gemm probe cannot see dominate.

**Specialising `orgqr` is worth at most 1.5x, and the measured price of not doing it is ~11%.** Applying Q to an identity does `2n^3` against a specialised `4n^3/3` at m=n=k. The identity fill plus copy-back cost ~11% of float n=1024 and ~17–22% of float n=64, below that theoretical ratio, against a 2.3–111x margin over the vendor across most of the range.

**32 device entry functions that could never launch.** `larft_forward_columnwise_batched` took `use_device` as a runtime bool, so it instantiated both implementations for every `(Tag, T, WG)`; `geqrf` passes a literal `false`, so `larft_forward_columnwise_wg_device<GeqrfWyTag, ...>` was 4 types x 4 work-group rungs x 2 forms = 32 entry functions compiled, ptxas'd and device-linked into `batchlas_extensions_cta` — the slowest-linking library in the tree — and never launched (nsys: no `(bool)1` variant in any WP5 run). They included the highest-register kernel in the whole WP5 set (cdouble, 90 registers, 208 B stack frame). `UseDevice` is now a template parameter, the runtime wrapper retained for `ormqr` whose choice really is a getenv. **880 → 848 entry functions, device link 125.45 s → 116.63 s.**

**Three uncoalesced fills and one needlessly strided operand.** `OrgqrIdentityKernel`, `OrgqrCopyBackKernel` and `pack_v_panel_batched` all launched `sycl::range<3>(batch, rows, cols)` and read `idx[2]` as the **column**; `sycl::id<3>` makes dim 2 fastest-varying and every operand is column-major, so a warp touched 32 sectors instead of 4. Separately the blocked driver handed `V` to both trailing GEMMs at the **parent** `ld = m` although `V` is scratch it owns outright — at j0=992 of a 1024-column factorisation, a 32-row panel whose columns were 4 KB apart, the recorded "native GEMM collapses on strided ld" shape. `V` is now packed at `ld = mp`. Clean same-session A/B (fixes reverted, rebuilt, measured, restored; all relsd <= 0.11%): `orgqr` float n=512 b=512 1.196x, n=1024 b=128 1.118x, n=2048 b=32 1.059x; `geqrf` float n=512 b=512 1.054x. 16 of 16 cells improved or were neutral, and the gain is ~1.00x wherever the cell is GEMM-bound — which is most of the FP64 and complex grid.

**The `route_gemm.hh` `batch >= 64` gate (`route_gemm.hh:48`) costs the vendor build.** At N=2048 b=32 six `double` trailing-GEMM cells go to cuBLAS and the vendor build is **1.14–1.25x slower** than the vendor-free build on the same shapes. Small, but a real `preferred()` edge on a shape `geqrf` issues.

**The complex deficit is outside WP5 and was not attempted.** A vendor-free build pays 2.55x (float), 1.00x (double), 2.61x (cfloat), 2.01x (cdouble) on the BLAS-3 core, essentially all of it G1 (4.81x / 1.00x / 4.99x / 3.12x; G3 is 1.06x / 1.00x / 1.02x / 0.95x). Closing it needs a **transposed** wide-scalar/register GEMM — WP2 territory; `route_gemm.hh:43-45` still refuses complex outright. double is 1.00x because both builds run the identical native kernel; separate processes and separate `.so`s agree to 0.03% (11.8008 vs 11.8012 ms), the strongest internal control in the baseline directory.

## Correctness findings

### The 48 KiB launch hole

`geqrf` shipped inside WP4's recorded 48 KB launch hole. A resident-leaf launch asking for **exactly** 49,152 B of local memory is refused by the CUDA backend (`CUDA_ERROR_INVALID_VALUE` at `enqueueKernelLaunch`). WP4 had written down the condition that reopens the hole — "one group algorithm added anywhere in the body — a `reduce_over_group` ... reintroduces the hole" — and `geqr2_panel_device` runs two `reduce_over_group` calls per reflector. Measured cold, one process per point, through the public facade: 48,896 B pass / **49,152 B fail** / 49,664 B pass, all four types, reached at 384x32, 384x16, 192x32 and 192x16 respectively. A byte threshold, not a shape or a type.

**How it hid: by execution order.** The attribute the UR CUDA adapter sets is sticky per `CUfunction` and one instantiation serves every panel shape, so any earlier launch of a larger panel raises the cap for the rest of the process. `geqrf_tests`' own blocked ladder reaches 100x32 (51,200 B) before 96x32 and is green either way; it took `orgqr_tests` asking for cdouble 96x96 as the *first* blocked shape in its process to expose it. No amount of shape coverage inside a single process would have closed it.

Fixed by adopting potrf's band and pad verbatim (`kGeqrfHoleLo = 47104`, `kGeqrfHoleHi = 49664`, `kGeqrfHolePadTo = 49920`), applied in **three** places so the table and the launcher cannot disagree: the resident `local_accessor` allocation, `geqrf_cta_fits`, and `geqrf_cta_max_elems_for_slm` (through `geqrf_hole_safe_budget`, inert at this box's 97,280 B budget, so no pinned capacity number moves). Guarded by `GeqrfTest.ResidentLeafLaunchHoleAt48KiB`, declared **first** in the file because it is only discriminating while it is the first resident launch of its type in the process. Nothing in GoogleTest can assert that ordering; the guard is declaration order plus a comment.

### A residual test cannot guard a convention

Kernel break **K3** replaced LAPACK's real-beta `larfg` convention with `internal::larfg`'s phase-preserving one. **Every residual column stayed green for every type** — `||QRx-Ax||`, `||Q^H Qx-x||`, and the same with the explicit Q, to 1e-15 / 1e-6 — because a phase-preserving factorisation is a perfectly good QR. It is simply not the one `ormqr`, `orgqr`, `ormbr`, `sy2sb`, `band_reduction`, netlib and cuSOLVER all agree on. Only the elementwise columns saw it: `dimag` 0.94–0.97, `dF` 1.4–1.8, `dtau` 0.19–0.71.

Before the repair pass the **only** test that could see this was `NativeFactorMatchesTheVendorElementwise`, which opens with `GTEST_SKIP` in a vendor-free build. **So in the build this work package exists for, the real-scalar half of `geqrf`'s drop-in contract had no guard at all.** `ConventionMatchesReferenceLapackWithoutAVendor` closes it against an independent host `xGEQR2` written from the LAPACK reference; re-running the break (BR1) with it in place is **RED for all four types in `build-novendor`**, where before it shipped green.

Two secondary results. (a) BR1 turned a *few* residual tests red this time (`BlockedResidualAndOrthogonality` float/double, `ShortFinalPanelStraddlesTheBlockWidth` double) because dropping the sign choice causes cancellation in `alpha - beta` on some data — but the CTA tier stayed green and **both complex types stayed green**. A residual test catches this break *sometimes*; the convention test catches it deterministically. (b) **`zgeqr2` applies `conj(tau)`, not `tau`**, because reducing from the left applies `H^H`. The first host reference used `tau` and disagreed with the kernel by 1–4% for cfloat/cdouble while being exact for float/double — the same signature as kernel breaks K1/KE. For a real `T` the conjugate is the identity, so this entire defect class is **invisible to half the type list by construction**.

The control, over 10 shapes x 4 types x up to 2 tiers: **`dF` 3.2e-06 (float) / 8.2e-15 (double) / 2.0e-06 (cfloat) / 3.8e-15 (cdouble)** against the vendor's own `geqrf` output, with **`dtau` 3–6x looser — 9.2e-06 / 3.1e-14 / 1.3e-05 / 3.1e-14** (`kernels/README.md` §3 — the `dF` row alone is not the drop-in tolerance; quoting it as "dF/dtau" understates `tau` by 3–6x). `dimag` is exactly 0 everywhere. The native factor **is** elementwise the factorisation cuSOLVER produces, `tau` included. Treat those figures as order-of-magnitude rather than pinned constants: the committed `kernels/run_v.txt` disagrees with its own summary table in both directions (its worst `dF` is 3.7e-06 float at 128x128 b=64 and 9.5e-15 double; its worst `dtau` is 1.6e-05 / 1.6e-14 / 1.6e-05 / 2.6e-14), so the table and the file are from different runs.

### The `orgqr_buffer_size` latent defect

`orgqr_buffer_size` gated "did a native tier fire?" on `native_need == 0` — the exact defect the same change had deliberately removed from `geqrf_buffer_size` 170 lines above, with a comment explaining why a **zero workspace is a legitimate answer** (it is exactly what the CTA tier reports). It was unreachable today only because `orgqr_blocked_layout` unconditionally allocates `m*n*batch`, and reachable the moment a specialised in-place `orgqr` lands — which both `orgqr_native.hh` and `orgqr_blocked.cc` explicitly contemplate. Now uses `native_fired`, matching its sibling. Recorded because "a zero-sized workspace means no native route" is the same conflation the CTA tier's zero workspace creates everywhere in this family.

### The short-final-panel vacuity

Reference break 1 (drop the **last** reflector) is **green for float and double** on a square matrix — residual bit-identical at 4.072e-07 / 1.615e-15 — and red for complex (2.137e-02). On a square matrix the final reflector acts on a 1x1 trailing block, and LAPACK's `larfg` returns `tau = 0` there for a real scalar but a non-zero `tau` for a complex one, because it must still rotate R's diagonal onto the real axis (`|tau[k-1]|` measured as 0.000000e+00 real, 1.553246e+00 complex). At 300x200 the same break is red for **every** type. **A short-final-panel regression test written on a square real matrix guards nothing** — precisely the shape class that produced the silent `sy2sb` stage-1 failure. Use `m > n`, a middle panel, or complex; break 5 (drop a *middle* reflector) is red for all four types and is the standing check.

A related vacuity is recorded rather than fixed: break **N1** (`ib` → `nb` in the larft/pack-V calls) turns nothing red **by construction**. `supports()` requires `m >= n`, so `k == n`, so a short final panel has `j0 + ib == k`, therefore `n2 == 0` and the driver breaks out *before* the WY update — `larft` is never handed a short panel. The short-final-panel error class exists in this driver only at the **leaf** (break KD), not in the trailing update, which is the opposite of where sy2sb's bug was.

### Break sweeps

12 breaks in the experiment harness (5 reference + 7 kernel) and 13 against the shipped suite (9 + 4 from the repair pass). The ones that carry information:

| break | what it deleted | outcome |
|---|---|---|
| KA | the 48 KB hole pad removed from the resident leaf's `local_accessor` | `ResidentLeafLaunchHoleAt48KiB`, all 4 types, and **only** that test; cold-filtered per type, 4/4 fail |
| K1 / KE | `conj(tau)` in the panel apply | **complex only** red (qr 6.0e-02–4.5e-01, `dF` 1.6–1.8); float/double green — correct null |
| K2 | `T^H` → `T` in the WY trailing update | `cta` green (it has no WY update — correct); `blocked` red for every type |
| K3 / BR1 | LAPACK's beta sign choice | residuals **green**, `dimag`/`dF`/`dtau` red — see above |
| K5 | `tau`'s batch stride `k` → the panel's `ib` | red everywhere, but only after the checker was fixed — see below |
| K7 | a sub-view of the **caller's** matrix loses its explicit stride | red by 25 to 91 orders of magnitude |
| KB | blocked sub-view built with `nr` instead of the parent `ld` | 33 rows / 9 tests, all 4 types |
| KC | panel loop `j0 < k` → `j0 + nb <= k` (short final panel dropped) | 62 rows / 8 tests, all 4 types |
| KG | `tau` batch stride `k` → `ib` in the leaf call | 68 rows / 8 tests. **Item 0 is unaffected**, which is why the checker walks every batch item |
| BR3 | the division arm of the reciprocal guard | `SubnormalScaleColumnsTakeTheDivisionPath`, all 4 types, and nothing else |
| BR4 | `native_tier_preferred` removed entirely | `NativeTierTieBreak...`, float and double — the two types with a measured crossover — and nothing else |
| BR4b | the same window moved **into `supports()`** | red on the *intended* assertion (`supports(cta, sh_hi)` was false), which is what proves the test's second half is not vacuous |

**The checker itself was defective and the sweep had to be run twice.** The first K5 run printed `qr=4.788e-07` — green — with `tau` poisoned to -12345 for most batch items. The probes overflowed to NaN and `std::max(0.0, NaN)` returns `0.0`, so a NaN residual read as a **perfect** one. `qrcheck.cpp` now uses a NaN-propagating `nanmax` in all four probes and K5 turns fully red. **The same defect is still present in `experiments/wp5_qr/baseline/wp5qr.cpp`** and in anything derived from it; `qrbench.cpp` has the fix and reported 0 of 456 timed rows `BAD`.

**Two kernel breaks turned nothing red, and both are reported rather than hidden.** K4 deleted barrier B2 (between every work-item's read of `A(j,j)` as alpha and work-item 0's write of beta). It is required by the SYCL memory model — `reduce_over_group` converges control flow but is not specified to order memory — and on this compiler and device the intervening reductions happen to carry a barrier. The barrier stays; the honest statement is that the harness **cannot prove it is needed**. K6 changed W1/W2's batch stride to the current panel's `nb*n2` and is a **bad break rather than a missing guard**: W1/W2 are private scratch, read and written only by the three GEMMs through the same view, so any stride they agree on is arithmetically fine. A stride break is only meaningful on a view whose stride is fixed by someone else — which is K7.

**And one null that is a property of a whole suite.** Every kernel break above left `tests/orgqr_tests.cc` **green in the vendor build** (break N2). That suite pins no route, so its facade `geqrf`/`orgqr` resolve to cuSOLVER and no native kernel runs: as a guard on WP5's kernels it is a **null** in a vendor-present build and discriminates only in `build-novendor`. `tests/geqrf_tests.cc` calls the direct entry points precisely so it does not have that property. The residual bound is measured, not comfortable: tightening it (break B0) turns 16 rows red, so the shipped constants carry 5.2x and 2.2x of margin — not the 40–200x that `potrf_tests.cc:180-300` records as wide enough to hide an accuracy defect.

### Suite status

`geqrf_tests` (new, `blas` label) is 80/0 in `build/` and 72/0 in `build-novendor/`. The vendor-free burn-down moved **26 of 54 → 30 of 55**: `backend_dispatch_tests` (13/13), `syev_two_stage_tests` (20/20) and `sytrd_sy2sb_tests` (2/2) now pass vendor-free, and nothing newly failed. They are **not** the four suites the WP5 brief predicted. `orgqr_tests` went 16 → 8 failures and `ormqr_tests` 24 → 16, each losing exactly its CUDA rows; `ormqr_cta_tests` (2) and `ormqr_blocked_tests` (30) are unchanged because their references are netlib `geqrf`/`ormqr` on a host queue and `ormqr_vendor_or_throw`. All four still carry `Backend::NETLIB` rows that no CUDA kernel can fix. The route diff over the repair pass is one substitution and one addition, both inside the new test: **zero vendor-present decisions moved.**

One harness trap in the gate itself, since every number above is a `ctest` count: the label selection is a **single** `-L "blas|ortho"` flag, never two. Repeated `-L` flags AND together, select **zero** tests, and exit 0 — a green run that ran nothing.

## Open debts

1. **The vendor-present default is still cuSOLVER, everywhere, for `geqrf` and `orgqr`.** The 3.24x / 7.85x geomeans are unrealised. Flipping a cell is gated on more than a kernel-level win — this tree has turned a 2.16x kernel win into an 11% `gesvd` loss — and needs an end-to-end harness (`ortho_benchmark`, a `syev` path) that WP5 did not run. **The single largest piece of value left on the table.**
2. **The tier crossover is measured on SQUARE shapes only, and `native_tier_preferred` gates on `n`.** That is a mechanism argument (CTA's serial cost is its per-reflector chain, `k = min(m,n)` long, and `geqrf_panel_wg` derives the work-group from `n` alone), not a measured one. A tall skinny panel — m=512, n=32, float — is CTA-eligible and has **no measured cell**. Left on CTA deliberately.
3. **cfloat 97..110 is extrapolated, not measured**; cdouble has no measured cell above n=64 against a ceiling of 77. See [cta-vs-blocked-crossover](#cta-vs-blocked-crossover).
4. **`geqrf_panel_wg(n, max_wg)` never looks at `m`.** It returns wg=256 for a column that may hold 64 elements, so at double n=64 at least 193 of 256 work-items execute zero loop iterations and still pay a full 256-wide `reduce_over_group`. That kernel is **100.0% of GPU time** in the campaign's worst `geqrf` loss (double n=64, 0.53x, 187 GFLOP/s = 14.5% of FP64 peak). Its own comment says the number "is NOT a tuned number". Untouched.
5. **`larft_forward_columnwise_wg_legacy` is 15.2% of `geqrf` float n=1024 and 20.3% of the losing cdouble n=256 cell**, doing ~1/60 of the arithmetic of the GEMM beside it — at cdouble n=256, twice what the panel factorisation it exists to accelerate costs. Two mechanisms: the `(j,col)` loop re-reads two full V columns per pair (`O(ib^2/2)` work-group barriers per panel), and the T-update recurrence runs serially on `lid == 0` with 255 lanes idle. The body is inherited verbatim from `ormqr_blocked.cc`; WP5 is what put it on `geqrf`'s critical path. Not rewritten.
6. **`geqrf_cta`'s capacity has no blocks-per-SM term.** A `m*n*sizeof(T) <= local_mem/2` shape for the *capacity* would track the measured float crossover almost exactly and is untried.
7. **The `!dev_is_zero(r)` half of the reciprocal guard is untested and may be wrong.** `r == 0` needs `|alpha - beta| > 2e323`, i.e. `alpha - beta` must itself have overflowed to inf — reachable at input ~1e308, where `vfactor` becomes inf and `v` becomes exactly zero: finite, but **not a correct reflector**. The shipped test covers only the overflow-of-reciprocal half. Worth knowing about the half that *is* covered: before `SubnormalScaleColumnsTakeTheDivisionPath` the division arm had **never executed** in any test, harness or benchmark in this tree. Reaching the guard at all requires subnormal input, at which magnitude no tight residual is possible, so that test asserts finiteness plus orthogonality at an explicitly justified loose bound.
8. **`geqrf_buffer_size` builds its shape twice and makes 6 uncached SYCL `get_info` calls per API call.** This lands on `band_reduction.cc:595`, which calls `geqrf(...).wait()` once per step, and on `sytrd_sy2sb.cc:504`, which calls `geqrf(...)` once per step **without** a `.wait()` (the source note says both wait; the code does not) — `O(n^2/kd^2)` steps, ~500 for n=1024. Pure host overhead, no wrong answer, and **not measured**. Measure before acting.
9. **`resolve_ormqr_block_size` still returns the float-only 16/16/24/48/56 ladder keyed on `A.rows()`** (`include/batchlas/tuning_params.hh:45-47`, `include/batchlas/blas/functions/ormqr.hh:219-227`) for every `ormqr` caller that passes no hint. Measured wrong for three of four types, costing 1.11–1.55x. `geqrf` and `orgqr` bypass it with their own type-keyed widths; nothing else does. Read the *source* header, not the generated one: the `configure_file` copy at `build/include/batchlas/tuning_params.hh` says **16/32/64/128/128** and is never compiled, because `src/CMakeLists.txt` puts `${PROJECT_SOURCE_DIR}/include` ahead of `${PROJECT_BINARY_DIR}/include`. The harness prints the width it actually used, and it prints 16/24/48/56.
10. **The CTA tier's workspace is zero but callers still pay the blocked layout.** The facade takes `max` over every *supported* native tier and `supports()` deliberately puts no lower extent bound on the Blocked arm, so a caller at n=64 batch=8192 pays 168 MB (float) / 671 MB (cdouble) even though the route it takes is CTA. Sizing W1/W2 on `n - nb` rather than `n` took ~28% off; the remainder is the deliberate `max` policy. (The vendor `orgqr` it replaces asks for 1164 MB / 4644 MB at that cell.)
11. **`potrf` had WP5's dispatch gap. CLOSED at P7** ([potrf.md#native_tier_preferred](potrf.md#native_tier_preferred)); the reading below is what the debt was. `potrf` carries the same two native tiers ({CTA, Blocked}) and the same all-false `preferred()`, so its vendor-free walk still returns the first *supported* native route from a static order array that cannot follow a crossover — the exact defect `native_tier_preferred` was added for. `getrf` and `getrs` have since declared the hook; `route_potrf.hh` has not. Nobody has measured whether `potrf`'s vendor-free tier choice is wrong, which is the first step, not the fix.
12. **`resolve_ormqr_route` is called with two arguments** (`ormqr.hh:209`), taking the `vendor_available = true` default, so `ormqr` never reaches `route_resolve.hh:38-49`'s vendor-free fallback. It gets away with it only because its `preferred()` is native-first. `geqrf` and `orgqr` pass the argument explicitly; do not inherit the omission.

## Raw evidence

Raw data is preserved at the git tag `perf-evidence/vendor-independence` and is retrievable with `git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| WP5 index, repair pass, the ten unsettled questions | `experiments/wp5_qr/README.md` |
| vendor baseline, saturation, orgqr-via-ormqr viability, block widths, trailing-GEMM routing | `experiments/wp5_qr/baseline/README.md` |
| block-width ladders, both builds, all four types | `experiments/wp5_qr/baseline/summary_nb.txt`, `summary_nb2.txt`, `nb.csv`, `nb2.csv` |
| vendor `geqrf` sweep and saturation ladders | `experiments/wp5_qr/baseline/sweep_raw.txt`, `sat.csv`, `sat2.csv`, `summary.txt` |
| trailing-GEMM pair on real sub-views; the 18-panel prediction | `experiments/wp5_qr/baseline/gemmtrail.csv`, `panelsum.csv`, `summary_gt.txt`, `summary_ps.txt` |
| resolved Route and KernelVariant, observed not reasoned | `experiments/wp5_qr/baseline/routeq_qr.csv`, `variants.csv` |
| reference breaks 1–5 and the `tau = 0` finding | `experiments/wp5_qr/baseline/break.csv`, `break2.txt` |
| kernel harness, control, 5 reference + 7 kernel breaks | `experiments/wp5_qr/kernels/README.md`, `breaks_ref.txt`, `breaks_kernel.txt`, `run_v.txt`, `run_nv.txt` |
| the measured grid against cuSOLVER/cuBLAS; method and discard rule | `experiments/wp5_qr/bench/README.md` |
| order / batch / tall / orgqr-batch sweeps | `experiments/wp5_qr/bench/order.csv`, `batch.csv`, `tall.csv`, `orgqr_batch.csv`, `*_summary.txt` |
| the CTA-vs-blocked tier sweep, with the excluded pins | `experiments/wp5_qr/bench/tier.csv`, `tier_summary.txt` |
| nsys kernel splits, winning and losing cells | `experiments/wp5_qr/bench/nsys_split.md`, `kernsum/*_kern.txt` |
| ormqr WY trmm-vs-gemm gate | `experiments/TRMM_SYRK_BATCHED_KERNELS.md` |
| campaign narrative, "WP5 has landed" | `VENDOR_INDEPENDENCE_PLAN.md` |

## The shipped `geqrf` window

What the per-type order floor in `preferred()` (`route_geqrf.hh`) resolves to at the one shape
both test files probe, why a window in `preferred()` can silently overrule the tier hook, and
how much of that the integration-level test can actually see.

### What 96x96 resolves to, per type

96 is simultaneously **double's floor**, **float's last CTA order** and **below cdouble's
floor**, which is why it is the probe shape in `tests/geqrf_tests.cc` G9 and in the pure-layer
route tests. `preferred()` carries a PER-TYPE order floor, so the vendor-present answer at
96x96 is a per-type fact and **not a blanket "vendor"**. Ratios are native-over-vendor from
[`small-n-baseline.md#geqrf`](small-n-baseline.md#geqrf), n = 96 row, batch 8192.

| T | 96x96 resolves to | ratio | why this cell |
|---|---|---|---|
| float | `native:cta` | **2.69x** | 96 is the LAST float order on CTA |
| cfloat | `native:cta` | **2.28x** | no measured CTA/blocked crossover, and 9,216 scalars fit the 12,160 cfloat tile |
| double | `native:blocked` | **1.16x** | double's CTA crossover is n > 48 |
| cdouble | **VENDOR** | 0.49x | the cdouble floor is n >= 256 |

cdouble 96x96 is 0.49x native and BELOW the cdouble floor of 256; routing it native ships a
measured loss. The tile-fit arithmetic for cfloat is the CTA capacity from
[cta-capacity](#cta-capacity): 96*96 = 9,216 scalars against a cfloat tile of 12,160 elems
(square n = 110).

The tier half of the answer is **the crossover AND the tile fit, both read from the device**:
on a device whose tile cannot hold 96x96 the blocked arm is the right answer for every type,
so nothing here is hardcoded to this box.

### A non-empty `preferred()` pre-empts `native_tier_preferred`

`resolve_route`'s `automatic()` walks `kGeqrfOrder` testing `supports(r) && preferred(r)` and
**RETURNS on the first hit** (`route_resolve.hh:35-37`), before `native_tier_preferred` is
consulted at all. CTA leads that order, so a window that answers true for CTA hands it every
shape it can hold, whatever the tier hook says.

Measured cost of getting this wrong: the window shipped `native:cta` at **double n = 96**,
where **blocked is 1.37x faster** and the resulting route is **0.848x** the vendor — a loss
against the arm it replaced. See the second sweep in
[cta-vs-blocked-crossover](#cta-vs-blocked-crossover): double 96, default 23.86 ms (blocked)
against 32.75 ms (cta).

**The invariant that guards it.** Before the order floor shipped, `tests/geqrf_tests.cc` G9b
block (3) asserted `is_vendor(...)`, on the reasoning that `native_tier_preferred` is consulted
only on the vendor-free walk while `preferred()` is consulted always — true then, because
`preferred()` was all-false, so the vendor-present answer could not be native at all. With a
window in place the invariant it was really guarding survives, and is stronger: **whichever arm
the vendor-present walk lands on, it must be the SAME arm the vendor-free tie-break picks.**
That is what "the tier hook did not leak" means once native can win, and it is what catches the
pre-emption defect above. Concretely: `preferred()` must answer true for exactly one tier, and
the resolved `present.algo` must equal `RouteTable::best_native_tier(shape).algo`.

### How the window is guarded

The integration-level check (G9b block (3), `NativeTierTieBreakPicksTheFasterNativeVendorFree`)
is **one instantiation deep**. Its probe shapes are chosen for the TIER crossover, not for the
order window, so only a type whose probe shapes land INSIDE its order window can fire the
assertion. Today that is **float only** — double probes at n = 48 and 64, both under its floor
of 96, so the check takes the vendor early-exit and asserts nothing. **Verified by planting the
defect: only `GeqrfTest/4` went red**, and `GeqrfTest/5` passed with the defect in place.

The per-type tier cover therefore lives in the pure layer, where shapes are free:
`RouteGeqrf.PreferredIsTheMeasuredOrderFloorAndTheTallClause` pins **float 96 -> CTA, float 128
-> Blocked, double 96 -> Blocked and cfloat 256 -> Blocked**.

## The shipped `orgqr` ceiling

Why `route_orgqr.hh`'s `preferred()` is `rows <= 512 && cols <= 512`, every type. Relocated
verbatim from that predicate's comment block; the grid it summarises is
[`small-n-baseline.md`](small-n-baseline.md#orgqr), and the losing cells that bracket it are in
[`orgqr` grid](#orgqr-grid) above.

**Read every ratio here as "beats the per-item loop", NOT as "beats cuSOLVER".** `cublas.cc`
dispatches `cusolverDnXorgqr` once PER BATCH ITEM on an out-of-order sub-queue
(`cublas.cc:1414-1419`), so the vendor arm is `batch` launches deep and the comparison is
against a structure, not against a kernel.

**That is exactly why the window has no floor.** The measured margin is smallest at the largest
order and never approaches the flip gate:

| T | n = 512 | n = 256 | n = 64 | n <= 16 |
|---|---|---|---|---|
| float | 3.64 | 5.30 | 27.10 | >= 181 |
| cfloat | 2.54 | 4.04 | 15.09 | >= 172 |
| double | 4.61 | 8.46 | 27.09 | **>= 97.72** |
| cdouble | 2.77 | 4.75 | 11.31 | >= 42 |

66 square cells measured over four types and orders 4..512, zero losses, minimum 2.54.

The `double` floor is the one number reconciled on the move. The predicate comment carried it as
`>= 98`, which overreaches its own measurement: the kept cell is **97.72**
(`benchmarks/results/factor_baseline_orgqr_double.csv`, `double 16x16 b32768`, vendor 1642.61 ms
against native 16.8097 ms), the same figure the small-n grid's `16 (b32768)` row records. The
other three floors sit below their own `n = 16` `b32768` cells — float 209.57, cfloat 172.40,
cdouble 42.25 — and so do not overreach.

**The 512 ceiling is load-bearing, and it is the bracket.** There is no losing cell inside the
measured range, so the bound comes from the cells ABOVE it, which [`orgqr` grid](#orgqr-grid)
records as losses: cfloat n = 1024 is 0.82x (0.88 at batch 256, and the record does not claim it
crosses), cdouble n = 1024 is 0.78x, and at n = 2048 every type loses — float 0.41x, cfloat
0.31x, cdouble 0.46x. An unbounded `is_native(r)` would route all of those native. Only float
n = 1024 recovers with batch (0.84 / 1.11 / 1.27 / 1.33 at batch 32..256), and one type crossing
is not a window.

**Workspace, which the window does not weigh.** The vendor arm also costs 3.3x the workspace —
4,870 MB against 1,476 MB at cdouble n = 64, batch 8192 — which a caller near the memory ceiling
will feel. That advantage reverses above n = 1024 in the same direction the speed advantage
does; the reversal cells are in [`orgqr` grid](#orgqr-grid).

## CTA capacity: the `INT_MAX` reading

Relocated from `tests/route_vocabulary_tests.cc` (the shape helper above `geqrf_cta_elems`),
which is where the consequence for a *test* shape was written down. The budget and the
per-type element counts are the same ones as [cta-capacity](#cta-capacity); what follows is
the device reading behind them and why a permissive test shape makes half an assertion
disappear.

**The capacity this box actually reports is NOT the square side.** The shape builder sets both
`cta_max_m` and `cta_max_elems` from `geqrf_cta_max_*_for_slm`, and `..._max_m_for_slm` returns
`min(elems, INT_MAX)` (`geqrf_cta.cc:176-179`) — so `cta_max_m == cta_max_elems` and the
**area** is the only binding bound. Budget `101,376 - 4,096 = 97,280 B`: float 24,320 elems,
double and cfloat 12,160, cdouble 6,080.

**A permissive capacity makes the tier half of a route test vacuous for the complex types.**
The permissive `4096 / 1<<24` pair used for the other ops' shapes in that file would make
EVERY cell CTA-eligible. For float and double that still leaves something to assert, because
their `native_tier_preferred` carries a real column cap (96 and 48, see
[the-third-predicate](#the-third-predicate)). For cfloat and cdouble it does not: their
`native_tier_preferred` returns a `1 << 30` column cap, so **the fit gate is the ONLY thing
that ever sends cfloat or cdouble to the blocked arm**. A test that hands them an unbounded
tile can never observe `Algorithm::Blocked`, and the tier column of its table asserts nothing.
Hence `geqrf_dev_shape<T>()` reports this box's real budget rather than the permissive shape.

## The `geqrf` order floor and the tall-panel clause

The two windows `route_geqrf.hh`'s `preferred()` ships, and every cell that bounds them. The
per-cell grid both tables are derived from is [`small-n-baseline.md#geqrf`](small-n-baseline.md#geqrf)
and is **not** repeated here; what is here is the selection — which cell became an edge, and
which measured cell was refused as one.

### Why the floors sit where they do

cuBLAS `geqrfBatched` is unblocked and saturates at ~380 GFLOP/s (float) **regardless of n**
(see [the-vendor-baseline](#the-vendor-baseline)), so the native arm pulls away as n grows.
Below the floor the reverse holds, because the native panel kernel is the whole cost at a size
where there is no trailing work to amortise it.

The floors are the first order clearing the repository's flip gate, `t_native <= 0.90 t_vendor`
(ratio >= 1.11), at the top of the measured batch ladder, with the order below it measured as a
loss:

| T | floor | ratio at floor | bracketing loss below |
|---|---|---|---|
| `float` | 64 | 1.71 | 48: 1.02   33: 0.76 |
| `cfloat` | 48 | 1.74 | 33: 0.69   32: 0.62 |
| `double` | 96 | 1.16 | 65: 0.66   64: 0.58 |
| `cdouble` | 256 | 1.50 | 192: 1.06   129: 0.58 |

**float n = 48 (1.02) and cdouble n = 192 (1.06) are inside the gate's dead band and are
deliberately excluded**: both are single cells that do not clear 1.11, and a window edge without
a clearing measurement is a guess.

### Tall panels cross over earlier

They are also the shape the callers actually issue: `sytrd_sy2sb.cc:509` and
`band_reduction.cc:603` factorise an `m x kd` panel with `kd` typically 32, where every square
cell loses. A floor on `cols()` alone leaves that shape on the vendor while the measurement says
native is 1.6-6.4x there. Ratios are native-over-vendor at the top of the batch ladder — b16384
for the `512x*` and `128x32` rows, b8192 for `1024x128`:

| shape | aspect | float | cfloat | double | cdouble |
|---|---|---|---|---|---|
| 128 x 32 | 4x | 2.23 | 3.79 | 0.68 | 0.68 |
| 512 x 32 | 16x | 2.68 | 3.39 | 1.58 | 2.16 |
| 512 x 64 | 8x | 3.175 | 4.102 | 2.373 | 1.734 |
| 1024 x 128 | 8x | 7.16 | 5.090 | 6.418 | *excluded* (1.691 at b8192, paging; 2.96 at b4096) |

Hence the second clause, on the panel's **aspect ratio**, and the ratio is per type. `128 x 32`
is the whole reason: at 4x the 32-bit types win 2.23-3.79 and the 64-bit types **lose** at 0.68,
so a type-independent aspect floor of 4 would route two measured losses native. The 64-bit floor
is **8x**, bracketed on both sides — `1024 x 128` (8x) wins 6.418 for double and `512 x 64` (8x)
wins 2.373 / 1.734 for double / cdouble, while `128 x 32` (4x) loses 0.68 for both.

**Two edges are unbracketed and are debts, not evidence.** The 32-bit aspect floor of 4x is not
bracketed below: no tall cell narrower than 4x was measured for any type, so 4 is the smallest
measured aspect and not a demonstrated boundary. Neither is `rows() >= 128`, for the same reason
— `128 x 32` is simply the shortest tall panel in the grid.

**Reconciliation on the move out of the code.** The table above is re-derived from
`benchmarks/results/factor_baseline_geqrf_*.csv`, and four cells of the version that shipped in
the header were wrong: cfloat `512x64` read 3.70 against a measured 4.102, cfloat `1024x128`
read 5.07 against 5.090, float `512x64` read 3.17 against 3.175, and double / cdouble were shown
as `-` at `512x64` and `1024x128` although all four of those cells are measured (2.373 / 1.734
and 6.418 / 1.691). The corrections all move in the argument's own direction: cdouble at 8x wins
1.734 at `512x64`, which the header's `-` had left the 64-bit aspect floor resting on double
alone. The one cell that is genuinely not usable is cdouble `1024x128` b8192 — 32 GB, paging,
1.691 against 2.130 and 2.956 on the two rungs below it, excluded on
[`small-n-baseline.md#geqrf`](small-n-baseline.md#geqrf)'s own oversubscription rule.

### Why the window answers for exactly one tier

**The tier hook must not be pre-empted.** `resolve_route`'s `automatic()` walks `kGeqrfOrder`
testing `supports(r) && preferred(r)` and RETURNS on the first hit, before
`native_tier_preferred` is consulted at all. CTA leads that order, so a window that answers true
for CTA hands it every shape it can hold, whatever the tier hook says. Measured cost of getting
this wrong, at the double floor (n = 96): **CTA 65.19 ms against blocked 47.70 and vendor 55.27,
i.e. the window shipped 0.848x — a loss against the arm it replaced** — while float n = 128 took
CTA's 2.12x instead of blocked's 3.62x. `tests/geqrf_tests.cc` G9b exists for exactly this and
caught it. The mechanism and the invariant that guards it are in
[a-non-empty-preferred-pre-empts-native_tier_preferred](#a-non-empty-preferred-pre-empts-native_tier_preferred);
the numbers above are the ones that were in the header.

Two figures on that line disagree with other records and neither changes the conclusion. Commit
`f4e63fb`'s own message gives the vendor time as 55.18, not 55.27; **55.27 is the one consistent
with the 0.848x it also quotes** (55.18 / 65.19 = 0.846), so 55.27 is kept. The independent pin
at float n = 128 in [`small-n-baseline.md`](small-n-baseline.md#where-this-baseline-disagrees-with-the-record)
reads `cta` 15.419 ms / `blocked` 9.349 ms / vendor 31.024 ms, i.e. 2.01x against 3.32x rather
than 2.12x against 3.62x — a different session and batch, same direction, same size of mistake.

**Why `best_native_tier(s)` and not `native_tier_preferred(r, s)` composed directly.** For the
complex types that hook returns a `1 << 30` column cap, meaning "CTA wherever it FITS" (see
[the-third-predicate](#the-third-predicate)), so its **Blocked arm is false at every real
order**. Composing it directly would answer false for *both* tiers at, say, cfloat 256x256 —
where CTA cannot hold the tile (65,536 scalars against a cfloat capacity of 12,160) and Blocked
measures **7.51x** — and hand a large measured win back to the vendor. `best_native_tier`
resolves the fit first and consults the hook only among the tiers that can serve.

---

## The occupancy rule

**P7, 2026-09-10.** `geqrf_cta_max_elems_for_slm<T>(budget, min_blocks_per_sm)` divides
the device budget by an occupancy target before converting it to an element count. The
hole clamp is applied **after** the division: a scaled budget can land inside the 48 KB
band even when the whole one did not.

### Why geqrf's target is 2 and not 4

`resident::kMinBlocksPerSm` is 4 and potrf and getrf use it. geqrf uses 2
(`kGeqrfMinBlocksPerSm`), and the reason is a general one worth stating plainly: **the
occupancy rule answers "how many work-groups fit", not "is the better-occupancy arm
faster".** For this op the alternative to the CTA tier is not a leaner kernel — it is the
blocked driver, which runs *the same resident panel* plus `larft` and three trailing
GEMMs. An occupancy-limited kernel is still the best kernel when the alternative is a
multi-launch driver.

At target 4 the advertised area falls below the CTA-vs-Blocked crossover
`native_tier_preferred` already ships (96 columns for float), i.e. the capacity would
refuse shapes the measured tier table calls faster. Measured, native arm, large batch:

| type | m x n | batch | before | at target 4 | ratio |
|---|---|---|---|---|---|
| cfloat | 64 x 64 | 8192 | 4.9460 | 9.0565 | **1.831 SLOWER** |
| float | 96 x 96 | 4096 | 4.9748 | 5.6843 | **1.143 SLOWER** |

Both are routed cells — geqrf's `preferred()` window is non-empty for float and cfloat —
so those were user-visible losses, not vendor-free-only ones. At target 2 both return to
baseline (cfloat 64: 4.9470, ratio 1.000; float 96: 4.9698, ratio 0.999) and the ceiling
lands at 108 square for float, inside the plan's measured 96..112 crossover window.

At this box's 97,280 B budget, target 2 (note 48,640 falls inside the launch-hole band and
is clamped to 47,104, which is where the odd-looking figures come from):

| type | advertised elems (target 2) | square | resident elems (target 1) | square |
|---|---|---|---|---|
| float | **11,776** | 108 | 24,320 | 155 |
| double | **5,888** | 76 | 12,160 | 110 |
| complex\<float\> | **5,888** | 76 | 12,160 | 110 |
| complex\<double\> | **2,944** | 54 | 6,080 | 77 |

Both `cta_max_m` and `cta_max_elems` survive, and the capacity stays an AREA: collapsing
it to a scalar order makes the tier column of the geqrf and route-vocabulary tests vacuous
for cfloat and cdouble, whose `native_tier_preferred` is "CTA wherever it fits".

### Settling the target on the tall panels

**P7 follow-up, 2026-09-11.** The paragraph above was argued from square cells only. The
review's objection was that the clamp moves the *tall* panels — and those are the shape
the eigen drivers issue on every call (`sytrd_sy2sb.cc:509`, `ortho.cc`, `band_reduction.cc:603`
hand `geqrf` an `m x nb` panel with `nb` 32..64). A host probe against the shipped table
confirmed the routing move: at target 2 the default `Auto` route goes `native:cta ->
native:blocked` for float `512x32`, `384x32`, `256x64`, for cfloat `256x32`, and for
double `256x32`. **Measured, the move costs nothing.**

Raw rows: [`benchmarks/results/p7_occupancy_geqrf_tall.csv`](../../benchmarks/results/p7_occupancy_geqrf_tall.csv)
(144 rows, 9-11 reps, interleaved, `rel_sd <= 0.05`, residual and `||Q^H Q - I||` checked
on items 0 and batch-1 in double promotion). Ratio is `blocked_ms / cta_ms`, so **> 1
means CTA ahead**, matching [cta-vs-blocked-crossover](#cta-vs-blocked-crossover).

| type | m x n | elems | batch | vendor | CTA | Blocked | blk/cta |
|---|---|---|---|---|---|---|---|
| float | 512 x 32 | 16,384 | 8192 | 21.8490 | 8.2560 | 8.2622 | 1.001 |
| float | 384 x 32 | 12,288 | 8192 | 15.6254 | 4.0916 | 4.0982 | 1.002 |
| float | 256 x 32 | 8,192 | 8192 | 9.4998 | 2.3654 | 2.3749 | 1.004 |
| float | 256 x 64 | 16,384 | 8192 | 40.3957 | 15.0023 | 10.5431 | **0.703** |
| cfloat | 256 x 32 | 8,192 | 8192 | 20.2912 | 6.7715 | 6.7776 | 1.001 |
| double | 256 x 32 | 8,192 | 8192 | 21.6106 | 20.6838 | 19.3423 | **0.935** |
| float | 1024 x 128 | 131,072 | 4096 | 366.7172 | *never CTA-eligible* | 63.9487 | — |

Three dead ties and two Blocked wins; **CTA wins no cell the clamp moves.** `1024x128` is
the control and did not move: 131,072 elements is above the resident ceiling at every
target, so it is Blocked at 1, 2 and 4 alike.

**Why the ties are ties, and it is not noise.** `geqrf_blocked`'s block width on this box
is **nb = 32** (nb = 16 for double: `geqrf_blocked_debug_params`). At `n <= nb` the blocked
route factorises the whole panel in **one leaf call** — the same `geqr2_panel_device`
launch the CTA tier makes, taken through `geqrf_leaf_fits` at the *whole* budget, so it
keeps residency. `cta -> blocked` at `512x32` / `384x32` / `256x32` / `320x32` / `192x32`
is therefore not a kernel change at all; the 0.1-0.4% is the driver's host-side framing.
The residual and orthogonality columns are bit-equal between the two arms at every one of
those cells, which is the check that says so. At `n > nb` the driver splits — float
`256x64` becomes two `256x32` panels plus `larft` and a trailing GEMM — and that is
**faster** than the one-shot `256x64` resident panel, by 1.42x, stable across the batch
ladder (b4096 0.729, b8192 0.703, b16384 0.690). Double `256x32` splits into two `256x16`
panels and wins 1.07x (b4096 0.940, b8192 0.935, b16384 0.933; the cell reproduces to
three digits on a re-measure).

**Target 4 is refuted, with fresh numbers.** The two cells quoted in the table above
re-measure at 5.7432 / 4.9769 for float `96x96` (**1.154 CTA ahead**) and 9.0424 / 4.9606
for cfloat `64x64` (**1.823 CTA ahead**), against the 1.143 and 1.831 recorded at P7. Both
are routed cells. Raising geqrf to its siblings' 4 would hand both to the blocked driver.

**Decision: `kGeqrfMinBlocksPerSm` stays 2.** Neither the constant nor the tall-panel
clause moves. `GeqrfTest.OccupancyTargetIsPinned` fails if the value
moves, so the constant cannot drift back to its siblings' 4 unremarked. The tall clause in particular needs no re-cut: at every `m x nb` panel in the
grid with `nb` in 32..64, the arm the table routes to is never slower than the arm it
passed over.

#### What the clamp does cost, and where the real cut belongs

The bracketing is not one-sided and the debt is recorded rather than argued away. A single
scalar occupancy target is a proxy for a **per-type crossover area**, and it lands 4-28%
below the measured one. Ladders at fixed n, `blocked_ms / cta_ms`:

| type | n | m (elems) → ratio | crossover bracket |
|---|---|---|---|
| float | 64 | 128 (8,192) 1.551 · 192 (12,288) 1.231 · **224 (14,336) 0.706** · 256 (16,384) 0.703 | 12,288 .. 14,336 |
| float | 96 | 96 (9,216) 1.154 · 128 (12,288) 1.191 · **160 (15,360) 0.672** · 192 (18,432) 0.690 | 12,288 .. 15,360 |
| cfloat | 48/64/96 | 64x64 (4,096) 1.823 · 128x48 (6,144) 1.925 · **128x64 (8,192) 0.913** · 96x96 (9,216) 0.924 · 160x64 (10,240) 0.957 | 6,144 .. 8,192 |
| double | 32 | 192 (6,144) 1.043 · 224 (7,168) 0.949 · 256 (8,192) 0.935 · 288 (9,216) 1.148 · 320 (10,240) 1.139 · 352 (11,264) 1.128 | **non-monotone** |

float `n = 32`, cfloat `n = 32` and cdouble `n = 32` are absent from this table on purpose:
`n <= nb` makes both arms the same launch, so they are ties by construction and bracket
nothing. **double at n = 32 is genuinely non-monotone** — a narrow Blocked dip at
m = 224..256 with CTA ahead on both sides — and every cell on that ladder is within 1.15x,
so it brackets nothing either. It is quoted because it was re-measured at 11 reps
specifically to check the dip, and the dip reproduced.

Cost of the shipped target at **routed** cells it cuts below the crossover: float `128x96`
1.191, float `192x64` 1.231, cfloat `128x48` **1.925**, double `192..352 x 32` 1.04-1.15.
Cost of target **1** at routed cells: float `224x64` 1.417, `256x64` 1.423, `192x96` 1.450,
`160x96` 1.487, double `224..256 x 32` 1.07. Cost of target **4**: everything target 2
costs, plus float `96x96` 1.154 and cfloat `64x64` 1.823. **4 is strictly worse than 2; 1
and 2 trade cells of comparable size in opposite directions**, and 2 is the one that is
neutral-or-better on the driver shapes, which is why it stays.

**The debt.** The largest single cell above is cfloat `128x48` at 1.925, and it is on the
wrong side of a ceiling that misses by 4% (5,888 advertised against a crossover above
6,144). Chasing it by moving the occupancy target is the wrong lever: the target scales
every type's ceiling by the same factor, and the four crossovers are not in that ratio. The
right cut is a measured **per-type area** in `native_tier_preferred`, where a speed cutoff
belongs — `route_geqrf.hh`'s own header says `supports()` is correctness only, and
`GeqrfTest`'s tie-break test already asserts that neither arm may lose `supports()` across
the crossover for exactly this reason. That is a routing change with its own grid (double
is not settleable from the six cells here) and is **not** taken as part of this settlement.

### The `geqrf_blocked_debug_params` key

`geqrf_blocked_debug_params<T>(ctx, m, n)` packs **nb in the low 16 bits and the leading
panel's leaf in the high 16** (`1` = resident, `2` = global); the whole word is **0** when
the driver is absent. It answers `geqrf_leaf_fits` at the whole SLM budget, not
`geqrf_cta_fits` at the occupancy-scaled one -- see [the panel leaf is not the tier
ceiling](#the-panel-leaf-is-not-the-tier-ceiling). `tests/geqrf_tests.cc` compares the
high half against bare `1u`/`2u`, so this is the only definition of those two values in
the tree. The `getrf` hook uses the same encoding; its record is at
[lu.md](lu.md#one-spelling-per-ceiling).

### The panel leaf is not the tier ceiling

`geqrf_cta_fits` is the TIER's predicate and is occupancy-scaled; `geqrf_leaf_fits` is the
residency one and is asked at the whole budget. `geqrf_panel_factorize` and
`geqrf_blocked.cc`'s leading-panel tag use the latter, so no blocked panel loses residency
to an occupancy target. `GeqrfTest.ResidentLeafLaunchHoleAt48KiB` moved to the leaf
predicate and the leaf entry point for the same reason: a 48 KB tile is far above the
occupancy-scaled tier ceiling, so under the new rule only the resident leaf can reach the
hole at all. It must still be the FIRST test in its file.

### Packed resident panels

`geqr2_panel_device` is templated on `GeqrfScope`. Under `SubGroup` the phase barriers are
sub-group barriers, `team`/`nteams` collapse to 0/1, and — the part that matters here —
both `reduce_over_group` calls become butterflies (`geqrf_sg_max`, `geqrf_sg_sum`). That
removes the static shared allocation that makes this kernel, alone among the three,
actually trip the 48 KB hole. Packing is offered while `m <= 32 && n <= 32`.

| type | m x n | batch | vendor | before | after | ratio |
|---|---|---|---|---|---|---|
| float | 8 x 8 | 32768 | 0.0613 | 0.7287 | 0.1466 | **0.201** |
| float | 16 x 16 | 32768 | 0.3501 | 1.6156 | 0.4700 | **0.291** |
| float | 32 x 32 | 16384 | 1.4448 | 1.9716 | 1.2037 | **0.611** |
| double | 32 x 32 | 16384 | 4.5973 | 21.6526 | 6.9274 | **0.320** |
| cfloat | 32 x 32 | 16384 | 1.6531 | 2.6749 | 2.2372 | **0.836** |

1.2-5.0x. The largest factors are float 8x8 (4.97x) and double 32x32 (3.13x), which is
consistent with the mechanism: those are the shapes where the old launcher gave a
64-element column a reduction over 256 lanes, through the work-group collective design
rule R4 already recorded as 1.5-4.7x slower for FP64.

### The one cell this cost

**cdouble 64 x 64, batch 8192, native arm: 59.1719 -> 98.4222 ms, 1.66x SLOWER.** 4,096
elements is above cdouble's advertised 2,944 and fits only at target 1 (65,536 B, 1.48
work-groups per SM), so the shape leaves CTA for the blocked driver. cuBLAS serves it in
31.4838 ms — native loses to the vendor either way — and cdouble's `preferred()` floor is
256 columns, so `Auto` never routed this cell native. Visible only in a vendor-free build
or under a forced route. Per design rule R10 double is measured and reported, not gated.

## The tiny tier (WP6 / P1): square n <= 32 in registers

`src/extensions/geqrf_tiny.cc`, `Algorithm::Tiny`, first in `kGeqrfOrder`. One matrix per
`SubGroupPartition<N>` with `N` in {8, 16, 32}; lane `r` owns row `r` in a compile-time
`D rA[N]`. **Two** sub-groups per work-group -- `tiny_device.hh`'s shared `kTinyWgSize`,
64 work-items -- carrying `64 / N` matrices. Reachable only by an explicit `{Native, Tiny}`
pin (`BATCHLAS_GEQRF_ROUTE=tiny`) or by forcing: `preferred()` and
`native_tier_preferred()` both answer **false** for the tier.

**NO TIMING HAS BEEN TAKEN.** The implementation PR is correctness and residency only.
Nothing on this page is a speed claim about the tier, and its window stays closed until a
grid exists.

### The tiny ceiling predicate

`geqrf_tiny_max_n_for_slm<T>(budget)` is the **one** predicate for the tier ceiling — the
shape builder, the entry point and the tests all call it, and forking it would let
`supports()` advertise an order the launcher refuses. It returns 0 when no bucket fits; the
type ceiling is **32**, and **16 for `complex<double>`** ([the cdouble N=32
cell](#the-cdouble-n32-cell)). The value is **not monotone in the bucket width** — the
chunk rule halves the tile as N grows ([the chunk
width](#the-chunk-width-and-the-one-cell-that-is-pinned)) — so it is a walk with a `break`,
which is what makes the range it names contiguous.

#### Why the Tiny arm is spelled out

`native_tier_preferred`'s `default:` returns **true**, and Tiny leads `kGeqrfOrder`. A
default-true answer would therefore hand the tier the vendor-free walk
(`route_resolve.hh:40-50`) and every bare `BATCHLAS_GEQRF_ROUTE=native` pin
(`route_resolve.hh:78-86`) before a single cell had been measured. The explicit `false`
leaves it reachable by an explicit `{Native, Tiny}` pin, which `route_resolve.hh` honours
regardless of this hook -- all a benchmark arm needs. Flipping it to true belongs in the
SAME commit that widens `preferred()`'s window from the grid.

The `supports()` gate is square-only (`m == n && n <= tiny_max_n`): the register array IS
the matrix, so a tall panel is the CTA tier's, and `tiny_max_n` is an ORDER, not an area --
the footprint is a work-group's, not a matrix's.

**What it replaces.** The CTA tier holds the tile in local memory and, for each
reflector, runs a separate 5-step butterfly per TRAILING COLUMN (`geqrf_cta_device.hh`'s
apply): at n = 32 that is **496 dependent shuffle chains per matrix**, each reducing a
column that has exactly one element per lane, plus a re-read of the trailing columns from
local memory on every reflector. The tiny kernel reduces ONCE per column -- the
Householder scalars, replicated to every lane by an order-symmetric butterfly, so no
publish/broadcast pair is needed -- and forms the whole trailing update as an elementwise
product into one local tile whose columns are then summed. The matrix never leaves the
register file between the load and the store.

### The tiny tier register table

`scripts/register_probe.sh out.log '' batchlas_extensions_cta` on sm_89, at the shipped
work-group width (64) and chunk widths. **The third argument is load-bearing**: the
script's default target is `batchlas_sycl`, which contains none of these kernels, and a
clean report from the wrong library reads exactly like a clean report from the right one.
Each kernel appears twice (`<name>` and `<name>_with_offset`); the two agreed in every
cell.

`blocks/SM = min( 65536 / (ceil8(regs) * 64), 1536 / 64, 24 )`; occupancy is
`blocks * 64 / 1536`.

| T | N | C | registers | ceil8 | regs x 64 | blocks/SM | occupancy | frame | spill |
|---|---|---|---|---|---|---|---|---|---|
| float | 8 | 8 | 64 | 64 | 4,096 | 16 | 66.7% | 0 | 0 / 0 |
| float | 16 | 16 | 91 | 96 | 5,824 | 10 | 41.7% | 0 | 0 / 0 |
| float | 32 | 32 | 142 | 144 | 9,088 | 7 | 29.2% | 0 | 0 / 0 |
| double | 8 | 8 | 84 | 88 | 5,376 | 11 | 45.8% | 0 | 0 / 0 |
| double | 16 | 16 | 124 | 128 | 7,936 | 8 | 33.3% | 0 | 0 / 0 |
| double | 32 | 16 | 193 | 200 | 12,352 | 5 | 20.8% | 0 | 0 / 0 |
| cfloat | 8 | 8 | 72 | 72 | 4,608 | 14 | 58.3% | 0 | 0 / 0 |
| cfloat | 16 | 16 | 128 | 128 | 8,192 | 8 | 33.3% | 0 | 0 / 0 |
| cfloat | 32 | 16 | 168 | 168 | 10,752 | 6 | 25.0% | 0 | 0 / 0 |
| cdouble | 8 | 8 | 110 | 112 | 7,040 | 9 | 37.5% | 0 | 0 / 0 |
| cdouble | 16 | 8 | 162 | 168 | 10,368 | 6 | 25.0% | 0 | 0 / 0 |

Eleven instantiations, every one at zero spill and zero stack frame, every one at or above
R1's four-blocks-per-SM floor. `ptxas` reports **0 bytes of static shared** for every
`GeqrfTinyKernel` (no `bytes smem` field appears on any of the 22 entry lines), which is
what keeps the (47,104, 49,664] launch hole structurally unreachable for this TU; the
dynamic local request is 2,560-8,960 B per work-group, so none of `geqrf_cta.cc`'s
hole-padding machinery is carried here.

The table lives in `geqrf_tiny_device.hh` as `GeqrfTinyRegs<D>`, keyed on the **device**
scalar, and it is the single source of truth: the chunk-width rule reads it, the hard
`regs x wg <= 65536` gate is derived from it (`max cell + 8`), and the R1 residency gate
is a `static_assert` per cell.

The register MODEL -- `R(N, words) = N*words + 88`, calibrated on `trsm_native.cc`'s
measured table -- survives only as the fallback for a cell with no probe row. It is wrong
in **both** directions: it over-predicts at N = 8 and 16 (96 against a measured 64 for
float N=8) and under-predicts at N = 32 (120 against 142 for float) -- roughly **-30% at
N = 8 and +25% at N = 32**. Letting it choose the chunk width while the probe gated the
launch was two sources of truth for one number, and that is what the move fixes.

`kGeqrfTinyRegMargin = 8` is the headroom added to the worst cell before the hard
`regs x wg <= 65536` gate is derived from it. The hard gate has enormous slack at this
tier's width: at a 64-wide work-group it permits **1,024 registers per work-item**, so it
can never fire, and the R1 four-blocks-per-SM `static_assert` per cell is the gate that
actually binds.

`ptxas` allocates registers in banks of eight, which is why every residency formula on
this page rounds first: at 92 registers a block costs **96** per work-item, and a rule
that reads 92 admits a block the hardware refuses.

### The launch shape: 64 work-items, not 128

An earlier revision ran four sub-groups (128 work-items) on the stated grounds that "a
wider work-group amortises the tile allocation over more matrices". It does not: the
`local_accessor` is sized `M * per_matrix`, exactly linear in `M` with no fixed term, so
there is nothing to amortise. The three effects that are real all point the other way --
register quantisation (`blocks/SM = 65536 / (ceil8(regs) * WG)`), local memory per
work-group, and the launch tail's wasted partitions.

Same probed register counts (registers are a property of the body, and no
`reqd_work_group_size` is declared, so no `.maxntid` lets ptxas trade against a launch
bound), two launch widths:

| T | N | blocks x 64 = threads | blocks x 128 = threads | occupancy 64 | occupancy 128 |
|---|---|---|---|---|---|
| float | 8 | 16 / 1,024 | 8 / 1,024 | 66.7% | 66.7% |
| float | 16 | 10 / 640 | 5 / 640 | 41.7% | 41.7% |
| float | 32 | 7 / 448 | 3 / 384 | **29.2%** | 25.0% |
| double | 8 | 11 / 704 | 5 / 640 | **45.8%** | 41.7% |
| double | 16 | 8 / 512 | 4 / 512 | 33.3% | 33.3% |
| double | 32 | 5 / 320 | 2 / 256 | **20.8%** | 16.7% |
| cfloat | 8 | 14 / 896 | 7 / 896 | 58.3% | 58.3% |
| cfloat | 16 | 8 / 512 | 4 / 512 | 33.3% | 33.3% |
| cfloat | 32 | 6 / 384 | 3 / 384 | 25.0% | 25.0% |
| cdouble | 8 | 9 / 576 | 4 / 512 | **37.5%** | 33.3% |
| cdouble | 16 | 6 / 384 | 3 / 384 | 25.0% | 25.0% |

Equal or better in all eleven cells, strictly better in four. It also halves bytes per
work-group, and re-uses `tiny_device.hh`'s shared `kTinyWgSize` instead of forking a second
geometry constant. `getrf_tiny.cc`'s four-sub-group rationale ("pure register work with no
local memory, so the width costs nothing but occupancy granularity") does not transfer:
geqrf is the one tiny arm that HAS local memory.

100% occupancy (<= 42 registers per work-item) is out of reach for every cell but float
N=8's neighbourhood, because `rA[N]` alone is `N*words` registers -- 32 for float N=32, 64
for double N=32, 128 for cdouble N=16 -- before any working set.

### The chunk width, and the one cell that is pinned

C is derived, not fixed: the largest of {N, N/2, N/4} at which
`blocks_by_slm >= blocks_by_regs`, both sides compile-time and the register side read from
the probe. Local memory per matrix is `N*(C+1) + C` scalars -- the product tile plus a
SEPARATE C-element `y` vector, which is exactly why two barriers per chunk cover all three
hazards including the inter-chunk WAR.

| T | N | C | matrices/WG | bytes/WG | blocks by SLM | blocks by regs |
|---|---|---|---|---|---|---|
| float | 8 | 8 | 8 | 2,560 | 38 | 16 |
| float | 16 | 16 | 4 | 4,608 | 21 | 10 |
| float | 32 | 32 | 2 | 8,704 | 11 | 7 |
| double | 8 | 8 | 8 | 5,120 | 19 | 11 |
| double | 16 | 16 | 4 | 9,216 | 10 | 8 |
| double | 32 | 16 | 2 | 8,960 | 10 | 5 |
| cfloat | 8 | 8 | 8 | 5,120 | 19 | 14 |
| cfloat | 16 | 16 | 4 | 9,216 | 10 | 8 |
| cfloat | 32 | 16 | 2 | 8,960 | 10 | 6 |
| cdouble | 8 | 8 | 8 | 10,240 | 9 | 9 |
| cdouble | 16 | 8 | 4 | 9,728 | 10 | 6 |

Local memory is never the binding residency limit at the shipped widths, which is what the
derived rule exists to guarantee.

**double N=32 is PINNED to C=16, and the re-probe is why.** Halving the work-group to 64
halves bytes/WG, which lets the derived rule offer that cell C=32 (SLM admits 5 blocks
against 5 by registers at the C=16 register count -- the tie the rule accepts). The
prediction was that the single-chunk body would be no worse than float N=32's, which was
clean. It was worse:

| double N=32 | registers | ceil8 | blocks/SM at WG=64 | barriers per matrix |
|---|---|---|---|---|
| C = 16 (shipped) | 193 | 200 | 5 | 124 |
| C = 32 (re-probe) | **218** | 224 | **4** | 62 |

Both are at zero frame and zero spill, so this is not the unroll defect below; the wider
chunk simply doubles the two elementwise phases and ptxas answers with 25 more registers.
Halving the barriers costs a resident block, and 4 blocks/SM is R1's floor rather than
comfortably above it, so the cell keeps the narrow chunk. `GeqrfTinyRegs<double>::pin[2]`
records it and a `static_assert` re-derives it, so the pin cannot silently rot.

**The tile's leading dimension is `C + 1`, and the odd stride is per scalar width.** The
tile's two hot accesses are a unit-stride row walk (phase 1 writes `sP[lane*SLDA + kk]`)
and an all-lanes-one-address broadcast (phase 3 reads `sy[kk]`), so the `+1` exists to
spread the N rows of a column over distinct banks:

| scalar width | phases per warp-wide access | banks touched by row r | covers 32 banks |
|---|---|---|---|
| 4 B | 1 x 32 lanes | `gcd(C+1, 32) = 1`, so r mod 32 | r = 0..31 |
| 8 B | 2 x 16 lanes | `2r mod 32` | r = 0..15 |
| 16 B | 4 x 8 lanes | `4r mod 32` | r = 0..7 |

Each row covers all 32 banks exactly once at its own width. The reference local-memory
budget the compile-time derivation is taken against is `kGeqrfTinyReferenceSlm = 97,280`
B, this box's per-SM figure; every RUNTIME decision reads `LOCAL_MEM_SIZE` from the
device through `resident::device_slm_budget` instead.

### The stack frame is the gate for this kernel, not the spill counter

`scripts/register_probe.sh`'s own header says to gate on `0 bytes spill stores / 0 bytes
spill loads` and never on stack frame, because 220 of 376 entry functions in this tree
carry a non-zero frame with zero spills. **For this kernel that advice inverts**, and the
first probe is why:

| T | N | C | first probe | after `#pragma clang loop unroll(full)` |
|---|---|---|---|---|
| float | 32 | 32 | 142 regs, 0 B frame, 0 spill | unchanged |
| double | 32 | 16 | 158 regs, **256 B frame**, 0 spill | 193 regs, 0 B frame, 0 spill |
| cfloat | 32 | 16 | 158 regs, **256 B frame**, 0 spill | 168 regs, 0 B frame, 0 spill |
| cdouble | 32 | 16 | 140 regs, **544 B frame**, 0 spill | not instantiated |

256 B is exactly `sizeof(double) * 32`, i.e. `rA[N]` itself. `#pragma unroll` is a HINT
that clang weighs against a code-size threshold; at N = 32 with C = N/2 the doubled
phase-2 body crosses it, the column loop survives, `rA[j]` becomes a **dynamic index**,
and the front end places the array in local memory. That is not a ptxas *spill*, so the
spill counter stayed at zero and the recommended gate reported the kernel healthy while
its entire thesis had been undone. `unroll(full)` on every loop whose index reaches `rA`
is the fix. The float N=32 cell was clean on the first probe (C = N there, one chunk per
column), which is exactly why probing the float row alone would have missed this.

The `-Wpass-failed=transform-warning` line `loop not unrolled` on the column loop is the
compile-time signal for the same defect, and it is emitted.

### The cdouble N=32 cell

Not instantiated: `geqrf_tiny_type_ceiling<complex<double>>()` is 16, and the N=32 arm
sits under `if constexpr` so no kernel is emitted. This is a **budget decision with a live
counter-example**. `trsm_native.cc` ships a `complex<double>` N=32 body with `x[32]` --
the same 128-register array -- at a measured 226 registers and zero spill on this
toolchain today, and the first probe above compiled this kernel's cdouble N=32 cell (at
`unroll(full)` it would need re-measuring, since 140 registers there came with a 544 B
frame). So the cell is reachable; it is left out to hold the instantiation count at 11.
It is also the largest unclaimed shape in the band: the P0 baseline has cuBLAS at
5,758.5 us for cdouble n=32 at b8192 against a 282.6 us DRAM roof, 20.4x above it, with
the CTA arm at 0.33x -- the worst vendor-relative cell in the whole grid. First extension
after v1, not a closed question.

### R7: the device link

`geqrf_tiny`'s 22 entry functions cost **19.4 s of the library's 182.9 s of `ptxas`**,
or 10.6% — inside R7's 15% budget for this tier alone. The AGGREGATE for all three tiny
tiers is over budget, and both the figure and the written justification R7 then requires
live in exactly one place:
`docs/perf/lu.md#r7-the-device-link-all-three-tiny-tiers-landed`. It is a property of
`batchlas_extensions_cta` as a whole, so it is not restated here.

### What the design gives up, and what to A/B first

* **Phase 2 wastes lanes.** Only `min(C, n-1-j)` of the N lanes are active while a chunk's
  columns are summed, so at C = N/2 at most half the partition works during part of the
  apply. That is now confined to three cells (double N=32, cfloat N=32, cdouble N=16) and
  to the last few columns of any cell. Splitting each column's sum across `N/w` lanes and
  combining with `log2` shuffles is the fix, and is deliberately not in v1.
* **Barriers per column are `2 * ceil((n-1-j)/C)`** -- exactly 2 wherever C = N (8 of the
  11 cells), and 4 falling to 2 in the three cells where C = N/2. The plan budgeted 4 per
  column everywhere; the saving comes from the butterfly norm letting every lane recompute
  the Householder scalars, not from the chunking.
* **A padded `ld` may split DRAM sectors when several matrices share a sub-group.** At
  N = 8 a sub-group's 32 lanes read four segments belonging to four different matrices,
  and at the fixtures' `ld = n + 5` those start unaligned. The `ld = n` and `ld = n + 5`
  rows at n = 8 and 16 are what would expose it; if it measures large the answer is a
  routing gate on `ld`, not a kernel change.
* **double and cdouble cannot reach the DRAM roof.** Ada runs FP64 at 1/64, so the apply's
  FP64-class warp instructions put double n=32 several times above its roof. Report those
  cells against the VENDOR only; R10 is doing real work here.

### Why bit-identity against the CTA route is not the padding test

The plan asked for "n = 5 inside N = 8 is bit-identical to the CTA route at n = 5". That
is unobtainable by construction: the CTA route reduces the column norm over 32 sub-group
lanes (`geqrf_sg_sum`, `geqrf_cta_device.hh:169-186`) and the tiny kernel over an N-lane
partition butterfly, so the association orders differ in the last ulp. The shipped guard
is stronger and obtainable -- NaN in the `ld` pad, in the stride pad and in every row and
column beyond `n`, then a finite correct factor AND an unchanged NaN pad -- and it arms
the store-guard break at the same time.

### Why the elementwise scan is the m x n window only

`G7`'s native-vs-vendor elementwise comparison in `tests/geqrf_tests.cc` scans the
`m x n` window of each item and nothing else. Widening it to the whole buffer is not
conservative -- it silently converts a RELATIVE bound into an absolute one.

`p.buf` is allocated with the poison fill `mk<T>(-9.75e3, 4.5e3)` and only the `m x n`
window is ever written, so a scan over the whole buffer takes `scale` from the `ld` and
stride padding -- magnitude **9.75e3** -- rather than from the factor, whose entries are
**O(1)**. That is roughly **1000x** looser than the tolerance reads, and it admitted a
float disagreement of about **0.07** on the very property the test exists to pin.

The padding cannot compensate in the other direction either: it is bit-identical in both
runs, so it contributes **0** to `worst` while inflating `scale`. The bound is therefore
one-sided in the wrong direction, which is why the narrow scan is NECESSARY and not
merely tighter.


### The fixture's tolerance floor, and why it is new

`residual_tol` / `orth_tol` in `tests/geqrf_tests.cc` were `0.5 (m + k) eps`. That term is
a stand-in for the backward-error bound's constant times m n, and it stops standing in
below about m + k = 16: at m = k = 1 it demands a relative residual of ONE eps, and at
m = k = 2 two. Measured, on correct output:

| shape | type | observed | old tolerance |
|---|---|---|---|
| 1x1 | cfloat | res 1.47e-07 (1.2 eps), orth 1.99e-07 (1.7 eps) | 1.19e-07 |
| 2x2 | float | res 3.10e-07 (2.6 eps), orth 3.56e-07 (3.0 eps) | 2.38e-07 |

A floor of 8 eps was added. It loosens NO shape this file tested before -- the smallest
was 16x16, i.e. m + k = 32 and a 16 eps tolerance -- because nothing here reached n < 8
until the tiny band. The evidence that the kernel and not the tolerance was at fault is
`TinyIsNoWorseThanTheCtaRouteAtTinyOrders`, which runs both tiers on identical data for
n = 1..12 and requires the tiny residual and orthogonality to stay within **2x** of the
CTA route's, or under the 8 eps floor.

**What "no worse" means there, exactly.** The two kernels differ only in the association
order of one column norm — 32 sub-group lanes against an N-lane partition butterfly —
which licenses an O(n eps) RELATIVE difference in the reflector, so the ratio must sit
near one rather than inside a free multiple. The **4x** this test first carried let the
register kernel lose two bits of the answer and still pass under a name claiming it loses
none; the slack is now **2.0**. The floor is the other half of the claim, and at these
orders it is the half that binds: both tiers land below 8 eps here, where a ratio says
more about the noise than about the kernel.

### Break sweeps: the tiny tier

Each break was planted, rebuilt, run and restored. Expected-vs-observed:

| # | break | expected | observed |
|---|---|---|---|
| b | store guard `k < N` instead of `k < n` (pass `N` to `tiny_store_full`) | the new pad-unchanged assertion in `check_one` goes red | **RED.** `tiny/solo: the kernel wrote outside its m x n window, at buffer offset 1 (item 0, column 1, row 0)` at n=1; then `TinyPackedBatchMatchesSolo` and `TinyPaddingIsInertUnderNaNPoison` too, and `tiny/zero-column` at 0.878 residual against 9.54e-07 |
| c | drop the `lane < j` term from the `vr` select, so eliminated rows re-enter the reflector | residual red at small n | **RED, all four types.** float `\|\|QR-A\|\|/\|\|A\|\|` = 0.2469 vs 9.54e-07 at 3x3; double 0.2469 vs 1.78e-15; cfloat and cdouble 0.4766 |
| d | restore `if (prob >= batch) return;` in place of the `live` predicate | garbage or a hang in the partial last work-group at N < 32 | **GREEN -- the break was NOT observed.** See below |
| e | drop the `sg_id *` term from `tiny_partition_id` | S-1 of every S partitions serve another matrix's slot | **RED, all four types.** `packed item 4 of 35 differs from its solo run at (0,0), n=5` |

**Break (d) did not fire, and that is the finding.** It was planted at batch = 35 with a
packing of 16, i.e. a last work-group whose sub-group 0 holds three live partitions and
one dead one -- exactly the configuration the "no early return" rule exists for -- and
every test in the file stayed green. The idiom really is undefined (a sub-group barrier
reached by only part of a sub-group), but this toolchain lowers
`sycl::group_barrier(sub_group)` in a way that tolerates it on sm_89, so **no numerical
test on this box can distinguish it from the correct code.** A guard nobody has seen fail
is not a guard, so the rule is guarded by its source text instead:
`GeqrfTinySource.KernelBodyHasNoEarlyReturn` scans the kernel body between the
`parallel_for` and its closing brace for any `return`, and the same planted break turns it
red with no rebuild:

```
src/extensions/geqrf_tiny.cc:131 returns from inside the kernel body ...
                if (prob >= batch) return;
```

The companion check, `GeqrfTinySource.SynchronisesAtSubGroupScopeOnly`, asserts the TU
contains no `get_group()` and no `reduce_over_group` for the same reason: a work-group
barrier here is a race rather than a launch failure, and a work-group `reduce_over_group`
adds static shared whose cost lands on a NEIGHBOURING kernel's cold first launch.

#### The synthesis pass: four more breaks

Planted against the WG=64 tree, after the chunk rule was moved onto the probe.

| # | break | expected | observed |
|---|---|---|---|
| f | delete one of the two `sycl::group_barrier(sg)` calls in the chunk body | `ChunkBodyCarriesExactlyTwoSubGroupBarriers` red with **no rebuild** | **RED.** `Expected equality of these values: sg_barriers / Which is: 1 / 2` |
| f' | the same break, **rebuilt and run numerically** | ambiguous; the chunk body is warp-uniform on sm_89 | **GREEN, 34 of 34.** Every numerical tiny test passes with a missing barrier |
| g | widen both norm butterflies from `<N>(part, ...)` to `<32>(sg, ...)` | `TinyNeighbourNaNDoesNotLeakAcrossPartitions` red | **RED**, and it names the mechanism: `item 5 changed when a NEIGHBOUR was poisoned with NaN at (0,0) n=5 pack=8`. 27 of 41 tiny cases red overall |
| h | rescale the reflector: store `2v` and `tau/4` (the same `H = I - tau v v^H`) | `TinyTauMatchesLapackeElementwise` red, residual green | **RED on tau**, exactly 4x: `tau[0] of item 0 at n=5 disagrees with LAPACKE (0.3764505 vs 1.5058020)`. **Residual ALSO red** -- see below |
| i | LAPACK's beta sign choice, `-SIGN(nrm, alphr)` -> `+SIGN` | tau red, residual green (a valid QR with a sign-flipped row of R) | **Both red.** tau red as expected; the residual red too, because same-sign `alpha - beta` is catastrophic cancellation -- which is LAPACK's own reason for the sign |

**Break (f') is the point of the source check, and it is now measured rather than argued.**
With one of the two barriers deleted the whole numerical tiny suite -- 34 cases across four
types, including the packed-vs-solo bit-identity oracle -- stays green. Control flow in the
chunk body is warp-uniform, so on sm_89 the missing barrier costs nothing; on any device
where a sub-group is not a warp it is a hard race with no test to catch it. The count and
the spelling are therefore guarded by source text.

**Breaks (h) and (i) correct the rationale D9 was written from, and the correction is
worth recording.** The claim was that a residual and an orthogonality probe are blind to
`tau` on the REAL types, so `tau` there is unguarded. That is **not** true once the LAPACK
storage convention is taken into account: `v(j) = 1` is implicit, so for a given stored
factor `tau` is uniquely determined, and the file's own `host_form_Q` reconstructs `Q` from
`(tau, v)` and so does constrain it. Neither break is invisible to the residual.

What the LAPACKE comparison *does* add is an **independent oracle**. Every other numerical
check in `geqrf_tests.cc` measures the kernel against `host_form_Q` in the same file, so a
convention error shared by the kernel and that oracle would pass every one of them. The
elementwise `tau` check is the only assertion in the file that cannot have that property.
It is kept for that reason, not for the reason it was proposed.

### The two partition butterflies

`geqrf_tiny_device.hh`'s `geqrf_tiny_reduce_fmax` and `geqrf_tiny_reduce_sum` are
hand-rolled XOR butterflies over the PARTITION — the shape `steqr_cta_device.hh`'s
`partition_reduce_fmax` established — for two reasons, neither cosmetic:

1. There is **no `reduce_over_group` overload for `SubGroupPartition`** anywhere in the
   tree, and a WORK-GROUP `reduce_over_group` emits static shared, which re-opens the
   `(47104, 49664]` launch hole for every kernel sharing the `CUfunction`. Same mechanism
   as [the 48 KiB launch hole](#the-48-kib-launch-hole), and the reason
   `GeqrfTinySource.SynchronisesAtSubGroupScopeOnly` scans the TU for the token.
2. They are **ALL-reduces on purpose.** The butterfly is order-symmetric, so every lane
   leaves with a BIT-IDENTICAL result and can recompute the Householder scalars for
   itself; that replication is what removes the publish/broadcast barrier pair. Widening
   either from `<N>(part, ...)` to `<32>(sg, ...)` leaks across partitions — break (g) of
   [the synthesis pass](#the-synthesis-pass-four-more-breaks).

`geqrf_tiny_slm_elems(n_pad, c)` is `n_pad * (c + 1) + c` scalars: the `N x (C+1)` product
tile plus a SEPARATE `c`-element `y` vector. `y` being a separate array, and not a row of
the tile, is exactly why two barriers per chunk suffice — the inter-chunk WAR on the tile
and the RAW on `y` are then covered by different barriers. Their count and placement are
guarded by source text (`ChunkBodyCarriesExactlyTwoSubGroupBarriers`), because break (f')
showed the numerical suite cannot see a missing one on sm_89.

`kGeqrfTinyRegOverhead = 88` is the register model's constant term and
`kGeqrfTinyRegMargin = 8` the hard gate's headroom. The model survives only as the
fallback for a cell with no probe row, for the reason recorded at [the tiny tier register
table](#the-tiny-tier-register-table). `GeqrfTinyRegs<D>`'s primary template IS that
"no probe row" case: zero in `at` is the sentinel the model fallback keys on, zero in
`probed_chunk` disables the fixed-point assertion for that cell, and `pin` overrides the
derived chunk width. `pin` is **not** a free knob — the derivation is blind to a `C` that
costs a block by pushing the REGISTER count up, which is what
[double N=32](#the-chunk-width-and-the-one-cell-that-is-pinned) measures.

`geqrf_tiny_matrices_per_wg<N>()` goes through `resident::pack_matrices_per_wg` rather
than being spelled `kGeqrfTinyWg / N`, so the power-of-two guarantee and the
max-work-group clamp stay in the one place that owns them; the byte arguments are nominal
because the lane bound is the binding one here. `kGeqrfTinyReferenceSlm = 97280` is a
COMPILE-TIME derivation constant only — every runtime decision reads `LOCAL_MEM_SIZE` from
the device through `resident::device_slm_budget`.

`reqd_work_group_size` is deliberately **not** declared anywhere in this tier: it emits
`.maxntid`, which lets `ptxas` trade registers against a launch bound and so makes the
probed table a function of the launch shape rather than of the kernel body. Only
`reqd_sub_group_size(32)` is declared. That is what licenses the same-register-counts
column of [the launch shape](#the-launch-shape-64-work-items-not-128).

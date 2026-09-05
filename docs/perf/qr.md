# QR: geqrf, orgqr, ormqr, and the third routing predicate (WP5)

Native SYCL `geqrf` (two tiers) and `orgqr` (one tier), the `ormqr` they are built on, and the routing that selects between them.

All timings: GPU 1 of a 2x RTX 4090 box (sm_89, 128 SMs), `CUDA_VISIBLE_DEVICES=1`, `WARM_S=1.5`, medians of interleaved A/B, cells with relative sd > 10% discarded, nothing timed under `BATCHLAS_KERNEL_TRACE`. "Vendor-free" always means the **build** (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`), never an env var inside a build that still links cuSOLVER: `route_resolve.hh:76-175` falls through to `automatic()` when a forced route is unsupported and `automatic()` returns `{Vendor, Auto}` (:129), so a forced-route A/B inside one build can silently be vendor-vs-vendor. `local_mem_size` here is 101,376 B; every capacity below derives from that minus the standard 4,096 B reserve, i.e. a 97,280 B budget. The generated `device_limits.hh`'s 49,152 is hardcoded for any `nvidia_gpu_sm_*` architecture with no device query at all (`cmake/BatchLASDetectSYCL.cmake:44-45`) and is 2.06x wrong on this box; nothing in this family reads it — `geqrf_cta.cc:31`'s `kGeqrfReferenceSlmBudget = 97280` answers only the "at this repository's reference budget" convenience overloads, and every real decision reads the device through `geqrf_route.hh:51-121`.

## What ships

### Route arms

| op | arms, in `order` sequence | `preferred()` |
|---|---|---|
| `geqrf` | `{Native, CTA}`, `{Native, Blocked}`, `{Vendor, Auto}` (`route_geqrf.hh:33-199`) | **false everywhere** (`route_geqrf.hh:73-367`) |
| `orgqr` | `{Native, Blocked}`, `{Vendor, Auto}` (`route_orgqr.hh:21-111`) | **false everywhere** (`route_orgqr.hh:60-235`) |
| `ormqr` | `{Native, Blocked}`, `{Vendor, Auto}` (`route_ormqr.hh:45-48`) | `is_native(r) && supports(r, s)` (`route_ormqr.hh:77-79`) |

`geqrf` and `orgqr` ship **route-neutral**: a vendor-present build takes cuSOLVER for every shape. The kernels are reachable only from a vendor-free build (`route_resolve.hh:38-127`), from `BATCHLAS_GEQRF_ROUTE` / `BATCHLAS_ORGQR_ROUTE`, or from the direct entry points `geqrf_cta_dispatch` / `geqrf_blocked_dispatch` / `orgqr_blocked_dispatch`. The 3.24x (`geqrf`) and 7.85x (`orgqr`) geomeans below are therefore **unrealised in the default build** — debt 1.

`ormqr` is the exception: `preferred()` is native-first, so a supported blocked `ormqr` runs natively in every build. That predates WP5 (no shape ever sent a supported blocked `ormqr` to the vendor) and is why `orgqr`'s native arm — an identity fill plus a routed `ormqr` — works at all.

`supports()` for both new tables is correctness-only. Gates that matter: `m >= n` for both `geqrf` native arms (handed a wide view the trailing update walks past the bottom of the panel, `route_geqrf.hh:46-232`); `n <= m` for `orgqr` (Q's columns live in C^m, `route_orgqr.hh:39-157`); GPU-only for both; and heterogeneous batch for both, refused because nothing in this tree gets heterogeneous-batch QR right (netlib included — `netlib_lapack.cc:1430-1442` hoists m and n out of its loop, and its `orgqr` at :1472-1477 hoists m, n and k).

**The sub-group gate is `geqrf`'s alone, and `orgqr` deliberately has none.** `geqrf` tests sub-group size 32 **enumerated** from `sycl::info::device::sub_group_sizes` (`route_geqrf.hh:25-175`, `queue-impl.cc:339-345`), never inferred from `get_property(MAX_SUB_GROUP_SIZE)` — that returns `sub_group_sizes()[0]`, so the weak test refuses a `{8,16,32}` device and accepts a `{64}` one, a launch abort for a kernel carrying `[[sycl::reqd_sub_group_size(32)]]`. `orgqr`'s shape builder sets no `has_sg32` and no SLM capacity at all (`orgqr_route.hh:76-83`), because `ormqr_blocked` carries no `reqd_sub_group_size` and holds nothing resident: a sub-group field there would be a decorative input. Do not read the `geqrf` gate list as covering both tables.

### CTA capacity

`geqrf`'s CTA tier holds the whole `m x n` panel in a `local_accessor`, so its ceiling is an **area**, `m*n <= cta_max_elems` in int64 (`route_geqrf.hh:59-297`). `cta_max_m` is also tested but is not independently binding with the shipped layout — there is one tile and no per-row resident array, so the largest admissible m at n = 1 *is* the area bound (`geqrf_cta.cc:171-310`). It is kept as a separate number because it is what moves if a per-row array (a staged `v`, a norm cache) is ever added. On this box: float 24,320 elems (square n = 155), double and cfloat 12,160 (n = 110), cdouble 6,080 (n = 77).

Above the area bound the blocked driver serves the shape. Its panel leaf is the same device body (`geqrf_cta_device.hh`) instantiated against a global pointer instead of a `local_accessor`, chosen per panel by the same `geqrf_cta_fits` predicate the route table's capacity uses — so the ceiling the table advertises and the allocation the launcher makes cannot disagree.

### The third predicate

`preferred()` cannot express "which of two **native** tiers". It is consulted by the loop above the vendor-free walk and runs regardless of `vendor_available`, so a window written to fix the vendor-free tier choice also moves vendor-**present** traffic, including where cuSOLVER beats both natives. WP5 added an optional third predicate, `RouteTable::native_tier_preferred`, detected with a `requires` expression and defaulting to `true` so every table that does not declare it keeps its old answer (`route_resolve.hh:18-83`). It is consulted **only** on the vendor-free walk, which is now two passes (`route_resolve.hh:38-127`); `gemm`, `trsm`, `potrf` and `gesvd` were untouched by construction at WP5 and the route diff confirmed it. Since then WP6 has declared the hook for `getrf` (`route_getrf.hh:78`) and `getrs`. **`potrf` still has not**, although it has the same two native tiers and the same all-false `preferred()` — see [open-debts](#open-debts).

The shipped `geqrf` window, verbatim (`route_geqrf.hh:79-473`):

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

The transposed panel GEMM `W1 = V^H A22` is the largest single kernel in **all three** blocked cells profiled. `src/sycl/gemm_kernels.cc:464-482` short-circuits every transposed form to `max_dim <= 32 ? Direct : Tiled16` before the register ladder; the TN/NT/TT register forms need `m >= 128 && n >= 32 && k >= 128`, and `m` here *is* the block width, so reaching them costs a block width of 128 — the worst width end to end. Complex cannot reach them at any width for **two** independent reasons, and the first is the load-bearing one: the whole register ladder sits inside `if constexpr (std::is_same_v<T, float>)` (`:471`), and the gate additionally tests `transA == Transpose::Trans` (`:472`) while a complex panel update is `ConjTrans`. **There is no block width at which geqrf's transposed panel gemm reaches a register kernel and is also a good block width.** G3 does reach the good kernel for both float and cdouble, so G3 is not the problem. The `double n=64` column is `native:cta`, which has no trailing update at all: that loss is the panel kernel at FP64 rate (1:64 of FP32 on this card) and no GEMM change can touch it.

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

`wy_trmm_applicable` (`ormqr_blocked.cc:49-124`) chooses the trmm tile kernel over a GEMM for the WY apply; shipped predicate `route_has_tile_kernel && !is_complex<T> && ib <= 64`. Measured on `ormqr_blocked_benchmark`, `Side::Left`, `ConjTrans`, ABBA-ordered against `BATCHLAS_ORMQR_WY=gemm`, batch 256 (128 for cdouble), nb in {16,32,64}, each figure the mean of two runs with the whole sweep repeated at a second measurement window (**the n set is not recorded in the source note — unverified**); ratios are gemm/trmm, so > 1.00 is trmm ahead: float 1.006–1.046x, double 1.004–1.016x, **cfloat 0.944–0.995x**, cdouble 0.958–1.010x. A 16-row tile variant moved every type in the direction the tile-reuse argument predicts (double 1.013–1.036x, cdouble 0.996–1.018x, cfloat 0.946–0.983x) but **did not change the gate**: closing to parity is not a reason to switch a call site, and cfloat stays behind because a complex multiply is four real ones. `ib <= 64` predates this and stays — past it the tile kernel measured 0.83x–0.97x in float. Read that as a local dip and not a monotone cutoff: the same table has the tile kernel back ahead at m >= 512 (1.07–1.32x). No `ormqr` block width reaches there, so the gate costs nothing today. netlib is excluded because OpenBLAS's `?trmm` is weak on a 16x16 triangle against 128 right-hand sides (0.336–1.199x, worst at n=128 ib=16); ROCm because `rocblas_?trmm` is a per-batch vendor loop against a strided-batched GEMM.

## Negative results

**The harnesses that would have lied, and why the measurement uses none of them.** `benchmarks/geqrf_benchmark.cc` counts flops as `2mn^2 + (2/3)n^3` — the **wrong sign** on the second term (LAPACK's `geqrf` is `2mn^2 - (2/3)n^3`) — is registered float/double only with **no complex**, and never checks the answer. `benchmarks/gemm_benchmark` allocates operands at `ld == rows`, which is structurally incapable of seeing the sub-view question this driver's trailing GEMMs pose. Separately, WP4's `phase2.cpp` defined its *own* `Blocked<T>` class instead of timing the shipped code: it was 2x slower and contradicted the real numbers by a factor of two. Everything on this page times the **public API** with the workspace the facade's own `*_buffer_size` asked for, and checks the residual in the same process — five apparent wins entered the WP4 record because a racing kernel was fast and wrong.

**"WP5 will be decided by the panel factorisation" is refuted for the shipped blocked driver.** The claim came with a 63.2x-headroom estimate computed against a **routed** trailing update; the shipped driver's largest GEMM lands on Tiled16. Measured, the panel is 9.3% (float n=1024), 2.5% (cdouble n=1024) and 10.6% at the losing cdouble n=256 cell, against ~71% for the trailing update. The one place the panel *is* the whole cost is the CTA tier — 100.0% one kernel — which is exactly where FP64 loses.

**Tuning the block width on the trailing GEMMs alone is the wrong instrument.** Measuring the G1+G3 pair in isolation says wider is always better and shows a float cliff at nb=128 — effective throughput 6896 → 18,906 GFLOP/s, because `m >= 128` finally admits `Tiled128x32RegisterK32TN`. **End to end that cliff does not survive**: nb=128 is the worst width tested in both builds, 83.0 ms against 36.8 ms at nb=32, because the panel and `larft` costs a per-gemm probe cannot see dominate.

**Specialising `orgqr` is worth at most 1.5x, and the measured price of not doing it is ~11%.** Applying Q to an identity does `2n^3` against a specialised `4n^3/3` at m=n=k. The identity fill plus copy-back cost ~11% of float n=1024 and ~17–22% of float n=64, below that theoretical ratio, against a 2.3–111x margin over the vendor across most of the range.

**32 device entry functions that could never launch.** `larft_forward_columnwise_batched` took `use_device` as a runtime bool, so it instantiated both implementations for every `(Tag, T, WG)`; `geqrf` passes a literal `false`, so `larft_forward_columnwise_wg_device<GeqrfWyTag, ...>` was 4 types x 4 work-group rungs x 2 forms = 32 entry functions compiled, ptxas'd and device-linked into `batchlas_extensions_cta` — the slowest-linking library in the tree — and never launched (nsys: no `(bool)1` variant in any WP5 run). They included the highest-register kernel in the whole WP5 set (cdouble, 90 registers, 208 B stack frame). `UseDevice` is now a template parameter, the runtime wrapper retained for `ormqr` whose choice really is a getenv. **880 → 848 entry functions, device link 125.45 s → 116.63 s.**

**Three uncoalesced fills and one needlessly strided operand.** `OrgqrIdentityKernel`, `OrgqrCopyBackKernel` and `pack_v_panel_batched` all launched `sycl::range<3>(batch, rows, cols)` and read `idx[2]` as the **column**; `sycl::id<3>` makes dim 2 fastest-varying and every operand is column-major, so a warp touched 32 sectors instead of 4. Separately the blocked driver handed `V` to both trailing GEMMs at the **parent** `ld = m` although `V` is scratch it owns outright — at j0=992 of a 1024-column factorisation, a 32-row panel whose columns were 4 KB apart, the recorded "native GEMM collapses on strided ld" shape. `V` is now packed at `ld = mp`. Clean same-session A/B (fixes reverted, rebuilt, measured, restored; all relsd <= 0.11%): `orgqr` float n=512 b=512 1.196x, n=1024 b=128 1.118x, n=2048 b=32 1.059x; `geqrf` float n=512 b=512 1.054x. 16 of 16 cells improved or were neutral, and the gain is ~1.00x wherever the cell is GEMM-bound — which is most of the FP64 and complex grid.

**The `route_gemm.hh` `batch >= 64` gate (`route_gemm.hh:48`) costs the vendor build.** At N=2048 b=32 six `double` trailing-GEMM cells go to cuBLAS and the vendor build is **1.14–1.25x slower** than the vendor-free build on the same shapes. Small, but a real `preferred()` edge on a shape `geqrf` issues.

**The complex deficit is outside WP5 and was not attempted.** A vendor-free build pays 2.55x (float), 1.00x (double), 2.61x (cfloat), 2.01x (cdouble) on the BLAS-3 core, essentially all of it G1 (4.81x / 1.00x / 4.99x / 3.12x; G3 is 1.06x / 1.00x / 1.02x / 0.95x). Closing it needs a **transposed** wide-scalar/register GEMM — WP2 territory; `route_gemm.hh:43-114` still refuses complex outright. double is 1.00x because both builds run the identical native kernel; separate processes and separate `.so`s agree to 0.03% (11.8008 vs 11.8012 ms), the strongest internal control in the baseline directory.

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
9. **`resolve_ormqr_block_size` still returns the float-only 16/16/24/48/56 ladder keyed on `A.rows()`** (`include/batchlas/tuning_params.hh:45-49`, `include/batchlas/blas/functions/ormqr.hh:219-227`) for every `ormqr` caller that passes no hint. Measured wrong for three of four types, costing 1.11–1.55x. `geqrf` and `orgqr` bypass it with their own type-keyed widths; nothing else does. Read the *source* header, not the generated one: the `configure_file` copy at `build/include/batchlas/tuning_params.hh` says **16/32/64/128/128** and is never compiled, because `src/CMakeLists.txt` puts `${PROJECT_SOURCE_DIR}/include` ahead of `${PROJECT_BINARY_DIR}/include`. The harness prints the width it actually used, and it prints 16/24/48/56.
10. **The CTA tier's workspace is zero but callers still pay the blocked layout.** The facade takes `max` over every *supported* native tier and `supports()` deliberately puts no lower extent bound on the Blocked arm, so a caller at n=64 batch=8192 pays 168 MB (float) / 671 MB (cdouble) even though the route it takes is CTA. Sizing W1/W2 on `n - nb` rather than `n` took ~28% off; the remainder is the deliberate `max` policy. (The vendor `orgqr` it replaces asks for 1164 MB / 4644 MB at that cell.)
11. **`potrf` has WP5's dispatch gap and has not closed it.** `potrf` carries the same two native tiers ({CTA, Blocked}) and the same all-false `preferred()`, so its vendor-free walk still returns the first *supported* native route from a static order array that cannot follow a crossover — the exact defect `native_tier_preferred` was added for. `getrf` and `getrs` have since declared the hook; `route_potrf.hh` has not. Nobody has measured whether `potrf`'s vendor-free tier choice is wrong, which is the first step, not the fix.
12. **`resolve_ormqr_route` is called with two arguments** (`ormqr.hh:209`), taking the `vendor_available = true` default, so `ormqr` never reaches `route_resolve.hh:38-127`'s vendor-free fallback. It gets away with it only because its `preferred()` is native-first. `geqrf` and `orgqr` pass the argument explicitly; do not inherit the omission.

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

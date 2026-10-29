# QR: geqrf, orgqr and ormqr {#perf_qr}

> **Status:** current · GPU 1 of a 2x RTX 4090 (sm_89), CUDA 13.2 · measured 2026-09-10 to 2026-10-23; square and tall sm_89 rows transcribed (run `transcribed-424a45bc`, untimed), wide float/cfloat rows deep (2026-10-24); sm_120 tables in `tuned/README.md`

Native QR families (`geqrf`: `tiny`, `cta`, `blocked`; `orgqr` and `ormqr`: `blocked`) against their `vendor` family: what `can_run` admits, what the shipped `tuned/{geqrf,orgqr,ormqr}.*.txt` rows run, and the measurements behind each edge. Selection is flat (@ref design_flat_selection).

## Conventions

- Timings: `CUDA_VISIBLE_DEVICES=1`, `WARM_S=1.5`, medians of interleaved A/B, cells with relative sd above 10% discarded. Ratios are `t_vendor / t_native` (> 1 native ahead) unless stated.
- "Vendor-free" is `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`. A pin inside a vendor build measures one family, not what a vendor-free Auto gets.
- SLM budget: `local_mem_size` 101,376 B minus the 4,096 B reserve = 97,280 B, read into `select::Device::slm_budget`. The build-time 49,152 B for `nvidia_gpu_sm_*` is not read by this family.

## QR: route arms

| op | families (`choice.hh`, tie-break order) | `can_run` (correctness only) |
|---|---|---|
| `geqrf` | `tiny`, `cta`, `blocked`, `vendor` | native: no batch mask, `d.is_gpu && d.has_sg32`, homogeneous, `m, n, batch >= 1`; `tiny` adds `m == n && n <= geqrf_tiny_max_n_for_slm<T>(budget)`; `cta` is `geqrf_cta_fits<T>(m, n, budget)`; `blocked` needs `geqrf_blocked_available<T>()` and `geqrf_cta_max_elems_for_slm<T>(budget) >= 1`; `vendor` is `d.has_vendor` and homogeneous (`src/ops/geqrf/can_run.hh`) |
| `orgqr` | `blocked`, `vendor` | `blocked`: GPU, homogeneous, `n <= m`; `vendor`: `d.has_vendor`, homogeneous (`src/ops/orgqr/orgqr.cc`) |
| `ormqr` | `blocked`, `vendor` | not complex `Trans` (thrown earlier by `throw_if_undefined`), homogeneous A and C, GPU / vendor; no batch term ([defect 16](../design/known-defects.md#defect-16-blocked-qr-past-the-grid)) |

Keys: `geqrf` `form` (sq / tall / wide), `n`, `aspect = max(m,n)/min(m,n)` (integer division); `orgqr` `m`, `n`. Family that runs under Auto, sm_89, vendor build:

| type | square | tall (`m > n`) | wide (`m < n`, deep rows) |
|---|---|---|---|
| `float` | `tiny` 4..16, 21..32; `vendor` 1..3, 17..20, 33..63; `cta` 64..96; `blocked` from 97 | `cta` 64..96 (from 32 at aspect >= 4); `blocked` from 97; `vendor` below | native, except `vendor` at n = 64, aspect >= 4 |
| `cfloat` | `tiny` 5..8, 11..16, 24..32; `vendor` 1..4, 9..10, 17..23, 33..47; `cta` from 48 | `cta` from 48 (from 32 at aspect >= 4); `vendor` below | native, except `vendor` at n = 64, aspect 64 |
| `double` | `vendor` below 76; `blocked` from 76 | `blocked` from 76; at aspect >= 8 `cta` 32..48, `blocked` from 49 | `vendor` (untimed) |
| `cdouble` | `vendor` below 256; `cta` from 256 | `cta` from 256 (from 32 at aspect >= 8); `vendor` below | `vendor` (untimed) |

- Rows rank `tiny` first past 32 too; `can_run` refuses it there. Where `cta` is ranked first and its area does not fit ([CTA capacity](#cta-capacity)), `blocked` runs; that is the only way the complex types reach `blocked`.
- `orgqr` ranks `blocked` first for `m <= 512`, `vendor` from 513 ([the shipped orgqr ceiling](#the-shipped-orgqr-ceiling)). `ormqr` ranks `blocked` first everywhere except complex `trans=T` (an argument error); `orgqr`'s `blocked` (identity fill plus the public `ormqr`) relies on that.
- Vendor-free, the next native entry runs: `tiny` in the float and cfloat gaps, `cta` for square double n <= 48 and cdouble n <= 255, `blocked` for square double 49..75 ([the native tiny tie-break](#the-native-tiny-tie-break)).
- Pins `BATCHLAS_{GEQRF,ORGQR,ORMQR}_ROUTE` = `auto | native | vendor | <family>`; a family `can_run` refuses throws `invalid_argument`. `geqrf_cta_dispatch`, `geqrf_blocked_dispatch` and `orgqr_blocked_dispatch` bypass selection; `geqrf_cta_dispatch` forces `GeqrfPanelLeaf::Resident`.
- `d.has_sg32` enumerates `sub_group_sizes`, never `MAX_SUB_GROUP_SIZE` (which accepts a `{64}` device). `orgqr` needs neither it nor the SLM budget: `ormqr_blocked` holds nothing resident.
- Guards: `GeqrfCandidates.AutoReadsTheTranscribedTable` (every edge through Auto and the `native` walk at the device's real budget; skips unless the budget is 97,280 B) and `GeqrfTranscribedTable.RowsHoldTheTranscribedPreference` (both devices' rows).

### CTA capacity

`cta` holds the whole `m x n` panel in a `local_accessor`. The ceiling is an area, `m*n <= cta_max_elems` (int64), tested by `geqrf_cta_fits` (`src/ops/geqrf/kernels/geqrf_cta.cc`), which `can_run` and the CTA entry both call; the per-row bound is `min(elems, INT_MAX)` and never binds. The tier divides the budget by `kGeqrfMinBlocksPerSm = 2` ([QR: the occupancy rule](#qr-the-occupancy-rule)):

| type | tier capacity (elems) | square n | residency, whole budget (elems) | square n |
|---|---|---|---|---|
| float | 11,776 | 108 | 24,320 | 155 |
| double, cfloat | 5,888 | 76 | 12,160 | 110 |
| cdouble | 2,944 | 54 | 6,080 | 77 |

Above the tier's area `blocked` serves the shape. Tests must use the device's real budget: with a permissive one the complex types never reach `blocked`.

### Block-width evidence

`geqrf` and `orgqr` compute their own block width, clamped to `k = min(m,n)`: **16 for double, 32 otherwise** (`geqrf_blocked.cc`, `orgqr_blocked.cc`); multiples of 16, never below 32 for complex (the wide-scalar GEMM gates on `min_dim >= 32`; nb = 24 fell to Tiled16 at 1.72-2.30x). `ormqr` WY apply on an identity, vendor-free, n = 1024 batch 64, median ms:

| nb | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 16 | 45.36 | **123.65** | 101.53 | 1547.54 |
| 32 | **36.82** | 135.65 | **81.54** | **1061.24** |
| 56 (shipped `ormqr` ladder) | 45.48 | 174.82 | 113.18 | 1333.81 |
| 128 | 83.04 | 303.45 | 296.93 | 1936.75 |

`ormqr` itself still uses the float-only `tuning::ormqr_block_size_for_n` ladder (open debt 6).

### CTA vs blocked crossover

Vendor-free, `blocked_ms / cta_ms` (> 1 CTA ahead), batch 8192 at n <= 64, 4096 above; cells where the area does not fit are n/a; n <= nb is a null (one panel on the resident leaf).

| type | n=32 | n=48 | n=64 | n=80 | n=96 | n=112 | n=128 |
|---|---|---|---|---|---|---|---|
| float | null | 2.686 | 2.034 | 2.037 | **1.294** | **0.821** | 0.699 |
| double | **1.049** | **0.983** | 0.922 | 0.772 | 0.731 | n/a | n/a |
| cfloat | null | 3.171 | 2.093 | 1.253 | 1.079 | n/a | n/a |
| cdouble | null | 2.589 | 1.929 | n/a | n/a | n/a | n/a |

Float is bracketed 96 / 112; double 32 / 64, with the 48 tie resolved to CTA (zero workspace). The complex edges are not bracketed (open debt 3). Mechanism: above about 50 KB the tile allows one work-group per SM and the per-reflector barrier chain has nothing to overlap (float n = 96 is 36,864 B, n = 112 is 50,176 B).

### QR: the vendor baseline

cuBLAS `geqrfBatched` is latency-bound (about 380 GFLOP/s float, 105 cdouble, at any n), so native pulls away with n (n = 1024 batch 128: float 41x, cdouble 4.3x; full grid [small-n-baseline.md](small-n-baseline.md#geqrf)). The FP64 losses at n = 64 are batch-driven: double 4.37x at batch 32, **0.53x** from batch 2048 on; quote 0.53x.

cuSOLVER `orgqr` is not batched: `backend::orgqr_vendor` loops `cusolverDn?orgqr` / `?ungqr` per item with batch-linear workspace (4.6 GB at cdouble n = 64 batch 8192). A native win is "beats the per-item loop", never "beats cuSOLVER".

### `ormqr` WY `trmm` gate

`wy_trmm_applicable` (`src/ops/ormqr/kernels/ormqr_blocked.cc`) takes the trmm tile kernel for the WY apply when `select::level3_tile_substitution_enabled<B, T> && !is_complex<T> && ib <= 64`; `BATCHLAS_ORMQR_WY=gemm|trmm` pins it. No table sees this derived leg. gemm/trmm at batch 256, nb 16..64: float 1.006-1.046x, double 1.004-1.016x, cfloat 0.944-0.995x (hence real only); past `ib = 64` float dips to 0.83-0.97x. netlib and ROCm `?trmm` are excluded (weak small triangles, per-batch loop).

## QR: negative results

`benchmarks/geqrf_benchmark.cc` (wrong flop term, no answer check) and `gemm_benchmark` (`ld == rows`) are not evidence; time the public API with its `*_buffer_size` workspace.

| Alternative | Result | Verdict |
|---|---|---|
| Optimise the panel factorisation | panel is 2.5-10.6% of GPU time at n >= 256; trailing GEMMs about 71% (transposed Tiled16 GEMM 47-70%) | refuted; panel is 100% only in the FP64 CTA tier |
| Tune block width on trailing GEMMs alone | nb = 128 wins in isolation, is the worst width end to end (83.0 vs 36.8 ms) | wrong instrument |
| Specialise `orgqr` for the identity | fill and copy-back are about 11% at float n = 1024 | at most 1.5x |
| Close the complex deficit in `blocked` | the cost is G1, the transposed GEMM (4.81x / 4.99x / 3.12x for float / cfloat / cdouble vendor-free) | needs a transposed wide-scalar or register GEMM |

## Correctness findings

### The 48 KiB launch hole

A resident-leaf launch asking for exactly 49,152 B of local memory fails in the CUDA backend (`CUDA_ERROR_INVALID_VALUE`): 48,896 B pass, 49,152 B fail, 49,664 B pass, all four types, whenever the body contains a group algorithm. It hides because the attribute is sticky per `CUfunction`: an earlier larger launch raises the cap for the process. The resident `local_accessor`, `geqrf_cta_fits` and `geqrf_cta_max_elems_for_slm` pad the band (`kGeqrfHoleLo = 47104`, `kGeqrfHoleHi = 49664`, `kGeqrfHolePadTo = 49920`, `geqrf_cta.cc`). `GeqrfTest.ResidentLeafLaunchHoleAt48KiB` must be the first test in its file (declaration order). Kernels that must avoid static shared use hand-rolled butterflies, never `reduce_over_group`.

### A residual test cannot guard a convention

A phase-preserving `larfg` sign in place of LAPACK's real beta left every residual green: still a valid QR, not the one `ormqr`, `orgqr`, `sy2sb`, netlib and cuSOLVER expect. `ConventionMatchesReferenceLapackWithoutAVendor` (independent host `xGEQR2`) guards it vendor-free; `NativeFactorMatchesTheVendorElementwise` with a vendor. `zgeqr2` applies `conj(tau)`, invisible on real types. Elementwise `dF` against the vendor is 3e-6 / 8e-15 / 2e-6 / 4e-15 (float / double / cfloat / cdouble); `dtau` is 3-6x looser.

### The `orgqr_buffer_size` latent defect

A zero native workspace is a legitimate answer (the CTA tier reports it), so the size query must not treat 0 as "no native tier". It returns exactly the chosen family's need; a native call must never be sized by the batch-linear vendor workspace.

### ormqr: one route resolution for the call and its size query

`ormqr` and `ormqr_buffer_size` make one choice from the same inputs (`key_of`, `choice.hh`, `can_run`): the call through `select::run`, the query through `select::pick`; the blocked width comes from `block_size`, so the queried bytes are the bytes the call checks. `block_size_hint` exists because `ormqr_block_size_for_n` is keyed on `A.rows()`, wrong for a tall skinny panel; it is clamped to `[1, k]` and ignored by `vendor`.

### Break sweeps

| break | outcome |
|---|---|
| drop the 48 KB hole pad | only `ResidentLeafLaunchHoleAt48KiB`, all 4 types |
| drop `conj(tau)` in the panel apply | complex only (real is a correct null) |
| `T^H` to `T` in the WY update | `cta` green (no WY update), `blocked` red for every type |
| LAPACK's beta sign | residuals green; elementwise `dimag`, `dF`, `dtau` red |
| `tau` batch stride, sub-view `ld`, panel loop bound | red everywhere (item 0 unaffected: check every item) |
| stop the blocked trailing update at column m | only the wide cases ([geqrf wide shapes](#geqrf-wide-shapes-m-less-than-n)) |

- `std::max(0.0, NaN)` returns 0: a checker built on it passed `tau = -12345`.
- Dropping the last reflector is green on square real matrices (LAPACK's 1x1 `tau` is 0) and red for complex; a short-final-panel test needs `m > n`, a middle panel or complex. `GeqrfTest.WideFactorsTheWholeMatrix` is the case that sees an `ib`/`nb` mix-up in the last panel.
- The barrier between reads of `A(j,j)` and the beta write stays red on nothing; the SYCL memory model requires it.
- `tests/orgqr_tests.cc` pins no route, so in a vendor build it exercises cuSOLVER; it discriminates kernel breaks only vendor-free.

## QR: L2 batch chunking {#qr-l2-batch-chunking}

> **Status:** current · RTX 4090 (sm_89, 128 SMs, 72 MB L2), CUDA 13.2 · measured 2026-10-23

The blocked geqrf, orgqr and ormqr drivers can run the batch in parts of `tuning::blocked_chunk_items` items (item sizes: geqrf A + V + W1 + W2, never split when k <= nb; orgqr 2mn + m nb + 2 nb n; ormqr nq k + C + V + 2W). Callers that size once at their largest operands (sytrd, band_reduction) make each driver halve its chunk until a part fits; the ormqr facade therefore skips its workspace check for `blocked`.

**Rule:** `GEQRF_`, `ORGQR_` and `ORMQR_BLOCKED_L2_CHUNK_PERCENT` = 0 (no L2 chunking). Every part is capped at `max_groups_yz / 2` items (`select::grid_capped_chunk`, `src/select/grid_chunk.hh`); without the cap a batch past the grid left items unfactored ([defect 16](../design/known-defects.md#defect-16-blocked-qr-past-the-grid)).

Cold A/B, pinned blocked, chunked / off:

| op | type | n | batch | 75 % | 150 % | 300 % |
|---|---|---:|---:|---:|---:|---:|
| geqrf | cfloat | 128 | 4096 | 1.572 | 1.261 | 1.077 |
| orgqr | cfloat | 128 | 4096 | 1.866 | 1.473 | 1.241 |
| ormqr L C | cfloat | 64 | 16384 | 1.137 | 0.847 | 0.955 |

geqrf and orgqr lose at every percent (the parts under-fill the trailing gemms). ormqr alone wins at 150 %, but orgqr applies the routed ormqr and lost 1.017-1.484x with it chunked inside, syev blocked n = 256 2.8%; a per-caller rule (chunk only a top-level ormqr) is open.

## Open debts

1. **No window is a timed table row** except the wide float/cfloat rows. Widening one needs an end-to-end gate (a 2.16x kernel win once became an 11% `gesvd` loss); fix with a `tools/tune/batchlas_tune` retune of `geqrf`, `orgqr`, `ormqr`.
2. **The CTA/blocked crossover is a single occupancy target, not a per-type area**: it lands 4-28% below the measured crossover; worst cell cfloat 128x48, where CTA is 1.925x faster but the fit sends it to `blocked` ([what the clamp costs](#settling-the-target-on-the-tall-panels)). Fix: a retune timing `cta` against `blocked` on the tall grid.
3. **cfloat 97..110 is extrapolated; cdouble has no measured cell above n = 64.** The capacity (cfloat 76, cdouble 54) sends both bands to `blocked`.
4. **`geqrf_panel_wg(n, max_wg)` never looks at `m`**: at double n = 64 at least 193 of 256 work-items idle through a 256-wide reduction; that kernel is 100% of GPU time in the worst `geqrf` loss (0.53x).
5. **`larft_forward_columnwise_wg_legacy`** is 15-20% of GPU time (float n = 1024, cdouble n = 256) for about 1/60 of the GEMM's arithmetic (serial T-update on `lid == 0`); `larft` plus `pack_v` is about 20-22%.
6. **`ormqr`'s width ladder is float-only and keyed on `A.rows()`** (`block_size()` in `ormqr.cc`, `tuning_params.hh`), costing 1.11-1.55x for the other types ([block-width evidence](#block-width-evidence)); a derived leg, so no table retune fixes it.
7. **The `!dev_is_zero(r)` half of the reciprocal guard is untested** (needs `alpha - beta` to overflow, input ~1e308).
8. **`geqrf_buffer_size_bound`** reserves the blocked layout even where `cta` runs (168 MB float, 671 MB cdouble at n = 64 batch 8192); sizing W1/W2 on `n - nb` took about 28% off. Per-call host overhead of the eigen drivers' ~500 `geqrf` calls per n = 1024 solve is unmeasured.
9. **double and cdouble rank `cta` ahead of `tiny` vendor-free**, though `tiny` measured better (0.33-1.13x of vendor against 0.14-0.67x) at n = 8..28; cdouble has no grid.
10. **Wide FP64 rows and all sm_120 wide rows** stay transcribed.
11. **Tiny band interiors** (cfloat 6, 7, 12..15, 26..31; float 5..16, 22..31) were measured once under the degraded guard ([the caveat](#a-measurement-caveat-that-is-part-of-the-result)); the edges were re-cut.

## Raw evidence

Git tag `perf-evidence/vendor-independence`, `experiments/wp5_qr/` (`bench/` holds `order.csv`, `batch.csv`, `tall.csv`, `orgqr_batch.csv`, `tier.csv`, `tier_summary.txt`, `nsys_split.md`; `kernels/` and `baseline/` the break sweeps). In the tree: `benchmarks/results/p7_occupancy_geqrf_tall.csv`, `p5_reg_leaf_{un,}capped.csv`, `p5_auto_after_flip.csv`, `factor_baseline_orgqr_double.csv`; ledger `benchmarks/results/tuning/ledger/geqrf.sm_89.jsonl`.

## The shipped `orgqr` ceiling

`tuned/orgqr.sm_89.txt` ranks `blocked` first up to m = 512 for every type, `vendor` from 513. Below the ceiling native wins all 66 square cells (orders 4..512), minimum 2.54x (cfloat 512); at n = 512 / 64: float 3.64 / 27.1, cfloat 2.54 / 15.1, double 4.61 / 27.1, cdouble 2.77 / 11.3. Above it cfloat n = 1024 is 0.82x, cdouble 0.78x, and every type but double loses at n = 2048 (float 0.41x, cfloat 0.31x, cdouble 0.46x). Only float n = 1024 recovers with batch (0.84 to 1.33 at batch 32..256), and the rows have no batch key. Grid: [small-n-baseline.md](small-n-baseline.md#orgqr).

## The `geqrf` order floor and the tall-panel clause

The square floor is the first order that clears the flip gate (ratio >= 1.11) at the top of the batch ladder, with the order below measured as a loss:

| T | floor | ratio at floor | bracketing loss below |
|---|---|---|---|
| `float` | 64 | 1.71 | 48: 1.02; 33: 0.76 |
| `cfloat` | 48 | 1.74 | 33: 0.69 |
| `double` | 76 | 1.26 | 72: 1.105 (b16384), 1.088 (b32768) |
| `cdouble` | 256 | 1.50 | 192: 1.06; 129: 0.58 |

The double floor depends on the register panel leaf ([the double order floor](#the-double-order-floor-moves-to-76)).

The eigen drivers issue `m x kd` panels, kd typically 32 (`sytrd_sy2sb.cc`, `band_reduction.cc`), where every square cell loses; tall panels cross earlier. Native over vendor at saturation:

| shape | aspect | float | cfloat | double | cdouble |
|---|---|---|---|---|---|
| 128 x 32 | 4 | 2.23 | 3.79 | 0.68 | 0.68 |
| 512 x 64 | 8 | 3.18 | 4.10 | 2.37 | 1.73 |
| 1024 x 128 | 8 | 7.16 | 5.09 | 6.42 | 2.96 (b4096) |

So the aspect floor is 4 for the 32-bit types and 8 for the 64-bit types. Unbracketed: no tall cell narrower than 4x, and nothing shorter than 128 rows.

## QR: the occupancy rule

`geqrf_cta_max_elems_for_slm<T>(budget, min_blocks_per_sm)` divides the budget by an occupancy target before converting to elements; the hole clamp applies after the division. `geqrf` uses 2 (`kGeqrfMinBlocksPerSm`), not `resident::kMinBlocksPerSm = 4`: its alternative is the multi-launch blocked driver, so an occupancy-limited CTA kernel can still be best. At target 4 the advertised area falls below the crossover: cfloat 64x64 b8192 **1.83x slower**, float 96x96 b4096 **1.14-1.15x slower**.

### Settling the target on the tall panels

At target 2, Auto moves float 512x32, 384x32, 256x64, cfloat 256x32 and double 256x32 from `cta` to `blocked` at no cost (`p7_occupancy_geqrf_tall.csv`, batch 8192, `blocked_ms / cta_ms`): 1.001-1.004 at n <= nb (the same leaf launch, bit-equal), **0.703** float 256x64 and **0.935** double 256x32 (the driver splits the panel). CTA wins no cell the clamp moves. `GeqrfTest.OccupancyTargetIsPinned` fails if the value moves.

What the clamp costs: crossovers sit at float n = 64 between m = 192 (1.231) and 224 (0.706), float n = 96 between 128 (1.191) and 160 (0.672), cfloat between 128x48 (1.925) and 128x64 (0.913), so target 2 sends float 128x96, 192x64 and cfloat 128x48 to `blocked` early (open debt 2); target 1 costs float 224x64..160x96 1.42-1.49x.

`geqrf_blocked_debug_params<T>(ctx, m, n)` packs nb in the low 16 bits and the leading panel's leaf in the high 16 (`1` resident, `2` global; 0 when the driver is absent), answering `geqrf_leaf_fits` at the whole budget. Same encoding as `getrf` ([lu.md](lu.md#one-spelling-per-ceiling)).

### QR: the panel leaf is not the tier ceiling

`geqrf_cta_fits` is the tier's predicate (occupancy-scaled); `geqrf_leaf_fits` is residency at the whole budget. `geqrf_panel_factorize` and the blocked driver use the latter, so no blocked panel loses residency to the occupancy target, and only the resident leaf can reach the 48 KB hole.

`geqr2_panel_device` is templated on `GeqrfScope`; under `SubGroup` (packed panels, `m <= 32 && n <= 32`) barriers and reductions are sub-group butterflies (`geqrf_sg_max`, `geqrf_sg_sum`), 1.2-5.0x faster.

## The tiny tier (WP6 / P1): square n <= 32 in registers {#the-tiny-tier-wp6--p1-square-n--32-in-registers}

`src/ops/geqrf/kernels/geqrf_tiny.cc`: one matrix per `SubGroupPartition<N>`, N in {8, 16, 32}; lane `r` owns row `r` in `D rA[N]`; two sub-groups per work-group (64 work-items) carry `64 / N` matrices. One reduction per column, Householder scalars recomputed in every lane by an order-symmetric butterfly, trailing update through one local tile. `BATCHLAS_GEQRF_ROUTE=tiny` pins it for any square shape `can_run` admits.

### The tiny ceiling predicate

`geqrf_tiny_max_n_for_slm<T>(budget)` is the one ceiling predicate (`can_run`, the entry point and the tests). It returns 0 when no bucket fits; the type ceiling is 32, and 16 for `complex<double>`. It walks buckets with a `break`, so the range stays contiguous.

### The tiny tier register table

`scripts/register_probe.sh out.log '' batchlas_extensions_cta` at work-group width 64; **the third argument is load-bearing** (the default target contains none of these kernels and reports clean). `blocks/SM = min(65536 / (ceil8(regs) * 64), 24)`.

| T | N | C | registers | blocks/SM |
|---|---|---|---|---|
| float | 8 / 16 / 32 | 8 / 16 / 32 | 64 / 91 / 142 | 16 / 10 / 7 |
| double | 8 / 16 / 32 | 8 / 16 / 16 | 84 / 124 / 193 | 11 / 8 / 5 |
| cfloat | 8 / 16 / 32 | 8 / 16 / 16 | 72 / 128 / 168 | 14 / 8 / 6 |
| cdouble | 8 / 16 | 8 / 8 | 110 / 162 | 9 / 6 |

All eleven instantiations: zero spill, zero stack frame, at or above four blocks/SM, 0 B static shared. The table is `GeqrfTinyRegs<D>` in `geqrf_tiny_device.hh`, the single source: the chunk rule reads it and the residency gate is a `static_assert` per cell. The fallback model `N*words + 88` is off by -30% to +25%; ptxas allocates in banks of eight, so round first.

### The launch shape: 64 work-items, not 128

The `local_accessor` is linear in matrices per group with no fixed term, so a wider group amortises nothing; 64 is equal or better in occupancy in all eleven cells and strictly better in four.

### The chunk width, and the one cell that is pinned

C is the largest of {N, N/2, N/4} with `blocks_by_slm >= blocks_by_regs`; local memory per matrix is `N*(C+1) + C` scalars (odd leading dimension `C + 1` for banks; separate `y`, so two barriers per chunk cover all hazards). **double N = 32 is pinned to C = 16**: C = 32 would need 218 registers and drop to 4 blocks/SM; `GeqrfTinyRegs<double>::pin[2]` records it.

### The stack frame is the gate for this kernel, not the spill counter

With 0 spill, double and cfloat N = 32 first showed a 256 B frame (cdouble 544 B): `#pragma unroll` is a hint, `rA[j]` became a dynamic index and the array moved to local memory. Every loop indexing `rA` uses `#pragma clang loop unroll(full)`; the compile-time signal is `-Wpass-failed=transform-warning`.

### The cdouble N=32 cell

Not instantiated (`geqrf_tiny_type_ceiling<complex<double>>()` is 16) to keep 11 instantiations. It is the largest unclaimed cell: cuBLAS takes 5,758 us at cdouble n = 32 b8192 against a 283 us DRAM roof, and CTA is at 0.33x.

### Why bit-identity against the CTA route is not the padding test

CTA reduces over 32 lanes, tiny over N, so the last ulp differs. The guard poisons the `ld` pad, stride pad and everything beyond `n` with NaN and checks a finite factor and an unchanged pad.

### Why the elementwise scan is the m x n window only

A whole-buffer scan takes its scale from the poisoned padding (`mk<T>(-9.75e3, 4.5e3)`), is about 1000x looser, and admitted a float disagreement of 0.07.

### The fixture's tolerance floor, and why it is new

`residual_tol` and `orth_tol` are `0.5 (m + k) eps` with a floor of 8 eps: below m + k = 16 the linear term no longer covers the backward error (1x1 cfloat 1.2 eps, 2x2 float 2.6 eps). `TinyIsNoWorseThanTheCtaRouteAtTinyOrders` requires tiny within **2x** of CTA (4x let a kernel lose two bits).

### Break sweeps: the tiny tier

Red: store guard `k < N` for `k < n`; dropping `lane < j` from the `vr` select; dropping `sg_id *` from `tiny_partition_id` (packed item differs from its solo run).

#### The synthesis pass: four more breaks {#the-synthesis-pass-four-more-breaks}

- Deleting one of two sub-group barriers in the chunk body is numerically green (warp-uniform on sm_89); `ChunkBodyCarriesExactlyTwoSubGroupBarriers` guards it by source.
- Widening the norm butterflies to `<32>(sg, ...)` is red (a neighbour's NaN leaks).
- Storing `2v` and `tau/4` is red on `tau` against LAPACKE (the only independent `tau` oracle).
- An early `return` for dead partitions did not fire on this toolchain; `GeqrfTinySource.KernelBodyHasNoEarlyReturn` and `SynchronisesAtSubGroupScopeOnly` guard the rule by source.

### The two partition butterflies

`geqrf_tiny_reduce_fmax` and `geqrf_tiny_reduce_sum` are hand-rolled XOR all-reduces over the partition: no `reduce_over_group` exists for `SubGroupPartition`, and a work-group one emits static shared (the 48 KB hole). Being order-symmetric, every lane recomputes the Householder scalars, removing a broadcast barrier pair. Only `reqd_sub_group_size(32)` is declared; `reqd_work_group_size` would make the probed table depend on the launch.

## The register panel leaf (WP6 / P5) {#the-register-panel-leaf-wp6--p5}

`geqr2_panel_reg_device<D, N>` (`src/ops/geqrf/geqrf_panel_reg_device.hh`) is a third panel leaf in `geqrf_panel_factorize`: one panel row per work-item, the `m x N` panel in registers, work-group `roundup(m, 32)`. Shipped for `double` only (N = 16), reached from the blocked driver inside the height window; the fused apply was never built.

### The register panel leaf width and height table

Standalone `-Xcuda-ptxas -v`; all 0 B frame and spill.

| scalar | N | registers | max work-group | ptxas | shipped |
|---|---|---|---|---|---|
| `float` | 32 | 168 | 384 | 39.6 s | no |
| `double` | 16 | 136 | 480 | 6.3 s | **yes** |
| `complex<float>` | 32 | 180 | 352 | 46.0 s | no |
| `complex<double>` | 32 | 255 (ceiling) | 256 | 60.5 s | no |

Only double ships on device-link budget (`batchlas_extensions_cta` is at about 183 s of ptxas; double adds 3.4%, the others 22-33%). Turning a scalar on is `cols = 0` to non-zero in `GeqrfPanelRegPlan`, which also requires a measured `launch_max_rows`.

### The height ceiling the AOT probe got wrong

The shipped kernel uses more than the standalone 136 registers (real count in (157.5, 170.6]), so the derived 448-row ceiling failed: double n = 16 launches to m = 384 and fails from 385 (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, an abort outside a fixture). `GeqrfPanelRegPlan<double>::launch_max_rows = 384` is measured; `RegisterPanelLeafRunsAtItsOwnHeightCeiling` guards it.

### The panel height window

Double, `factor_bench`, nine interleaved reps, `t_blocked / t_reg` at saturated batch (`p5_reg_leaf_{un,}capped.csv`). Single `m x 16` panel:

| panel rows | 16 | 32 | 64 | 128 | 144 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|---|---|
| `t_blocked / t_reg` | 0.98 | 1.09 | **2.17** | **1.10** | 0.89 | 0.74 | 0.48 | 0.38 |

`kGeqrfPanelRegPolicyRows = 128` is the crossover; `geqrf_panel_reg_preferred` is the policy, `geqrf_panel_reg_fits` the capability (to 384 rows, reachable by pinning `GeqrfPanelLeaf::Register`). Square through the driver, capped: n = 16 0.98, 64 1.40, 128 1.18, 160 1.10, 192 1.07, 512 1.01; uncapped loses from n = 224.

### The double order floor moves to 76 {#the-double-order-floor-moves-to-76}

`tuned/geqrf.sm_89.txt` ranks `blocked` first for square double from n = 76 (was 96). `t_vendor / t_native` with the register leaf:

| n | 64 | 72 | 76 | 80 | 96 |
|---|---|---|---|---|---|
| batch 16384 | 1.100 | 1.105 | 1.263 | 1.400 | 1.697 |
| batch 32768 | 1.009 | 1.088 | 1.241 | 1.376 | — |

Without the leaf n = 80 reads 1.045. Auto, route read back from coverage (`p5_auto_after_flip.csv`): 76x76 1.26, 128x128 2.23, 512x512 5.01.

### QR: the register leaf A/B

`BATCHLAS_GEQRF_LEAF = auto | reg` selects the panel leaf per call (`Settings::selection`, read in `geqrf_blocked.cc`). A null A/B is what a dead pin looks like: confirm the kernel name (nsys `GeqrfPanelRegKernel`) before reading a ratio.

## The tiny `geqrf` window {#the-tiny-geqrf-window}

`factor_bench`, interleaved arms, medians of 9 (15 on edges), batch 32768 (saturation moves the answer: cfloat n = 9 is 1.99x at batch 2048, 1.02x at 32768). Ratio against the vendor; gate 1.11.

```
float             4 <= n <= 16,  21 <= n <= 32
complex<float>    5 <= n <=  8,  11 <= n <= 16,  24 <= n <= 32
double, cdouble   refused
```

| edge | float | cfloat |
|---|---|---|
| bottom | 4: 1.586 (3: vendor) | 4: 1.061 / 1.157 / 0.981 refused; 5..8 admitted, 8: 1.977 |
| N=16 bucket fill | 9..12: 1.59-2.38 | 9: 1.026, 10: 1.117 / 1.111 / 1.103 refused; 11: 1.341 |
| 16 / 17 | 3.711 / 0.812 | 1.691 / 0.553 |
| 20 / 21 | 1.043 / 1.215 | 0.752 / 0.873 |
| 23 / 24 | — | 1.067 / 1.122 (x3, admitted) |
| 32 | 3.281 | 1.865 |

- The holes are the bucket fill penalty: N in {8, 16, 32} only, and every order in a bucket runs the bucket's trip count, so n = 17..20 and the bucket bottoms lose. Do not tidy them away.
- FP64 is refused: double 0.33-0.95x at n = 8..28, 1.13x only at n = 32.
- Grid points sit on both sides of every edge (`grid_n` in `src/ops/geqrf/choice.hh`), so nearest-row lookup cannot move one. Guards: `AutoReadsTheTranscribedTable` and `RowsHoldTheTranscribedPreference` (float 17, 21); `CanRunFalsePinsThrow` and `PinnedCandidatesStraddleTheirLimits` arm the `can_run` terms.

### A measurement caveat that is part of the result

The grid was first taken while NVML was broken (kernel module 595.84 against userspace 595.91.07), so `benchmarks/gpu_guard.sh` could not see utilisation or clocks and a stand-in guard, reading `/proc/<pid>/fd`, could not see other users' jobs. One cell was contaminated (cfloat n = 23: 0.148x, then 1.067x on every guarded run), both passing `rel_sd < 10%`. Every edge was re-taken under the real guard and reproduced within 0.01x except cfloat 24, which moved the band from 25..32 to 24..32. `rel_sd` does not detect a foreign process; the guard does.

### The native tiny tie-break

Vendor-free and under a `native` pin, the rows rank `tiny` second after `vendor`, so `tiny` takes the window's gaps. At batch 32768 (tiny / CTA): float 3 1.61 / 0.54, cfloat 4 0.99 / 0.38, cfloat 10 1.10 / 0.51; cfloat 17..21 rank `cta` before `tiny` (17: 0.55 / 0.76, 20: 0.75 / 0.83). Still losing natively: float and cfloat 17..20, and n = 33..40 on CTA (0.69-0.90).

## QR: the shared WY machinery in larft_wy.hh

`src/kernels/common/larft_wy.hh` holds `larft` (the `ib x ib` T factor, Forward/Columnwise) and `pack_v` (the unit-lower V panel), shared by `ormqr_blocked.cc`, `geqrf` and `sytrd_sy2sb.cc`.

### larft_wy.hh: why the WY helpers are shared

One copy serves every caller. The legacy manual-reduction `larft` is the default; the device-BLAS one is reached only through `ormqr`'s `BATCHLAS_ORMQR_IMPL`.

### larft_wy.hh: UseDevice is a template parameter

A runtime `bool` instantiated both implementations per `(Tag, T, WG)` (32 dead entry functions, 9 s of device link). Callers passing a literal use the `_t` form.

### larft_wy.hh: pack_v indexes the row fastest

`sycl::id<3>` makes dim 2 fastest and the operands are column-major, so the row goes in dim 2 (else 32 sectors per warp instead of 4). Same for `OrgqrIdentityKernel` and `OrgqrCopyBackKernel`; worth 1.06-1.20x on `orgqr` at n >= 512.

## geqrf wide shapes, m less than n {#geqrf-wide-shapes-m-less-than-n}

A wide `geqrf` writes the k = m reflectors below the diagonal and applies \f$Q^H\f$ to every column, so R is upper trapezoidal m x n; tau holds `min(m, n)` per item. `cta` and `blocked` take any m x n (LAPACK's own m < n path; no new kernel, no separate `ormqr` pass); `tiny` is square-only. `orgqr` and `ormqr` consume the leading m x m reflector block. `GeqrfTest.WideFactorsTheWholeMatrix` forms Q with the routed `ormqr` and checks \f$\|A_0 - Q\,\mathrm{triu}(F)\|\f$ over all n columns.

Native beats cuSOLVER in every measured cell (2026-10-23, float and cfloat, 2.2x at 48x64 batch 1024 to 132x at 128x8192 batch 128). The sm_89 float and cfloat wide rows are deep (keys n in {64, 1024, 8192}, aspect in {1, 4, 64}; the ms are in `tuned/geqrf.sm_89.txt`): native everywhere except float n = 64 at aspect 4 and 64 and cfloat n = 64 at aspect 64, where the vendor leads (float 1.28x and 1.20x, cfloat 3.0x). n = 8192 aspect 1 is not measured (cuSOLVER exceeds the cell timeout; lookups resolve to the aspect-4 row, `blocked`); n = 2 keys have no valid shape and keep their transcribed rows.

Armed break: stopping the blocked trailing update at column m turns red, on all four types and nothing square or tall, `WideFactorsTheWholeMatrix`, the wide rows of `NativeFactorMatchesTheVendorElementwise`, `PinnedCandidatesStraddleTheirLimits` and `ExactWorkspaceInAPoisonedArena`.

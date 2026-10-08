# QR: geqrf, orgqr and ormqr

> **Status:** current · GPU 1 of a 2x RTX 4090 (sm_89), CUDA 13.2 · measured 2026-09-10 to 2026-09-26; tables transcribed 2026-10-05 (`source=transcribed:424a45bc`, untimed)

Native QR families (`geqrf`: `tiny`, `cta`, `blocked`; `orgqr` and `ormqr`: `blocked`), each against its `vendor` family: what `can_run` admits, which family the shipped `tuned/{geqrf,orgqr,ormqr}.*.txt` rows rank first, and the measurements behind the rows. Selection is flat (@ref design_flat_selection). The windows are rows transcribed from the deleted route-era predicates; none has been re-timed as a table.

## Conventions

- Timings: `CUDA_VISIBLE_DEVICES=1`, `WARM_S=1.5`, medians of interleaved A/B, cells with relative sd above 10% discarded, nothing timed under `BATCHLAS_KERNEL_TRACE`.
- "Vendor-free" is `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`. A pin inside a vendor build (`BATCHLAS_GEQRF_ROUTE=cta`) measures one family, not what a vendor-free user gets on Auto.
- `local_mem_size` is 101,376 B. Every capacity below uses the budget 97,280 B (minus the 4,096 B reserve), read once into `select::Device::slm_budget` (`src/select/select.cc:222`). The build-time table value 49,152 B for any `nvidia_gpu_sm_*` (`cmake/BatchLASDetectSYCL.cmake:94-96`) is 2.06x low here, and nothing in this family reads it.

## QR: route arms

Families in tie-break order, the `can_run` that admits them, and the sm_89 rows. The sm_120 tables hold the same rows. @ref selection_tables lists the tables per op.

| op | families (`choice.hh`) | `can_run` (correctness only) |
|---|---|---|
| `geqrf` | `tiny`, `cta`, `blocked`, `vendor` (`src/ops/geqrf/choice.hh:15-24`) | native: `d.is_gpu && d.has_sg32 && !A.is_heterogeneous() && m >= n && n >= 1 && batch >= 1` (`can_run.hh:29`); `tiny` adds `m == n && n <= geqrf_tiny_max_n_for_slm<T>(budget)` (:32); `cta` is `geqrf_cta_fits<T>(m, n, budget)` (:33); `blocked` needs `geqrf_cta_max_elems_for_slm<T>(budget) >= 1` (:35-38); `vendor` is `d.has_vendor` (:39) |
| `orgqr` | `blocked`, `vendor`; last resort `vendor`, then `blocked` (`choice.hh:14-22`) | `blocked`: `d.is_gpu && orgqr_blocked_available<T>() && !heterogeneous && A.cols() <= A.rows()` (`orgqr.cc:43-44`); `vendor`: `d.has_vendor` (:46) |
| `ormqr` | `blocked`, `vendor` (`choice.hh:14-20`) | not complex `Trans`, and `d.is_gpu` / `d.has_vendor` (`ormqr.cc:49-55`) |

Auto takes the first entry of the row that `can_run` admits. Keys are `form`, `n` (columns) and `aspect` = `max(m,n) / min(m,n)` by integer division. First-ranked family per type:

| type | square | tall (`m > n`) | wide |
|---|---|---|---|
| `float` | `tiny` at n 4..16 and 21..32; `vendor` at 1..3, 17..20, 33..63; `cta` at 64..96; `blocked` from 97 | `cta` at n 64..96, and at n 32..96 when aspect >= 4; `blocked` from 97; `vendor` below | `vendor` |
| `cfloat` | `tiny` at 5..8, 11..16, 24..32; `vendor` at 1..4, 9..10, 17..23, 33..47; `cta` from 48 | `cta` from n 48, and from 32 when aspect >= 4; `vendor` below | `vendor` |
| `double` | `vendor` below 76; `blocked` from 76 | `blocked` from n 76; when aspect >= 8, `cta` at n 32..48 and `blocked` from 49; `vendor` below | `vendor` |
| `cdouble` | `vendor` below 256; `cta` from 256 | `cta` from n 256, and from 32 when aspect >= 8; `vendor` below | `vendor` |

- Where a row ranks `cta` first and `can_run` refuses it (past CTA's area, [CTA capacity](#cta-capacity)), `blocked` runs. For the complex types that is the only way `blocked` is selected.
- `orgqr` rows rank `blocked` first at every `m <= 512` and `vendor` from `m = 513`, all four types ([the shipped orgqr ceiling](#the-shipped-orgqr-ceiling)). `ormqr` rows rank `blocked` then `vendor` (the complex `trans=T` `vendor`-only rows are rejected earlier by `throw_if_undefined`, `ormqr.cc:59-64`); `ormqr` is native-first in every build, which `orgqr`'s `blocked` family (identity fill plus the public `ormqr`) relies on.
- Vendor-free, the second entry runs. For `geqrf` that is `tiny` in the float and cfloat gaps, `cta` for square double n <= 48 and cdouble n <= 255, `blocked` for square double 49..75, and `cta` then `blocked` elsewhere ([the native tiny tie-break](#the-native-tiny-tie-break)).
- Pins `BATCHLAS_{GEQRF,ORGQR,ORMQR}_ROUTE` take `auto`, `native`, `vendor` or a family spelling; a family `can_run` refuses throws `invalid_argument` (R6). `geqrf_cta_dispatch`, `geqrf_blocked_dispatch` and `orgqr_blocked_dispatch` bypass selection.

> **Note:** The 3.24x (`geqrf`) and 7.85x (`orgqr`) geomeans span the whole n = 4..512 grid. A vendor-present build realises only the in-window part. Full grid: [`small-n-baseline.md`](small-n-baseline.md#geqrf).

**Correctness terms.**

- `m >= n` for native `geqrf`: a wide view walks past the bottom of the panel in the trailing update. `n <= m` for `orgqr`: Q's columns live in \f$C^m\f$.
- GPU-only for both. Heterogeneous batches are refused by both (nothing in this tree gets heterogeneous-batch QR right, netlib included, `netlib_lapack.cc:1430-1443`, :1472-1477).
- `d.has_sg32` is `geqrf`'s alone. It enumerates `sycl::info::device::sub_group_sizes` (`src/select/select.cc:220`, `src/util/queue-impl.cc:334-340`) and is never inferred from `MAX_SUB_GROUP_SIZE`, which returns the first size and so accepts a `{64}` device; a `reqd_sub_group_size(32)` kernel there aborts at launch. `orgqr`'s `can_run` has neither `has_sg32` nor the SLM budget, because `ormqr_blocked` has no `reqd_sub_group_size` and holds nothing resident.

### CTA capacity

`geqrf`'s CTA family holds the whole `m x n` panel in a `local_accessor`. The ceiling is an area, `m*n <= cta_max_elems` in int64, tested by `geqrf_cta_fits` (`src/extensions/geqrf_cta.cc:333-345`); `can_run` (`can_run.hh:33`) and the CTA entry point (`geqrf_cta.cc:512`) both call it. The per-row bound is not binding (`geqrf_cta_max_m_for_slm` is `min(elems, INT_MAX)`, `geqrf_cta.cc:324-328`).

The tier is occupancy-scaled: `geqrf_cta_fits` divides the budget by `kGeqrfMinBlocksPerSm = 2` (`src/extensions/geqrf_native.hh:22`).

| type | tier capacity (elems) | square n | residency (whole budget) | square n |
|---|---|---|---|---|
| float | 11,776 | 108 | 24,320 | 155 |
| double, cfloat | 5,888 | 76 | 12,160 | 110 |
| cdouble | 2,944 | 54 | 6,080 | 77 |

Above the tier's area, `blocked` serves the shape. Its panel leaf is the same device body (`geqrf_cta_device.hh`) against a global pointer, chosen per panel by `geqrf_leaf_fits` (`geqrf_cta.cc:351-353`), which is `geqrf_cta_fits` at target 1 ([QR: the panel leaf is not the tier ceiling](#qr-the-panel-leaf-is-not-the-tier-ceiling)). Target-2 rationale: [QR: the occupancy rule](#qr-the-occupancy-rule).

### The CTA/blocked tier order (formerly the third predicate)

sm_89 rows rank `cta` before `blocked` for square float n <= 96 and `blocked` first from 97. Double: `cta` first up to n = 48, `blocked` from 49. Both complex types rank `cta` first everywhere, so the fit alone sends them to `blocked`. Tall rows use the same n edges.

> **Note:** `experiments/wp5_qr/README.md` §3a says "blocked ahead from n ~= 104 (float) and n ~= 48 (double)". The shipped edge keeps the last measured CTA-ahead cell on CTA (float n <= 96), and resolves the double n = 48 0.983 tie to CTA, because CTA needs zero workspace while the blocked driver allocates `m*nb*batch` of V, T and WY scratch.

### Block-width evidence

`geqrf` and `orgqr` compute their own block width, keyed on type and clamped to `k = min(m,n)`: **16 for double, 32 otherwise** (`geqrf_blocked.cc:41-181`, `orgqr_blocked.cc:36-107`). Neither inherits `tuning::ormqr_block_size_for_n`.

`ormqr` WY apply (on an identity), `BATCHLAS_TUNE_ORMQR_BLOCK_SIZE` forced, median ms, vendor-free, n=1024 batch=64 (`]` shipped `ormqr_block_size_for_n`, `*` best):

| nb | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 16 | 45.36 | **123.65\*** | 101.53 | 1547.54 |
| 32 | **36.82\*** | 135.65 | **81.54\*** | **1061.24\*** |
| 56 `]` | 45.48 | 174.82 | 113.18 | 1333.81 |
| 128 | 83.04 | 303.45 | 296.93 | 1936.75 |

The shipped ladder costs float 1.24x, double 1.41x, cfloat 1.39x, cdouble 1.26x here; at n=256 batch=512 the best is 16/16/16/32 and the shipped width costs 1.11x / 1.32x / 1.26x / 1.55x. The shipped buckets are a CUDA/float optimum applied to all four types (`ormqr_blocked` has no type axis in `tune.py`). Multiples of 16 only (the trailing G1 uses the block width as `m`, Tiled16 vendor-free; 24 and 56 lose everywhere); never below 32 for complex (nb=24 falls to Tiled16, 1.72-2.30x); not wider than 32 ([QR: negative results](#qr-negative-results)); `nb=16` beats 32 for double by 1.10-1.32x, so both drivers key on type.

## Measured boundaries {#qr-measured-boundaries}

### CTA vs blocked crossover

`tier_summary.txt`, vendor-free, `BATCHLAS_GEQRF_ROUTE=cta` against `=blocked`, every pin verified. Ratio is `blocked_ms / cta_ms`, so **> 1 means CTA ahead**. Batch 8192 at n <= 64, 4096 at n >= 80. Twenty of 44 (type, n) cells had the `cta` pin resolve to `native:blocked` (`m*n > cta_max_elems`) and are excluded (n/a).

| type | n=32 | n=48 | n=64 | n=80 | n=96 | n=112 | n=128 |
|---|---|---|---|---|---|---|---|
| float | 1.002 (null) | 2.686 | 2.034 | 2.037 | **1.294** | **0.821** | 0.699 |
| double | **1.049** | **0.983** | 0.922 | 0.772 | 0.731 | n/a | n/a |
| cfloat | 1.002 (null) | 3.171 | 2.093 | 1.253 | 1.079 | n/a | n/a |
| cdouble | 0.995 (null) | 2.589 | 1.929 | n/a | n/a | n/a | n/a |

Cells with n <= nb are null: the blocked driver is one panel on the resident leaf, so 1.00x is an identity.

**Bracketing.** Float is bracketed on both sides (96: CTA 1.294; 112: blocked 1.218). Double is bracketed by 32 (CTA 1.049) and 64 (blocked 1.085), with 48 a tie resolved to CTA. **The complex boundaries are not bracketed**: cfloat's last measured cell is n = 96 (1.079) against a ceiling of 110, with a collapsing margin; cdouble's is n = 64 (1.929) against 77.

Shipped default vs the other tier forced (three reps, ms): float 112 9.15 (blocked) vs 11.98, **1.31x**; float 128 9.97 vs 15.39, **1.54x**; double 80 18.49 (blocked) vs 24.07, **1.30x**; double 110 30.55 vs 42.52, **1.39x**; float 96 5.00 (cta) vs 6.02, 1.20x; double 48 19.34 (cta) vs **19.02**, tie; cfloat 96 10.64 (cta) vs **10.72**, tie.

**Mechanism.** `geqrf_cta`'s capacity is a byte budget with no blocks-per-SM term. Above about 50 KB the tile allows one work-group per SM and the per-reflector barrier chain has nothing to overlap with: float n=96 is 36,864 B (2 blocks/SM, CTA ahead 1.294), n=112 is 50,176 B (1 block/SM, CTA behind 0.821). Not checked with an occupancy counter. [The occupancy rule](#qr-the-occupancy-rule) adds the term.

### QR: the vendor baseline

cuBLAS `geqrfBatched` saturates at about 380-390 GFLOP/s (float) and 105-110 (cdouble) regardless of n; it is a latency-bound, one-column-at-a-time routine. Its wall time is nearly independent of batch at n >= 512 (float n=2048: 21,361 ms at batch 32, 23,151 ms at batch 256), so the ms column is a valid absolute target. At n=1024 the native ceilings are 3564, 1079, 1683 and 132 GFLOP/s against ~380-390, ~200, ~205 and ~105-110 (9.2x, 5.4x, 8.2x, 1.2x).

cuSOLVER `orgqr` is not batched: `backend::orgqr_vendor` (`src/backends/cublas.cc`) calls `cusolverDn{S,D}orgqr` / `{C,Z}ungqr` once per batch item, with workspace the single-item size times the batch (1164 MB for float n=64 b=8192, 4644 MB for cdouble). Any win over it is "beats the per-item loop", never "beats cuSOLVER".

### `geqrf` order and batch grid

`order.csv`, vendor ms to vendor-free ms (ratio; native ahead in bold). Native wins 25 of 36 cells, geomean 3.24x. Full grid: `experiments/wp5_qr/bench/order.csv`.

| n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32, 8192 | 0.8 to 1.0 (0.78x) | 2.3 to 10.8 (0.21x) | 1.0 to 1.4 (0.71x) | 5.8 to 17.6 (0.33x) |
| 64, 8192 | 6.3 to 2.9 (**2.14x**) | 15.7 to 29.6 (0.53x) | 13.1 to 5.0 (**2.64x**) | 31.5 to 58.9 (0.54x) |
| 256, 2048 | 121.4 to 21.6 (**5.62x**) | 228.7 to 72.3 (**3.17x**) | 227.1 to 44.4 (**5.11x**) | 434.9 to 520.6 (0.84x) |
| 1024, 128 | 2112.0 to 51.4 (**41.08x**) | 4290.9 to 169.9 (**25.26x**) | 3428.6 to 108.9 (**31.49x**) | 5993.9 to 1392.2 (**4.31x**) |

n = 2048, batch 32 reaches 181x / 85x / 103x / 14.9x. The batch schedule varies with n, so order crossovers read off this table are confounded with batch; the clean statements come from `tier.csv` and the fixed-n blocks of `batch.csv`.

**Batch dominates for FP64.** At n = 64 the double vendor/native ratio is 4.37x at batch 32, 1.16x at 512, then **0.46x / 0.53x / 0.53x** at 2048 / 8192 / 16384 (cdouble 0.55 / 0.53 / 0.52). Both arms are launch-bound below batch ~512. **Quote 0.53x**, not the 4-5x at batch 32.

**Tall panels**, the shape the callers issue (`band_reduction.cc:595`, `sytrd_sy2sb.cc:504`), `tall.csv`: native wins 26 of 32 cells, geomeans 3.34x / 2.37x / 2.44x / 0.96x (float / double / cfloat / cdouble), and the margin grows with m at fixed n (128x32 b4096 1.57 / 0.62 / 3.31 / 0.71; 1024x128 b256 10.85 / 8.30 / 7.27 / 1.67).

### `orgqr` grid

Native wins 31 of 36 cells, geomean 7.85x. Ratio is cuSOLVER's per-item loop divided by native.

| n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 32, 8192 | **123.2x** | **114.1x** | **65.1x** | **12.4x** |
| 128, 4096 | **17.1x** | **30.8x** | **10.6x** | **6.0x** |
| 512, 512 | **3.8x** | **6.0x** | **2.3x** | **1.7x** |
| 1024, 128 | **1.26x** | **2.44x** | 0.82x | 0.78x |
| 2048, 32 | 0.41x | **1.34x** | 0.31x | 0.46x |

On the batch axis (`orgqr_batch.csv`) float n=1024 goes 0.84x / **1.11x** / **1.27x** / **1.33x** at b = 32/64/128/256 and is still rising. cfloat n=1024 climbs 0.55 to 0.88x; b > 512 does not fit in 24 GB, so no crossing is claimed. At n=2048 the loss is real (float 0.33 to 0.47x, cfloat 0.26 to 0.35x, cdouble 0.44 to 0.46x). **Double never loses at any measured (n, batch).** The vendor arm is linear in batch (within 4%).

**Workspace reverses where speed does.** Native needs 3.3-6.7x less at small n and large batch (cdouble n=64 b=8192: 4870 MB vendor, 1476 MB native) and 2.7-5.5x more at n >= 1024 (float n=2048 b=32: 103 MB vs 562 MB).

### Where the time goes

nsys `cuda_gpu_kern_sum`, vendor-free (not committed, not timings; agree with `order.csv` to about 4%). Share of GPU time, float n=1024 (41x win) / cdouble n=1024 (4.3x win) / cdouble n=256 (0.84x loss): transposed GEMM (Tiled16) **46.6% / 69.7% / 51.3%**; NN GEMM 24.3 / 21.4 / 15.6%; `larft` + `pack_v` 19.7 / 6.5 / 22.3%; panel 9.3 / 2.5 / 10.6%. The double n=64 loss (0.53x) is the `cta` family: panel kernel 100.0% at FP64 rate (1:64), so no GEMM change touches it. The trailing update calls the public `gemm` (`src/ops/geqrf/geqrf.cc:39-47`) and the transcribed gemm rows reproduce these choices until the gemm retune ([gemm.md](gemm.md#choices-flat-selection-p34)). `orgqr`'s bottleneck is `larft` at 49.0% (float n=64 b=8192) and the transposed GEMM at 41.9% (float n=1024); identity fill and copy-back cost about 11% at float n=1024 and 17-22% at n=64.

> **Warning:** `cuda_gpu_kern_sum` aggregates by kernel name and GEMM kernels carry no caller tag, so an `orgqr` capture mixes in `geqrf`'s trailing updates from the untimed factor build. Use `SYNTH=1` (host-fabricated reflectors, `tau = 2/(v^H v)`); every number above is post-`SYNTH`.

### `ormqr` WY `trmm` gate

`wy_trmm_applicable` (`src/extensions/ormqr_blocked.cc:50-62`) picks the trmm tile kernel over a GEMM for the WY apply inside `blocked`: `route_has_tile_kernel && !is_complex<T> && ib <= 64`, where `route_has_tile_kernel` is `select::level3_tile_route_available<B, T>` (`src/select/vendor.hh:94-96`). It is a leg derived inside the family, so no table sees it. `BATCHLAS_ORMQR_WY=gemm|trmm` pins it (`ormqr_blocked.cc:32-46`).

On `ormqr_blocked_benchmark`, `Side::Left`, `ConjTrans`, batch 256 (128 for cdouble), nb in {16,32,64}, gemm/trmm (> 1 means trmm ahead): float 1.006-1.046x, double 1.004-1.016x, **cfloat 0.944-0.995x**, cdouble 0.958-1.010x. The n set was not recorded, so this is unverified. `ib <= 64` stays: past it the tile kernel measured 0.83-0.97x in float (a local dip; it is ahead again at m >= 512, 1.07-1.32x), and no `ormqr` block width reaches there. netlib is excluded (OpenBLAS `?trmm` is weak on a 16x16 triangle, 0.336-1.199x), as is ROCm (`rocblas_?trmm` is a per-batch loop).

## QR: negative results

The benchmark harnesses `benchmarks/geqrf_benchmark.cc` (wrong sign on the second flop term, float/double only, no answer check) and `benchmarks/gemm_benchmark` (`ld == rows`, blind to sub-view trailing GEMMs) are not evidence; the numbers here time the public API with the workspace its `*_buffer_size` asked for and check the residual in-process. The vendor build is 1.14-1.25x slower than vendor-free at N=2048 b=32 because six double trailing-GEMM cells hit the gemm `batch >= 64` edge (gemm retune target).

| Alternative | Result | Verdict |
|---|---|---|
| "The panel factorisation decides WP5" | measured panel share is 9.3% (float n=1024), 2.5% (cdouble n=1024), 10.6% (cdouble n=256); trailing update about 71%; panel is 100% only in the CTA tier, where FP64 loses | refuted |
| Tune block width on trailing GEMMs alone | G1+G3 in isolation shows a float cliff at nb=128 (6896 to 18,906 GFLOP/s); end to end nb=128 is the worst width (83.0 vs 36.8 ms at nb=32) | wrong instrument |
| Uncoalesced fills / strided V | `OrgqrIdentityKernel`, `OrgqrCopyBackKernel`, `pack_v_panel_batched` read `idx[2]` as the column; V was handed to the trailing GEMMs at `ld = m`, now packed at `ld = mp`. `orgqr` float n=512 b=512 1.196x, n=1024 b=128 1.118x, n=2048 b=32 1.059x; `geqrf` float n=512 b=512 1.054x | fixed |
| Specialise `orgqr` for the identity | `2n^3` vs `4n^3/3`; fill and copy-back cost about 11% (float n=1024) | worth at most 1.5x |
| Close the complex deficit | vendor-free pays 2.55x / 1.00x / 2.61x / 2.01x on the BLAS-3 core, almost all from G1 (4.81x / 1.00x / 4.99x / 3.12x) | needs a transposed wide-scalar or register GEMM |

## Correctness findings

### The 48 KiB launch hole

A resident-leaf launch asking for exactly 49,152 B of local memory fails in the CUDA backend (`CUDA_ERROR_INVALID_VALUE` at `enqueueKernelLaunch`). It is a byte threshold, not a shape or type condition: 48,896 B pass, **49,152 B fail**, 49,664 B pass, all four types (cold, one process per point). The reopening condition is any group algorithm in the body; `geqr2_panel_device` runs two `reduce_over_group` calls per reflector. **It hid** because the attribute is sticky per `CUfunction` and one instantiation serves every panel shape: any earlier larger launch raises the cap for the rest of the process (`geqrf_tests` reaches 100x32, 51,200 B, before 96x32).

The fix adopts potrf's band and pad (`kGeqrfHoleLo = 47104`, `kGeqrfHoleHi = 49664`, `kGeqrfHolePadTo = 49920`) in the resident `local_accessor`, `geqrf_cta_fits` and `geqrf_cta_max_elems_for_slm` (via `geqrf_hole_safe_budget`, inert at 97,280 B). `GeqrfTest.ResidentLeafLaunchHoleAt48KiB` must be the first test in its file; GoogleTest cannot assert that, so it is declaration order plus a comment.

### A residual test cannot guard a convention

Break K3 replaced LAPACK's real-beta `larfg` sign convention with `internal::larfg`'s phase-preserving one. **Every residual stayed green for every type**, because a phase-preserving factorisation is still a valid QR, just not the one `ormqr`, `orgqr`, `ormbr`, `sy2sb`, netlib and cuSOLVER use. Only the elementwise columns caught it (`dimag` 0.94-0.97, `dF` 1.4-1.8, `dtau` 0.19-0.71). `ConventionMatchesReferenceLapackWithoutAVendor` (independent host `xGEQR2`) makes the break red for all four types vendor-free; `NativeFactorMatchesTheVendorElementwise` skips there.

- `zgeqr2` applies `conj(tau)`, not `tau` (a host reference using `tau` disagreed by 1-4% for cfloat and cdouble); for real types the conjugate is the identity, so this class is invisible to half the type list.
- Control over 10 shapes x 4 types: `dF` 3.2e-06 / 8.2e-15 / 2.0e-06 / 3.8e-15 (float / double / cfloat / cdouble) against the vendor `geqrf`; `dtau` is 3-6x looser. Quote `dtau` separately from `dF`; order-of-magnitude only (`kernels/run_v.txt` disagrees with its own summary).

### The `orgqr_buffer_size` latent defect

`orgqr_buffer_size` treated `native_need == 0` as "no native tier fired", but a zero workspace is a legitimate answer (the CTA tier reports it). It was unreachable only because `orgqr_blocked_layout` always allocates `m*n*batch`. The size query now returns exactly the chosen family's need (R5). The vendor workspace is batch-linear (about 4.6 GB at cdouble n = 64, batch = 8192) and a native call must never be sized by it.

### ormqr: one route resolution for the call and its size query

`ormqr` and `ormqr_buffer_size` (`src/ops/ormqr/ormqr.cc`) make one choice from the same inputs (`key_of`, `choice.hh`, `can_run`): the call goes through `select::run`, the size query through `select::pick`, and `run` is `pick` plus the trace scope. The blocked WY width comes from `block_size`, so the query's bytes are the bytes the call checks. The route-era pair could disagree (vendor ran, blocked size returned: "insufficient workspace", 2560 against 276,480 B); a pin that fails `can_run` now throws (R6).

`block_size_hint` exists because `tuning::ormqr_block_size_for_n` is keyed on `A.rows()`, wrong for a tall skinny panel. It is clamped to `[1, k]`; the vendor family ignores it.

### Short final panels on square matrices

Reference break 1 (drop the last reflector) is green for float and double on a square matrix and red for complex: the final reflector acts on a 1x1 block, where LAPACK returns `tau = 0` for a real scalar but non-zero for a complex one. **A short-final-panel regression test on a square real matrix guards nothing**; use `m > n`, a middle panel, or complex. Break 5 (drop a middle reflector) is red for all four types and is the standing check. Break N1 (`ib` to `nb` in the larft and pack_v calls) cannot turn anything red: `can_run` requires `m >= n`, so a short final panel has `n2 == 0` and the driver exits before the WY update; the error class exists only at the leaf (break KD).

### Break sweeps

Twelve experiment-harness breaks (5 reference, 7 kernel) and 13 against the shipped suite. The informative ones:

| break | what it deleted | outcome |
|---|---|---|
| KA | 48 KB hole pad on the resident leaf | `ResidentLeafLaunchHoleAt48KiB`, all 4 types, only that test |
| K1 / KE | `conj(tau)` in the panel apply | complex only red (correct null for real) |
| K2 | `T^H` to `T` in the WY update | `cta` green (no WY update); `blocked` red for every type |
| K3 / BR1 | LAPACK's beta sign choice | residuals green; `dimag`, `dF`, `dtau` red |
| K5 / KG / KB / KC | `tau` batch stride `k` to `ib`; sub-view with `nr` not parent `ld`; panel loop bound | red everywhere (KG: 68 rows / 8 tests, item 0 unaffected, so the checker walks every item) |
| BR3 / BR4 | division arm of the reciprocal guard / `native_tier_preferred` removed | one named test each (`SubnormalScaleColumnsTakeTheDivisionPath`; `NativeTierTieBreak...`) |

- **The checker was defective**: the first K5 run was green with `tau` poisoned to -12345, because probes overflowed to NaN and `std::max(0.0, NaN)` returns 0. `qrcheck.cpp` now uses a NaN-propagating `nanmax`; `experiments/wp5_qr/baseline/wp5qr.cpp` still has the defect.
- **K4** (barrier B2 between reads of `A(j,j)` and the beta write) stays red on nothing; the SYCL memory model requires it, so it stays. **K6** (W1/W2 stride) is a bad break: private scratch accepts any agreed stride.
- **`tests/orgqr_tests.cc` stays green under every kernel break** in the vendor build: it pins no route, so calls resolve to cuSOLVER. It discriminates only in `build-novendor`.

### Suite status

`geqrf_tests` is 80/0 in `build/` and 72/0 in `build-novendor/`. The vendor-free suite went from 26 of 54 to 30 of 55 passing, nothing newly failing; `orgqr_tests` 16 to 8 failures, `ormqr_tests` 24 to 16 (CUDA rows only). The route diff changed zero vendor-present decisions.

> **Warning:** The label selection in the gate must be one `-L "blas|ortho"` flag. Repeated `-L` flags AND together, select zero tests, and exit 0.

## Open debts

1. **No window is a timed table row, and only the in-window part of the native wins is realised.** Widening one needs an end-to-end gate (a 2.16x kernel win once became an 11% `gesvd` loss). Fix: a `tools/tune/batchlas_tune` retune of `geqrf`, `orgqr`, `ormqr`.
2. **The tier crossover is measured on square shapes**; tall rows reuse the square edges (P7's tall ladders found no cell where the chosen arm is slower). The per-type crossover area is not a key the rows can see.
3. **cfloat 97..110 is extrapolated; cdouble has no measured cell above n=64** (ceiling 77). The capacity (cfloat 76, cdouble 54) refuses both bands, so `blocked` serves them.
4. **`geqrf_panel_wg(n, max_wg)` never looks at `m`**: at double n=64 at least 193 of 256 work-items run zero iterations yet pay a 256-wide `reduce_over_group`; that kernel is 100.0% of GPU time in the worst `geqrf` loss (0.53x, 14.5% of FP64 peak).
5. **`larft_forward_columnwise_wg_legacy` is 15.2% of float n=1024 and 20.3% of cdouble n=256** for about 1/60 of the GEMM's arithmetic (it re-reads two V columns per pair; the T-update is serial on `lid == 0`).
7. **The `!dev_is_zero(r)` half of the reciprocal guard is untested** (`r == 0` needs `alpha - beta` to overflow, input ~1e308).
8. **Per-call host overhead of the eigen drivers' `geqrf` calls is unmeasured** (`sytrd_sy2sb.cc` calls it about 500 times at n=1024).
9. **`ormqr`'s `blocked` width ladder is float-only and keyed on `A.rows()`** (`block_size()`, `ormqr.cc:70-74`; `tuning_params.hh:139-145`, :232-235), costing 1.11-1.55x for three types. It is a derived leg, so no table retune fixes it. `build/include/batchlas/tuning_params.hh` is never compiled (`src/CMakeLists.txt` puts `${PROJECT_SOURCE_DIR}/include` first).
10. **Callers that size once at a bounding panel pay the blocked layout**: `geqrf_buffer_size_bound` (`geqrf.cc:97-107`) reserves 168 MB (float) or 671 MB (cdouble) at n=64 batch=8192 even where `cta` runs; sizing W1/W2 on `n - nb` took about 28% off.
14. **double and cdouble rank `cta` ahead of `tiny` in the vendor-free walk**, though the tiny grid measured CTA and Blocked at 0.14-0.67x of vendor at n = 8..28 against `tiny` 0.33-1.13x. One retune or measured edit fixes it; cdouble has no grid.

Closed: the occupancy-scaled tier capacity (P7), the `potrf` tier order, the `ormqr` route resolution (P5), the tiny window.

## Raw evidence

Git tag `perf-evidence/vendor-independence` (`git show perf-evidence/vendor-independence:<path>`), under `experiments/wp5_qr/`: `README.md` (index, repair pass); `baseline/` (`README.md`, `summary_nb*.txt`, `nb*.csv`, `sat*.csv`, `gemmtrail.csv`, `panelsum.csv`, `routeq_qr.csv`, `variants.csv`, `break*.csv`/`.txt`); `kernels/` (`README.md`, `breaks_ref.txt`, `breaks_kernel.txt`, `run_v.txt`, `run_nv.txt`); `bench/` (`README.md` for method and discard rule, `order.csv`, `batch.csv`, `tall.csv`, `orgqr_batch.csv`, `tier.csv`, `tier_summary.txt`, `nsys_split.md`, `kernsum/*_kern.txt`). ormqr WY trmm vs gemm: `experiments/TRMM_SYRK_BATCHED_KERNELS.md`. In the tree: `benchmarks/results/p7_occupancy_geqrf_tall.csv`, `p5_reg_leaf_{un,}capped.csv`, `p5_auto_after_flip.csv`, `factor_baseline_orgqr_double.csv`.

## The shipped `geqrf` window

### What 96x96 resolves to, per type

96 is a probe shape in `GeqrfCandidates.AutoReadsTheTranscribedTable` (`tests/geqrf_candidates_tests.cc:609`). Per type, native over vendor ([`small-n-baseline.md#geqrf`](small-n-baseline.md#geqrf), n = 96, batch 8192): float row `tiny, cta, blocked, vendor` runs `cta`, **2.69x** (last float order ranked `cta` first); cfloat same row runs `blocked`, **2.28x** (CTA measured; the cfloat area is 5,888 elems and 9,216 does not fit; P7 measured blocked ahead, 0.924); double row `blocked, tiny, cta, vendor` runs `blocked`, **1.16x** (1.70x with the register leaf, [what the shipped route does now](#what-the-shipped-route-does-now)); cdouble row `vendor, cta, tiny, blocked` runs `vendor`, 0.49x (floor n >= 256; ranking it native would ship a loss). The tier half of the answer is the row order and the tile fit, read from the device; nothing is hardcoded to this box.

### How the window is guarded

`GeqrfCandidates.AutoReadsTheTranscribedTable` runs every edge shape through Auto and the `native` walk (floors, tall clause, tiny window and gaps, CTA/blocked crossover, area fit); `GeqrfTranscribedTable.RowsHoldTheOldPreference` (:820) reads both devices' rows. The first skips unless the budget is 97,280 B and the table is the transcription. A route-era `preferred()` that answered true for CTA once pre-empted the tier hook and shipped CTA at double n = 96, where blocked is 1.37x faster; a table row cannot have this defect, and the remaining risk is a row ranking `cta` first past the crossover, which the guards read.

## The shipped `orgqr` ceiling

`tuned/orgqr.*.txt` ranks `blocked` first only up to m = 512, every type, both devices (`m=512` rows `blocked, vendor`; `m=513` rows `vendor, blocked`), transcribing `rows <= 512 && cols <= 512` (`cols` is implied by `n <= m`). Grid: [`small-n-baseline.md`](small-n-baseline.md#orgqr); bracketing losses: [`orgqr` grid](#orgqr-grid). Every ratio is "beats the per-item loop", not "beats cuSOLVER".

The window has no floor: the margin is smallest at the largest order. Native over vendor at n = 512 / 256 / 64 / n <= 16: float 3.64 / 5.30 / 27.10 / >= 181; cfloat 2.54 / 4.04 / 15.09 / >= 172; double 4.61 / 8.46 / 27.09 / **>= 97.72**; cdouble 2.77 / 4.75 / 11.31 / >= 42. That is 66 square cells, orders 4..512, zero losses, minimum 2.54. The double floor is from `benchmarks/results/factor_baseline_orgqr_double.csv` (`double 16x16 b32768`: vendor 1642.61 ms, native 16.8097 ms); the predicate comment's `>= 98` overreaches by 0.28.

The 512 ceiling is the load-bearing bracket: above it cfloat n = 1024 is 0.82x, cdouble 0.78x, every type at n = 2048 loses (float 0.41x, cfloat 0.31x, cdouble 0.46x). Only float n = 1024 recovers with batch (0.84 / 1.11 / 1.27 / 1.33 at batch 32..256); one type crossing is not a window, and the rows have no batch key (`key_names` is `m:log n:log:2`, `src/ops/orgqr/choice.hh:25`). The window does not weigh the vendor arm's 3.3x workspace.

## CTA capacity: the `INT_MAX` reading

The device reading behind [CTA capacity](#cta-capacity); a test that gives `geqrf` a permissive capacity makes half its assertions vacuous. The row bound equals the element bound, so the area binds: budget 97,280 B gives float 24,320 elems, double and cfloat 12,160, cdouble 6,080 at residency, half at target 2. The complex rows rank `cta` before `blocked` at every order, so only the fit sends them to `blocked` and an unbounded-budget test can never observe it (float and double keep a real column edge, 96 and 48). Hence `GeqrfCandidates.AutoReadsTheTranscribedTable` runs only at the real budget (`this->budget() != 97280` skips, `tests/geqrf_candidates_tests.cc:612`); its cfloat 77x77 and cdouble 96x96 vendor-free rows expect `blocked` from the fit alone.

## The `geqrf` order floor and the tall-panel clause

The `geqrf` rows carry two windows, cut from the route-era `preferred()`: `cols() >= floor_n || (rows() >= 128 && cols() >= 32 && rows() >= tall_aspect * cols())`, transcribed as `n >= 32 && aspect >= tall_aspect`. [QR: route arms](#qr-route-arms) lists the first-ranked family per type; the per-cell grid is in [`small-n-baseline.md#geqrf`](small-n-baseline.md#geqrf).

### Why the floors sit where they do

cuBLAS `geqrfBatched` saturates near 380 GFLOP/s regardless of n, so the native arm pulls away as n grows. The floor is the first order that clears the flip gate `t_native <= 0.90 t_vendor` (ratio >= 1.11) at the top of the batch ladder, with the order below measured as a loss:

| T | floor | ratio at floor | bracketing loss below |
|---|---|---|---|
| `float` | 64 | 1.71 | 48: 1.02; 33: 0.76 |
| `cfloat` | 48 | 1.74 | 33: 0.69; 32: 0.62 |
| `double` | **76** (was 96) | 1.26 | 72: 1.11; 64: 1.01 |
| `cdouble` | 256 | 1.50 | 192: 1.06; 129: 0.58 |

Float n = 48 (1.02) and cdouble n = 192 (1.06) sit in the gate's dead band. The double row is re-measured with the register leaf ([the double order floor moves to 76](#the-double-order-floor-moves-to-76)).

### Tall panels cross over earlier

Callers issue `m x kd` panels with kd typically 32 (`sytrd_sy2sb.cc:509`, `band_reduction.cc:603`), and every square cell at that width loses. Native over vendor at the top of the batch ladder (b16384 for `512x*` and `128x32`, b8192 for `1024x128`):

| shape | aspect | float | cfloat | double | cdouble |
|---|---|---|---|---|---|
| 128 x 32 | 4x | 2.23 | 3.79 | 0.68 | 0.68 |
| 512 x 32 | 16x | 2.68 | 3.39 | 1.58 | 2.16 |
| 512 x 64 | 8x | 3.175 | 4.102 | 2.373 | 1.734 |
| 1024 x 128 | 8x | 7.16 | 5.090 | 6.418 | *excluded* (1.691 at b8192, paging; 2.96 at b4096) |

The aspect floor is per type: at 4x the 32-bit types win 2.23-3.79 and the 64-bit types lose at 0.68, so a type-independent floor of 4 would route two measured losses native. The 64-bit floor is 8x, bracketed on both sides (1024 x 128 wins 6.418 for double; 512 x 64 wins 2.373 and 1.734; 128 x 32 loses 0.68). Unbracketed debts: no tall cell narrower than 4x was measured, and `rows() >= 128` is unbracketed (128 x 32 is the shortest tall panel in the grid).

### Why the window answers for exactly one tier

A route-era window that answered true for whichever tier led pre-empted the tier hook (double n = 96: CTA 65.19 ms against blocked 47.70 and vendor 55.27). Rows rank one native family first, by construction.

**The fit comes before the order.** For complex types the order is "CTA wherever it fits". At cfloat 256x256 CTA cannot hold the tile (65,536 scalars vs 12,160 at residency) and blocked measures 7.51x; `select::choose` skips every entry `can_run` refuses, so the row `tiny, cta, blocked, vendor` runs `blocked` without a special rule.

## QR: the occupancy rule

`geqrf_cta_max_elems_for_slm<T>(budget, min_blocks_per_sm)` divides the device budget by an occupancy target before converting to an element count. The hole clamp is applied after the division, so a scaled budget can land inside the 48 KB band even when the whole one did not.

### Why geqrf's target is 2 and not 4

`resident::kMinBlocksPerSm` is 4 (potrf and getrf); geqrf uses 2 (`kGeqrfMinBlocksPerSm`), because its alternative to the CTA tier is the multi-launch blocked driver, so an occupancy-limited kernel is still best. At target 4 the advertised area falls below the CTA-vs-blocked crossover: cfloat 64x64 b8192 goes 4.9460 to 9.0565 ms (**1.831 slower**) and float 96x96 b4096 4.9748 to 5.6843 ms (**1.143 slower**), both routed cells. At target 2 both return to baseline (1.000, 0.999). The float ceiling is 108, inside the measured 96..112 crossover.

### Settling the target on the tall panels

The tall panels (`sytrd_sy2sb.cc:509`, `ortho.cc`, `band_reduction.cc:603`, `m x nb`, nb 32..64) are what the eigen drivers issue on every call. At target 2 Auto goes `cta` to `blocked` for float 512x32, 384x32 and 256x64, cfloat 256x32 and double 256x32, and the move costs nothing (`benchmarks/results/p7_occupancy_geqrf_tall.csv`, 144 rows, interleaved, residual and orthogonality checked). `blocked_ms / cta_ms` (> 1 means CTA ahead), batch 8192: float 512x32 1.001, 384x32 1.002, 256x32 1.004, **256x64 0.703**; cfloat 256x32 1.001; **double 256x32 0.935**. Float 1024x128 (131,072 elems, batch 4096) is the control: never CTA-eligible, blocked 63.9 ms vs vendor 366.7. CTA wins no cell that the clamp moves.

**Why the ties are ties.** The blocked width here is nb = 32 (16 for double, `geqrf_blocked_debug_params`). At n <= nb the blocked route factorises the whole panel in one leaf call, the same `geqr2_panel_device` launch as the CTA tier through `geqrf_leaf_fits` at the whole budget; the 0.1-0.4% is host-side framing and results are bit-equal. At n > nb the driver splits: float 256x64 becomes two 256x32 panels plus `larft` and a trailing GEMM and is 1.42x faster than the one-shot panel (b4096 0.729, b8192 0.703, b16384 0.690); double 256x32 splits into two 256x16 panels and wins 1.07x.

**Target 4 is refuted with fresh numbers**: float 96x96 is 1.154 CTA-ahead (was 1.143) and cfloat 64x64 1.823 (was 1.831); both would go to blocked.

**Decision: `kGeqrfMinBlocksPerSm` stays 2.** `GeqrfTest.OccupancyTargetIsPinned` fails if the value moves. At every m x nb panel with nb in 32..64 the routed arm is never slower than the one passed over.

#### What the clamp does cost, and where the real cut belongs

A single scalar target is a proxy for a per-type crossover area and lands 4-28% below the measured one. Crossover brackets at fixed n, `blocked_ms / cta_ms`: float n=64 between m=192 (12,288 elems, 1.231) and m=224 (14,336, **0.706**); float n=96 between 128 (12,288, 1.191) and 160 (15,360, **0.672**); cfloat between 128x48 (6,144, 1.925) and 128x64 (8,192, **0.913**); double n=32 is non-monotone (0.935..1.148 over m = 192..352, within 1.15x; the dip at m = 224..256 reproduced at 11 reps).

Routed cells cut below the crossover by target 2: float 128x96 (1.191), 192x64 (1.231), cfloat 128x48 (**1.925**), double 192..352 x 32 (1.04-1.15). Target 1 costs float 224x64 (1.417), 256x64 (1.423), 192x96 (1.450), 160x96 (1.487) and double 224..256 x 32 (1.07). Target 4 costs everything target 2 does plus float 96x96 and cfloat 64x64, so it is strictly worse; 2 is neutral-or-better on the driver shapes.

> **Note:** The largest debt is cfloat 128x48 at 1.925: a ceiling that misses by 4% (5,888 advertised against a crossover above 6,144). The right cut is a measured per-type area in the table (speed cutoffs belong in rows, R4; `can_run` is correctness only, R3); `n` and `aspect` together can express an area, \f$m n \approx n^2 \cdot \mathrm{aspect}\f$. The fix is a retune that times `cta` against `blocked` on the tall grid.

### The `geqrf_blocked_debug_params` key

`geqrf_blocked_debug_params<T>(ctx, m, n)` packs nb in the low 16 bits and the leading panel's leaf in the high 16 (`1` = resident, `2` = global); the word is 0 when the driver is absent. It answers `geqrf_leaf_fits` at the whole SLM budget, not `geqrf_cta_fits` at the scaled one. `tests/geqrf_tests.cc` compares the high half against `1u`/`2u`. The `getrf` hook uses the same encoding ([lu.md](lu.md#one-spelling-per-ceiling)).

### QR: the panel leaf is not the tier ceiling

`geqrf_cta_fits` is the tier's predicate (occupancy-scaled); `geqrf_leaf_fits` is the residency predicate at the whole budget. `geqrf_panel_factorize` and the leading-panel tag in `geqrf_blocked.cc` use the latter, so no blocked panel loses residency to the occupancy target. `GeqrfTest.ResidentLeafLaunchHoleAt48KiB` uses the leaf predicate for the same reason: a 48 KB tile is far above the scaled tier ceiling, so only the resident leaf can reach the hole.

### Packed resident panels

`geqr2_panel_device` is templated on `GeqrfScope`. Under `SubGroup` the barriers are sub-group barriers, `team`/`nteams` collapse to 0/1, and both `reduce_over_group` calls become butterflies (`geqrf_sg_max`, `geqrf_sg_sum`), removing the static shared allocation that alone trips the 48 KB hole. Packing is offered while `m <= 32 && n <= 32`.

Gain 1.2-5.0x (ms before to after, batch 16384 unless noted): float 8x8 b32768 0.7287 to 0.1466 (vendor 0.0613); float 32x32 1.9716 to 1.2037 (vendor 1.4448); double 32x32 21.6526 to 6.9274 (vendor 4.5973); cfloat 32x32 2.6749 to 2.2372. It is largest for float 8x8 and double 32x32, where the old launcher gave a 64-element column a reduction over 256 lanes.

**The one cell this cost:** cdouble 64 x 64, batch 8192, native arm 59.17 to 98.42 ms (**1.66x slower**). 4,096 elements is above cdouble's advertised 2,944 and fits only at target 1, so the shape leaves CTA for blocked. cuBLAS serves it in 31.48 ms, so native loses either way; the cdouble rows rank `vendor` first below 256, so it shows only vendor-free or under a pin.

## The tiny tier (WP6 / P1): square n <= 32 in registers

`src/extensions/geqrf_tiny.cc`, the `tiny` family (first in `geqrf`'s list, `src/ops/geqrf/choice.hh:15`). One matrix per `SubGroupPartition<N>`, N in {8, 16, 32}; lane `r` owns row `r` in a compile-time `D rA[N]`. Each work-group has two sub-groups (`kTinyWgSize` = 64 work-items), carrying `64 / N` matrices. Auto runs the tier where a row ranks it first ([the tiny `geqrf` window](#the-tiny-geqrf-window)); vendor-free it takes the gaps. `BATCHLAS_GEQRF_ROUTE=tiny` pins it for any square shape `can_run` admits.

It replaces CTA's per-reflector, per-trailing-column butterfly (496 dependent shuffle chains per matrix at n = 32): one reduction per column, Householder scalars replicated to every lane by an order-symmetric butterfly, the trailing update an elementwise product into one local tile. The matrix stays in registers from load to store.

### The tiny ceiling predicate

`geqrf_tiny_max_n_for_slm<T>(budget)` is the one predicate for the ceiling (`can_run` at `can_run.hh:32`, the entry point and the tests all call it). It returns 0 when no bucket fits. The type ceiling is **32**, and **16 for `complex<double>`** ([the cdouble N=32 cell](#the-cdouble-n32-cell)). It is a walk with a `break`, which keeps the range contiguous, and is not monotone in bucket width because the chunk rule halves the tile as N grows. The `can_run` term is square-only: the register array is the matrix, and the ceiling is an order because the footprint is per work-group. An untimed family is never ranked first by a measured row; the tier reaches Auto only through rows a grid wrote.

### The tiny tier register table

`scripts/register_probe.sh out.log '' batchlas_extensions_cta` on sm_89 at work-group width 64. **The third argument is load-bearing**: the default target `batchlas_sycl` contains none of these kernels, and a clean report from the wrong library looks the same. `blocks/SM = min( 65536 / (ceil8(regs) * 64), 1536 / 64, 24 )`.

| T | N | C | registers | blocks/SM | occupancy |
|---|---|---|---|---|---|
| float | 8 / 16 / 32 | 8 / 16 / 32 | 64 / 91 / 142 | 16 / 10 / 7 | 66.7 / 41.7 / 29.2% |
| double | 8 / 16 / 32 | 8 / 16 / 16 | 84 / 124 / 193 | 11 / 8 / 5 | 45.8 / 33.3 / 20.8% |
| cfloat | 8 / 16 / 32 | 8 / 16 / 16 | 72 / 128 / 168 | 14 / 8 / 6 | 58.3 / 33.3 / 25.0% |
| cdouble | 8 / 16 | 8 / 8 | 110 / 162 | 9 / 6 | 37.5 / 25.0% |

All eleven instantiations have zero spill and zero stack frame and sit at or above the four-blocks-per-SM floor. Static shared is 0 B for every `GeqrfTinyKernel`, so the 48 KB hole is unreachable from this TU; the dynamic local request is 2,560-8,960 B per work-group.

The table lives in `geqrf_tiny_device.hh` as `GeqrfTinyRegs<D>`, keyed on the device scalar, and is the single source of truth: the chunk rule reads it, the hard `regs x wg <= 65536` gate is derived from it (`max cell + 8`; it can never fire at this width), and the R1 residency gate is a `static_assert` per cell (the binding one). The fallback model `R(N, words) = N*words + 88` is wrong in both directions (about -30% at N = 8, +25% at N = 32); `ptxas` allocates in banks of eight, so every residency formula rounds first.

### The launch shape: 64 work-items, not 128

The `local_accessor` is exactly linear in M with no fixed term, so a wider group amortises nothing. 64 work-items is equal or better in occupancy in all eleven cells and strictly better in four (float N=32 29.2% vs 25.0%; double N=8 45.8% vs 41.7%; double N=32 20.8% vs 16.7%; cdouble N=8 37.5% vs 33.3%), and halves bytes per group. `getrf_tiny.cc`'s four-sub-group rationale does not transfer: geqrf is the one tiny arm that uses local memory.

### The chunk width, and the one cell that is pinned

C is the largest of {N, N/2, N/4} at which `blocks_by_slm >= blocks_by_regs` (both compile-time; the register side is read from the probe). Local memory per matrix is `N*(C+1) + C` scalars (product tile plus a separate C-element `y`, which is why two barriers per chunk cover all hazards). C = N except double N=32, cfloat N=32 (C=16) and cdouble N=16 (C=8); local memory is never the binding residency limit.

**double N=32 is pinned to C=16.** C=32 would need 218 registers (ceil8 224) and 4 blocks/SM instead of 193 (200) and 5, for 62 instead of 124 barriers per matrix. 4 blocks/SM is R1's floor, so the pin stays; `GeqrfTinyRegs<double>::pin[2]` records it and a `static_assert` re-derives it.

**The tile's leading dimension is `C + 1`**: the odd stride spreads a column's rows across banks. The compile-time derivation uses `kGeqrfTinyReferenceSlm = 97,280` B; runtime decisions read `LOCAL_MEM_SIZE` through `resident::device_slm_budget`.

### The stack frame is the gate for this kernel, not the spill counter

`register_probe.sh` advises gating on zero spill and ignoring stack frame; for this kernel that inverts. The first probe showed double and cfloat N=32 with a **256 B frame** (cdouble 544 B) and 0 spill: `#pragma unroll` is a hint, and at N = 32 with C = N/2 the doubled phase-2 body crosses clang's threshold, `rA[j]` becomes a dynamic index and the array moves to local memory. Fix: `#pragma clang loop unroll(full)` on every loop whose index reaches `rA` (then 193 / 168 regs, 0 B). The compile-time signal is `-Wpass-failed=transform-warning` "loop not unrolled".

### The cdouble N=32 cell

Not instantiated: `geqrf_tiny_type_ceiling<complex<double>>()` is 16. `trsm_native.cc` ships a `complex<double>` N=32 body with the same 128-register array at 226 registers and zero spill, so the cell is reachable; it is left out to keep the instantiation count at 11. It is the largest unclaimed cell (cuBLAS at cdouble n=32 b8192 takes 5,758.5 us against a 282.6 us DRAM roof; the CTA arm is at 0.33x) and the first extension after v1.

### R7: the device link

`geqrf_tiny`'s 22 entry functions cost 19.4 s of `batchlas_extensions_cta`'s 182.9 s of `ptxas` (10.6%), inside R7's 15% for this tier; the aggregate for all three tiny tiers is over budget ([lu.md](lu.md#r7-the-device-link-all-three-tiny-tiers-landed)).

### What the design gives up, and what to A/B first

- Phase 2 wastes lanes (only `min(C, n-1-j)` of N active; double N=32, cfloat N=32, cdouble N=16). Barriers per column are `2 * ceil((n-1-j)/C)`.
- A padded `ld` may split DRAM sectors when several matrices share a sub-group (N = 8, `ld = n + 5`); if it measures large, the fix is a routing gate on `ld`.
- double and cdouble cannot reach the DRAM roof (FP64 is 1/64 on Ada); report them against the vendor only.

### Why bit-identity against the CTA route is not the padding test

The CTA route reduces the column norm over 32 sub-group lanes (`geqrf_sg_sum`, `geqrf_cta_device.hh:169-186`); the tiny kernel uses an N-lane butterfly, so the last ulp differs and bit-identity is unobtainable. The shipped guard: NaN in the `ld` pad, the stride pad and every row and column beyond `n`, then a finite correct factor and an unchanged NaN pad.

### Why the elementwise scan is the m x n window only

`G7` in `tests/geqrf_tests.cc` poison-fills the buffer with `mk<T>(-9.75e3, 4.5e3)` and scans only the `m x n` window: a whole-buffer scan takes its scale from the padding, is about 1000x looser than the tolerance, and admitted a float disagreement of about 0.07.

### The fixture's tolerance floor, and why it is new

`residual_tol` and `orth_tol` were `0.5 (m + k) eps`, which stops standing in for the backward-error constant below m + k = 16 (1x1 cfloat residual 1.2 eps, 2x2 float 2.6 eps). A floor of 8 eps was added; it loosens no shape tested before (smallest 16x16). `TinyIsNoWorseThanTheCtaRouteAtTinyOrders` requires the tiny residual and orthogonality within **2x** of CTA's (n = 1..12) or under the floor; 4x let a register kernel lose two bits.

### Break sweeps: the tiny tier

Breaks planted, rebuilt, run, restored: store guard `k < N` instead of `k < n` **RED** (wrote outside the window); drop `lane < j` from the `vr` select **RED, all four types**; drop `sg_id *` from `tiny_partition_id` **RED, all four types** (packed item 4 differs from its solo run); restore `if (prob >= batch) return;` **GREEN, not observed** (below).

#### The synthesis pass: four more breaks

Against the WG=64 tree: deleting one of two `sycl::group_barrier(sg)` in the chunk body is red in the source test `ChunkBodyCarriesExactlyTwoSubGroupBarriers` but **GREEN numerically, 34 of 34** (the chunk body is warp-uniform on sm_89, so missing barriers are invisible to the numerical suite; the count is guarded by source). Widening the norm butterflies to `<32>(sg, ...)` is **RED** (a neighbour's NaN changes item 5; 27 of 41 tiny cases). Storing `2v` and `tau/4` is **RED on tau** (exactly 4x against LAPACKE); LAPACK's beta sign to `+SIGN` is red on tau and residual.

- **The early-return break did not fire**: planted at batch = 35 with a packing of 16 (a dead partition in sub-group 0), the undefined idiom is tolerated by this toolchain's `group_barrier(sub_group)` on sm_89. The rule is guarded by source text: `GeqrfTinySource.KernelBodyHasNoEarlyReturn` scans the body for `return`, and `GeqrfTinySource.SynchronisesAtSubGroupScopeOnly` rejects `get_group()` and work-group `reduce_over_group` (a work-group barrier here is a race, and work-group static shared lands on a neighbouring kernel's cold launch).
- The residual is not blind to `tau` on the real types (`v(j) = 1` is implicit, so `host_form_Q` reconstructs Q from it). The LAPACKE `tau` comparison stays because it is the only assertion with an independent oracle.

### The two partition butterflies

`geqrf_tiny_reduce_fmax` and `geqrf_tiny_reduce_sum` (`geqrf_tiny_device.hh`) are hand-rolled XOR butterflies over the partition (after `steqr_cta_device.hh`'s `partition_reduce_fmax`). No `reduce_over_group` overload exists for `SubGroupPartition`, and a work-group one emits static shared and reopens the 48 KB hole. They are all-reduces on purpose: the butterfly is order-symmetric, so every lane recomputes the Householder scalars itself, removing the publish/broadcast barrier pair; widening either to `<32>(sg, ...)` leaks across partitions.

`geqrf_tiny_slm_elems(n_pad, c)` is `n_pad * (c + 1) + c`; `kGeqrfTinyRegOverhead = 88`, `kGeqrfTinyRegMargin = 8`. In `GeqrfTinyRegs<D>`, zero in `at` selects the model fallback and `pin` overrides the chunk width (not a free knob: the derivation cannot see a C that costs a block by raising registers). `reqd_work_group_size` is deliberately not declared (it emits `.maxntid` and makes the probed table depend on the launch); only `reqd_sub_group_size(32)` is.

## The register panel leaf (WP6 / P5)

`geqr2_panel_reg_device<D, N>` (`src/extensions/geqrf_panel_reg_device.hh`) is a third panel leaf beside the resident and global ones in `geqrf_panel_factorize`: one panel row per work-item, `rA[N]` holds the whole `m x N` panel in registers between one coalesced load and one store, work-group `roundup(m, 32)`, three barriers per column.

### What landed and what did not

Landed: `GeqrfPanelRegKernel<T, N>` (`double` only), reachable from the blocked driver (Auto picks it inside a measured height window), and the `double` order floor 96 to 76 ([below](#the-double-order-floor-moves-to-76)). Not built: `LarfApplyRegKernel`, `geqrf_fused_driver`, `orgqr_fused`, `ormqr_fused`, a `fused` family, a `larft` replacement (open debt 5).

The fused apply was dropped on a reading of the tree: `geqrf_blocked.cc` exits its panel loop at `n2 <= 0`, so with nb = 32 an `m x 32` panel runs one `geqrf_panel_factorize` launch with no `pack_v`, `larft` or trailing GEMM. The tall-panel targets (float 128x32, 512x32, cfloat 512x32) are exactly those shapes; only double at nb = 16 splits them.

### The register panel leaf width and height table

Standalone `-Xcuda-ptxas -v` link; every row 0 B stack frame, 0 B spill.

| scalar | N | registers | max work-group | ptxas, both entry copies | shipped |
|---|---|---|---|---|---|
| `float` | 32 | 168 | 384 | 39.6 s | no |
| `double` | 16 | 136 | 480 | 6.3 s | **yes** |
| `complex<float>` | 32 | 180 | 352 | 46.0 s | no |
| `complex<double>` | 32 | 255 (ptxas ceiling) | 256 | 60.5 s | no |

`probed_cols` pins the width each row was measured at; `geqrf_panel_reg_row_is_current<D>()` fails to compile if the plan's width moves.

**Why only `double` ships.** R7: `batchlas_extensions_cta` is at about 183 s of `ptxas` with the tiny tiers already over budget. `double` adds 6.3 s (+3.4%); float +21.6%, cfloat +25.1%, cdouble +33.1%. Cost is O(N^2): loops must be fully unrolled or `rA[j]` becomes dynamic; cdouble is clamped at 255 registers (one block per SM). Float and cfloat could fit with N = 16 and two 16-wide register panels per 32-wide blocked panel (not v1). Turning a scalar on is `cols = 0` to `cols = 32` in `GeqrfPanelRegPlan`.

**Not served:** any scalar except `double`; panels taller than 384 rows (cannot launch) or 128 (not worth launching; [the height ceiling](#the-height-ceiling-the-aot-probe-got-wrong), [the panel height window](#the-panel-height-window)). Occupancy is not gated at compile time: the leaf is 2.17x at m = 64 and 0.38x at m = 384.

### The height ceiling the AOT probe got wrong

`GeqrfPanelRegPlan<double>::probed_regs = 136` came from a standalone link, but the kernel inside `libbatchlas_extensions_cta` does not use 136 registers (`cuobjdump -res-usage` finds no entry; these kernels ship as device IR), so `65536 / ceil8(136)` = 448 advertised a height the device refuses. Bisected on the shipped library (double, n = 16): m = 256..**384** launches (work-group up to 384); m = **385**..448 fails with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. The real count is in `(157.5, 170.6]` (160 or 168). The ceiling is now the measured constant `GeqrfPanelRegPlan<D>::launch_max_rows`, with a `static_assert` requiring one for any scalar with non-zero `cols`.

- The launch aborts rather than throws: in `factor_bench` nine grid cells returned "NO OUTPUT rc=134". Inside the test fixture it is catchable.
- `RegisterPanelLeafCapabilityIsSelfConsistent` passed on the broken ceiling because nothing asked the device to run `max_m`; `RegisterPanelLeafRunsAtItsOwnHeightCeiling` is the guard, and the residual ladder now runs to 384.

### The panel height window

`factor_bench`, GPU 1 under `gpu_guard.sh`, nine interleaved reps per arm, three arms in one process (`vendor`, `blocked`, `blocked/leaf=reg`), one process per cell, double; `t_blocked / t_reg` at saturated batch. Raw rows: `benchmarks/results/p5_reg_leaf_uncapped.csv`, `benchmarks/results/p5_reg_leaf_capped.csv`.

One row per work-item makes a tall panel a wide group, so the leaf inverts with height. Single `m x 16` panel:

| panel rows | 16 | 32 | 64 | 128 | 144 | 160 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|---|---|---|
| `t_blocked / t_reg` | 0.98 | 1.09 | **2.17** | **1.10** | 0.89 | 0.89 | 0.74 | 0.48 | 0.38 |

`kGeqrfPanelRegPolicyRows = 128` is the crossover and `geqrf_panel_reg_preferred` the policy built on it. `fits` and `preferred` are two predicates on purpose: the leaf launches to 384 rows and a caller pinning `GeqrfPanelLeaf::Register` reaches all of them, while the driver and Auto consult `preferred` (gating on `fits` measures 0.38-0.89x at `mp >= 144`).

Square `m = n` through the blocked driver (capped / uncapped): n=16 **0.98** / 0.98; 64 1.40 / 1.39; 128 1.18 / 1.18; 160 **1.10** / 1.08; 192 1.07 / 1.02; 256 1.04 / 0.93; 512 1.01 / 0.93. The uncapped policy loses at every n >= 224; the capped one never loses outside its bottom edge. Bracketing non-winners for the ratio >= 1.11 window:

- Below: square n = 16 at 0.98. `m >= 32` would recover it (32x16 reads 1.09) but rests on two cells and would split every driver's short final panel onto a second leaf. Not shipped.
- Above: square n = 160 at 1.10 and n = 192 at 1.07.
- Capability edge: every tall cell from 144x16 to 512x16 reads 1.000 (both arms run the same kernel), the control showing the A/B is not measuring itself.

`double n = 64` is not a win against the arm that ships there: the leaf is 1.40x against blocked, but n = 64 resolves to the vendor, and against cuSOLVER the register arm reads 1.10x at batch 16384 and 1.01x at 32768, failing the 1.11 gate.

### The double order floor moves to 76

The double floor was 96 in the route-era `preferred()` and is now **76**: the transcribed `tuned/geqrf.double.*.txt` rows rank `blocked` first from square n = 76 (n = 75 ranks `vendor, blocked, ...`). `t_vendor / t_native` with the register leaf in the native arm:

| n | 64 | 72 | 76 | 80 | 96 |
|---|---|---|---|---|---|
| batch 16384 | 1.100 | **1.105** | 1.263 | 1.400 | 1.697 |
| batch 32768 | **1.009** | **1.088** | 1.241 | 1.376 | — |

n = 72 and 64 are non-winners at both batches; n = 76 is the lowest order that clears at both. The vendor arm is still unsaturated (1.77-1.82x per doubling of batch, native 1.99x), so ratios fall with batch; the floor is gated at batch 32768, the largest that fits. **The flip depends on the leaf**: without it the same cells read 1.045 at n = 80 and 1.029 at batch 32768.

Non-square shapes newly admitted (the tall clause `rows >= 8 * cols` does not cover them), `t_vendor / t_native`: 160 x 80 b8192 1.53; 256 x 80 b8192 1.93; 1024 x 80 b2048 2.91; 512 x 88 2.25. 256 x 80 (aspect 3.2) took the vendor and lost 1.93x.

### What the shipped route does now

Default route, `--arms=vendor,auto`, resolved route read back from `BATCHLAS_COVERAGE_OUT` (`benchmarks/results/p5_auto_after_flip.csv`). Double, `t_vendor / t_auto` (resolved): 64x64 and 72x72 b16384 1.00 (vendor); 76x76 1.26, 80x80 1.40, 96x96 1.70, 128x128 2.23 (b16384, blocked); 256x256 b4096 3.42; 512x512 b1024 5.01; 256x80 b8192 1.93. 512x16 and 64x16 sit under both the floor and the tall clause's 32-column bound and stay on the vendor.

`geqrf_cta_dispatch` asks for `GeqrfPanelLeaf::Resident` explicitly: a pinned `cta` that silently ran another kernel is the defect that assertion exists to catch.

### What this measurement could NOT establish

- Any type but double (float, cfloat, cdouble carry `cols = 0` on R7 grounds); `orgqr` and `ormqr` end to end (not re-measured); the exact register count (bounded to (157.5, 170.6]).
- The fused apply: never built, so P5's kill clause (`docs/design/small-n-factorization-plan.md`) was never testable. What shipped is its fallback, half-shipped (the larft fix was never made; the leaf swap is double only, `src/extensions/geqrf_panel_reg_device.hh:28-54`).
- The larft removal (open debt 5): nsys of 128 x 128 double at batch 8 puts `LarftKernelName` at 34%, but batch 8 is unsaturated; size the work against the larft plus pack_v pair, about 20-22% ([where the time goes](#where-the-time-goes)).

### QR: the register leaf A/B

`BATCHLAS_GEQRF_LEAF = auto | reg` selects the panel leaf per call (`Settings::selection`, consumed in `geqrf_blocked.cc`), mirroring `BATCHLAS_GETRF_LEAF`.

> **Warning:** A null A/B is what a dead pin looks like. The first full grid was thrown away: `BATCHLAS_BUILD_BENCHMARKS` was OFF in the `dev-tests` cache, so `factor_bench` was stale and its `/leaf=reg` arm set getrf's variable, which geqrf ignores; both arms ran the resident leaf and every cell read 0.999 to 1.004. The tell was an nsys `cuda_gpu_kern_sum` showing `GeqrfPanelResidentKernel` only. **Confirm an A/B by kernel name before reading its ratio.**

### Break sweeps: the register panel leaf

R9, double, `GeqrfTest/5`. Every break below was **RED**: dropping `r < j` from the `v` select (residual 63.1 vs 1.78e-15 at 7 x 5); store guard `k < n` to `k < N`; dropping `m <= max_m` from `geqrf_panel_reg_fits` (the launch aborts: "The kernel uses 136 registers per work-item"); `kGeqrfAutoPrefersRegisterLeaf = true` without the height policy (four other guards too); `launch_max_rows` 384 to 448 (red at m = 416, `UR_RESULT_ERROR_OUT_OF_RESOURCES`, caught); `geqrf_panel_reg_preferred` gating on `fits` only (`AutoTakesTheRegisterLeafExactlyInsideTheHeightWindow`); double order floor 76 to 96 (red at the `at=76` probe, now read by `GeqrfTranscribedTable.RowsHoldTheOldPreference`).

Removing barrier B3 is red, but **only from m = 129, n = 16** (the first height with five sub-groups) and green for every `m <= 128`; the guard exists only because the ladder reaches past 128.

The register leaf's residual is not compared to the resident leaf element-wise (32-lane butterfly plus a serial scan over `ceil(m/32)` slots, so the last ulp differs). `RegisterPanelLeafAgreesWithTheResidentLeaf` asserts normwise agreement at `64 * eps * m * max|R|`; both leaves call `geqrf_larfg_scalars` verbatim.

## The tiny `geqrf` window

The register tier was written, tested and never timed; the route-era predicates kept it out of Auto, so the whole square band went to CTA, a native loss (n=32, batch 8192: float 0.78x, double 0.21x, cfloat 0.71x, cdouble 0.33x; [the order and batch grid](#geqrf-order-and-batch-grid)).

### The grid

R8: interleaved arms inside one `factor_bench` process, medians of 9 (15 on confirmation cells). Ratios are against the **vendor** arm; flip gate `ratio >= 1.11`. **Saturation changes the answer**: cfloat n=9 reads 1.985x / 1.305x / **1.015x** at batch 2048 / 8192 / 32768, cfloat n=21 1.263x / 0.794x / 0.873x, float n=9 2.824x / 1.795x / 1.593x. Everything below is at batch 32768.

Edge cells (float / cfloat): n=4 1.586 / 1.061; n=8 2.964 / 1.977; n=16 3.711 / 1.678; n=17 0.812 / 0.553; n=18 0.885 / 0.623; n=20 1.043 / 0.752; n=21 1.225 / 0.873; n=23 — / 1.053; n=24 1.403 / 1.122; n=28 1.906 / 1.800; n=30 2.150 / 2.321; n=32 3.284 / 1.865. Float n=9..12: 1.593 / 1.866 / 2.189 / 2.384; cfloat 9..13: 1.026 / 1.117 / 1.341 / 1.252 / 1.495.

### The window that ships

```
float             4 <= n <= 16,  21 <= n <= 32
complex<float>    5 <= n <=  8,  11 <= n <= 16,  24 <= n <= 32
double, cdouble   refused
```

These are the square rows of `tuned/geqrf.<dtype>.sm_89.txt` (and sm_120) that rank `tiny` first, transcribed from the route-era `tiny_window()` and untimed as table rows. Grid points sit on both sides of every edge (n = 3, 4, 16, 17, 20, 21, 32, 33, ... in `src/ops/geqrf/choice.hh:39-42`), so nearest-row lookup cannot move an edge.

**The holes are measured; do not tidy them away.** `tiny_n_is_legal` admits N in {8, 16, 32} only (partition butterflies), and the loop `continue`s rather than `break`s, which keeps the array in registers with zero spill. So every order from 17 to 32 runs the full 32-iteration trip count, and the same fill penalty hits the bottom of each bucket (n = 9, 10 in N=16; n = 4 in N=8). Float absorbs it; complex does not. The same mechanism is the n = 17 cliff in `getrf` ([lu.md](lu.md)).

**fp64 is refused on evidence**: double measured 0.901x, 0.414x, 0.949x, 0.332x, 0.414x, 0.512x at n = 8..28 and 1.134x at n = 32; one marginal point is not a window. Vendor-free the tier is still the best native arm (CTA and Blocked read 0.14-0.67x).

### A measurement caveat that is part of the result

These grids were first taken while NVML was broken (kernel module 595.84 against userspace 595.91.07), so `benchmarks/gpu_guard.sh` could not poll utilisation or the start-of-run SM clock; a stand-in guard checked foreign processes through `/proc` file descriptors. A cold or throttled run could not be detected: **cfloat n=23 read 0.148x in one run and 1.053x in another**, both passing `rel_sd < 10%`. Two deliberate consequences: the complex window uses a 1.25 margin, not the 1.11 flip gate (every admitted cell cleared 1.25x, every refused cell measured at most 1.122x; cfloat n=10 at 1.117x and n=24 at 1.122x are left out on purpose), and the float window keeps 1.11 because float reproduced tightly (n=17: 0.812x twice; n=21: 1.211x then 1.225x). Both were re-cut after the fix ([the re-cut](#the-re-cut-against-a-working-guard)).

### Arming the window

Four breaks against `geqrf_tests` on the route-era `tiny_window()`: admit n = 17..20 (a measured float loss) **RED**; let CTA answer inside the window too (R8b) **RED**; fire the window with `tiny_max_n == 0` green; drop the square-only gate green (`supports()` already refused both, so the property was defended twice; a green break is evidence about the test, not the code).

Under flat selection the two live breaks are a float row in 17..20 ranking `tiny` first and a window row ranking `cta` ahead of `tiny`; `GeqrfCandidates.AutoReadsTheTranscribedTable` runs both edges through Auto and the `native` walk, and `GeqrfTranscribedTable.RowsHoldTheOldPreference` reads float 17 and 21 directly. The other two terms live only in `can_run` (`can_run.hh:32`), armed with a pin by `GeqrfCandidates.CanRunFalsePinsThrow` (`tests/geqrf_candidates_tests.cc:505`) and `PinnedCandidatesStraddleTheirLimits` (:346).

## The re-cut, against a working guard

Every cell was re-taken through the real guard ("exclusive for the whole run" on all 33 cells). The stand-in read `/proc/<pid>/fd`, which the kernel permits only for the reader's own processes, so another user's jobs on this shared box were invisible to it.

**Almost nothing changed**: every edge cell reproduced to within about 0.01x (float 17 0.812 to 0.805, float 21 1.225 to 1.215, float 32 3.284 to 3.281, cfloat 16 1.678 to 1.691). **One cell was contaminated**: cfloat n=23 read 0.148x on the stand-in and 1.067x on every other run.

Three guard-clean runs: cfloat 4 reads 1.061 / 1.157 / **0.981**, cfloat 10 1.117 / 1.111 / **1.103**, cfloat 24 1.122 / 1.122 / 1.119. n = 4 crosses the gate both ways and n = 10 sits on it; both are correctly refused. n = 24 stays above the gate in all three (within 0.3%), so the top band moves from 25..32 to **24..32**, the only predicate change from the re-cut.

**Still owed:** the band interiors (cfloat 6, 7, 12..15, 26..31; float 5..16, 22..31) were measured once with the stand-in, mostly above 1.4x; refresh when the op is next touched.

### The native tiny tie-break

2026-09-26. The vendor-free walk (and benchviz's pinned `native` arm) used the vs-vendor tiny window for native-vs-native too, so in the window's gaps it fell through to CTA (cfloat n = 4 plotted at 0.38x). At batch 32768, `t_vendor / t_arm` (tiny / CTA / Blocked): float 3 1.61 / 0.54 / 0.54; float 17 0.80 / 0.77 / 0.77; cfloat 4 0.99 / 0.38 / 0.38; cfloat 10 1.10 / 0.51 / 0.51; cfloat 17 0.55 / **0.76** / 0.77; cfloat 20 0.75 / **0.83** / 0.84; cfloat 22 0.92 / 0.91 / 0.91.

In the transcribed rows tiny is ranked second after `vendor`: every float square row up to 32 ranks `tiny` second or first, cfloat 17..21 rows rank `vendor, cta, tiny, blocked`, and the other cfloat rows up to 32 rank `tiny` first or second. The vendor-free walk and a `native` pin follow those rows.

Still losing: float and cfloat 17..20 (best 0.80 and 0.83) and n = 33..40 on CTA (float 0.78 to 0.90, cfloat 0.69 at 33). A tiny bucket for N = 4 would need its own register probe. The float n = 64 dip is not one: CTA is the best native tier there (batch 16384: CTA 1.72x, Blocked 0.95x) and cuSOLVER has a strong kernel at that order.

## QR: the shared WY machinery in larft_wy.hh

`src/extensions/larft_wy.hh` holds `larft` (the `ib x ib` triangular factor T of a block of Householder reflectors, Forward/Columnwise) and `pack_v` (the unit-lower V panel from `geqrf`'s packed output). The header keeps the invariants.

### larft_wy.hh: why the WY helpers are shared

The header holds `ormqr_blocked.cc`'s bodies (moved verbatim); a third private copy remains in `sytrd_sy2sb.cc` (around line 233), unreachable from its loop and left for a separate change. The legacy manual-reduction `larft` is the default in `ormqr` and the only one `geqrf` uses; the device-BLAS one is reached only through `BATCHLAS_ORMQR_IMPL`.

### larft_wy.hh: UseDevice is a template parameter

`UseDevice` is a template parameter: as a runtime `bool` it instantiated both implementations for every `(Tag, T, WG)`, and `geqrf` (literal `false`) compiled and linked 32 device entry functions it never launched. Entry functions fell 880 to 848, the device link 125.45 s to 116.63 s. The runtime wrapper remains only for `ormqr` (a getenv); callers passing a literal must use the `_t` form.

### larft_wy.hh: pack_v indexes the row fastest

`sycl::id<3>` makes dim 2 the fastest-varying index and both operands are column-major, so with the column in dim 2 a warp read and wrote 32 sectors instead of 4. Before the swap `pack_v` on a 17.3 MB job (float m = n = 1024, batch 128, nb = 32) took 63.7 us median per instance, 3.4x the DRAM floor (the 72 MB L2 hid most of the amplification; about 8x at larger m or batch). The same fix covers `OrgqrIdentityKernel` and `OrgqrCopyBackKernel`.

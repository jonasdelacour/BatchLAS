# SYEV {#perf_syev}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · checked 2026-10-29

Routing, tuning constants and the measurements behind them for the dense symmetric/Hermitian eigensolver
`syev`. Unless a line says otherwise: device 1, cuSOLVER 12.2.0.1, 2026-08-03..08, µs per matrix.

[TOC]

## syev: what ships today

The families live in `src/ops/syev/choice.hh`: `cta`, `cta_fused` and `jacobi` (the n ≤ 32 sub-group
solvers), `blocked`, `two_stage` and `vendor`. `select::choose` takes the first runnable entry of the
nearest row of `tuned/syev.<arch>.txt` (`sm_89`, `sm_120`), keyed on dtype, `jobz`, `n` and `batch`.
`can_run` (`src/ops/syev/syev.cc`) is correctness only: native families need a non-NETLIB GPU backend, a
homogeneous unmasked batch and square A with n ≥ 1; the small three also need n ≤ 32 (`kSmallMaxN`) and a
32-wide sub-group, and `cta`/`cta_fused` are refused on ROCm (steqr_cta gives wrong eigenvalues on gfx1200).
Uplo is not a key: Upper is [mirrored into Lower](#syev-the-upper-to-lower-mirror-for-lower-only-providers).
The sm_89 table is `transcribed` (identical across batch); sm_120 is deep-measured and batch matters there.

| decision | where | rule |
|---|---|---|
| eigenvectors, n > 32 | `jobz=V` rows | float: blocked to 448, two-stage 449..1024, vendor above. double: blocked to 448, then vendor. complex<float>: blocked to 512, then vendor. complex<double>: blocked to 256, then vendor |
| eigenvalues only, n > 32 | `jobz=N` rows | blocked to 320, two-stage above, every type |
| n ≤ 32 | both modes | double: `jacobi`. float: `jacobi` to 8, `cta_fused` above. complex<float>: `cta_fused` to 8, `cta` above. complex<double>: `cta`, except `jobz=V` n = 25..32 `vendor` first |
| vendor-free, n > 448 | `vendor` skipped | [the vendor-free double band](#syev-vendor-free-double-band) |
| batch parts | `syev_blocked_chunk`, `syev_two_stage_chunk` | [L2 chunking](#syev-l2-batch-chunking) and the scratch-budget cap |
| values-only blocked | `syev_blocked.stebz_evals` | `stebz` bisection off the tridiagonal, no `stedc` |
| complex panel width | `sytrd_block_size_default<T>` (`syev_blocked.cc`) | 32 for complex at 256 < n ≤ 512 |
| complex trailing update | `rank2k_trailing_update_supported` (`sytrd_blocked.cc`) | float and complex<float> (CUDA + cuBLAS) use rank-2k; double, complex<double> keep the GEMM pair |
| sb2st back-transform | `sb2st_back_tile_for` / `_subs_for` (`sytrd_sb2st_hh.cc`) | complex<float> n = 512: tile 2, subs 4; else the heuristic |
| block sizes | `tuning_params.hh` | ORMQR 16/16/24/48/56, SYTRD 8/8/16/8/48 (n ≤ 64/128/256/512/else); `SY2SB_ORMQR_NB_LARGE = 32` |
| stedc | `STEDC_*`, `plan_stedc_levels` | leaf threshold 32 (a hard cap), merge variant 2 (FusedCta), threads-per-root 8, wg multiplier 8 |
| two-stage band width | `choose_two_stage_kd` | kd = 32 |

**Pinning.** `BATCHLAS_SYEV_ROUTE=<family>` (also `native`, `vendor`, `auto`) or `select::ScopedPin`; a pin
`can_run` refuses, or an unknown word, throws. `BATCHLAS_SYEV_PROVIDER`, `_SMALL_KERNEL` and `_CTA_MAX_N` are
retired and not read.

## syev: eigenvector routing at saturation

The 2026-08-04 float grid (`12963a8`, median of 3, largest batch all providers fit) set the shape: blocked
wins to n = 256 (n = 64: 1.23× over vendor), two-stage at 512..1024 (n = 1024: 2441.9 against vendor 2706.8),
vendor at 2048 (1.65×, but measured at batch 64, unsaturated: the weakest boundary). The per-type numbers
below (2026-08-07, median of 5, blocked at its best `nb`) placed the crossovers.

### syev: the blocked-over-cuSOLVER headline measurement

Quoted on the front page. Float, eigenvectors:

| n | batch | blocked | two_stage | vendor | winner |
|---|---|---|---|---|---|
| 320 | 819 | **67.84** | 113.39 | 203.00 | blocked (3.0× over vendor) |
| 448 | 585 | **195.27** | 262.79 | 400.59 | blocked (2.1×) |
| 512 | 512 | 326.71 | 332.55 | 504.40 | tie |
| 1024 | 128 | 3296.80 | 2614.70 | 2552.10 | tie (vendor/two-stage) |

The float crossover is 448; two-stage wins 449..1024 by at most 1.06×, a neutral band kept as is.
Per type (blocked / vendor): complex<float> wins to 448, crosses at 512 (686.19 / 706.96), vendor at 640
(1.07×). double: blocked at 384 (771 / 875), tie at 448, vendor from 512. complex<double>: blocked at 256
(650 / 723), vendor from 288 (1131 / 975). Two-stage never wins complex eigenvectors.

> **Warning:** the complex<double> boundary is an FP64-1:64 artifact (the native panel pays far more than
> cuSOLVER) and 288..512 is a dead heat inside the 1.10× band. Re-measure on a data-center GPU.

The complex panel symv no longer emits the C99 Annex G multiply (`__mulsc3` plus an `isnan` branch;
`e4186908`, 1.22×–1.29× on the panel). The remaining complex gap is the panel itself,
see [the open panel symv](#syev-open-debt-the-single-read-panel-symv).

### syev: eigenvalues-only routing

Float, 2026-08-04: blocked wins n = 64..320 (n = 320: 48.16 against two-stage 50.35), two-stage from 512
(n = 1024: 908 against vendor 2494, 2.75×), and two-stage is also the complex winner above 320, so no per-type
split. These predate [WP1](#syev-stebz-values-only-in-the-blocked-solver-wp1), which made blocked up to 1.87×
faster at n ≤ 320; the 320 | 512 boundary is not re-derived. Do not add a batch floor: the old `batch >= 256`
term sent n = 1024 at batch 254 to the vendor (2.75× loss).

### syev: the Upper-to-Lower mirror for Lower-only providers

Blocked and two-stage are Lower-only and take Upper by mirroring (\f$O(n^2)\f$ before an \f$O(n^3)\f$ solve).
Upper/Lower measured 1.003×–1.020× (float, n = 128..1024, both modes), so a native Upper reduction was rejected.
`mirror_upper_to_lower` (`src/ops/syev/uplo_mirror.hh`, `kernels/uplo_mirror.cc`) writes
\f$A_{ji} := \overline{A_{ij}}\f$ in place; it leaves the diagonal alone and does not zero a complex diagonal's
imaginary part, as the Lower path assumes. The header is declaration-only, with explicit instantiations in the
`.cc`, because a SYCL kernel name needs one definition. Testing trap: `Matrix::Random(..., symmetric=true)` makes
Upper and Lower interchangeable, so the fixture in `tests/syev_blocked_tests.cc` poisons the strictly-lower
triangle and asserts a Lower-read solve differs ("fixture is vacuous" otherwise).

## syev: small-n kernel choice

### syev: the 2026-08-03 small-n bake-off

Build `7911847`, median of 5 process repeats at each cell's knee batch; a margin below 1.10× is neutral.

| type | n | mode | winner | runner-up | ratio | vs cuSOLVER |
|---|---|---|---|---|---|---|
| double | 4 | vectors | jacobi | cta_fused | 3.05× | 15.17× |
| double | 32 | vectors | jacobi | vendor | 1.37× | 1.37× |
| float | 8 | vectors | jacobi | cta_fused | 1.37× | 8.42× |
| float | 16 | vectors | cta_fused | jacobi | 1.22× | 4.10× |
| float | 32 | vectors | cta_fused | tridiag_ref | 1.11× | 1.79× |

Jacobi wins every double cell (1.37×–3.59×); `syev_cta` won none. Fused with its tuned multiplier beats Jacobi
for float from n = 7, but the 8 | 9 boundary stays for Jacobi's relative accuracy on graded input (4.5e-07
against 2.7e+28 for `syev_cta`). FP64 at 1/64 inflates Jacobi's double margin; re-gate on a 1:2 FP64 GPU
([steqr small-n](steqr.md#small-n-syev-routing-measured-not-shipped)).

Complex (2026-08-07, vectors): complex<float> fused wins n ≤ 8 (n = 8: 1.25×), `cta` from 9 (n = 32: 0.383
against fused 0.595, Jacobi 2.48, vendor 0.471), so the float rule is wrong for complex. complex<double>: fused
leads `cta` by only 1.03×–1.08× (not split); the vendor wins `jobz=V` from n = 26 (1.23×) to 32 (1.42×), an FP64
artifact; complex<float> never crosses (`cta` 1.23× over vendor at 32).

### syev: the fused CTA kernel design

`syev_cta` runs `sytrd_cta`, `steqr_cta`, `ormqx_cta` plus pack kernels, each round-tripping global memory;
`syev_cta_fused` keeps one problem in one sub-group partition from load to store with the same device code
(`sytrd_cta_device.hh`, `steqr_cta_device.hh`). The reduction always runs Upper (Lower input is symmetrised on
load); eigenvalues are placed by rank, so no sort kernel. Hermitian input becomes real tridiagonal by
\f$T' = S^H T S\f$, \f$S_{i+1} = S_i\,\bar e_i / |e_i|\f$, reapplied before `ormqx_cta(QL, Left, NoTrans)`.
Real input follows DSYEV (explicit Q seeds `steqr`; lane c owns column c). The tile ld is P + 1 (bank conflicts).

### syev: the Jacobi kernel design and its accuracy argument

`syev_jacobi_cta` is cyclic two-sided Jacobi, one `SubGroupPartition<P>` per problem, A and Z in local memory.
It rotates only when \f$|a_{pq}| > \mathrm{tol}\cdot\sqrt{|a_{pp}|\,|a_{qq}|}\f$ (Demmel and Veselić; LAWN 169),
so graded input keeps small relative error (proved for SPD; indefinite input is still correct); an absolute test
would forfeit that. Round-robin pairing over an even-padded index space is cyclic-by-rows (Hari and Begović
Kovač, ETNA 46, 2017); pairs are precomputed in packed 16-bit slots, and `rot_count == 0` is the converged bit.

### syev: the LOBPCG projected-solve knob

Retired 2026-10-05. `BATCHLAS_SYEV_CTA_MAX_N` sent LOBPCG's projected Rayleigh–Ritz solves to the vendor
(1.03×–1.15× on LOBPCG n = 256) but shipped off, because it flips a near-tie in
`ILUKTests.SyevxInstrumentationAndPreconditioner` (`lose_count == 0`). Reviving it needs an owner decision:
tolerate that tie or pin the test's provider.

## syev: tuning constants and the shadowed header

### syev: the shadowed tuning header

The committed `include/batchlas/tuning_params.hh` is the only copy; CMake generates none. A generated build-tree
copy used to be shadowed by it and hid a retune, so it was removed. A retune takes effect only once ported by
hand into the committed header. Every accessor reads `BATCHLAS_TUNE_*` on every call, buffer-size queries
included, so never change one between a query and its call.

### syev: the 2026-08-03 block-size sweep in the syev context

Superseded by the 2026-08-07 retune; still the only syev-context SYTRD point at n = 512: blocked, float, SYTRD
24 gave 339.9 against 380.8 at 8 (the shipped LARGE value is 8, see
[remaining items](#syev-remaining-unmeasured-items)). ORMQR 16 was optimal at n = 32..128; 128 cost 2.1×–3.1×.
Raw: `benchmarks/results/syev_retune_phaseB_float.jsonl`.

### syev: the 2026-08-07 constant retune

`924b3a59`, A/B'd end to end at the consumers, CUDA/float: syev n=64 b=4096 1.087×, n=512 b=512 1.106×;
gesvd n=128 b=1024 1.146×; everything else within 1%.

- ORMQR to 16/16/24/48/56 (`gebrd` has its own `GEBRD_BLOCK_SIZE_*`). SYTRD to 8/8/16/8/48; whether n = 512 was
  measured for LARGE = 8 is not recorded.
- `SY2SB_ORMQR_NB_LARGE = 32`: the `sytrd_sy2sb.cc` shape gate fires only at n ≥ 1024 and batch ≥ 32; the hint is
  worth 1.10×–1.12× at n = 512. It shadows `ORMQR_BLOCK_SIZE_*` on syev's hot path.
- stedc threads-per-root 8, wg multiplier 8, set from syev: stedc's own winners cost syev 2.7% at n = 256.
- `SB2ST_BACK_*` stay 0: the heuristic is within 0.6% of the tuned pair, and any pair outside tile {1,2,4,8} ×
  subs {4,8,16} silently falls through to the slower tiled kernel.

Tuning is float-only, so per-type corrections live at the consumer, where a retune cannot overwrite them.

### syev: complex panel width in the 256 to 512 bucket

The harness SYTRD value 8 costs complex 1.16×–1.20×: a complex panel column moves twice the bytes and four times
the flops, so it needs a wider panel to amortise `latrd_lower_panel`'s barrier chain. complex<float>, vectors,
blocked, median of 5, 2026-08-07:

| n / batch | nb=8 | nb=16 | nb=24 | nb=32 | nb=48 |
| --- | --- | --- | --- | --- | --- |
| 320 / 819 | 197.25 | 173.24 | **166.27** | 168.55 | 175.92 |
| 512 / 512 | 942.68 | 838.73 | 815.30 | **802.74** | 811.89 |

nb = 32 is within 1.5% of optimal across the bucket; float spreads only 1.04×, so it keeps the harness value.

### syev: the two-stage band width kd

kd = 32 is optimal at every n ≥ 256 in eigenvector mode, disproving the MAGMA prediction of 96–128 after
`f7f3c57`. Total ms, vectors, median of 3, 2026-08-04:

| n / batch | kd=16 | kd=32 | kd=64 | kd=128 | blocked |
|---|---|---|---|---|---|
| 256 / 1024 | 69.5 | **61.3** | 65.7 | 75.7 | **40.9** |
| 1024 / 128 | 445.3 | **369.9** | 433.2 | 715.3 | 475.5 |

Wide kd wins only at n = 128, where blocked wins anyway. Both modes share `choose_two_stage_kd` (see
[sytrd: the Householder chase](sytrd.md#sytrd-the-householder-chase-against-the-givens-chase) and
[the dense-to-band hint](sytrd.md#sytrd-the-dense-to-band-ormqr-block-width-hint)). "Two-stage wins at
n ≥ 1024" is shape-dependent: at n = 2048 / batch 32, grid `latrd` makes blocked 1.22× faster than two-stage.

### syev: latrd grid gate confirmed in eigenvector mode

`latrd_grid_min_n = 768` (derived in values mode) holds in vector mode too. legacy/grid at batch 1 / 8 / 64:
n = 512 1.012 / 1.018 / 0.956; n = 768 1.379 / 1.375 / 1.090; n = 256 0.734 / 0.734 / 0.788. Same crossover, so
the constant is not split; the win shrinks once the batch saturates the SMs.

## syev: the 2026-08-08 implementation plan, measured

Work packages from a profile at saturating batch, each with an end-to-end accept gate (device 1,
`--warmup=2 --min_iters=5`). WP1, WP4 and WP3 shipped (WP3 missed its 1.05× gate at 1.043×); WP2 was rejected
([rejected ideas](#syev-rejected-ideas)); WP5 and WP6 are [open](#syev-open-debts-and-unmeasured-items).
Accuracy runs for WP1 and WP3 are not recorded.

### syev: harness unblockers (WP0)

`BATCHLAS_LATRD_GRID_GROUPS` cannot force the grid `latrd` path at batch ≥ 128: it is clamped to
`MAX_COMPUTE_UNITS / batch`. `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` bypasses the cap but can deadlock (the software
grid barrier hangs silently past residency; run under a timeout) and has never been exercised.

### syev: stebz values-only in the blocked solver (WP1)

Values mode ran a full eigenvector divide-and-conquer and discarded Z; `stebz` bisection now runs off the real
tridiagonal (`f28a3fcd`), and the eigenvector workspace is no longer allocated. Batch 1024, before to after:
float n = 64 1.87×, n = 256 1.44×, n = 320 1.36×; cfloat 1.44× / 1.23× / 1.16× (stedc is real arithmetic, a
smaller share of the complex solve).

### syev: per-type sb2st back-transform geometry (WP4)

Two-stage, vectors, n = 512, batch 512: forced tile 2 / subs 4 is 1.139× faster for cfloat (971.17 to 852.77)
and 1.15× slower for float (334.28 to 384.30), so the rule is per type (`8c24626f`) and `n == 512` exactly, the
only size measured. Measure any complex stage-2 change against 852, not 971.

### syev: her2k trailing update for complex float (WP3)

For complex, `sytrd_blocked`'s trailing update issued two full GEMMs where float issues one `syr2k`. `her2k`
(one batched GEMM into scratch plus a Hermitian fold, `405b4120`) is 1.33× on the primitive (\f$n_2\f$ = 480,
batch 512) and 1.043× end to end (cfloat, blocked, n = 256, batch 1024); float unchanged.

- The host-loop fallback (`BATCHLAS_HER2K_ROUTE=vendor`) is 1.23× slower than the GEMM pair at that shape.
- When the expansion does not fit `GLOBAL_MEM_SIZE / 4` (`src/expansion_budget.hh`), `her2k_vendor` silently drops
  to a per-item host loop; scratch is \f$n_2^2\cdot\text{batch}\cdot\text{sizeof}(T)\f$.
  `BATCHLAS_EXPAND_MAX_BYTES=1` forces it. complex<double> is not admitted (unmeasured).

## syev: the stedc merge-variant and leaf-cliff regression

PR 55 (`0bb92fb`) cost syev 1.05×–3.25× (float vectors n = 320: 74.53 to 242.43); the fix restored all eight
sizes to within 0.96×–1.05× (2026-08-05). Two defects, both now guarded:

- **Merge variant.** `STEDC_MERGE_VARIANT_*` had been flipped from FusedCta (2) to Fused (1) on a claim that 2
  was wrong and slower; neither reproduced (the NaN was the deadlock fixed in `4bde59a`). FusedCta is 11–35%
  faster on `stedc`. Open: `FusedCtaConditionedHeavyDeflation` asserts only finite-and-sorted, never accuracy;
  strengthen it before trusting variant 2 on new hardware ([stedc tuning](stedc.md#stedc-current-tuning-values)).
- **Leaf cliff.** `plan_stedc_levels` searched leaves in [threshold/2, 2·threshold] and chose leaf 40 at
  n = 320. `steqr` takes `steqr_cta` only for n ≤ the sub-group width: alone, n = 32 is 0.262 µs, n = 36 3.761
  (14× for 12.5% more width). The leaf is now capped at the threshold (`hi = threshold`), and 32 is the
  sub-group width, an invariant. `StedcLevelPlan.*` asserts the plan shape directly, because a bad leaf still
  gives correct eigenvalues.

## syev: L2 batch chunking {#syev-l2-batch-chunking}

`syev_cta` and `syev_blocked` run the batch in parts of `tuning::blocked_chunk_items` items, an item counted as
three n × n matrices, not the part workspace (the cta workspace is 435 KB per item at float n = 16, mostly
`steqr_wg`'s rotation store the CTA path never uses; sizing by it made parts 10× slower). `cta_fused` is one
kernel and not chunked. Every part is also capped at `max_groups_yz / 2` items (`select::grid_capped_chunk`,
[defect 21](../design/known-defects.md#defect-21-range-rounded-past-the-grid)).

**Rule:** `SYEV_CTA_L2_CHUNK_PERCENT = 150`, `SYEV_BLOCKED_L2_CHUNK_PERCENT = 0` (only the grid cap splits).
Cold A/B, 2026-10-23, pinned route, time relative to `off` (no chunking, before the grid cap):

| route | type | n | batch | 75 % | 150 % |
|---|---|---:|---:|---:|---:|
| cta | float | 16 | 16384 | 0.910 | 0.911 |
| cta | float | 16 | 65536 | 0.825 | 1.026 |
| cta | cfloat | 16 | 131072 | 1.037 | 0.963 |
| blocked | float | 128 | 1024 | 1.069 | 1.056 |
| blocked | float | 256 | 2048 | 1.674 | 1.209 |
| blocked | cfloat | 128 | 16384 | 1.330 | 1.116 |

The cta gains past batch 32767 now come from the grid cap; re-measured against the capped baseline, the shipped
rule is 0.999–1.001 (the 0.911 was single-pass noise). Blocked loses at every percent (n = 256 parts under-fill
sytrd and stedc). stedc and two-stage have no L2 rule. Open: `steqr_buffer_size` reserves the `steqr_wg`
rotation store on the CTA path.

## syev: the vendor-free double band, n = 449 to 1600 {#syev-vendor-free-double-band}

Vendor-free (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`), checked 2026-10-25, correctness only, nothing timed. The
fullerene workloads are double: C120 is n = 480 at batch about 1819, C300 n = 1200 at batch about 76.

**Routes.** No table or `can_run` change was needed: every row in the band lists native families after
`vendor`, so `select::choose` takes the next entry (sm_89: `blocked` for `jobz=V`, `two_stage` for `jobz=N`;
sm_120: whatever the deep row ranks next), never the last resort. The CPU (NETLIB) queue has no route and throws
`NoRouteError` by design.

**Workspace at the fullerene batch.** Blocked once sized its workspace for the whole batch: 22831 MiB at C120,
which with A does not fit 24 GB (the test passed only because managed memory oversubscribes). Each driver now
chunks its own batch and sizes workspace for one part: its own rule first, then `budget_capped_chunk`
(`src/ops/syev/kernels/chunk_budget.hh`) when the largest part passes `Device::scratch_budget`
(min(`MAX_MEM_ALLOC_SIZE`, `GLOBAL_MEM_SIZE` / 4), 6 GB on the 4090). The cap sizes candidates with the driver's
own part layout, bisects for the most items per part, then returns the fewest near-even parts sized to fit (C120
blocked: 454, 454, 454, 457). The buffer-size query and the launch share the chunk function. C300 fits as one
call; on sm_120 so does C120, so the test seam `ops::syev::detail::ScopedChunkBudget` forces the cap.

A `batch_range` view has no pointer array, but `two_stage`'s panel `geqrf` reads one on the vendor, so a split
two-stage call reserves one array (largest part's length) after the part workspace.

**Tests** (`tests/syev_tests.cc`, `DoubleBand*`, `BatchPartsUnderAScratchBudget`,
`NativeFamiliesFitTheScratchBudgetAtFullereneShapes`, `VendorFreeCpuHasNoRoute`): graded Hermitian input,
condition 10⁶, mostly padded ld/stride with a poisoned unnamed triangle; the saturating cases are heavy-tier.

**Still to time:** vendor-free `blocked` against `two_stage` for `jobz=V` at n = 480..1600, batch ≥ 128 and the
fullerene batches, and the parts at C120 against one oversubscribed call.

## syev: open debts and unmeasured items

**Workspace footprint (open defect).** `syev_cta` n = 32 with vectors needs 1.5–2.2 MiB per matrix (about 500×
the data): batch 4096 takes 7.5 GB, 16384 the whole card (and regresses). Eigenvalues-only is unaffected. It
caps usable batch near 4096 on 24 GB, caused a real OOM on the shared box, and is uninvestigated.

### syev: open debt, the single-read panel symv

Plan WP5, open (the agent guide's "open 2.7×"). The `latrd` panel is 35–71% of every blocked solve; at n = 512,
batch 256 it moves 2.3× ideal L2 and 12–16× L1TEX traffic at 10.8% SM throughput, because each element is read
twice (row walk and column walk, lanes `lda` apart). Proposed: stage 32×32 tiles once and form
\f$y_R \mathrel{+}= A_{RC} v_C\f$ and \f$y_C \mathrel{+}= A_{RC}^H v_R\f$ from that copy (reflectors are sequential,
so a MAGMA multi-block symv is no drop-in). The DRAM floor caps it at 2.7× (cfloat n = 512 end to end 1.82×).
**Gate:** ≥ 1.6× on the panel at n = 512, `ncu` L1TEX falling toward 2×.

### syev: open debt, complex stage-2 occupancy

Plan WP6, open. At n = 512, batch 512 the float back-transform is saturated (SM 93.7%, occupancy 82.8%) and the
complex one is not (35.3% / 49.8%); the chase runs 4.85× float against a 4× arithmetic ratio. Halved occupancy
signals register pressure. Next: retile `unmqr_hb2st_wave` per type (against 852), and check whether `__mulsc3`
is on the chase's hot path. Reaching a 2.2× complex/float ratio would be about 1.49× on the complex two-stage
solve, ahead of blocked and vendor at n = 512; routing waits for the measurement.

### syev: open debt, stebz reports no convergence status

The values-mode paths of `syev_blocked` and `syev_two_stage` use `stebz`, whose three exits (tolerance met,
budget exhausted, midpoint stalled) look alike, so both clear `info` to 0 for every item. `stebz` needs a
convergence flag (see also [stedc: open debts](stedc.md#stedc-open-debts)).

### syev: remaining unmeasured items

- The n = 2048 eigenvector row (unsaturated), the values-mode 320 | 512 boundary, and one routing sweep per type
  and mode after WP5.
- SYTRD LARGE = 8 against the syev-context 24 at n = 512; check with `BATCHLAS_TUNE_SYTRD_BLOCK_SIZE`.
- double XLARGE: SYTRD 16 and ORMQR 32 beat the old values (1.05×–1.07×) before `924b3a59`; not re-measured.
- Profile `stedc` internally (20.7% of the float vector solve at n = 256).
- The sm_89 table is an untimed transcription; rocSOLVER is unmeasured.

## syev: rejected ideas

| idea | result | verdict |
|---|---|---|
| ORMQR = 128 (the "shadowed header" regression theory) | 1.50× slower at n = 512, 2.1×–3.1× at n = 64..128; 16 has always shipped | refuted |
| `cta` above n = 32 (WP2, reverted `14bad498`) | 85× (n = 33) to 211× (n = 64) slower than blocked | rejected |
| block Jacobi as a speed path | 2–11× slower; accuracy-only | rejected |

## syev: measurement traps

- Contention inflates the vendor arm too (an overlapped run faked a 3.6× blocked win); check Upper/Lower ≥ 1.
- Rebuild clean after changing a widely-included struct: a stale `DeviceCaps` layout segfaulted `two_stage`.

## syev: raw data and provenance

Git LFS ([rules](README.md#new-raw-data-lives-in-benchmarksresults-in-git-lfs)):
`benchmarks/results/syev_retune_phaseA.jsonl` (small-n and per-type grids), `syev_retune_phaseB_float.jsonl`
(block-size sweep). Commits: `73eb5510` per-type routing, `924b3a59` retune, `f28a3fcd` / `8c24626f` /
`405b4120` WP1 / WP4 / WP3. Positional arguments of `syev_benchmark` are `n batch nb fuse jobz uplo` (nb 0 =
default); pin with `BATCHLAS_SYEV_ROUTE`. `BATCHLAS_KERNEL_TRACE` does not see cuBLAS; use
`nsys profile -t cuda -s none` and `nsys stats --report cuda_gpu_kern_sum`.

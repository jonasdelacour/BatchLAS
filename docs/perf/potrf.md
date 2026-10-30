# POTRF and POSV {#perf_potrf}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2

Batched Cholesky (`potrf`) and the positive-definite solve (`posv`): the native families, their capacity ceilings, the
measurements behind each shipped window and constant, correctness traps and open debts. Selection is flat
(@ref design_flat_selection); every window is a row of `tuned/potrf.*` or `tuned/posv.*`.

## Selection {#selection-since-flat-kernel-selection-phase-2}

Candidates (`src/ops/potrf/choice.hh`): `tiny`, `cta`, `lpanel:panel=8`, `lpanel:panel=16` (float only), `blocked`,
`vendor`, chosen in `src/ops/potrf/potrf.cc`.

* **Auto** takes the first runnable entry of the nearest row of `tuned/potrf.<device>.txt`. Keys: `dtype`, `uplo`
  (exact), `n` (log, weight 3), `batch` (log). A device without a table borrows the nearest one with a warning; the CPU
  never borrows. Last resort: `blocked`, then `vendor`.
* **Tables** come from `scripts/sweep_to_table.py` over `benchmarks/results/tuning/ledger/potrf.<device>.jsonl`;
  provenance in `tuned/README.md`. `BATCHLAS_TUNED_DIR=<dir>` replaces them without a rebuild.
* **Pins:** `BATCHLAS_POTRF_ROUTE=auto|native|vendor|<spelling>` (`lpanel:panel=8` or `lpanel:8`). A pin that does not
  parse or cannot run the shape throws `std::invalid_argument`; `native` or `vendor` with no runnable candidate of that
  class falls back to Auto with a warning. A vendor-free build and a bare `native` pin take the row's first runnable
  non-vendor entry (`PotrfCandidates.BareNativePicksTheBestRunnableNonVendor`).
* `BATCHLAS_POTRF_NB` and `BATCHLAS_POTRF_W` are blocked-driver tuning constants; selection never reads them.

### `can_run()` terms

`can_run()` (`potrf.cc:36-60`) is correctness only:

* Every native family: no batch mask, `d.is_gpu`, `d.has_sg32`, a homogeneous batch, `n >= 1`, `batch >= 1`.
* `tiny`: `n <= potrf_tiny_max_n<T>()` (32; 16 for cdouble) and `max_wg >= 64`.
* `cta`: `n <= potrf_cta_max_n_for_slm<T>(slm_budget)`.
* `lpanel`: `Uplo::Lower` and `n <= potrf_lpanel_max_n_for_slm<T>(...)`.
* `blocked`: `potrf_blocked_available<T>()` and `cta_max_n >= 1`; either triangle, no lower bound on `n`.
* `vendor`: `d.has_vendor` and a homogeneous batch.

Upper on `tiny` and `cta` is a load/store transform (`S(i,c) = conj(A(c,i))`).

### Current rankings (sm_89)

`tuned/potrf.sm_89.txt` (241 `converted` rows from the forced-route archive, 12 `deep`), first choice:

* `tiny` at `n <= 32` for float and cfloat, `n <= 16` for fp64 (except `L` n = 1 and double 9-10, cdouble 9: `vendor`).
* float `L`: `cta` at most cells of 33..65, `vendor` 72..384, `blocked` at 512. cfloat `L`: `cta` at 33, `vendor` above.
  double: `vendor` 17..256, `blocked` at 512. cdouble `L`: `cta` 20..32, `vendor` from 128.
* Float Upper 96..512 (`deep`): `blocked` at n = 384 batch 512 and at n = 512, `vendor` elsewhere.
* `lpanel` never ranks first on sm_89 (no current-kernel timings). On sm_120 `lpanel:panel=16` leads 21 of 189 float
  `L` rows.

## posv selection {#posv-selection-since-flat-kernel-selection-phase-3}

Three families (`src/ops/posv/choice.hh`, chosen in `src/ops/posv/posv.cc`), no vendor family: `tiny`, the
[fused factor-and-solve kernel](#the-fused-posv-tier); `cta`, the public `potrf` then
[one fused solve kernel](#the-fused-potrs-solve); `blocked`, the public `potrf` then two public `trsm`. Each child selects its own kernel,
so `cta` or `blocked` can run cuSOLVER `potrf` underneath. Keys: `dtype`, `uplo`, `n` (log, weight 3), `nrhs` and
`batch` (log). Last resort: `blocked`.

* **`can_run()`:** no batch mask, sub-group 32, homogeneous batch, `n, nrhs, batch >= 1`. `tiny` adds
  `n <= posv_tiny_max_n<T>()` (16 for cdouble, 32 otherwise), `nrhs <= 4` and `max_wg >= 64`. `cta` adds
  `ops::potrs::fused_fits`: `nrhs <= kGetrsFusedMaxRhs` (8) and `n * nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)`.
  `blocked` needs only a homogeneous batch.
* **sm_89 table** is transcribed from the deleted router's windows and untimed (`transcribed:7e71a6e0`): `tiny` at every
  `n <= 32` (16 for cdouble) and `nrhs <= 4`, except cfloat `n > 24` at `nrhs <= 2`, which ranks `cta`; `cta` above.
  Off-grid cells at the edges can differ: cfloat n = 25 and 26 at `nrhs <= 2` take the `tiny` n = 24 row.
* **sm_120 table** is deep-measured; the earlier 3329-cell seed sweep is `benchmarks/results/routing/sm120_posv_sweep.jsonl`.
* An empty problem or a heterogeneous batch throws `batchlas::invalid_argument` before `choose()`; no pin can take them.
* **Pins:** `BATCHLAS_POSV_ROUTE=auto|native|vendor|tiny|cta|blocked`. Bare `native` is Auto's pick; `vendor` warns and
  falls back to Auto; `native:<family>` throws.

## Families and constants {#potrf-what-ships}

Kernels: `src/ops/potrf/kernels/potrf_{tiny,cta,lpanel,blocked}.cc`. `tiny`, `cta`, `blocked` and `vendor` serve both
triangles; `lpanel` is Lower only.

| | float | double | cfloat | cdouble |
|---|---|---|---|---|
| `tiny` max `n` | 32 | 32 | 32 | 16 |
| `cta` advertised / resident `n` | **77** / 155 | **54** / 109 | **54** / 109 | **38** / 77 |
| `lpanel` advertised `n` | **744** | **368** | **368** | **180** |
| CTA `NB`/`TS` (`PotrfCtaConst`) | 8/4 | 8/4 | 8/4 | 8/2 |
| blocked `nb` / `W` (`PotrfBlockedConst`) | 128 / 128 | 96 / 32 | 96 / 32 | 64 / 16 |

No `nb` equals a fit ceiling; sizing `nb` from the fit ceiling is measurably wrong for every type ([`nb` and `W`](#nb-and-w)).

## The SLM budget and the fit ceilings {#the-slm-budget-and-the-fit-ceilings}

* `local_mem_size` is **101,376 B**; the budget is `local_mem_size - 4096` = **97,280 B** (`select::Device::slm_budget`).
  `kPotrfReferenceSlmBudget` (`potrf_cta.cc:50`) serves only `potrf_cta_max_n<T>()`.
* `device_limits.hh`'s 49,152 (`cmake/BatchLASDetectSYCL.cmake`) is wrong here by 2.06x; no decision reads it.
* The accessor slack is 256 B (at float `n = 155`, `ncu` reports 96,408 B).

Ceilings, the resident row launched cold and correct (`slm/maxn_fitcheck.csv`), pinned by
`MeasuredFitCeilings`:

| budget (B) | blocks/SM | float | double | cfloat | cdouble |
|---|---|---|---|---|---|
| 24,320 (advertised) | 4 | 77 | 54 | 54 | 38 |
| **97,280 (resident)** | **1** | **155** (96,540 B; 156 → 98,408) | **109** (110 → 98,108) | **109** (110 → 98,076) | **77** (78 → 99,056) |

## potrf: the 48 KB launch hole {#potrf-the-48-kb-launch-hole}

A dynamic local-memory request in `(49152 - static_shared, 49152]` fails cold with `CUDA_ERROR_INVALID_VALUE` at
`enqueueKernelLaunch`: the UR CUDA adapter raises `MaxDynamicSharedMemorySize` only when the dynamic request alone
exceeds 49,152. A kernel with zero static shared has no hole.

> **Warning:** the attribute is sticky per CUfunction, and potrf's kernel takes `n` at runtime. A 49,064 B launch fails
> cold and passes after a 65,536 B launch of the same function: the outcome depends on order.

* The pad (`src/ops/potrf/potrf_slm_hole.hh`, applied at `potrf_cta.cc:124`) maps `(47104, 49664]` to 49,920 B. It is
  inert today (no `smem` in any potrf instantiation) but stays: one `reduce_over_group` reintroduces the hole. geqr2
  uses the same band (@ref perf_qr).
* No test can cover the hole: any earlier launch above 48 KB in the process masks it.

## Register gate {#register-gate}

CTA kernel, `regprobe_any.sh` on `batchlas_extensions_cta`. Gate: **stack frame == 0 AND spill == 0 AND regs x WG <= 65536.**
Registers per work-item (sub-group / work-group launch) at the shipped `NB`/`TS`: float 64 / 56, double 94 / 80, cfloat
102 / 92, cdouble 128 / 109; every cell frame 0, spill 0; worst `regs x WG` 27,904 (cdouble), 2.35x headroom. A `break`
in the panel loop makes the trip count data-dependent, defeats `#pragma unroll` and puts `d[NB]` in a 128-byte frame: the
kernel predicates instead. Against cuSOLVER (batch 4096) float wins to about n = 64 and double never above n = 8.

### The L ladder {#the-l-ladder}

`L` (work-items per matrix) is `Ntiles_0 * TS^2 / 24`, capped at 256 (`potrf_cta.cc`): the elements the first trailing
update touches, at 24 per work-item (a tile count is the wrong unit because `TS` varies by type). Across 21 cells the rule
picks the best `L` or one within 1% in 19, within 5.5% in the other two (float n=48: L64 117.2 µs, 131.9 at L32, 120.8
at L128). **24 is a fitted constant**: re-measure it if `NB`, `TS` or the inner loop changes.

## The blocked driver {#the-blocked-driver}

Right-looking, `j = 0, nb, 2nb, ...` (Lower shown; Upper in [the blocked Upper driver](#the-blocked-upper-driver)):

* **leaf** `A(j:j+ib, j:j+ib)` by the CTA kernel on a sub-view;
* **fixup**: one kernel that merges local `info` into global and quenches failed items;
* **panel** `L21 = A21 L11^{-H}` through the injected public `trsm`;
* **trailing** `A22 -= L21 L21^H` through the injected public `gemm`, in `W`-wide column panels (a `W x W` diagonal block
  into scratch with a triangular fold, then a rectangle gemm).

The seams (`PotrfTrailingGemm`, `PotrfPanelSolve` in `potrf_native.hh`) are required; an empty seam throws. `potrf.cc`
passes the public `gemm` and `trsm`, so each child takes its own table's choice. The trailing update is 65-95% of a
vendor-free factorisation. The fold exists because `herk` has no native family and a square gemm would write the
triangle `potrf` must not touch.

### `nb` and `W` {#nb-and-w}

Whole driver, `batch = 128`, worst rel sd 1.3%; each `nb` bracketed:

* **float `nb = 128`:** n=512 2.462 ms (2.524 at 155, 2.634 at 96); n=1024 9.180 ms (11.604 at 155, 17.699 at 96).
  The trailing `k` is `nb`, and float's transposed register gemm needs `k >= 128`; above 128 only the leaf slows.
* **double `nb = 96`:** n=512 7.019 ms (7.258 at 109); n=1024 50.980 ms (53.338 at 64).
* **cfloat `nb = 96`:** n=1024 34.1 ms, best of six values. **cdouble `nb = 64`:** n=1024 550 ms (579 at 77).

`W`, whole potrf, `n = 1024`, worst rel sd 4.9%: float **128** 16.785 ms (17.863 at 32, 18.654 at 64, 20.495 at 96;
n=2048 46.510 against 50.879 at 64); cfloat **32** 54.00 (54.63 at 16, 57.40 at 64); double 32 is 1.1% behind 16,
inside noise. The curve is non-monotone (a gemm-selection cliff): do not interpolate. cdouble was not swept: `k = 64`
reaches no register gemm at any `W`.

* **Reproduction trap:** `potrf_blocked_params` rounds `nb` down to a multiple of `trsm_cta_max_n<T>()` = 32, including
  a `BATCHLAS_POTRF_NB` value: 48/80/109/155 collapse onto 32/64/96/128.
* **Trans versus ConjTrans:** real types pass `Trans` (`ConjTrans` selects a 1.8x slower kernel). Complex must pass
  `ConjTrans`: `Trans` takes the residual from 4.0e-07 to 1.9e-02.

### Blocked driver measured against cuSOLVER {#blocked-driver-measured-against-cusolver}

Whole `potrf`, Lower, `cuSOLVER / potrf` with both seams native (vendor-free) / routed: float n=1024 b256 **1.108 /
1.646**, n=512 0.614 / 0.909; double n=1024 **1.014 / 1.012**; cfloat n=1024 0.436 / **1.040**; cdouble n=1024 0.311 /
0.767. Real types pass cuSOLVER as `n` grows; complex diverges (the complex trailing gemm is 97.6% of cdouble n=1024).
`n <= 256` loses for every type: at `n <= nb` the driver is the leaf plus a fixup.

### potrf: L2 batch chunking {#potrf-l2-batch-chunking}

The blocked driver factors the batch in parts (`potrf_blocked_chunk`) by the rule of
[LU: L2 batch chunking](lu.md#lu-l2-batch-chunking): `BLOCKED_L2_CHUNK_PERCENT` = 75% of L2, rounded up to a CU
multiple, one pass below two chunks, `info` offset and workspace per part. Differences: an item is its lower triangle,
`n(n+1)/2 x sizeof(T)`; and a single block (`n <= nb`) never splits. The chunk loop serves Upper too.

* **sm_120** (RTX PRO 6000 Max-Q, 128 MiB L2, 2026-10-21), float pinned `blocked`, on/off: geomean 0.948 over 21
  cells, down to 0.853 (n=256 b32768); the one loss is n=128 just past two chunks (b6392, +3%). Forced chunk sizes bracket
  75%: the best part holds 0.75-1.5 L2 of triangles (n=256: 376 1.085, **752 0.866**, 1128 0.916).
* **sm_89** (2026-10-23, cold): float n=128 b16384 0.969, n=256 b2048 0.912; cfloat n=128 b4096 0.876.
* **The shared `info`.** Without a caller `info`, every part reuses the pool's `ws.info`, so it is zeroed per part
  (after the previous part's last fixup on an out-of-order queue); else a late `info[b] != 0` silently quenches the next
  part's item `b`. Guard: `BlockedChunksRezeroSharedInfoPerPart`.

### The panel-solve verdict {#the-panel-solve-verdict}

With the gemm seam at vendor, the native panel `trsm` beats the vendor's in every cell: float 1.31-1.42x (n 512-2048),
double 1.13-1.19x, cfloat 1.91x, cdouble 2.69x (n = 512).

## Negative results {#potrf-negative-results}

| Alternative | Result | Verdict |
|---|---|---|
| CTA `NB = 16` | slower in 18 of 20 cells (float n=96 1162 → 2374 µs); 156-206 regs | rejected; a dual instantiation doubles device link cost |
| fold-free trailing update (product straight into `A`) | 11% cheaper | wrong: clobbers the other triangle |
| fusing launches, fewer panels | GPU 97-99.7% busy | at most 3% |
| packing `L21` contiguous (strided-`ld` gemm collapse) | `sub/flat` 0.89-1.09 on potrf's shapes | no gain; the collapse needs a register kernel potrf does not reach |

## potrf: correctness findings {#potrf-correctness-findings}

1. **The panel trsm raced on multi-sub-group work-groups.** The trsm CTA kernel staged its triangle lane-strided and read
   the diagonal back with no barrier (`src/ops/trsm/kernels/trsm_native.cc:186` now has it). Every earlier test ran at
   `wg = 32`, where the race cannot show. Guard: `TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly`.
2. **`beta = 0` must not read C.** The `W x W` diagonal gemm writes uninitialised `ws.product` with `beta = 0`; a native
   epilogue that reads `prior` turns `0 * NaN` into NaN. Fixed in the gemm epilogues
   ([known-defects #11](../design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero)). Guard:
   `PotrfBlockedTest.BlockedDoesNotReadUninitialisedWorkspace`, pinning `direct` and `tiled`.
3. **Test oracles:** residual tests poison the other triangle, check the first and last item at non-natural
   `ld`/stride, plant NaN pivots (a negative pivot divides to a finite number), and normalise the original diagonal to
   `+1` so a stale-pivot read goes red.
4. **Scope:** `group_barrier(sg)` where several sub-groups own one matrix races silently, so `potrf_cta_launch_params`
   derives `Scope`, and `WorkGroup` with `G != 1` throws.
5. **`info`:** the driver merges the leaf's local index as `if (info[b] == 0 && leaf[b] != 0) info[b] = j + leaf[b]`; the
   quench writes an identity diagonal as well as a zero panel, because a zero `L11` diagonal gives `0/0`.

Workspace sizes come from `potrf_cta_buffer_size` and `potrf_blocked_buffer_size` through `BumpAllocator::measuring()`.

## Open debts {#potrf-open-debts}

1. **Complex vendor-free is 0.31-0.51x cuSOLVER**, from the complex trailing gemm; a register-tiled complex gemm is
   worth about 2.7x on vendor-free cdouble potrf. The highest-value follow-on.
2. **`n <= 256` loses for every type** in the blocked driver.
3. **L2 chunking retune pending:** the sm_120 `blocked` rows predate chunking; at cfloat n=128 b32768 pinned `blocked`
   runs 255.8 ns/item against Auto's 283.8.
4. **LPanel is not established on sm_89:** no current-kernel timing, `NB = 16` untimed, no fp64 grid, no `ptxas`/`ncu`
   evidence for the occupancy argument.
5. **The trsm windows** were measured on the racing kernel above `q*batch ~ 65k`; they survive as untimed rows of
   `tuned/trsm.sm_89.txt`.
6. **Out-of-order queues are untested:** five edges rely on `if (!ctx.in_order()) ctx.wait()`.
7. **The accuracy bound is loose:** `4*n*eps` against a true worst case of `(0.2, 1] * n * eps`.
8. **Upper coverage in `potrf_tests` is lighter than Lower** outside the blocked driver.
9. **The route diff was never run** (`BATCHLAS_COVERAGE_OUT` per side, `scripts/route_diff.sh`).
10. **B6 is unfalsifiable on this hardware** ([LPanel armed breaks](#armed-breaks-r9--lpanel)): record it as
    *unfalsifiable*, never *covered*.
11. **posv:** the composed `blocked` arm under Auto is 1.5-5.6x slower than its vendor-pinned form at float `n <= 32`
    (a child picks a losing native tier; `trsm` `Side::Left` is the first suspect, @ref perf_trsm); posv Upper and
    `nrhs` 2-3 are untimed; the cdouble posv tier spills ([register bound](#the-posv-tiny-tier-is-not-register-resident-for-cdouble)).

## The blocked Upper driver {#the-blocked-upper-driver}

`blocked` serves `Uplo::Upper` with the same right-looking schedule on transposed block rows, no copies (closing
known-defects #19): CTA Upper leaf on A11; panel solve trsm Left/Upper/ConjTrans on A12; diagonal tile `-U^H U` (gemm
ConjTrans, NoTrans) folded into the upper triangle; rest of the block row `C -= U^H Ur`; the failure quench on the panel
row. `lpanel` stays Lower-only: its left-looking update reads the factored lower triangle.

The lower triangle is neither read nor written (unlike cuSOLVER). Guards: `PotrfBlockedTest.BlockedUpper*` and
`FacadeBlockedUpper`.

sm_89, `factor_bench`, 12 reps: cuSOLVER's Upper is much slower than its Lower from batch 512, so blocked Upper wins
float n = 384 and 512 at batch 512 (2.540 vs 2.901 ms, 4.691 vs 6.293) and loses at batch 128 (n=512 1.261 vs 1.229)
and for cfloat everywhere (0.35-0.94x). The float Upper rows of `tuned/potrf.sm_89.txt` follow.

## Raw evidence

Tag `perf-evidence/vendor-independence` (`git show <tag>:<path>`): `experiments/wp4_potrf/` (`slm/` budget, ceilings and
hole; `regbaseline/` register probe; `README.md` CTA versus cuSOLVER and the `L` grid; `phase2*/` correctness, `W` and
performance grids), with `WP4_POTRF_SPEC.md` and its corrections.

## potrf: the occupancy rule {#potrf-the-occupancy-rule}

`potrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` divides the budget by an occupancy target before walking the
footprint. The default target is 4 (`resident::kMinBlocksPerSm`, rule R1), so the tier advertises what fits in a quarter
of the budget; "can one work-group hold this at all" uses 1. Figures: [the shipped ceilings](#the-shipped-ceilings-pinned-in-both-scales).

### The occupancy clamp on nb {#the-occupancy-clamp-on-nb}

`potrf_blocked_params` clamps `nb` to the advertised ceiling while `n <= kPotrfOccupancyNbMaxOrder` (256), and to the
resident ceiling above. `nb` is also the trailing `k`: halving it buys leaf occupancy and pays in the gemm. Native arm,
large batch, after / before: float n=128 b2048 **0.706**, n=256 b1024 **0.931**, n=512 0.999; cfloat n=128 **0.665**;
double n=128 **0.788**.

Without the 256 threshold n >= 384 regresses (float 512 1.209x, cfloat 512 1.382x, double 512 1.097x). The one cost:
float n = 96 b2048 goes 0.531 → 0.775 ms, but cuSOLVER takes 0.271 ms there, so Auto is unaffected.

### The shared helper: budget, slice, walk and pack {#the-shared-helper-budget-slice-walk-and-pack}

`src/util/resident_capacity.hh` (`constexpr`, no SYCL headers) holds the capacity rule for potrf, getrf and geqrf:
`device_slm_budget` (local memory less a 4096 B reserve), `kMinBlocksPerSm = 4` (geqrf uses 2), `occupancy_budget`
(the slice one work-group may own), `resident_max_n` (the largest `n` that fits with every smaller `n` fitting) and
`pack_matrices_per_wg` (matrices per work-group when one sub-group serves a matrix, capped by `max_pack = 4`).

* `resident_max_n` stops at the first miss: the 48 KB pad makes `bytes(n)` non-monotone, and a `continue` advertises a
  range with a hole in it (reachable only for budgets in **[49,664, 49,920)**).
* Products are `std::size_t` or `int64_t`: `(m|1)*n` overflows `int` at m ≈ 46,341, a reachable panel height.
* `G > 1` is correct only where every barrier is a sub-group barrier, unless the kernel is built for it
  ([LPanel](#why-lpanel-may-pack-on-work-group-barriers)).
* Test hooks pack two 16-bit fields (`potrf_cta_debug_launch`: G, L; `potrf_blocked_debug_params`: nb, W;
  `potrf_tiny_debug_launch`: G, S) and return 0 when the order does not fit.

### The contiguity rule and the synthetic table {#the-contiguity-rule-and-the-synthetic-table}

No budget on this box reaches the non-monotone case, so `tests/resident_capacity_tests.cc` arms the `break` with a
synthetic footprint of 128 B per `n` plus the pad (`(47,104, 49,664]` → 49,920, as in the potrf, getrf and geqrf CTA
kernels): 369..388 pad to 49,920 while 389 is 49,792 B. At a 49,850 B budget the walk must answer **368**; a skipping
walk answers 389 and advertises twenty orders that cannot launch.

### The G-packing helper and its three limits {#the-g-packing-helper-and-its-three-limits}

`pack_matrices_per_wg` is bounded by local memory, the work-group width and the cap; the test has one row where each
binds alone. The 8-lane rows are load-bearing: at 32 lanes the cap and the 128-wide target agree, so a helper that ignored
`max_pack` would pass.

### The shipped ceilings, pinned in both scales {#the-shipped-ceilings-pinned-in-both-scales}

At 97,280 B (`tests/resident_capacity_tests.cc`). Advertised is what `can_run` admits; resident is what a blocked
driver's leaf can hold. Conflating them is the defect the pin catches.

| type | potrf adv / res | getrf adv / res | geqrf adv / res (elements) |
|---|---|---|---|
| float | **77** / 155 | **77** / 155 | **11,776** / 24,320 |
| double | **54** / 109 | **54** / 109 | **5,888** / 12,160 |
| complex\<float\> | **54** / 109 | **54** / 109 | **5,888** / 12,160 |
| complex\<double\> | **38** / 77 | **38** / 77 | **2,944** / 6,080 |

## potrf: the tiny tier {#potrf-the-tiny-tier}

`src/ops/potrf/kernels/potrf_tiny.cc`: one matrix per `SubGroupPartition<N>`, `N` in {8, 16, 32}. Lane `r` owns row `r`
in a compile-time `D rA[N]`; every cross-lane value is an indexed sub-group shuffle. Zero local memory, zero barriers,
zero static shared: the 48 KB hole is unreachable.

* **Geometry:** work-group 64 (`kTinySubGroups = 2`, a tuning constant), `G = 32/N` matrices per sub-group.
* `potrf_tiny_max_n<T>()` is a flat 32 (16 for cdouble). `potrf_tiny_buffer_size` is nonzero: an empty `info` span
  draws `batch` int32s of pool scratch.

### The tiny potrf window {#the-tiny-potrf-window}

`t_vendor / t_tiny`, sm_89. Lower float/cfloat at batch 65536 (`benchmarks/results/p9_potrf_tiny.csv`,
`p9_potrf_low.csv`); fp64 and Upper at batch 32768:

| n | 1 | 2 | 8 | 9 | 12 | 16 | 17 | 32 |
|---|---|---|---|---|---|---|---|---|
| float L | 3.92 | 5.02 | **10.64** | 3.02 | 3.74 | 3.07 | 2.06 | 2.91 |
| cfloat L | 3.43 | 4.34 | **7.43** | 2.10 | 1.57 | 1.62 | 1.23 | 1.30 |
| double L | 0.715 | 1.158 | **3.956** | 0.865 | 1.145 | 1.505 | 0.548 | 0.768 |
| cdouble L | 0.709 | 1.291 | **4.892** | 0.882 | 1.167 | 1.545 | — | — |

float and cfloat win at every order the tier holds (the ceiling is the tier's, not a loss). fp64 Lower: n = 1 is
overhead-bound, 9..11 fill the N = 16 bucket badly (n = 11: 1.049, 1.071), and from 17 the N = 32 bucket collapses
double. Upper wins every cell (float 3.2-28.7x, cfloat 1.7-20.2x, fp64 2.1-7.0x): cuSOLVER's Upper is about 2x its
Lower.

### Register probe (tiny tier) {#register-probe-tiny-tier}

Gate: frame 0, spill 0, `regs x 64 <= 65536`. `BATCHLAS_BUILD_DIR=<tree> scripts/register_probe.sh out.log ''
batchlas_extensions_cta` (sm_89; the link takes about 340 s). Every cell has frame 0 and spill 0; registers at N = 8 /
16 / 32: float 54 / 64 / 78, double 70 / 82 / 104, cfloat 64 / 112 / **166**, cdouble 84 / 128 / not instantiated.
`kTinyWorstProbedRegs = 176` with `static_assert(resident::sm89_fits(...))` (`potrf_tiny.cc:54-55`) makes widening the
work-group a compile error. cfloat N = 32 (6 blocks/SM, 12 resident matrices) is the first cell to check in any grid.

### The shared tiny-tier invariants {#the-shared-tiny-tier-invariants}

`src/kernels/common/tiny_device.hh` is shared by the potrf, getrf and geqrf tiny tiers. Each invariant breaks silently,
as a wrong answer or lost residency, never as a diagnostic.

1. **`rA[]` is never dynamically indexed or passed by reference**; helpers touch it only under `#pragma unroll` with
   compile-time bounds. A data-dependent `break` moves `rA` to a 64-288 B frame; predicate instead (`col_live`). Writes
   under a failure predicate are selects; `complex<double>` needs the component-wise `tiny_native::tiny_select` (`?:`
   over the 16-byte aggregate defeats SROA). A plain `?:` keeps all tiny cases green and only the register probe sees it:
   the gate is frame 0 AND spill 0, never a green suite.
2. **`N` must divide the sub-group**: `SubGroupPartition<P>` bases a chunk at `(lane / P) * P`, hence `{8, 16, 32}`.
3. **Every lane of a partition reaches every partition collective.** Dead partitions load the identity and suppress
   their store; no tiny kernel uses `if (prob_id >= nb) return;`. On sm_89 this is guarded by source text only.

Also: a lane guard belongs inside a collective; complex values shuffle as two reals; `c <= lane` keeps the other
triangle untouched; `tiny_partition_id`'s `sg_id *` term is load-bearing (`TinyPackedBatchMatchesSolo`).

### The CTA comparison is elementwise, not bit-exact {#the-cta-comparison-is-elementwise-not-bit-exact}

At `n <= 8`, CTA runs one NB = 8 panel that matches this tier line for line, yet float `n = 2` Upper differs by one ULP:
the compiler contracts `a - b*conj(c)` into an FMA on one side only. The test asserts an elementwise `8*eps` bound. The
bit-exact oracles are tiny-versus-tiny (`TinyPaddingIsInert`, `TinyPackedBatchMatchesSolo`).

### What the geometry test can and cannot check {#what-the-geometry-test-can-and-cannot-check}

`TinyLaunchGeometryIsReported` checks the work-group against `MAX_WORK_GROUP_SIZE`, the two-sub-group floor, and that the
bucket read back as `32/G` holds `n` and is the tightest of 8/16/32. It cannot check a pinned `S`: hook and launcher share
one helper, so a check re-derived from the hook cannot fail.

## The LPanel tier {#the-lpanel-tier}

`src/ops/potrf/kernels/potrf_lpanel.cc`, `potrf_lpanel_device.hh`: a left-looking, thread-per-row panel kernel. One
work-group holds an `n x NB` panel in local memory, factors it in place, and re-reads factored columns from global
memory. `NB = 8` for every type, `NB = 16` float only. Upper is refused by `can_run`.

```
slm_per_matrix(n, NB, sizeof(T)) = (n*NB + NB*NB) * sizeof(T) + 256
```

The 48 KB band applies to the total request (`G` steps down if the pad pushes it over). The ceiling is the smaller of the
local-memory slice and `MAX_WORK_GROUP_SIZE`; the second is required, because lane `row` carries registers across the `k`
loop, and an order above the work-group size is a silent race.

| | float | double | cfloat | cdouble |
|---|---|---|---|---|
| advertised (4 blocks/SM) | **744** | **368** | **368** | **180** |
| resident (1 block/SM) | 1024 (wg) | 1024 (wg) | 1024 (wg) | 750 |
| `NB = 16` advertised | 360 | — | — | — |

### Why LPanel may pack on work-group barriers {#why-lpanel-may-pack-on-work-group-barriers}

`potrf_cta.cc` packs `G > 1` only at `L == 32`, where its barriers are sub-group barriers. LPanel packs on work-group
barriers deliberately: every barrier in `potrf_lpanel_body` is a top-level `group_barrier(it.get_group())`, and a failed
item is a predicated skip. Hence:

* **No early return and no shortened `n`.** A dead slot runs the whole body with stores suppressed (`store`); `n = 0`
  would skip barriers and hang the group.
* **B6 is required:** the panel-end store writes columns the next panel's staging reads from other work-items
  ([open debt 10](#potrf-open-debts)).

### LPanel register gate {#lpanel-register-gate}

Per lane, `rp[NB] + rS[NB] + rA[NB]` is `3*NB` scalars: 24 for real types, 48 for cfloat, 96 for cdouble, so against
the 42-register full-occupancy line cfloat and cdouble lose occupancy. A source count; no `ptxas` probe has been run.

### The measured LPanel window {#the-measured-lpanel-window}

sm_89, `factor_bench`, arms interleaved, 7 reps, `vendor_ms / arm_ms` at two batches per cell
(`benchmarks/results/p3_lpanel_ab.csv`, `p3_lpanel_edges.csv`, `p3_auto_after_flip.csv`); taken before the current
table, whose sm_89 rows do not rank LPanel.

LPanel led float 37..256 (n=64 **1.888 / 2.226**; CTA led at 33 and 36) and cfloat 33..256 (n=64 **1.604 / 1.866**),
minimum in-window float 1.278 at n = 129 b2048. Float 384 and 512 clear the 1.11 gate only at the doubled batch (512:
1.079 / 1.322); cfloat does not fit above 368; tiny wins at n = 32; `blocked` was 0.22-0.94x of the vendor below 256. A
ratio of exactly 1.000 means the arm did not run the tier it names.

### The pivot-cell WAR race {#the-pivot-cell-war-race}

Every lane reads the pivot `sA[(j+i) + i*slda]` at the top of column `i`; the lane owning row `j+i` then overwrites it
with `sqrt(d)`. A warp that runs ahead publishes the root before a slower warp reads the pivot (residuals about 1e-04 at
`batch >= 1024`, nondeterministic). **Fix:** `B3b`, a work-group barrier between the `d` read and the scale. Guard:
`PanelStoresArePublishedAtSaturatingBatch` (n = 128, batch 1024, bit-equal items; batch 512 is not enough).

### Armed breaks (R9) — LPanel {#armed-breaks-r9--lpanel}

Planted, observed and restored. Red as predicted: deleting `B3` (`ResidualAcrossPanelBoundaries`, 19 cases), dropping
`dev_conj` when staging `sB` (complex only), dropping `ok &&` from `bad = ok && !(d > R(0))` (wrong `info`). Green:

* deleting `B6` (end of the panel loop): **not armed** ([open debt 10](#potrf-open-debts));
* relaxing the prefetch guard `row >= j + i` alone (the bound is enforced three times; relaxing the prefetch, publish and
  store guards together turns `OtherTriangleIsNeitherReadNorWritten` red for all types);
* giving a dead packed slot `n = 0` instead of `store = false` (whole warps exit, so nothing hangs).

A NaN poison cannot see a write (the stored value is computed from the NaN it read): the combined guard break was caught
only by a finite sentinel (`-999 + 777i`) and bit-equality.

## The fused potrs solve {#the-fused-potrs-solve}

posv's `cta` family runs the public `potrf`, then **one** kernel for both triangular solves (`potrs_fused_dispatch` in
`getrs_fused.cc`: the fused narrow-RHS getrs body without pivots, non-unit diagonal; Lower solves `L` in axpy form and
`L^H` in dot form). It shares getrs's resident-RHS capacity and `kGetrsFusedMaxRhs`. Replacing two trsm launches took
`t_vendor / t_native` (Lower, nrhs = 1) from float n=64 0.59 to **2.07** and cfloat 1.18 to **4.56**; fp64 is 1.5-4.5x
ahead of `blocked` at n = 33..256. `FusedSolveArmSolvesOnBothTriangles` pins potrf to `native`: unpinned Upper at `n >= 33`
resolves to cuSOLVER, which overwrites the other triangle and would hide a wrong-triangle read.

## The posv tiny launch bound {#the-posv-tiny-launch-bound}

`.minnctapersm` applies to `posv_tiny` through `PosvTinyBody`, swept per (type, NC, NR) over MinBlocks {1, 8, 12, 16},
batch 32768, and transcribed in `posv_tiny_min_blocks`: float takes 16 (flat within 2% after the local-memory rework),
cfloat 8 (n=24 nrhs=1 2.075 → 1.099 ms; 16 blocks is 2.3x slower at N = 32). fp64 is uncapped and unmeasured.

## The posv local-memory transpose {#the-posv-local-memory-transpose}

`posv_tiny` uses the column bucket `NC` (`{12, 16}` under N = 16, `{20, 24, 28, 32}` under N = 32), an `NR = 2` RHS
bucket (@ref perf_lu), and local memory for the two cross-lane reads that scale with `n`: the Cholesky column (each lane
stores its element, one `group_barrier(sg)`, vector loads double-buffered by parity, instead of `n - j - 1` shuffles per
step) and the backward-solve transpose (local memory may be indexed dynamically; registers may not).

The factor is bit-identical to `potrf_tiny`. cfloat, batch 32768: tiny beats the vendor by >= 1.23x everywhere and CTA
at `n <= 24` and at nrhs = 4; nrhs 1-2 above 24 go to CTA (n=32 nrhs 1: 2.78 vs 3.12 over vendor). float tiny is
1.3-3.5x CTA everywhere.

### posv: the backward solve costs a transpose {#posv-the-backward-solve-costs-a-transpose}

Lane `r` holds row `r` of `L`; the backward solve `L^H x = y` needs the column `L(i, r)`, which no lane holds. A
shuffle transpose costs `N(N-1)/2` shuffles (496 at N = 32), so it goes through [local memory](#the-posv-local-memory-transpose).

## The fused posv tier {#the-fused-posv-tier}

`src/ops/posv/kernels/posv_tiny.cc`: one launch for `A X = B`, Hermitian positive-definite `A`, `n <= 32` (16 for
cdouble), `nrhs <= 4`. It runs `potrf_tiny`'s recurrence, then both solves with `L` in registers; zero barriers apart from
the sub-group ones above. `can_run` drops it at `nrhs > 4`; those rows run `cta`.

### The measured posv window {#the-measured-posv-window}

`t_blocked / t_tiny` against the composed `blocked` arm under Auto (the incumbent), gate 1.11, batch 8192-32768
(`run_solve_grid.sh`; `benchmarks/results/p2_posv_{float,cfloat,double,cdouble}.csv`): float 3.76-21.84, cfloat
1.29-11.94, double 3.69-10.76, cdouble 3.08-7.24. The weakest, cfloat n = 32 nrhs = 4 b8192, rises to 1.87 at b131072.
Measure a flip against the arm it replaces: against a vendor-pinned composition cfloat n = 9 looks like a loss (0.79).

### posv tiny-tier register bound: assumed, then measured {#the-posv-tiny-tier-is-not-register-resident-for-cdouble}

`posv_tiny.cc` declares `kWorstRegsPerThread = 255`, the ISA ceiling. Two instantiations on the default Auto path are
not register-resident: `PosvTinyKernel<complex<double>, 16, 4>` (255 registers, 96 B frame, 184 B spill) and
`<complex<float>, 32, 4>` (149, 256 B frame); NR = 1 for cdouble is clean at 246. It is not a launch abort, only spill
traffic. Owed: why NR = 4 costs so much more, and whether the spill matters end to end (never timed at these shapes).

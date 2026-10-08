# POTRF and POSV

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2

Batched Cholesky (`potrf`) and the fused positive-definite solve (`posv`): the native families, their capacity
ceilings, routing windows, the measurements behind them, correctness findings and open debts. Selection is flat
(@ref design_flat_selection); each window is a row of `tuned/potrf.*` or `tuned/posv.*`.

## Selection {#selection-since-flat-kernel-selection-phase-2}

Candidates (`src/ops/potrf/choice.hh`): `tiny`, `cta`, `lpanel:panel=8`, `lpanel:panel=16` (float only), `blocked`,
`vendor`. The decision is made in `src/ops/potrf/potrf.cc`.

* **Auto** takes the first runnable entry of the nearest row of `tuned/potrf.<dtype>.<device>.txt`. Keys: `uplo`
  (exact), `n` (log distance, weight 3, since work scales as n^3 x batch), `batch` (log distance). A device without a
  table borrows the nearest one with a warning; the CPU never borrows. If nothing runs, the last resort is `blocked`,
  then `vendor`.
* **Tables are measured** by `scripts/sweep_to_table.py` from `benchmarks/results/routing/sm89_potrf_archive.jsonl` and
  `sm120_potrf_sweep{,_edges}.jsonl` (sm_120: [the Blackwell retune](blackwell.md)). Provenance: `tuned/README.md`.
  `BATCHLAS_TUNED_DIR=<dir>` replaces the tables without a rebuild.
* **`can_run()`** (`potrf.cc:32-57`) is correctness only. Speed thresholds live only in the tables.
* **Pins:** `BATCHLAS_POTRF_ROUTE=auto|native|vendor|<spelling>` (`lpanel:panel=8` or `lpanel:8`). A pin that does not
  parse or cannot run the shape throws `std::invalid_argument`. `native` or `vendor` with no runnable candidate of that
  class falls back to Auto with a warning. The old aliases (`native:tiny`, `native:cta`, `native:lpanel`,
  `native:blocked`, `lpanel`) throw.
* `BATCHLAS_POTRF_NB` and `BATCHLAS_POTRF_W` are tuning constants read by the blocked driver. Selection never reads them.
* `BATCHLAS_SELECT_TRACE=1` prints each decision; coverage `reached` rows carry the spelling in `chosen_algo`.

### `can_run()` terms

* Everything: square view (the entry throws on `m != n` first); `d.is_gpu`; `d.has_sg32`; a homogeneous batch; and
  `n >= 1 && batch >= 1` (the panel loop is undefined at `n = 0`).
* `tiny`: `n <= potrf_tiny_max_n<T>()` and `max_wg >= 64` (`:40-41`).
* `cta`: `n <= potrf_cta_max_n_for_slm<T>(slm_budget)` (`:43`).
* `lpanel`: `Uplo::Lower` and `n <= potrf_lpanel_max_n_for_slm<T>(...)` (`:44-48`).
* `blocked`: `Uplo::Lower`, `potrf_blocked_available<T>()`, `cta_max_n >= 1` (`:50-53`).
* `vendor`: `d.has_vendor` (`:55`).

There is no `uplo` term on `cta` or `tiny`: Upper is the same recurrence on `S(i,c) = conj(A(c,i))`, a load and store
transform. `blocked` has no lower order bound, because a correctness gate must not refuse a shape the driver serves.

### Current rankings (sm_89)

* `tiny` ranks first for every `n <= 32`, both `uplo`.
* `cta` ranks first at most cells of `33 <= n <= 65`; `vendor` at n = 44, 48 (batch 8192) and 64.
* `vendor` ranks first from n = 72 through 384; `blocked` at n = 512.
* Double: `tiny` first at `n <= 16`; `vendor` from n = 17 to `blocked` at n = 512. The margin is under 1%, inside the noise.
* `lpanel` never ranks first on sm_89: the archive has no current-kernel LPanel timings. It runs only when pinned.

### Known gaps

* `lpanel:panel=16` has never been timed on any device.
* sm_120 has `uplo=U` rows for float only; other types' Upper take the nearest `uplo=L` row.
* A table ranks a family only where its sweep timed it. Off-grid cells are guesses.
* `factor_bench`'s posv `composed` arm pins potrf to `tiny` only up to the type's tiny ceiling; above it, the pin is `native`.

## posv selection {#posv-selection-since-flat-kernel-selection-phase-3}

Three families, the same for all four dtypes (`src/ops/posv/choice.hh`, chosen in `src/ops/posv/posv.cc`):

* `tiny`: the fused factor-and-solve kernel (`posv_tiny_dispatch`).
* `cta`: the public `potrf`, then `potrs_fused_dispatch` for both solves.
* `blocked`: the public `potrf`, then two public `trsm`.

Each child selects its own kernel, so a `cta` or `blocked` posv can run cuSOLVER `potrf` underneath. There is no vendor
family. Keys: `uplo`, `n` (log, weight 3), `nrhs` and `batch` (log). Last resort: `blocked`.

* **`can_run()`:** sub-group 32, homogeneous batch, `n, nrhs, batch >= 1`. `tiny` adds `n <= posv_tiny_max_n<T>()` (16
  for cdouble, 32 otherwise), `nrhs <= 4` and `max_wg >= 64`. `cta` adds `nrhs <= kGetrsFusedMaxRhs` (8) and
  `n * nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)`. `blocked` is false on a heterogeneous batch.
* **sm_89 tables** are transcribed from the deleted router's windows and untimed (`source=transcribed:7e71a6e0`). Off-grid
  cells at the window edges can differ: cfloat n = 25 and 26 at `nrhs <= 2` take the `tiny` n = 24 row.
* **sm_120 tables** are measured: 3329 cells in `benchmarks/results/routing/sm120_posv_sweep.jsonl`.
* **Workspace:** `posv_tiny_buffer_size` for `tiny`; `potrf_buffer_size` for `cta` and `blocked`.
* An empty problem or a heterogeneous batch throws `batchlas::internal_error` before `choose()`; no pin can take them.
* **Pins:** `BATCHLAS_POSV_ROUTE=auto|native|vendor|tiny|cta|blocked`. `native:tiny`, `native:cta` and `native:blocked`
  throw. Bare `native` is Auto's pick; `vendor` warns and falls back to Auto.

## The RouteTable era (historical) {#the-routetable-era-historical}

Before flat selection, fixed windows sent `Uplo::Lower` float and complex float to Tiny (`n <= 32`), CTA (`n <= 256`,
float up to 35) and LPanel above that. The tables above replace them.

## potrf: what ships {#potrf-what-ships}

| family | source | orders | Uplo |
|---|---|---|---|
| `tiny` | `src/extensions/potrf_tiny.cc`, `tiny_device.hh` | `<= potrf_tiny_max_n<T>()` = 32; 16 for cdouble | both |
| `cta` | `src/extensions/potrf_cta.cc`, `potrf_cta_device.hh` | advertised **77 / 54 / 54 / 38** (float / double / cfloat / cdouble, 4 blocks/SM); resident 155 / 109 / 109 / 77 | both |
| `lpanel:panel=8`, `lpanel:panel=16` (float) | `src/extensions/potrf_lpanel.cc`, `potrf_lpanel_device.hh` | advertised **744 / 368 / 368 / 180** | Lower only |
| `blocked` | `src/extensions/potrf_blocked.cc` | any | Lower only |
| `vendor` | cuSOLVER | any | both |

| | float | double | cfloat | cdouble |
|---|---|---|---|---|
| CTA `NB`/`TS` (`potrf_cta.cc:38`) | 8/4 | 8/4 | 8/4 | 8/2 |
| CTA fit ceiling, advertised / resident | 77 / 155 | 54 / 109 | 54 / 109 | 38 / 77 |
| blocked `nb` (`potrf_blocked.cc:44`) | 128 | 96 | 96 | 64 |
| blocked `W` | 128 | 32 | 32 | 16 |

No `nb` equals a fit ceiling; sizing `nb` from the fit ceiling is measurably wrong for every type. A vendor-free build,
and a bare `native` pin, take the first runnable non-vendor entry of the row. A row with none falls to `blocked`.

## The SLM budget and the fit ceilings {#the-slm-budget-and-the-fit-ceilings}

* `local_mem_size` is **101,376 B**. The budget used is `local_mem_size - 4096` = **97,280 B**
  (`select::Device::slm_budget`). `kPotrfReferenceSlmBudget` (`potrf_cta.cc:47`) serves only `potrf_cta_max_n<T>()`.
* `device_limits.hh`'s 49,152 (set by `cmake/BatchLASDetectSYCL.cmake:45-46` for every `nvidia_gpu_sm_*` target) is wrong
  here by 2.06x. No decision reads it.
* The advertised ceiling is the first miss of a walk with `break`, not `continue`: `potrf_hole_padded` is not monotone
  (47,200 → 49,920 while 49,700 stays 49,700).
* The accessor slack is 256 B, not 64 B (at float `n = 155`, `ncu` reports 96,408 B). The ceilings are unchanged.

| `T` | ceiling `n` at 97,280 B | bytes at ceiling | first miss |
|---|---|---|---|
| `float` | **155** | 96,540 | 156 → 98,408 |
| `double` | **109** | 95,476 | 110 → 98,108 |
| `complex<float>` | **109** | 95,444 | 110 → 98,076 |
| `complex<double>` | **77** | 95,328 | 78 → 99,056 |

All four were launched cold and returned correct answers (`slm/maxn_fitcheck.csv`); `MeasuredFitCeilings` pins them. The
spec's {105, 74, 74, 52} would leave float `n` in 106..155 with no route vendor-free.

Ceilings at other budgets (the grid to cut at these cliffs):

| budget (B) | blocks/SM | float | double | cfloat | cdouble |
|---|---|---|---|---|---|
| 24,320 | 4 | 77 | 54 | 54 | 38 |
| 45,056 | 2 | 105 | 74 | 74 | 52 |
| **97,280 (shipped)** | **1** | **155** | **109** | **109** | **77** |
| 101,120 | 1 | 158 | 111 | 111 | 79 |

## potrf: the 48 KB launch hole {#potrf-the-48-kb-launch-hole}

A dynamic local-memory request in `(49152 - static_shared, 49152]` fails cold with `CUDA_ERROR_INVALID_VALUE` at
`enqueueKernelLaunch`, located to 8 B at work-group sizes 32 to 1024. The UR CUDA adapter raises
`MaxDynamicSharedMemorySize` only when the dynamic request alone exceeds 49,152. A kernel with zero static shared has no
hole (ceiling 101,376).

> **Warning:** the attribute is sticky per CUfunction. potrf's kernel takes `n` at runtime, so one CUfunction serves every
> `n`. A 49,064 B launch fails cold and passes after a 65,536 B launch of the same function. The outcome depends on order.

* The pad (`potrf_cta.cc:72-296`) maps a request in `(47104, 49664]` to 49,920 B. It is inert today (`ptxas` reports no
  `smem` for any of the eight potrf instantiations) but stays, because one `reduce_over_group` reintroduces the hole.
* No automated test can cover the hole: any earlier launch above 48 KB masks it.
* The geqr2 panel shipped without the pad and hit the hole; it now uses potrf's band ([qr.md](qr.md)).

## Register gate {#register-gate}

Probed on `batchlas_extensions_cta` with `regprobe_any.sh`. Gate: **stack frame == 0 AND spill == 0 AND regs x WG <= 65536.**

| `T` | NB/TS | regs SG/WG | frame | spill | worst `regs x WG` |
|---|---|---|---|---|---|
| `float` | 8/4 | 64 / 56 | 0 | 0/0 | 14,336 |
| `double` | 8/4 | 94 / 80 | 0 | 0/0 | 20,480 |
| `complex<float>` | 8/4 | 102 / 92 | 0 | 0/0 | 23,552 |
| `complex<double>` | 8/2 | 128 / 109 | 0 | 0/0 | **27,904** |

The worst cell has 2.35x headroom. A 128-byte frame was `d[NB]`: a `break` made the trip count data-dependent, so
`#pragma unroll` failed. The kernel predicates instead of breaking. Unrolling `NB = 16` clears the gate but costs
156-206 registers against 64-128 at `NB = 8`.

## CTA kernel measured against cuSOLVER {#cta-kernel-measured-against-cusolver}

`batch = 4096`, JIT-warmed. Ratio `t_vendor / t_native`; above 1 the native kernel wins.

| `n` | float | double | complex\<float\> | complex\<double\> |
|---|---|---|---|---|
| 8 | **2.75** | 1.07 | **1.79** | **1.47** |
| 16 | **1.88** | 0.64 | 1.05 | 0.76 |
| 32 | **1.85** | 0.87 | 0.91 | 1.00 |
| 48 | **1.26** | 0.83 | 0.67 | 0.87 |
| 64 | 1.00 | 0.58 | 0.40 | 0.57 |
| 96 | 0.51 | 0.49 | 0.37 | — |
| 155 | 0.63 | — | — | — |

The CTA kernel loses to cuSOLVER over most of the range `can_run` admits. Float wins up to about `n = 64`; double never
wins above `n = 8`. Float `n = 155` reaches 1.26 TFLOP/s (about 2.7% of FP32 peak): shared memory allows 1-2 resident
blocks per SM, so it binds before registers. Vendor-free, float `n = 128` pays about 2.9x the cuSOLVER runtime.

### The L ladder {#the-l-ladder}

`L` (work-items per matrix) is `Ntiles_0 * TS^2 / 24`, capped at 256 (`potrf_cta.cc:50-67, :383-392`): the elements the
first trailing update touches, at 24 per work-item. A tile count would be the wrong unit, because `TS` varies by type.

| cell (native µs, batch 4096) | L32 | L64 | L128 | L256 |
|---|---|---|---|---|
| float n=48 | 131.9 | 117.2\* | 120.8 | 240.7 |
| float n=128 | 7329 | 4577 | 3224 | 3053\* |
| float n=155 | 11301 | 6955 | 4838 | 4049\* |
| cdouble n=64 | 6826 | 5086 | 4711\* | 4859 |

`*` marks the rule's pick. Across 21 cells the rule picks the measured best, or one within 1%, in 19; the other two are
within 5.5%. **24 is a fitted constant**: re-measure it if `NB`, `TS` or the inner loop changes.

Negative result: `NB = 16` is slower in 18 of 20 cells (float n=96 1162 → 2374 µs). A dual instantiation was declined,
because it doubles device instantiations in a device-link-bound build.

## The blocked driver {#the-blocked-driver}

Right-looking, `Uplo::Lower`, `j = 0, nb, 2nb, ...`:

* **leaf** `A(j:j+ib, j:j+ib)` by the CTA kernel on a sub-view;
* **fixup**: one kernel that merges local-to-global `info` and quenches failed items;
* **panel** `L21 = A21 L11^{-H}` through the injected public `trsm`;
* **trailing** `A22 -= L21 L21^H` through the injected public `gemm`, cut into `W`-wide column panels (a `W x W` diagonal
  block into scratch with a triangular fold, then a direct rectangle gemm).

The seams are required `std::function` injections (`PotrfTrailingGemm`, `PotrfPanelSolve`,
`src/extensions/potrf_native.hh:107-118`); an empty seam throws. The launch passes the public `gemm` and `trsm`
(`src/ops/potrf/potrf.cc:70-78`), so each child takes its own table's choice. The trailing update is 65-95% of a
vendor-free blocked factorisation. The fold exists because `herk` has no native family and a square gemm would write the
upper triangle, which `potrf(Lower)` must not touch.

### `nb` and `W` {#nb-and-w}

Whole driver, `batch = 128`, worst rel sd 1.3%. Each pick is bracketed on both sides:

* **float `nb = 128`:** n=512 2.462 ms (against 2.524 at 155, 2.634 at 96); n=1024 9.180 ms (against 11.604 at 155, 17.699
  at 96).
* **double `nb = 96`:** n=512 7.019 ms (against 7.258 at 109); n=1024 50.980 ms (against 53.338 at 64).
* **cfloat `nb = 96`:** n=1024 34.1 ms, the best of six values swept.
* **cdouble `nb = 64`:** n=512 76.2 ms (against 79.5 at 77); n=1024 550 ms (against 579 at 77).

The float mechanism: the trailing `k` is `nb`, and float's only transposed register kernel needs `k >= 128`, so `nb < 128`
cannot reach it, while `nb > 128` only slows the leaf.

`W` for float is **128**. Whole-potrf ms, `W = 16 / 32 / 64 / 96 / 128`, worst rel sd 4.9%:

| arm | W=16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| float n=1024 b=256, seams native | 28.286 | 17.863 | 18.654 | 20.495 | **16.785** |
| float n=2048 b=128, seams native | 107.046 | 52.937 | 50.879 | 54.936 | **46.510** |
| float n=1024 b=256, seams vendor | 29.330 | 19.657 | 15.619 | 15.124 | **15.122** |
| double n=1024 b=256, native | **77.73** | 78.58 | 81.94 | 85.11 | 89.17 |
| cfloat n=1024 b=256, native | 54.63 | **54.00** | 57.40 | 59.95 | 63.45 |

The curve is non-monotonic (96 is worse than 64): a kernel-selection cliff, so do not interpolate. Double's 16 beats the
shipped 32 by 1.1%, inside the noise. cdouble was not re-swept: `k = nb = 64` reaches no register kernel at any `W`.

* **Reproduction trap:** `potrf_blocked_params` rounds `nb` down to a multiple of `trsm_cta_max_n<T>()` = 32, including a
  `BATCHLAS_POTRF_NB` value. A sweep of 48/80/109/155 collapses onto 32/64/96/128.
* **Trans versus ConjTrans:** real types pass `Trans`, complex pass `ConjTrans`. For real scalars these are the same
  product, but `ConjTrans` selects a slower kernel (1.77-1.86x), and `Trans` gives 1.81x at n=1024. Do **not** use `Trans`
  for complex: `A22 -= L21 L21^H` is Hermitian, and `Trans` takes the residual from 4.0e-07 to 1.9e-02.

### Blocked driver measured against cuSOLVER {#blocked-driver-measured-against-cusolver}

Whole-`potrf`, `Uplo::Lower`, 3 interleaved passes, medians, worst rel sd 5.3%. **nn** is both seams native (the
vendor-free build); **routed** is both seams choosing their own route. Ratios are `cuSOLVER / potrf`; above 1 the driver wins.

| type | n | batch | nn/vend | routed/vend |
|---|---|---|---|---|
| float | 512 | 256 | 0.614 | 0.909 |
| float | 1024 | 256 | **1.108** | **1.646** |
| float | 2048 | 128 | **1.396** | **2.194** |
| double | 1024 | 256 | **1.014** | **1.012** |
| cfloat | 1024 | 128 | 0.436 | **1.040** |
| cdouble | 1024 | 64 | 0.311 | 0.767 |

* Vendor-free float beats cuSOLVER at `n >= 1024`. Double is at parity for n = 512 to 2048 (0.988-1.057 nn).
* **Complex is not there:** 0.311-0.509 vendor-free, widening with `n`. The complex trailing gemm is 97.6% of the cdouble
  n=1024 call, at 0.40 TFLOP/s against cuBLAS's 1.18. With cuBLAS's gemm time substituted, the ratio is 0.89x instead of
  0.32x; a register-tiled complex GEMM is worth about 2.7x on vendor-free cdouble potrf.
* Small `n` (<= 256) loses for every type (0.311-0.825): at `n <= nb` the driver is the leaf plus one fixup launch.
* The crossover is in order, not batch: real types pass cuSOLVER as `n` grows; complex diverges.
* Two cells first taken at rel sd 8.7-12.5% were re-measured at saturating batch (float 1024/256: 1.129, against 1.108
  claimed). With the seams forced native in the vendor-present build, the time is within 0.02% of the vendor-free build.
* A single square gemm does not model the trailing update: the driver issues 217 `32 x 32 x 128` and `mr x 32 x 128`
  gemms, and nsys shows them at 10.63 ms against cuBLAS's 10.95 ms for one square gemm.

### The panel-solve verdict {#the-panel-solve-verdict}

With the gemm seam held at vendor, the panel solve is the only variable. `trsm=vendor / trsm=native` whole-potrf ratios:
float 1.31-1.42 (n = 512 to 2048), double 1.13-1.19, cfloat 1.91 and cdouble 2.69 at n = 512. Native wins every cell tried.
The panel solve is only 5-22% of a vendor-free blocked potrf, so a 2x faster panel solve is worth 3-11% end to end.

## Negative results {#potrf-negative-results}

* **The strided-`ld` GEMM collapse does not reproduce on potrf's trailing shapes** (`sub/flat` 0.89-1.09). It is real where
  a register kernel is reached (float `128x128x155`, `transB = Trans`: 1.43); potrf stays on the slower kernel.
* **A per-route `W` is not needed.** `W = 128` wins on both routes (1.06-1.14x native, 1.30x vendor).
* **Launch overhead is not the problem.** The GPU is 97-99.7% busy. Fusion, epilogue folding or fewer panels buy at most 3%.
* **The fold-free trailing update is 11% cheaper and wrong**: writing the `W x W` product straight into `A` clobbers the
  upper triangle.
* **The first benchmark campaign's "never faster" headline is superseded**: its large-batch cells were racing, and float
  ran at `W = 32`. Its nsys splits and complex-GEMM diagnosis stand.

## potrf: correctness findings {#potrf-correctness-findings}

**1. The panel trsm returned wrong answers on the default vendor-free path.** float and double, `n = 1024`, `batch = 256`:
69-75 of 256 items bad (float), 15/256 (double), non-deterministic; every failing column is `== 1 (mod nb)`.

* **Mechanism** (`src/sycl/trsm_native.cc`): the CTA kernel staged its triangle with a lane-strided loop, then read the
  diagonal back with no barrier. **Fix:** one `sycl::group_barrier(it.get_group())` after staging (`trsm_native.cc:211`).
  Max rel diff against vendor goes from 6.05e+16 to 4.27e-07.
* **How it hid:** trsm's work-group width comes from `{256,128,64,32}`, and every earlier test ran at `wg = 32`, one
  sub-group, where the race cannot show. Guard: `TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly`
  (`tests/trsm_tests.cc:538`).

**2. `beta = 0` read unwritten scratch.** The `W x W` diagonal gemm (`alpha = -1, beta = 0`) writes `ws.product`. A native
gemm's `LinearEpilogue::apply` reads `prior` unconditionally, and `0 * NaN = NaN`, although cuBLAS does not read C at
`beta = 0`.

```
BATCHLAS_POTRF_ROUTE=blocked                            -> 0/8 bad, rel resid 4.724e-07
BATCHLAS_POTRF_ROUTE=blocked BATCHLAS_GEMM_ROUTE=native -> 8/8 bad, rel resid 9.941e-01
```

**Fix:** `ctx->fill(ws.product.data(), T(0), ws.product.size())` (`potrf_blocked.cc:283-285`). Guard:
`PotrfBlockedTest.BlockedDoesNotReadUninitialisedWorkspace`, the only test of 101 that goes red without the fill.

**3. The facade test asserted on a route, not on execution.** It now compares bit-exactly against `potrf_cta_dispatch`;
a residual check cannot see low-bit differences.

**4. Guards that could not fail,** each now fixed: a lower-triangle-only residual (now poisons the upper triangle and
asserts it survives); a residual on batch item 0 only (now `max(item 0, item batch-1)` with `ld = n+7`,
`stride = ld*n+13`); a negative planted pivot, which divides to a finite number (now a NaN pivot); a `G > 1` claim that
nothing asserted (now `ASSERT_GT(packed_ns, 0)`); a residual bound 40-200x looser than the kernel (now `4*n*eps`).

**5. Design-level wrong answers the shape avoids.** The **stale pivot** would read the original diagonal, not the updated
Schur diagonal. The **scope mismatch** would use `group_barrier(sg)` where two or four sub-groups own one matrix; it races
without crashing, so `Scope` is derived by `potrf_cta_launch_params` and `WorkGroup` with `G != 1` throws. The test input
normalises the original diagonal to `+1`, so only the updated diagonal is negative. With that input the stale-pivot break
turns 26 of 42 residual tests red.

**6. `info`.** The leaf reports an index local to its sub-view. The driver merges it as
`if (info[b] == 0 && leaf[b] != 0) info[b] = j + leaf[b]`. The quench needs an identity diagonal as well as a zeroed
panel, because a zero `L11` diagonal gives `0/0 = NaN`. Shipped residuals are 4e-07 to 6e-07 (float, cfloat) and
5e-16 to 1e-15 (double, cdouble).

Workspace sizes come from `potrf_cta_buffer_size` and `potrf_blocked_buffer_size`, which replay the layout through
`BumpAllocator::measuring()`. Never sum by hand: `mempool.hh` checks capacity from the unaligned cursor.

## Open debts {#potrf-open-debts}

1. **Route-era windows unmeasured:** float `n > 256` at batch 512 is refuted (1.049 at n = 384, 1.079 at n = 512, both under
   the 1.11 gate); cfloat `257 <= n <= 368` is untested; cfloat above 368 is unreachable.
2. **Complex is 0.311-0.509x vendor-free**, outside this driver. A register-tiled complex GEMM is worth about 2.7x on
   vendor-free cdouble potrf. This is the highest-value follow-on.
3. **Small `n` (<= 256) loses for every type.** No blocked-driver tuning addresses it.
4. **The strided-`ld` cost of the trailing gemm is not isolated.** The cheap probe (pack `L21` contiguous per panel,
   about 0.5 ms at n = 1024) was never run.
5. **`Uplo::Upper` is unimplemented in the blocked driver** (`can_run` refuses it, `potrf.cc:50-53`). Ways out: mirror as
   syev does, or use a transposed schedule.
6. **The WP3 trsm windows** were measured on the racing kernel above `q*batch ~ 65k` and not re-run. They survive as
   untimed rows of `tuned/trsm.*.sm_89.txt`.
7. **The out-of-order-queue defect cannot be made to fail on this box.** Five dependent edges are guarded by
   `if (!ctx.in_order()) ctx.wait()`; no test builds an out-of-order `Queue`.
8. **The trailing update issues three kernels per column panel.** Fewer launches would gain at most about 3%.
9. **The 48 KB hole has no test and cannot have one** (sticky per-process attribute).
10. **Three dispatch guards have no test:** degenerate extents, non-GPU queue, no sub-group 32. They are `can_run` terms,
    and `PotrfCandidates.CanRunEqualsLaunch` compares them with the drivers.
11. **The accuracy test is loose.** The residual bound `4*n*eps` is looser than the true worst case `(0.2, 1] * n * eps`;
    a defect that costs less than about 4x in accuracy passes.
12. **`Uplo::Upper` coverage in `potrf_tests` is lighter than Lower.** 112 of 216 cases skip (host backends of a GPU kernel).
13. **The Phase 2 route diff was never run.** Owed: one `ctest` run per side under `BATCHLAS_COVERAGE_OUT`, compared with
    `scripts/route_diff.sh`, without a reconfigure.
14. **B6 is unfalsifiable on this hardware.** The barrier ending `potrf_lpanel_body`'s panel loop
    (`src/extensions/potrf_lpanel_device.hh:158-167`) measured green at batch 1024 to 16384, because the eight dependent
    prefetch loads cover the window on this card. Record it as *unfalsifiable*, never *covered*. Closing it needs a probe
    that widens the hazard window on purpose. See [armed breaks (R9) — LPanel](#armed-breaks-r9--lpanel).

## Raw evidence

* Tag `perf-evidence/vendor-independence`, read with `git show <tag>:<path>`: `experiments/wp4_potrf/` (`slm/` for the
  budget, fit ceilings and hole boundary; `regbaseline/` for the register probe; `README.md` for CTA versus cuSOLVER and
  the `L` grid; `phase2/`, `phase2_ab/`, `phase2_impl/` for the correctness, `W` and performance grids; `phase2_bench/` for
  the superseded first campaign).
* Spec and corrections: `WP4_POTRF_SPEC.md`, `WP4_POTRF_SPEC_CORRECTIONS.md`.

## potrf: the occupancy rule {#potrf-the-occupancy-rule}

`potrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` divides the budget by an occupancy target before walking the
footprint. The default target is 4 (`resident::kMinBlocksPerSm`, rule R1), so the tier advertises what fits in a quarter
of the budget. "Can one work-group hold this at all" uses `min_blocks_per_sm = 1`. The walk lives in
`src/util/resident_capacity.hh`; its `break` is armed by a synthetic non-monotone table, because no budget on this box
reaches the distinguishing case. The figures are in [the shipped ceilings](#the-shipped-ceilings-pinned-in-both-scales).

### potrf: native_tier_preferred {#potrf-native_tier_preferred}

A vendor-free build, and a bare `BATCHLAS_POTRF_ROUTE=native` pin, take the first runnable non-vendor entry of the row
(`PotrfCandidates.BareNativePicksTheBestRunnableNonVendor`). The CTA/Blocked crossover is a capacity limit: below it, the
blocked driver at `n <= nb` is the CTA leaf plus a fixup and cannot win; above it, CTA cannot hold the matrix. Every
family is ranked by a measured row, never by a default.

### The occupancy clamp on nb {#the-occupancy-clamp-on-nb}

`potrf_blocked_params` clamps `nb` to the advertised ceiling while `n <= kPotrfOccupancyNbMaxOrder` (256), and to the
resident ceiling above it. `nb` is also the trailing `k`, so halving it buys leaf occupancy and pays in the GEMM. Native
arm, large batch, after / before (below 1 is faster):

| type | n | batch | ratio |
|---|---|---|---|
| float | 128 | 2048 | **0.706** |
| float | 256 | 1024 | **0.931** |
| float | 512 | 512 | 0.999 |
| cfloat | 128 | 2048 | **0.665** |
| double | 128 | 2048 | **0.788** |

Without the threshold, n >= 384 regresses: float 512 1.209x, cfloat 512 1.382x, double 512 1.097x. Do not remove the
threshold. The one cost: float n = 96, batch 2048 goes 0.5312 → 0.7752 ms (1.46x slower), because order 96 is above the
advertised 77 and leaves the single CTA launch. cuSOLVER takes 0.2708 ms there, so Auto is unaffected.

### The shared helper: budget, slice, walk and pack {#the-shared-helper-budget-slice-walk-and-pack}

`src/util/resident_capacity.hh` holds the capacity rule for potrf, getrf and geqrf. It is `constexpr` and free of SYCL
headers. `select::Device` takes its `slm_budget` from `device_slm_budget` (`src/select/select.cc:222`).

| helper | what it answers |
|---|---|
| `device_slm_budget(local_mem_bytes, reserve_bytes = 4096)` | the runtime budget: local memory less the reserve the runtime keeps |
| `kMinBlocksPerSm = 4` | rule R1's occupancy target (geqrf uses 2) |
| `occupancy_budget(budget, min_blocks_per_sm)` | the slice of the budget one work-group may own |
| `resident_max_n(bytes_fn, budget, min_blocks_per_sm, n_hi = 4096)` | the largest `n` whose footprint fits, with every smaller `n` fitting too |
| `pack_matrices_per_wg(bytes, lanes, wg_budget, max_wg_size, target_wg_size = 128, max_pack = 4)` | how many matrices one work-group holds when one sub-group serves a matrix |

* `device_slm_budget` subtracts the reserve: 101,376 → 97,280 and 49,152 → 45,056.
* `resident_max_n` must stop at the first miss. Pads in the 48 KB hole make `bytes(n)` non-monotone, and a `continue`
  advertises a range with a hole in it. That case is reachable only for budgets in **[49,664, 49,920)**.
* Products are `std::size_t` or `int64_t`: `(m|1)*n` overflows `int` at m ≈ 46,341, a reachable panel height.
* `pack_matrices_per_wg` caps at 24 resident blocks per SM for a one-sub-group work-group (50% occupancy). `G > 1` is
  correct only where every barrier is a sub-group barrier; under a work-group barrier the packed matrices race with each
  other (see [why LPanel may pack](#why-lpanel-may-pack-on-work-group-barriers)).
* Test hooks pack two 16-bit fields (`potrf_cta_debug_launch`: G low, L high; `potrf_blocked_debug_params`: nb low, W
  high; `potrf_tiny_debug_launch`: G low, S high). All return 0 when the order does not fit.

### The contiguity rule and the synthetic table {#the-contiguity-rule-and-the-synthetic-table}

`tests/resident_capacity_tests.cc` arms the `break` rule against a synthetic footprint of 128 B per unit of `n`, plus the
48 KB pad (`kHoleLo = 47,104`, `kHoleHi = 49,664`, `kHolePadTo = 49,920`; identical in `potrf_cta.cc`, `getrf_cta.cc`,
`geqrf_cta.cc`):

| n | raw request (B) | after the pad |
|---|---|---|
| <= 368 | <= 47,104 | unchanged |
| 369..388 | 47,232..49,664 | raised to 49,920 |
| 389 | 49,792 | above the band, not padded |

So `bytes(389) < bytes(388)`. At a budget of 49,850 B the walk must answer **368**; a walk that skips misses answers 389
and advertises twenty orders that cannot launch. Arithmetic pinned alongside: `occupancy_budget(97280, 1)` = 97,280 and
`occupancy_budget(97280, 4)` = 24,320.

### The G-packing helper and its three limits {#the-g-packing-helper-and-its-three-limits}

`pack_matrices_per_wg` is bounded by local memory, the work-group width and the cap. At a 24,320 B slice:

| bytes | lanes | max_wg | target | cap | G | what binds |
|---|---|---|---|---|---|---|
| 8,192 | 32 | 1024 | 128 | 4 | 2 | local memory |
| 64 | 8 | 1024 | 128 | 4 | 4 | **the cap alone** (the width admits 16) |
| 64 | 8 | 1024 | 128 | 16 | 16 | cap raised, width admits 16 |
| 64 | 8 | 1024 | 32 | 16 | 4 | **the width alone**, cap raised |
| 64 | 8 | 64 | 128 | 16 | 8 | `max_wg`, cap raised |
| 4,096 | 32 | 64 | 128 | 4 | 2 | device max work-group size |

The 8-lane rows are the load-bearing ones: at 32 lanes the cap and the 128-wide target agree, so a helper that ignored
`max_pack` would pass the first rows too.

### The shipped ceilings, pinned in both scales {#the-shipped-ceilings-pinned-in-both-scales}

Pinned at the reference budget of **97,280 B** in `tests/resident_capacity_tests.cc`. The advertised figure is what
`can_run` admits; the resident figure is what a blocked driver's panel can hold. Conflating them is the defect the pin
catches.

| type | potrf advertised / resident | getrf advertised / resident | geqrf advertised / resident (elements) |
|---|---|---|---|
| float | **77** / 155 | **77** / 155 | **11,776** / 24,320 |
| double | **54** / 109 | **54** / 109 | **5,888** / 12,160 |
| complex\<float\> | **54** / 109 | **54** / 109 | **5,888** / 12,160 |
| complex\<double\> | **38** / 77 | **38** / 77 | **2,944** / 6,080 |

Each potrf figure is `resident::resident_max_n` over `potrf_hole_padded(potrf_slm_per_matrix(...))` with `(NB, TS)` from
`PotrfCtaConst`; `min_blocks = 4` gives the advertised row and `1` the resident row. The advertised ceiling is strictly
inside the resident one, and each fit predicate agrees with its ceiling on both sides. A larger budget admits more of
everything (97,280 → 101,376 raises all three).

## potrf: the tiny tier {#potrf-the-tiny-tier}

`src/extensions/potrf_tiny.cc`: one matrix per `SubGroupPartition<N>`, `N` in {8, 16, 32}. Lane `r` owns row `r` in a
compile-time `D rA[N]`, and every cross-lane value is an indexed sub-group shuffle. It uses **zero local memory, zero
barriers and zero static shared**, so the 48 KB hole is unreachable.

* **Geometry:** work-group 64 (`kTinySubGroups = 2`), `G = 32/N` partitions per sub-group: 8, 4 or 2 matrices per
  work-group at `N` = 8, 16 or 32.
* `potrf_tiny_max_n<T>()` is a flat 32 (16 for cdouble), not a budget walk. `potrf_tiny_buffer_size` is nonzero: an empty
  `info` span draws `batch` int32s of pool scratch.
* `potrf_tiny_debug_launch<T>` checks `S*G*N == wg`, `wg % 32 == 0` and `S >= 2`, and never pins S.

### The tiny potrf window {#the-tiny-potrf-window}

`t_vendor / t_tiny` at batch 65536, `Uplo::Lower`, 9 reps, one process per cell (`benchmarks/results/p9_potrf_tiny.csv`,
`p9_potrf_low.csv`):

| n | 1 | 2 | 4 | 8 | 9 | 12 | 16 | 17 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| **float** | 3.92 | 5.02 | 7.21 | **10.64** | 3.02 | 3.74 | 3.07 | 2.06 | 2.55 | 2.91 |
| **cfloat** | 3.43 | 4.34 | 6.02 | **7.43** | 2.10 | 1.57 | 1.62 | 1.23 | 1.26 | 1.30 |

* The window is `1 <= n <= 32`. n = 1 already wins 3.4-3.9x, so no lower bracket is needed.
* The ceiling is the tier's, not a measured loss: `can_run` refuses n = 33 (`potrf.cc:40`). A wider register kernel is
  untested, not refuted.
* float n = 3 was discarded: its tiny arm exceeded the 10% `rel_sd` gate.

### Unmeasured tiny windows: Upper and the double types {#unmeasured-tiny-windows-upper-and-the-double-types}

Superseded: Upper and fp64 were measured later. See [the extended window](#the-tiny-potrf-window-extended-to-upper-and-to-fp64).

### The posv window is not a no-op {#the-posv-window-is-not-a-no-op}

posv has no vendor family, so "not chosen" means the composed families. Against a vendor-pinned composition, cfloat looks
like it loses at half its orders. Against the incumbent it falls back to, it wins at every order (`t_incumbent / t_tiny`):

| n | 9 | 12 | 17 | 26 |
|---|---|---|---|---|
| vs vendor-pinned composition (nrhs=1) | 0.79 | 0.96 | 0.78 | 1.28 |
| **vs the incumbent under Auto** | **5.40** | **3.84** | **3.18** | **2.59** |

A window cut from the first row was built, shipped and measured. The excluded cells fell to 0.146-0.55x because they landed
on the composed arm, so it was reverted. Measure a flip against the arm it actually replaces.

### Register probe (tiny tier) {#register-probe-tiny-tier}

Gate: **stack frame == 0 AND spill == 0 AND regs x 64 <= 65536**. Probe: `BATCHLAS_BUILD_DIR=$PWD/build/presets/dev-tests
scripts/register_probe.sh out.log '' batchlas_extensions_cta` (sm_89). Every cell has frame 0 and spill 0:

| type | N | regs | regs x wg (64) |
|---|---|---|---|
| float | 8 / 16 / 32 | 54 / 64 / 78 | 3,456 / 4,096 / 4,992 |
| double | 8 / 16 / 32 | 70 / 82 / 104 | 4,480 / 5,248 / 6,656 |
| cfloat | 8 / 16 / 32 | 64 / 112 / 166 | 4,096 / 7,168 / 10,624 |
| cdouble | 8 / 16 | 84 / 128 | 5,376 / 8,192 |

cdouble N = 32 is not instantiated. The worst cell is cfloat N = 32 (166 registers). `kTinyWorstProbedRegs = 176` and a
`static_assert` on `kTinyWgSize * kTinyWorstProbedRegs <= 65536` make widening the work-group fail to compile. cfloat N = 32
has 6 blocks per SM at wg = 64 (25% occupancy, 12 resident matrices), against about 20 for `potrf_cta` at n = 32 float: the
first cell to check in any grid.

### Two negative results, found by the register probe and not by a timer {#two-negative-results-both-found-by-the-register-probe-and-not-by-a-timer}

1. **A batch-uniform early exit for n = 17..31 relocates `rA[]`.** The breaks `if (j >= n) break;` make the unroll
   data-dependent, so ptxas moves `rA` to local memory (frames of 64 to 288 bytes, zero spill). A predicate (`col_live`,
   `k < n`) restores frame 0.
2. **`complex<double>` needed a component-wise select.** A plain `?:` over the 16-byte aggregate is not a `select` in IR,
   so SROA declines to promote `rA`. `tiny_native::tiny_select` took cdouble N = 8 and 16 to frame 0.

### Armed breaks (R9) {#armed-breaks-r9}

Each guard was planted, observed red and restored; `potrf_tiny.cc` was checked byte-identical afterwards (md5).

| break planted | expected guard | observed |
|---|---|---|
| the rank-1 update broadcasts the pre-scale `rA[j]` | `TinyResidualBothTriangles` | red at n = 2 (residual 4.38e-06 against 9.54e-07) |
| `tiny_partition_id` loses its `sg_id *` term | `TinyPackedBatchMatchesSolo` | red; half the batch is poison |
| the `info` write drops its `alive &&` guard | `TinyInfoIsExactAndFirstFailureWins` | red; info = 8 where 3 is required |
| failure-path `tiny_select` replaced by a plain `?:` | **no test** | all 49 tiny cases green; the register probe catches it (frame 144 B and 272 B, spill 0) |

The last break is the one that matters for maintenance. The gate is frame 0 AND spill 0, never spill alone and never a
green suite. `TinyLaunchGeometryIsReported` checks invariants, never a pinned `S`; hook and launcher share one helper,
so a check re-derived from the hook's answer cannot fail.

potrf_tiny's 22 entry functions cost 12.3 s of the 174.8 s ptxas total (7.0%), inside the 15% budget (R7);
see [lu.md](lu.md#r7-the-device-link-all-three-tiny-tiers-landed).

### The shared tiny-tier invariants {#the-shared-tiny-tier-invariants}

`src/extensions/tiny_device.hh` is shared by the potrf, getrf and geqrf tiny tiers. Each invariant breaks silently, as a
wrong answer or a lost residency, never as a compiler diagnostic.

1. **`rA[]` is never dynamically indexed or passed by reference.** Helpers take a plain pointer and touch it only under
   `#pragma unroll` with compile-time bounds. Every write under a failure predicate is a select, never control flow.
   See [armed break 4](#armed-breaks-r9).
2. **`N` must divide the sub-group.** `SubGroupPartition<P>` bases a chunk at `(lane / P) * P`, which is why the ladder is
   `{8, 16, 32}`.
3. **Every lane of a partition must reach every partition collective.** Dead partitions load the identity and suppress
   their store. No tiny kernel uses `if (prob_id >= nb) return;`. On sm_89 this is guarded by source text only.

Further contracts: a lane guard belongs inside a collective, never around it; a complex value is shuffled as two reals;
the identity pad keeps a non-live row at exactly 0 contribution; `c <= lane` keeps the other triangle unread and unwritten;
`Uplo::Upper` is a load transform, not a second algorithm; `tiny_partition_id`'s `sg_id *` term is load-bearing; and
`tiny_load_pad_identity` is spelled per element so the register array's address is never taken. `kTinySubGroups = 2` is a
tuning constant, not a contract.

### The CTA comparison is elementwise, not bit-exact {#the-cta-comparison-is-elementwise-not-bit-exact}

At `n <= 8`, CTA runs one NB = 8 panel that matches this tier line for line. Bit-identity fails at float `n = 2`, Upper,
by one ULP, because the compiler contracts `a - b*conj(c)` into an FMA on one side only. The test asserts an elementwise
`8*eps` bound. The stable bit-exact oracles are tiny-versus-tiny (`TinyPaddingIsInert`, `TinyPackedBatchMatchesSolo`).

### What the geometry test can and cannot check {#what-the-geometry-test-can-and-cannot-check}

It checks the work-group against `MAX_WORK_GROUP_SIZE`, the two-sub-group floor, and that the bucket read back as `32/G`
holds `n` and is the tightest of 8/16/32. It cannot check a pinned `S`.

## The LPanel tier {#the-lpanel-tier}

`src/extensions/potrf_lpanel.cc` and `potrf_lpanel_device.hh`: a left-looking, thread-per-row panel kernel. One work-group
holds one `n x NB` panel in local memory, factors it in place, and re-reads the already-factored columns from global
memory. `NB = 8` for every type; `NB = 16` is float only. `Uplo::Upper` is refused by `can_run` (`potrf.cc:46`) because the
left-looking update reads the lower triangle; no in-tree caller asks for Upper.

### Footprint and the two caps {#footprint-and-the-two-caps}

```
slm_per_matrix(n, NB, sizeof(T)) = (n*NB + NB*NB) * sizeof(T) + 256
```

`sA` is the `n x NB` panel at `ld = n`; `sB` is the `NB x NB` broadcast block. The 48 KB band applies to the total request,
and `G` steps down if the pad pushes it over. The ceiling is the smaller of the local-memory slice and `MAX_WORK_GROUP_SIZE`.
The second is not slack: `potrf_lpanel_body` needs `L >= n`, because lane `row` carries registers across the whole `k`
loop, and an order above the work-group size is a silent race, not a launch failure.

| | float | double | complex\<float\> | complex\<double\> |
|---|---|---|---|---|
| LPanel advertised (4 blocks/SM) | **744** | **368** | **368** | **180** |
| LPanel resident (`min_blocks_per_sm = 1`) | 1024 (wg-capped) | 1024 (wg-capped) | 1024 (wg-capped) | 750 |
| CTA advertised, for comparison | 77 | 54 | 54 | 38 |
| LPanel `NB = 16` advertised (float only) | 360 | — | — | — |

For float, LPanel advertises the whole `32 < n <= 512` band in one launch, where CTA advertises 77. Without LPanel, float
`78 <= n <= 256` runs the blocked driver at `nb = 64`: two panels, two leaves, two fixups, a trsm, a gemm and a fold.

### Why LPanel may pack on work-group barriers {#why-lpanel-may-pack-on-work-group-barriers}

`potrf_cta.cc` packs `G > 1` only at `L == 32`, where its phase barriers are sub-group barriers. LPanel packs on
work-group barriers deliberately: every barrier in `potrf_lpanel_body` is `group_barrier(it.get_group())` at the top level,
and a failed item is a predicated skip, not a `break`. Two rules follow:

* **No early return and no shortened `n`.** A dead slot runs the whole body with its stores suppressed (the `store` flag).
  Giving it `n = 0` would skip the barriers and hang the group.
* **B6 is required.** The panel-end store writes columns that the next panel's staging reads from other work-items. See
  [open debts](#potrf-open-debts) item 14.

### LPanel register gate {#lpanel-register-gate}

The per-lane arrays `rp[NB] + rS[NB] + rA[NB]` are `3*NB` scalars: 24 for real types, 48 for `complex<float>`, 96 for
`complex<double>`. Against the 42-register line for full occupancy, cfloat reaches about two thirds of float's occupancy
and cdouble about a third. The gate is float first. No `ptxas` probe has been run for this tier; the numbers are a source
count.

### The measured LPanel window {#the-measured-lpanel-window}

`benchmarks/factor_bench`, arms interleaved, 7 reps, medians. Ratios are `vendor_ms / arm_ms`; **above 1 the arm wins**.
Raw CSVs: `benchmarks/results/p3_lpanel_ab.csv`, `p3_lpanel_edges.csv`, `p3_auto_after_flip.csv`.

| n | batch | float lpanel | float cta | float blocked | cfloat lpanel | cfloat cta | cfloat blocked |
|---:|---|---|---|---|---|---|---|
| 33 | 8k / 16k | 1.698 / 1.785 | **2.225 / 1.834** | 0.251 / 0.261 | **1.202 / 1.521** | 1.052 / 1.307 | 0.312 / 0.377 |
| 48 | 8k / 16k | **1.571 / 1.834** | 0.948 / 1.080 | 0.223 / 0.250 | **1.238 / 1.498** | 0.685 / 0.796 | 0.367 / 0.422 |
| 64 | 8k / 16k | **1.888 / 2.226** | 0.833 / 0.944 | 0.820 / 0.933 | **1.604 / 1.866** | (does not fit) | 0.537 / 0.585 |
| 128 | 4k / 8k | **1.568 / 1.920** | — | 0.527 / 0.614 | **1.651 / 1.857** | — | 0.702 / 0.766 |
| 256 | 2k / 4k | **1.617 / 2.121** | — | 0.750 / 0.935 | **1.258 / 1.469** | — | 0.863 / 0.890 |
| 384 | 0.5k / 1k | *1.049* / 1.241 | — | 0.838 / 0.929 | (does not fit) | — | 0.797 / 0.919 |
| 512 | 0.5k / 1k | *1.079* / 1.322 | — | 1.040 / 1.177 | (does not fit) | — | 0.948 / 1.079 |

The route-era window was `32 < n <= 256` (Lower, float and cfloat). For float, CTA led only at n = 33 (both batches) and
n = 36 (smaller batch). For cfloat, LPanel took every order from 33 up.

* **Edges.** Float 384 and 512 clear the gate only at the doubled batch, so 257..512 is left to the vendor. cfloat does not
  fit above 368. At n = 32 Tiny wins (float 2.187 / 2.591 against LPanel's 2.053 / 2.314 at batch 16k / 32k).
* **Blocked** is 0.22-0.94x of the vendor from 33 to 256, and reaches parity only at n = 512.
* **Ratios rise with batch,** and the window is gated on the smaller batch in each cell. The minimum in-window ratio is
  float 1.278 (n = 129, batch 2048).
* **Eight cells do not measure the tier they name** (cfloat `lpanel` at 384 and 512; cfloat `cta` at 64 and 65, past its
  ceiling). A ratio of exactly 1.000 is the signature.
* **End to end** (`--arms=vendor,auto`): float n = 33 **1.87x**, n = 64 **2.23x**, n = 128 **1.92x**, n = 256 **2.13x**;
  cfloat n = 64 **1.86x**, n = 256 **1.48x**.
* **Correctness:** 52 of 52 LPanel cases pass on CUDA; results are bit-stable for n in {64, 128, 256, 384, 512} up to batch
  16384. Timings were taken with the card held exclusively.

**Not established:** `NB = 16` has never been timed; `double` and `complex<double>` have no grid (ceilings 368 and 180,
so they may win); there is no `ncu`, register probe or nsys attribution, so the occupancy argument is arithmetic and the
single-launch schedule is not separated from it; and the blocked driver's leaf is still CTA.

### The pivot-cell WAR race {#the-pivot-cell-war-race}

The first kernel returned **wrong answers at large batch** while all 11 tests were green: at `batch >= 1024` (float
`n = 128`) residuals were about 1e-04, against a stable 1.3e-07 at `batch <= 512`, and nondeterministic on identical input.

**Mechanism.** Every lane reads the pivot `sA[(j+i) + i*slda]` at the top of column `i`. The lane that owns row `j+i`
then overwrites that cell with `sqrt(d)` with no barrier between the two. A warp that runs ahead publishes the root before
a slower warp has read it, so that warp scales its rows by the wrong factor. **Fix:** `B3b`, one work-group barrier between
the `d` read and the scale. The first wrong row is always a multiple of 32, so the reader was a warp: the defect needs
occupancy to appear. The guard `PanelStoresArePublishedAtSaturatingBatch` runs `n = 128, batch = 1024` with every item
identical and demands bit-equality with item 0; `batch = 512` was measured not to be enough.

### Armed breaks (R9) — LPanel {#armed-breaks-r9--lpanel}

Planted, built, observed and restored, with the files checked back by md5. Three of the seven predictions were wrong.

| # | the break | expected red | observed |
|---|---|---|---|
| 1 | delete `B6` (the barrier ending the panel loop) | `ResidualAcrossPanelBoundaries` | **green, not armed** (open debt 14) |
| 2 | delete `B3` (between publishing `sA` and the factorisation) | `ResidualAcrossPanelBoundaries` | red, 19 cases |
| 3 | drop `dev_conj` when staging `sB` | complex only | red for complex; real types green |
| 4 | relax the prefetch guard `row >= j + i` to `row >= j` | `OtherTriangleIsNeitherReadNorWritten` | **green**: the bound is enforced three times |
| 4b | relax the prefetch, publish and store guards together | as 4 | red for all four types |
| 5 | drop the `ok &&` from `bad = ok && !(d > R(0))` | `info` names the first planted failure | red: `info = 93` where 1 was planted |
| 7 | give a dead packed slot `n = 0` instead of `store = false` | `PackedBatchIsIndependentAndTheTailSlotsAreInert` hangs | **green**: whole warps exit, and integration now passes `n` unconditionally |

A NaN poison cannot see a write, because a value stored into the upper triangle is computed from the NaN it read. Break 4b
was caught only with a finite sentinel (`-999 + 777i`) and bit-equality: the NaN pass guards "not read", the finite pass
guards "not written". Routes 6, 8 and 9 armed code that no longer exists; under flat selection their property is a
property of the table rows.

## The tiny potrf window, extended to Upper and to fp64 {#the-tiny-potrf-window-extended-to-upper-and-to-fp64}

Both exclusions were untested and are now measured, and both are wins. Batch 32768, arms interleaved, ratios against the
vendor arm, taken with the degraded guard described in [the NVML caveat](qr.md#a-measurement-caveat-that-is-part-of-the-result).
sm_89 `uplo=U` rows rank `tiny` first at every order the tier holds; sm_120 `uplo=U` rows are float only.

### Uplo::Upper is the wider window {#uplo-upper-is-the-wider-window}

`factor_bench --uplo=upper` forms `Uᴴ U` for the residual; the Lower residual on an Upper factor scores a correct answer as
garbage.

| n | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 1 | 21.539x | 20.214x | 3.891x | 2.949x |
| 8 | **28.727x** | 13.251x | 6.987x | 6.310x |
| 16 | 11.018x | 3.462x | 2.317x | 2.127x |
| 17 | 3.192x | 1.738x | — | — |
| 32 | 3.314x | 2.118x | — | — |

Every measured cell wins. cuSOLVER is much worse at Upper than at Lower (float n = 16: 0.0730 ms against 0.0360 ms), while
the tier moves only from 0.01096 to 0.01263 ms. The window is `1 <= n <= tiny_max_n` for all four types on Upper.

### fp64 wins on Lower too, in two bands {#fp64-wins-on-lower-too-in-two-bands}

| n | 1 | 2 | 8 | 9 | 11 | 12 | 16 | 17 | 20 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| **double** | 0.715 | 1.158 | **3.956** | 0.865 | 1.049 | 1.145 | 1.505 | 0.548 | 0.590 | 0.768 |
| **cdouble** | 0.709 | 1.291 | **4.892** | 0.882 | 1.071 | 1.167 | 1.545 | — | — | — |

The window is **n = 2..8 and 12..16** for both types. n = 1 is overhead-bound. 2..8 fills the N = 8 bucket, 9..11 fills
N = 16 badly and loses, and 12..16 fills it well. From 17 on, the N = 32 bucket collapses double to 0.548-0.768x.

Four cells first read 1.000x and were vacuous: at cdouble n = 20 the pinned tier and the vendor both ran the vendor, because
the support predicate refused the pin. A parsed pin proves nothing about the route that ran; under flat selection that
pin throws. End to end, Auto tracks the pinned tier to three digits inside the window (double n = 6 3.017x, n = 16 1.505x;
at the excluded n = 10 Auto is the vendor, 1.000x).

### Arming both windows {#arming-both-windows}

Four breaks were planted against the deleted route-era predicate. Three went red: admitting the fp64 hole at n = 9..11,
refusing Upper, and letting Upper reach the Lower-only LPanel. The fourth, firing the window with no tier linked, stayed
green and is **unfalsifiable**, not covered. Under flat selection the fp64 hole and Upper rows are table data; for
example, `tuned/potrf.double.sm_89.txt` ranks `vendor` first at n = 9 and 10.

## The fused potrs solve {#the-fused-potrs-solve}

posv's `cta` family runs the public `potrf`, then **one** kernel for both triangular solves:
`sycl_getrs::potrs_fused_dispatch` (`getrs_fused.cc`), the fused narrow-RHS getrs body with no pivots and a non-unit
diagonal. Lower solves `L` in axpy form and `L^H` in dot form; Upper mirrors it. It shares getrs's resident-RHS capacity
and `kGetrsFusedMaxRhs`. Above the tiny window, sm_89 ranks `cta` then `blocked`.

Ratio `t_vendor / t_native`, Lower, nrhs = 1:

| n | float before | float after | cfloat before | cfloat after |
|---:|---:|---:|---:|---:|
| 64 | 0.59 | **2.07** | 1.18 | **4.56** |
| 128 | 0.79 | **1.62** | 3.35 | **7.03** |
| 256 | 1.02 | **1.46** | 11.3 | **15.4** |

Double and cdouble (batch 8192, CTA against Blocked) are 1.5-4.5x ahead of Blocked at n = 33 to 256, and ahead of cuSOLVER
in every cell. `FusedSolveArmSolvesOnBothTriangles` pins potrf to `native`: unpinned, Upper at `n >= 33` resolves to
cuSOLVER, which overwrites the unreferenced lower triangle, so a wrong-triangle read would otherwise stay green.

## The posv tiny launch bound {#the-posv-tiny-launch-bound}

The `.minnctapersm` bound applies to `posv_tiny` through `PosvTinyBody`, as for getrf's tiny tier
([lu.md](lu.md#the-tiny-launch-bound)). It is swept per (type, bucket, RHS width) over MinBlocks {1, 8, 12, 16}, batch
32768:

| cell | uncapped (ms) | chosen (ms, blocks) | gain |
|---|---:|---:|---:|
| float n=16 nrhs=1 | 0.0800 | 0.0713 (16) | 1.12x |
| float n=32 nrhs=1 | 0.534 | 0.403 (16) | 1.33x |
| cfloat n=24 nrhs=1 | 2.075 | 1.099 (8) | **1.89x** |
| cfloat n=32 nrhs=1 | 2.490 | 1.353 (8) | **1.84x** |
| cfloat n=32 nrhs=4 | 3.504 | 3.160 (8) | 1.11x |

A cap that is too tight costs as much as it saves: cfloat N = 32 at 16 blocks is 2.3x slower. Every cell is transcribed
from the sweep; `posv_tiny_min_blocks` holds the table. fp64 is uncapped and unmeasured.

## The posv local-memory transpose {#the-posv-local-memory-transpose}

Three levers from [lu.md](lu.md#the-column-bucket) applied to `posv_tiny`: the column bucket `NC` (`{12, 16}` under N = 16,
`{20, 24, 28, 32}` under N = 32), an `NR = 2` RHS bucket with every RHS loop stopping at `nrhs`, and local memory for the
two cross-lane reads that scale with `n`.

* **The Cholesky column.** Each lane stores its own element once, then one `group_barrier(sg)` and vector loads,
  double-buffered by parity. This replaced `n - j - 1` broadcast shuffles per step.
* **The backward-solve transpose.** Lane `i` stores its row and each lane loads the element at its own index. Local memory
  may be indexed dynamically; registers may not.

The factor is bit-identical to `potrf_tiny`. Launch bounds were re-swept per (type, NC, NR) and transcribed in
`posv_tiny_min_blocks`; float is flat within 2% and takes 16.

`t_vendor / t_tiny` and `t_vendor / t_cta` at batch 32768:

| posv cfloat n | nrhs 1 | nrhs 2 | nrhs 4 |
|---|---|---|---|
| 9 | 1.78 / 0.95 | 1.52 / 0.62 | 1.23 / 0.48 |
| 16 | 1.88 / 1.30 | 1.40 / 0.84 | 1.71 / 0.83 |
| 24 | 2.57 / 1.92 | 2.22 / 1.61 | 1.73 / 1.33 |
| 32 | 2.78 / 3.12 | 2.54 / 2.51 | 2.35 / 1.98 |

Every cfloat cell now beats the vendor by at least 1.23x. Against CTA, tiny wins 1.3-2.6x at `n <= 24`, and at nrhs = 4 up
to n = 32; nrhs = 1 above 24 goes to CTA. float is 1.84-5.32x the vendor everywhere and 1.3-3.5x over CTA.

**Current window** (read back from `tuned/posv.*.sm_89.txt`): float ranks `tiny` first at every `n <= 32` for nrhs 1, 2
and 4. cfloat ranks `tiny` first through n = 24 at every width, and at n = 32 only for nrhs = 4.

### posv: the backward solve costs a transpose {#posv-the-backward-solve-costs-a-transpose}

Lane `r` holds row `r` of `L`. The forward solve reads `L(r, i)`, local to the lane; the backward solve `L^H x = y` needs
the column `L(i, r)`, which no lane holds. The first kernel bought it with an on-the-fly shuffle transpose (`N(N-1)/2`
shuffles, 496 at N = 32). The transpose now goes through local memory
([see below](#the-posv-local-memory-transpose)).

## The fused posv tier {#the-fused-posv-tier}

`src/extensions/posv_tiny.cc`: one launch for `A X = B` with Hermitian positive-definite `A`, `n <= 32`, `nrhs <= 4`. It
runs `potrf_tiny`'s Cholesky recurrence, then both triangular solves with `L` still in registers, with zero local memory
and zero barriers. This is posv's `tiny` family, and on sm_89 every row inside the window ranks it first. `can_run` drops
it at `nrhs > 4`; those rows run `cta`.

### The measured posv window {#the-measured-posv-window}

Window: `n <= 32` for all types except cdouble (cap 16), `nrhs <= 4`. Ratio `t_blocked / t_tiny`, where the composed
`blocked` arm under Auto is the incumbent. The gate is 1.11. Four arms run interleaved (`run_solve_grid.sh`: tiny; blocked
with legs under Auto; vendor; native). Raw CSVs: `benchmarks/results/p2_posv_{float,cfloat,double,cdouble}.csv`.

| n | nrhs | float 8192 | float 16384 | float 32768 | cfloat 8192 | cfloat 16384 | cfloat 32768 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 11.27 | 16.37 | 21.84 | 8.06 | 9.39 | 11.94 |
| 9 | 4 | 7.73 | 8.56 | 9.52 | 2.99 | 3.00 | 3.14 |
| 17 | 1 | 7.06 | 7.69 | 7.60 | 3.29 | 3.20 | 3.16 |
| 24 | 4 | 5.10 | 4.52 | 4.76 | 1.55 | 1.72 | 1.92 |
| 32 | 1 | 5.36 | 5.13 | 5.75 | 2.16 | 2.25 | 2.35 |
| 32 | 4 | 3.76 | 3.77 | 4.14 | **1.29** | 1.52 | 1.69 |

Every cell clears the gate. The weakest, cfloat n = 32 nrhs = 4 at 1.29, rises with batch to 1.87 at 131072. Double and
cdouble (two batches per cell, 16384 and 32768) also clear it: double nrhs=1 at n = 4 / 8 / 16 / 32 gives
10.41 / 10.76, 8.03 / 8.19, 5.31 / 5.32, 3.69 / 3.70; cdouble nrhs=4 gives 7.18 / 7.24 at n = 4 and 3.08 / 3.11 at n = 16.
The double window is routed where it wins, by 2.12x at worst.

#### The composed arm under Auto is 1.5-5.6x slower than its vendor-pinned form

With `potrf` and `trsm` both vendor, the composition is 1.5-5.6x faster than under Auto at float nrhs = 1, n = 4 to 32
(batch 8192 to 32768). This is not a posv defect: `blocked` calls `potrf` and `trsm`, and one of them chose a native tier
that loses at `n <= 32` and large batch. `trsm` is the first suspect, since this is `Side::Left`, the side that
[trsm.md](trsm.md) records as having a float cliff. The fused tier still beats the composed arm where it loses to the
vendor-pinned form. The number to chase is `potrf` and `trsm` at `n <= 32`.

Not established: no measured non-winner at the top edge; `Uplo::Upper` has correctness tests at every order but no timing
row; and nrhs 2 and 3 were not timed.

### posv tiny-tier register bound: assumed, then measured {#the-posv-tiny-tier-is-not-register-resident-for-cdouble}

`posv_tiny.cc` declared `kWorstRegsPerThread = 256`, above the 255-register ISA ceiling, so no kernel could report it. The
probe (see [the probe's own gate](#the-probes-own-gate-could-not-fail)) found the tier's instantiations are not
register-resident:

| kernel | registers | stack frame | spill |
|---|---|---|---|
| `PosvTinyKernel<complex<double>, 16, 4>` | **255** | **96 B** | **184 B** |
| `PosvTinyKernel<complex<float>, 32, 4>` | 149 | **256 B** | 0 |
| `PosvTinyKernel<complex<double>, 16, 1>` | 246 | 0 | 0 |
| `PosvTinyKernel<double, 32, 4>` | 222 | 0 | 0 |
| every `GesvTinyKernel` | ≤ 182 | 0 | 0 |
| every `GetrfTinyKernel`, `PotrfTinyKernel`, `GeqrfTinyKernel` | ≤ 193 | 0 | 0 |

Both spilling instantiations are on the default Auto path. This is not a launch abort: 64 work-items need 8,192 registers
against 16,384 per sub-partition. The cost is spill traffic. `gesv_tiny.cc`'s 224 is a safe over-estimate.

Owed: why NR = 4 costs so much more than NR = 1 for cdouble (246 registers at NR = 1 against 255 plus spill at NR = 4);
whether the spill matters end to end (the cdouble window was never timed at these shapes); and whether float N = 32,
NR = 4's 256-byte frame is an unpromoted local array, which needs generated code to confirm. Until the end-to-end cost is
measured, the cdouble posv tier is routed on a residency claim its own probe refutes.

## The probe's own gate could not fail {#the-probes-own-gate-could-not-fail}

`scripts/register_probe.sh` set its spill flag on `Compiling entry function` and cleared it on `Function properties for`,
but ptxas prints the second line before the data line. The flag was therefore cleared before the spill test ran, and the
gate printed 0 for every input. Fixing it surfaced the cdouble spiller. Three more silent-zero defects were fixed in the
same pass: the stack frame was not reported (98 of 541 distinct kernels have a non-zero frame); unmatched entry names were
skipped while counting toward the denominator (they now fail the run); and a SIGPIPE abort truncated the report (fixed
with `sed -n '1,20p'`). The script fails loudly when the target has no `link.txt`, so pass `BATCHLAS_BUILD_DIR` when the
build tree is not `<repo>/build`. The link takes about 340 s.

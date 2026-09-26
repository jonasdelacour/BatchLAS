# POTRF: the CTA kernel, the blocked driver, and the 48 KB SLM launch hole (WP4)

Four native tiers ship, all linked and correct, and **two windows are routed**: a vendor-present build takes a
native tier for `Uplo::Lower`, `float` and `complex<float>`, at `n <= 256`, and cuSOLVER everywhere else. The tiers
also exist so a `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` build stops throwing `NoRouteError`.

All numbers: RTX 4090 (sm_89, 128 SM), one card held per campaign, under `experiments/gpu_guard.sh`. Ratios are
`vendor / native` — **> 1 means native wins** — unless the table says otherwise.

## What ships

| tier | file | orders | Uplo |
|---|---|---|---|
| `{Native, Tiny}` | `src/extensions/potrf_tiny.cc`, `tiny_device.hh` | `<= potrf_tiny_max_n<T>()` = 32, and 16 for cdouble | both |
| `{Native, CTA}` | `src/extensions/potrf_cta.cc`, `potrf_cta_device.hh` | `<= potrf_cta_max_n_for_slm<T>(local_mem - 4096)` = **77/54/54/38** here (advertised, at the default target of 4 blocks/SM); 155/109/109/77 is the *resident* ceiling, `min_blocks_per_sm = 1` | both |
| `{Native, LPanel}` | `src/extensions/potrf_lpanel.cc`, `potrf_lpanel_device.hh` | `<= potrf_lpanel_max_n_for_slm<T>(local_mem - 4096, max_wg)` = **744/368/368/180** here (advertised); see [the LPanel tier](#the-lpanel-tier) | **Lower only** |
| `{Native, Blocked}` | `src/extensions/potrf_blocked.cc` | any | **Lower only** |
| `{Vendor, Auto}` | cuSOLVER | any | both |

All four scalar types on every arm (cdouble n = 17..32 excepted on Tiny).

**`preferred()` is no longer false, and it now carries TWO windows** (`route_potrf.hh:80-110`), both
`Uplo::Lower` and both `float` and `complex<float>` only — `lpanel_types()` and an explicit `uplo` test gate the
whole predicate before either window is consulted:

* **`n <= 32` is the register tier.** `tiny_window(s)` is tested FIRST and, when it holds, `preferred()` answers
  for `Algorithm::Tiny` and for nothing else (`route_potrf.hh:87`). Its bound is
  `min(s.tiny_max_n, 32)`, so cdouble's lower instantiation cap would apply if the type reached the predicate at
  all. Grid: [the tiny potrf window](#the-tiny-potrf-window).
* **`32 < n <= 256` is CTA-or-LPanel.** `best_native_tier` picks one arm and `preferred()` additionally requires
  it to be the tier the grid measured at that order — CTA to 35 for float, 32 for cfloat, LPanel above. Grid:
  [the measured LPanel window](#the-measured-lpanel-window).

Everything outside both windows — `Upper`, `double`, `cdouble`, and **n > 256** — still takes cuSOLVER. Two of
those exclusions are *untested, not refuted*, with the grids that would settle them at
[unmeasured Tiny windows](#unmeasured-tiny-windows-upper-and-the-double-types).

**Tiny is FIRST in the candidate order and is now routed there.** See
[the-tiny-tier](#the-tiny-tier) for its register table and two negative results; at the tier edge, n = 32, it
measures **2.187 / 2.591x** the vendor at batch 16k / 32k (float), which is the cell that decided the boundary
against LPanel. The rest of the candidate order (`route_potrf.hh`) is *mostly* a capability ladder:
the blocked driver's diagonal leaf *is* the CTA kernel on a sub-view, so above `cta_max_n` only Blocked can serve.
**Below it the two arms overlap and the static array decides** — `supports(Blocked)` carries no lower order bound, so
for `Uplo::Lower` at `n <= cta_max_n` both are supported and the vendor-free walk takes CTA because it is listed
first. That is a tuned guess nobody has measured; see [open-debts](#open-debts) item 16 and the one data point that
exists (float n=128 b=512: CTA 0.293 ms against blocked 0.301 ms). Env var `BATCHLAS_POTRF_ROUTE` (`cta`/`blocked`/`native`/`vendor`/`native:cta`), synthesised from
`op_env_stem(Op::potrf)`; no legacy spelling exists and none must be invented. `BATCHLAS_POTRF_NB` /
`BATCHLAS_POTRF_W` are tuning, read once per process in the driver, never in the table.

| | float | double | complex\<float\> | complex\<double\> |
|---|---|---|---|---|
| CTA `NB`/`TS` (`potrf_cta.cc:38`) | 8/4 | 8/4 | 8/4 | 8/2 |
| CTA fit ceiling, **advertised** (4 blocks/SM, the default) | **77** | **54** | **54** | **38** |
| CTA fit ceiling, resident only (`min_blocks_per_sm = 1`) at 97,280 B | 155 | 109 | 109 | 77 |
| blocked `nb` (`potrf_blocked.cc:44`) | 128 | 96 | 96 | 64 |
| blocked `W` | **128** | 32 | 32 | 16 |

No `nb` equals the fit ceiling. Sizing `nb` from the fit ceiling — what the spec does — is measurably wrong for
every type.

### Route arms and the `supports()` gates

`supports()` (`route_potrf.hh:32-61`) is correctness-only; every gate means "wrong answer or cannot launch":

| line | gate | why it is correctness |
|---|---|---|
| `:200` | `s.m != s.n` | no factor of a non-square view; the kernel drives both extents from one order |
| `:207` | `!s.is_gpu` | nd_range kernel with a `local_accessor` tile; no host implementation |
| `:216` | `!s.has_sg32` | `[[sycl::reqd_sub_group_size(32)]]`; the launch is rejected without it. **Enumerated** from `sub_group_sizes` — `get_property(MAX_SUB_GROUP_SIZE)` returns `sub_group_sizes()[0]`, so syev's `max_sub_group >= 32` is wrong in both directions |
| `:234` | `s.heterogeneous_batch` | one launch, one `(order, ld, stride)` tuple; per-item active dims factorise the wrong order in place for every item after the first, and potrf has no batch walker |
| `:241` | `order() < 1 \|\| batch < 1` | panel loop and tile index map undefined |
| `:264-265` | CTA: `cta_max_n < 1` (absent from build), else `order() <= cta_max_n` | hard local-memory capacity |
| `:292-293` | Blocked: `uplo != Uplo::Lower`, else `blocked_available && cta_max_n >= 1` | the right-looking schedule is Lower-shaped; handed Upper it overwrites the wrong triangle |

Deliberately **no** `uplo` gate on the CTA arm: `A = U^H U` is the same recurrence on `S(i,c) = conj(A(c,i))`, a
load/store transform, swept under both `Uplo` by `ResidualBothTriangles` and
`OtherTriangleIsNeitherReadNorWritten`. Deliberately **no** lower order bound on the Blocked arm: in `supports()` it
would make a forced `blocked` at small `n` fall through `automatic()` to cuSOLVER and measure nothing. `PotrfShape`
adds `cta_max_n` (asked of the *device*), `blocked_available` (does the driver exist in this *build*) and
`has_sg32`; the builder is `src/backends/potrf_route.hh:20-55`, so the table stays pure.

### What `preferred()` answered before P3

**Superseded.** This section records the state up to P1/P7, when `preferred()` returned `false` unconditionally;
the shipped predicate is now the window at [the measured LPanel window](#the-measured-lpanel-window). What remains
true and load-bearing: un-preferred is not unroutable — `route_resolve.hh:38-49` still hands a **vendor-free**
caller any supported native route, which is why a window is about the vendor-present default and nothing else.

`route_diff.sh` across both landings shows zero changed non-potrf rows;
what changed is potrf's `native_route_supported` column, 0 → 1 (`route_potrf.hh:3-4`), and the Phase 2 capture is
+28 additions / 0 removals. **The Phase 2 triage delta's route diff was never re-run, and the reason recorded here
for skipping it was false.** It claimed the capture needs `-DBATCHLAS_ENABLE_COVERAGE=ON`, i.e. two
device-link-bound rebuilds. **There is no such option and there never was one that worked**: `cmake/BatchLASOptions.cmake:141-150`
records that it is deliberately absent, because the gate sits in `resolve_route`, an inline function template, so a
consumer TU compiled without the macro interposes its own uninstrumented copy and recording silently stops. It is
gated at RUNTIME on `$BATCHLAS_COVERAGE_OUT`, which `scripts/route_diff.sh` sets itself — "any ordinary build
works … no special configuration", in the script's own words. The skip therefore rests only on the second half of
the argument, that the diff touched no routing predicate (one `group_barrier`, two `fill`s, four in-order guards,
one tuning constant, comments). That is an argument from reading the diff, which is exactly what `route_diff.sh`
exists because it cannot establish. **The obligation is OWED**: one `ctest` run per side under
`BATCHLAS_COVERAGE_OUT`, no reconfigure. Note that [open debts](#open-debts) item 14 predicts the capture would
show nothing anyway — but "the instrument is blind here" is a result to record, not a reason to skip the run.

The grids below are the *input* to flipping a cell, not the decision. The gate is three-part:
`t_native <= 0.90 * t_vendor` at saturation, **and** no accuracy regression, **and** an end-to-end win.
(This repo once turned a 2.16x kernel win into an 11% gesvd loss.) For the LPanel window the first two were run —
see [the measured LPanel window](#the-measured-lpanel-window), which reports the grid, the bracketing non-winners
at n = 384 and 512, and an end-to-end A/B through the facade. **`ortho_benchmark` itself was not run** for potrf;
the end-to-end evidence is `factor_bench --arms=vendor,auto`, which exercises the routed path but not a caller.

## The SLM budget and the fit ceilings

`local_mem_size` reports **101,376 B**, and a kernel with 0 B static shared launches at exactly that (`cudaDeviceProp
sharedMemPerBlockOptin` agrees). `device_limits.hh`'s 49,152 is **hardcoded** by
`cmake/BatchLASDetectSYCL.cmake:45-46` for any `nvidia_gpu_sm_*` pattern — the detection routine never queries
`local_mem_size` — and is wrong here by 2.06x. Budget used: `local_mem_size - 4096` = **97,280 B**. `potrf_cta.cc:47`'s `kPotrfReferenceSlmBudget = 97280` is that figure frozen for the convenience overload `potrf_cta_max_n<T>()` **only**; `potrf_cta_dispatch`, `potrf_cta_debug_launch` and the route table all re-read `LOCAL_MEM_SIZE` from the device, which is why the hardcoded 49,152 never reaches a decision.

| `T` | ceiling `n` | bytes at ceiling | first miss |
|---|---|---|---|
| `float` | **155** | 96,540 | 156 → 98,408 |
| `double` | **109** | 95,476 | 110 → 98,108 |
| `complex<float>` | **109** | 95,444 | 110 → 98,076 |
| `complex<double>` | **77** | 95,328 | 78 → 99,056 |

Those bytes are the **shipped** `potrf_slm_per_matrix` (`potrf_cta.cc:58-67`), i.e. with the 256 B slack term.
`slm/README.md`'s table (96,368 / 95,336 / 95,272 / 95,132) and its "first miss" column predate that correction and
are the 64 B formula; `README.md` §2 carries the corrected ceiling column but its first-miss column is still the old
one. The **ceilings are identical under both formulas**, which is the point of the correction below.

All four ceilings were **launched cold, one process each, and returned the right answer** before the kernel existed
(`slm/maxn_fitcheck.csv`). `MeasuredFitCeilings` pins them against the budget-parameterised query, so it holds on any
machine. The spec's {105, 74, 74, 52} would leave float `n` in 106..155 with **no route at all** vendor-free.

Occupancy ladder at other budgets — the menu a `preferred()` grid should be cut against, since the cliffs are where
the answer changes:

| budget | blocks/SM | float | double | cfloat | cdouble |
|---|---|---|---|---|---|
| 24,320 | 4 | 77 | 54 | 54 | 38 |
| 32,853 | 3 | 89 | 63 | 63 | 45 |
| 45,056 (the spec's) | 2 | 105 | 74 | 74 | 52 |
| 49,920 | 2 | 111 | 78 | 78 | 55 |
| **97,280 (shipped)** | **1** | **155** | **109** | **109** | **77** |
| 101,120 (hard) | 1 | 158 | 111 | 111 | 79 |

Two formula corrections, both measured:

* **The accessor slack term was 64 B and 64 B was too small.** At float `n = 155` the raw accessor sum is 96,288 B,
  the old formula said 96,348, and `ncu` reports `launch__shared_mem_per_block_dynamic = 96,408` (static 0) — the
  launch asked **60 B more** than the number `supports()`, `potrf_cta_max_n_for_slm()` and `p.fits` are computed
  from. Real overhead is 120 B; the term is now 256 B and all four ceilings are unchanged. Nothing failed here only
  because the 4,096 B reserve left 872 B of headroom; on a device landing within ~120 B above the formula value it
  arrives as `CUDA_ERROR_INVALID_VALUE` at enqueue instead of the documented throw.
* **The advertised ceiling walked the raw size while the launcher gated on the hole-padded size.** One predicate
  now. The `break` in that walk is load-bearing: `potrf_hole_padded` is **not monotone** (47,200 → 49,920 while
  49,700 stays 49,700), and `supports()` spells the capacity as a contiguous `order <= cta_max_n`, so the ceiling
  must be the largest `n` for which *every* order up to `n` launches.

## The 48 KB launch hole

Measured cold: a dynamic local-memory request in `(49152 - static_shared, 49152]` fails with
`CUDA_ERROR_INVALID_VALUE` at `enqueueKernelLaunch`; boundary located to 8 B, identical at wg = 32/64/128/256/1024.
CUDA's non-opt-in per-block limit is 49,152 B for **static + dynamic**, and the UR CUDA adapter raises
`MaxDynamicSharedMemorySize` only when the *dynamic* request alone exceeds 49,152, so in the band where
`static + dynamic > 49152 >= dynamic` it neither fits nor opts in. A control kernel with zero static shared has no
hole and a ceiling of 101,376 — that control confirms the model.

**Order-dependent**: the attribute is sticky per CUfunction and potrf's kernel takes `n` as a *runtime* argument, so
one CUfunction serves every `n` — 49,064 B FAILs cold and passes after a 65,536 B launch of the same function. Pad
(`potrf_cta.cc:72-296`): request in `(47104, 49664]` → allocate 49,920. The band is +-2 KB rather than the probe's
256 B because static shared is not something this source controls; padding up costs occupancy only at float `n` in
~108..111, already at 1-2 resident blocks/SM.

**The pad is inert today, and that was measured**: `ptxas` reports no `smem` field for any of the eight potrf
instantiations, so static shared is 0 and `(49152 - 0, 49152]` is empty. Disabling the pad and re-running the
residual sweep — which includes float `n = 108..111` — passed green. (The source comment says `n = 110` asks
49,044 B, "squarely inside the hole"; that is the **64 B** formula. Under the shipped 256 B term the same launch
asks 49,236 B, which is above 49,152 and therefore *outside* the measured hole, though still inside the shipped
`(47104, 49664]` pad band. The green run remains evidence that the pad is inert; it is weaker evidence than the
comment claims, because the hole is empty anyway at zero static shared.) It stays because one `reduce_over_group` anywhere in the body
reintroduces it. **No automated test can cover it here**: the attribute is sticky per process, so any earlier >48 KB
launch masks it. A permanent hole in potrf's guard set.

**And the "future group collective" arrived.** WP5's `geqr2_panel_device` runs two `reduce_over_group` calls per
reflector, so its resident leaf carries static shared, sits in this hole, and **shipped without the pad** — a
reachable `CUDA_ERROR_INVALID_VALUE` found on `geqrf_tests`' first vendor-free run
(`VENDOR_INDEPENDENCE_PLAN.md:731-750`). The fix adopted potrf's band and constants verbatim in three places; see
[qr.md](qr.md). So this section is not a hypothetical: the condition potrf wrote down is the condition that later
bit, and the pad is the reason potrf itself did not.

## Register gate

Probed on `batchlas_extensions_cta` with `regprobe_any.sh` — the stock `scripts/register_probe.sh` hardcodes
`batchlas_sycl.dir/link.txt` and **would have reported clean for code it never compiled**.

| `T` | NB/TS | regs SG/WG | frame | spill | worst `regs x WG` |
|---|---|---|---|---|---|
| `float` | 8/4 | 64 / 56 | 0 | 0/0 | 14,336 |
| `double` | 8/4 | 94 / 80 | 0 | 0/0 | 20,480 |
| `complex<float>` | 8/4 | 102 / 92 | 0 | 0/0 | 23,552 |
| `complex<double>` | 8/2 | 128 / 109 | 0 | 0/0 | **27,904** |

The gate is three-condition (`frame == 0` **and** zero spill **and** `regs x WG <= 65536`), passing with 2.35x
headroom at the worst cell; `register_probe.sh`'s two-condition header is stale. 16 rows added versus the pre-potrf
baseline, all potrf, **zero non-potrf rows changed** — the unit's 16 pre-existing `complex<double>` spillers
(`gesvdj_cta`, `ormqr`, `syev`) are untouched, which is why the gate is scoped to the potrf entries.

* **A 128-byte stack frame was `d[NB]`, not `acc[TS][TS]`, and its cause was a `break`.** A `break` in the (P1) loop
  makes the trip count data-dependent, `#pragma unroll` fails, and `d[k]`/`d[c]` acquire a dynamic index — an array
  in `.local` with **zero reported spill**. Two other explanations were refuted first: routing `fma_acc`'s
  by-reference accumulator through a scalar temporary gave a byte-identical report, and dropping double to `TS = 2`
  (`acc[2][2]` = 32 B) left the frame at exactly 128. The kernel predicates instead.
* **Unrolling is not free.** Unrolled `NB = 16` clears the gate but costs 156-206 registers against 64-128 at
  `NB = 8` — 3 resident blocks against 8 for float at the 128-item work-group ceiling.

## CTA kernel measured against cuSOLVER

`batch = 4096` (saturation), JIT-warmed, two passes agreeing within a few percent.

| `n` | float | double | complex\<float\> | complex\<double\> |
|---|---|---|---|---|
| 8 | **2.75** | 1.07 | **1.79** | **1.47** |
| 16 | **1.88** | 0.64 | 1.05 | 0.76 |
| 32 | **1.85** | 0.87 | 0.91 | 1.00 |
| 48 | **1.26** | 0.83 | 0.67 | 0.87 |
| 64 | 1.00 | 0.58 | 0.40 | 0.57 |
| 96 | 0.51 | 0.49 | 0.37 | — |
| 128 | 0.36 | — | — | — |
| 155 | 0.63 | — | — | — |

**The CTA kernel loses to cuSOLVER over most of the range `supports()` advertises.** It wins for float to `n ~ 64`
and at `n = 8` for every type; above that it is 2-3x slower. Every boundary is bracketed by a measured non-winner:
float 1.26 at 48, 1.00 at 64, 0.51 at 96; double never wins above `n = 8`; cfloat 1.05 at 16 against 0.91 at 32.

Efficiency at float `n = 155`, `batch = 4096`: 5.08 GFLOP in 4,046 µs = **1.26 TFLOP/s, ~2.7% of this card's
~47 TFLOP/s FP32**. `sm__warps_active` is **8.3%** at `n = 96/128/155`; shared memory allows 1-2 resident blocks/SM
against `registers` 9-18, so shared memory binds by 9-18x. Vendor-free, a caller at float `n = 128` pays ~2.9x the
cuSOLVER runtime they replaced.

### The `L` ladder

`L` (work-items per matrix) derives from the **elements** the first trailing update touches, `Ntiles_0 * TS^2`, at
24 elements per work-item, capped at 256 (`potrf_cta.cc:50-67, :383-392`). It was a *tile*-count ladder justified
by a thread-limit argument `ncu` refutes — the thread limit never binds — and a tile count is the wrong unit anyway,
since `TS` varies across the type ladder: `complex<double>` at `TS = 2` has 4x the tiles at 1/4 the work each, so a
tile rule over-shoots it by two rungs.

Measured, `batch = 4096`, native µs, `*` marks the rule's pick:

| cell | L32 | L64 | L128 | L256 |
|---|---|---|---|---|
| float n=48 | 131.9 | 117.2\* | 120.8 | 240.7 |
| float n=64 | 318.2 | 274.5 | 242.6\* | 355.5 |
| float n=96 | 2128 | 1482 | 1171 | 1161\* |
| float n=128 | 7329 | 4577 | 3224 | 3053\* |
| float n=155 | 11301 | 6955 | 4838 | 4049\* |
| double n=96 | 7481 | 5976 | 5020 | 4812\* |
| cdouble n=64 | 6826 | 5086 | 4711\* | 4859 |

Across 21 cells the rule picks the measured best or within 1% in 19, within 5.5% in the other two (cfloat n=64;
cdouble n=32, a genuine ~5% regression, 642.4 → 680.0 µs, accepted for a rule worth 1.06-1.32x in eleven others).
The old ladder cost up to 1.27x (float n=96). **24 is a fitted constant**, the only number on that line not derived,
pinned by this grid alone; re-measure if `NB`, `TS` or the (P3) inner loop changes.

## The blocked driver

Right-looking, `Uplo::Lower`, `j = 0, nb, 2nb, ...`, `ib = min(nb, n-j)`, `m2 = n - j - ib`: **leaf**
`A(j:j+ib, j:j+ib)` by the CTA kernel on a sub-view (ld-insensitive — consecutive lanes hold consecutive rows at a
fixed column); **fixup**, one kernel doing the local→global `info` merge plus the failed-item quench; **panel**
`L21 = A21 L11^{-H}` through the injected routed trsm; **trailing** `A22 -= L21 L21^H` through the injected routed
gemm, cut into `W`-wide column panels — a `W x W` diagonal block into scratch plus a triangular fold, and a direct
rectangle gemm.

Both seams are `std::function` injections, a recorded defect class rather than style: WP3's V2 called
`sycl_gemm::gemm_custom` directly, bypassing `RouteTable<Op::gemm>`, so its trailing updates always got the native
kernel whether or not it was better — and potrf's trailing update is 65-95% of a vendor-free blocked factorisation.
An empty function means "use the native entry point", keeping the kernel layer free of the dispatch layer.

**Why gemm + explicit fold and not herk/syrk**: `herk` has no native arm at all (vendor-free it calls
`throw_no_vendor_route`) and `syrk`'s "cublasdx" route is silently a fallback that **writes both triangles**. A
square gemm over `A22` would also write the upper triangle, which LAPACK `potrf(Lower)` must not touch.

### `nb` and `W`

`nb`, whole driver, `batch = 128`, `n in {512, 1024}`, worst rel sd 1.3%, ms:

| type | grid | pick |
|---|---|---|
| float | n=512 native, nb 32/48/64/96/128/155: 3.754/3.194/2.929/2.634/**2.462**/2.524; n=1024 with a correct panel solve, nb 64/96/128/155: 19.993/17.699/**9.180**/11.604 | **128** |
| double | n=512 native, nb 32/48/64/80/96/109: 8.594/7.814/7.399/7.190/**7.019**/7.258; n=1024: 59.343/55.456/53.338/53.538/**50.980**/53.682 | **96** |
| cfloat | n=512 leaf+panel+trail: 7.38/6.32/5.75/5.47/**5.22**/5.40; n=1024 total: 43.4/42.8/36.7/38.1/**34.1**/37.9 | **96** |
| cdouble | n=512 total, nb 32/48/64/77: 81.3/78.6/**76.2**/79.5; n=1024 native: 576/749/**550**/579 | **64** |

Every pick is bracketed on both sides. float's mechanism: the trailing update's `k` **is** `nb`, and float's only
transposed register kernel (`Tiled128x32RegisterK32NT`) needs `m >= 128 && n >= 32 && k >= 128`, so `nb < 128`
cannot reach it while `nb > 128` only slows the leaf (0.335 ms at nb=128 against 0.391 at 155, n=512).

`W` for float is **128**, shipped as 32 first. Re-measured end to end on a *correct* factorisation at the shipped
`nb`, interleaved, 3 passes x 2 reps, worst rel sd 4.9%, whole-potrf ms:

| arm | W=16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| float n=512 b=256, seams native | 4.422 | 3.449 | 3.774 | 4.271 | **3.403** |
| float n=1024 b=256, native | 28.286 | 17.863 | 18.654 | 20.495 | **16.785** (1.06x over 32) |
| float n=2048 b=128, native | 107.046 | 52.937 | 50.879 | 54.936 | **46.510** (1.14x over 32) |
| float n=1024 b=256, seams vendor | 29.330 | 19.657 | 15.619 | 15.124 | **15.122** (1.30x over 32) |
| double n=1024 b=256, native | **77.73** | 78.58 | 81.94 | 85.11 | 89.17 |
| cfloat n=1024 b=256, native | 54.63 | **54.00** | 57.40 | 59.95 | 63.45 |

Same mechanism — the `W x W` diagonal gemm has `m = n = W`, `k = nb`, so only `W = 128` reaches the register kernel.
**The curve is non-monotonic (96 worse than 64), a kernel-selection cliff rather than the linear waste term; do not
interpolate it.** double's 16 beats the shipped 32 by 1.1%, inside the noise of the n=512 table its 32 came from;
left alone. cdouble was not re-swept: its `k = nb = 64` reaches no register kernel at any `W`. The shipped `(nb, W)`
pair is now measured **together**, which it never was: float n=1024 b=256 native at `W = 128`, `nb = 32/64/96/128`
gives 46.918/36.484/31.781/**16.452** — `nb = 128` by 1.93x.

**Reproduction trap:** `potrf_blocked_params` rounds `nb` down to a multiple of `trsm_cta_max_n<T>()` = 32, and that
applies to `BATCHLAS_POTRF_NB` too. A sweep of 48/80/109/155 collapses onto 32/64/96/128 and yields four cells
identical to their neighbours; read the recorded `nb` tables with those labels substituted.

**A free 1.81x for float from one enum value.** The trailing update passes `Transpose::Trans` for real types,
`ConjTrans` for complex — identical operations for a real scalar, different kernels, since `gemm_kernels.cc:464`'s
transposed short-circuit sends every `ConjTrans` to `Tiled16`. Kernel level 1.77-1.86x; end to end at n=1024,
16.629 → 9.180 ms = **1.81x** (1.00x at `nb <= 96`). It must **not** be done for complex — `A22 -= L21 L21^H` is
Hermitian — the substitution takes the residual from 4.0e-07 to 1.9e-02 for cfloat and cdouble.

## Blocked driver measured against cuSOLVER

Whole-`potrf` wall time, `Uplo::Lower`, JIT and clocks warmed and discarded, host-device page migration excluded
from the timer, 3 interleaved passes x 2 reps, medians, worst rel sd 5.3%, `bad = 0` in every arm. **nn** = both
seams native (the vendor-free build); **routed** = both seams choosing their own route.

| type | n | batch | cuSOLVER ms | nn ms | routed ms | nn/vend | routed/vend |
|---|---|---|---|---|---|---|---|
| float | 256 | 2048 | 3.497 | 5.902 | 5.039 | 0.593 | 0.694 |
| float | 512 | 256 | 2.082 | 3.393 | 2.291 | 0.614 | 0.909 |
| float | 1024 | 256 | 17.515 | 15.802 | 10.644 | **1.108** | **1.646** |
| float | 2048 | 128 | 64.857 | 46.473 | 29.560 | **1.396** | **2.194** |
| double | 256 | 512 | 3.471 | 4.208 | 4.239 | 0.825 | 0.819 |
| double | 512 | 256 | 12.471 | 11.794 | 11.778 | **1.057** | **1.059** |
| double | 1024 | 256 | 79.575 | 78.471 | 78.662 | **1.014** | **1.012** |
| double | 2048 | 64 | 149.454 | 151.223 | 151.226 | 0.988 | 0.988 |
| cfloat | 256 | 1024 | 2.669 | 5.240 | 3.726 | 0.509 | 0.716 |
| cfloat | 512 | 256 | 3.459 | 7.790 | 3.802 | 0.444 | 0.910 |
| cfloat | 1024 | 128 | 12.017 | 27.555 | 11.559 | 0.436 | **1.040** |
| cdouble | 256 | 256 | 6.521 | 15.287 | 7.592 | 0.427 | 0.859 |
| cdouble | 512 | 128 | 22.132 | 60.575 | 25.897 | 0.365 | 0.855 |
| cdouble | 1024 | 64 | 80.433 | 258.840 | 104.830 | 0.311 | 0.767 |

Two cells first taken at rel sd 8.7-12.5% were discarded and re-measured at a saturating batch (float 512→2048,
cfloat 256→1024). Reproduced independently through the **public `potrf` API**, no forced route, built against
`build/` and `build-novendor/`: float 256/2048 0.597 (claim 0.593), float 1024/256 1.129 (1.108), float 2048/128
1.401 (1.396), double 1024/256 1.025 (1.014), residuals identical between builds. Forced
`BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native` in the vendor-present build gives 15.787 ms against the
vendor-free build's 15.784 — 0.02%, so the forced-vs-resolved trap does not bite this driver.

* **Vendor-free float beats cuSOLVER at `n >= 1024`** — 1.108x at 1024, 1.396x at 2048, bracketed below by 0.614 at
  n=512. double is at parity across n=512..2048 (0.988-1.057). The WP4 goal, met for the real types.
* **Complex is not there**, 0.311-0.509 vendor-free, and the gap *widens* with `n` (cdouble 0.44 at n=128 → 0.28 at
  n=2048). Cause is outside this driver: `route_gemm.hh:43-45` returns false for complex and
  `gemm_kernels.cc:465` keeps the register ladder inside `if constexpr (is_same_v<T,float>)`, so every complex
  trailing gemm lands on `Tiled16`. At cdouble n=1024 that gemm is **97.6%** of the call and 2.95x slower than
  cuBLAS (0.40 TFLOP/s against 1.18, on a card whose FP64 ceiling is ~1.29). Substituting cuBLAS's gemm time into
  the unchanged breakdown gives 0.89x instead of 0.32x: **a register-tiled complex GEMM is worth ~2.7x on
  vendor-free cdouble potrf alone.**
* **Small `n` (<= 256) loses for every type (0.311-0.825), and it is the Phase 1 leaf.** float n=128 b=512: CTA
  0.293 ms, blocked 0.301 ms, cuSOLVER 0.140 ms — the blocked driver at `n <= nb` is the leaf plus one fixup launch.
* **The crossover is in ORDER, not batch.** Real types converge on and pass cuSOLVER as `n` grows; complex diverges.

nsys per-stage at float n=1024 b=256 (ms/call): trailing gemm **10.63 native against cuBLAS's 10.95** — parity, a 3%
win, on the shapes the driver issues. That contradicts the 0.13-0.18x recorded at kernel level, because the two
measured different shapes: one square `m2 x m2 x nb` gemm against the 217 `32 x 32 x 128` and `mr x 32 x 128` gemms
the W-decomposition issues. **A figure for "the trailing update" from a single square gemm does not transfer here.**

### The panel-solve verdict

Injected **routed trsm**, not a bespoke panel kernel. Gemm seam held at the vendor so the panel solve is the only
variable (whole-potrf ms, `vendor trsm / native trsm`, `bad = 0` everywhere):

| type | n | batch | trsm=vendor | trsm=native | ratio |
|---|---|---|---|---|---|
| float | 512 | 256 | 3.174 | 2.256 | 1.407 |
| float | 1024 | 256 | 15.144 | 10.663 | 1.420 |
| float | 2048 | 128 | 38.674 | 29.569 | 1.308 |
| double | 512 | 256 | 15.503 | 13.003 | 1.192 |
| double | 1024 | 256 | 99.482 | 87.893 | 1.132 |
| cfloat | 512 | 256 | 7.212 | 3.781 | 1.907 |
| cdouble | 512 | 128 | 69.349 | 25.770 | 2.691 |

Native wins in every cell tried. The *original* evidence ("46 of 48 panel cells") was taken on the racing kernel of
[correctness-findings](#correctness-findings) and is evidence of nothing; the table above is post-fix. A bespoke
kernel would also aim at the wrong stage — the panel solve is 5-22% of a vendor-free blocked potrf against 65-95%
for the trailing update, so a hypothetical 2x there is worth 3-11% end to end.

## Negative results

* **`NB = 16` for the CTA kernel is slower in 18 of 20 cells.** The hypothesis was sound and the prediction wrong:
  `ncu` does show the register cost is free above `n ~ 32` (`shared_mem` 2 vs `registers` 18 blocks/SM at float
  n=96), and halving the panel count still loses. Native µs, NB=8 → NB=16: float n=8 7.7 → 20.8 (**2.70x worse**),
  n=16 16.6 → 30.7, n=32 43.8 → 67.9, n=48 117.1 → 153.3, n=96 1162 → 2374 (**2.04x**), n=155 4046 → 4258; double
  n=96 4938 → 5400; cfloat n=8 12.0 → 26.4. Only double n=16 improves (142.7 → 131.6). Dual instantiation declined:
  it doubles potrf's device instantiations from 8 to 16 in a device-link-bound build, to choose between a winner and
  a loser.
* **A per-route `W` is not needed.** The observation (float loses 1.6x with the vendor gemm at `W = 32`) was
  confirmed; the remedy was refuted. `W = 128` wins on **both** routes — 1.06-1.14x native, 1.30x vendor.
* **Launch overhead is not the problem.** The driver issues 200-1000 launches per call (float n=1024: 8 leaves, 8
  fixups, 7 trsms, 217 gemms, 112 folds; cdouble n=1024: 960 gemms, 480 folds). nsys GPU time vs wall: float
  1024/256 nn 17.10 / 17.24 (**99.2% busy**), VV 19.17 / 19.83 (96.7%), cdouble 1024/256 nn 951.8 / 955.1 (99.7%).
  Fusion, folding into the epilogue, or cutting panels buys at most 3%.
* **The strided-`ld` GEMM collapse does not reproduce on potrf's trailing shapes.** WP3's 0.43-0.62x is a property
  of `Tiled128x128RegisterK8`, which the `ConjTrans` short-circuit means potrf never reaches: `sub/flat` is
  0.89-1.09 across all four types. The effect *is* real where a register kernel is reached (float `128x128x155` with
  `transB = Trans` measures `sub/flat = 1.43`) — potrf is immune only because it is stuck on the slowest kernel.
* **The fold-free trailing update is 11% cheaper and wrong.** Writing the `W x W` diagonal product straight into `A`
  costs 0.0650 vs 0.0737 ms (float trailing stage) and clobbers the upper triangle.
* **The first benchmark campaign's headline is superseded.** It concluded "never faster", geomean 0.74 routed / 0.52
  vendor-free, best reliably-correct cell 0.996x. Both causes were later fixed: its large-batch cells were racing
  and float ran at `W = 32`. Its *mechanism* work stands — nsys splits, the complex-GEMM diagnosis, the
  launch-overhead rejection, and the route-typo trap (an unrecognised `BATCHLAS_POTRF_ROUTE`, and `cta` above the
  ceiling, both silently resolve to vendor).

## Correctness findings

**1. The panel trsm returned wrong answers on the default vendor-free path.** `build-novendor`, no env, float and
double, `n = 1024`, `batch = 256`, condition number < 1.05, an input cuSOLVER factors to 1e-8/1e-16 in the same
process: 69/71/75 of 256 items bad over three reps (float), 15/256 (double), non-deterministic. Every failing column
`== 1 (mod nb)` — the first column of a panel, i.e. a diagonal block the previous panel's bad `L21` destroyed. Clean
at `batch <= 96` and `n <= 384`, which is why the implementation-time proof runs (batch 3-32) missed it. Localised
by holding one seam at a time: vendor/vendor clean, native gemm + vendor trsm clean, **vendor gemm + native trsm
19-29 bad**, native/native 61-65 bad. Not the leaf, which is bit-identical across all 256 items on the same sub-view.

Mechanism, `src/sycl/trsm_native.cc`: the V1 CTA kernel staged its triangle into local memory with a lane-strided
loop, then read the **diagonal** back to form reciprocals with **no barrier in between**. Element `idx` is written
by lane `idx % wg`; lane `s` reads a different lane's write for nearly every `s`. The same gap let lane 0's
`sDiv[0] = 0` land after another lane's store of 1, discarding the revert-to-division flag. Fix: one
`sycl::group_barrier(it.get_group())` after the staging loop (`trsm_native.cc:211`), the only functional line
changed in that file. A/B on full rebuilds: deleted → max rel diff vs vendor 6.05e+16, 127/128 items wrong; restored
→ 4.27e-07, 0/128.

**How it hid.** The trsm launcher picks its work-group width from `{256,128,64,32}`, taking the first with
`batch * ceil(q/wg) >= 4*CU` (512 here). Every trsm test that existed before this fix uses `batch <= 3` and `q <= 257`, so **every
one of them ran at `wg = 32`** — one sub-group, both loops in lock step, where the race cannot express itself. Nothing in
the suite ever left one sub-group and nothing said so; the blocked potrf panel solve is the first caller that does.
Deleting the barrier turns **exactly one of 92 trsm tests** red and leaves the whole 216-test potrf suite green.

**The regression test written with the fix was itself vacuous.** As first shipped,
`TrsmNativeCta.MultiSubGroupWorkGroupStagesItsTriangleCorrectly` called V1 directly at `n=16, q=1024, bs=128`: that
clears the work-group ladder and its anti-vacuity assertion (`wg > 32`) passes, but with the barrier deleted and the
library rebuilt **it still came back green**. Clearing the ladder is necessary, not sufficient. It is now
`TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly` (`tests/trsm_tests.cc:538`), driving order 48
through V2 so the *final* V1 block is order 16, at `q=976, batch=128`, against an independent multiply-back oracle;
RED with the barrier deleted, GREEN with it restored, both on full rebuilds. The repository's **fifth** recorded
guard that could not fail, and the first where the vacuous test shipped in the same change as its fix.

The `nb` rounding in `potrf_blocked_params` was originally containment for this defect (orders 48/77/80/109 failed
while 32/64/96/128 were clean, above `q*batch ~ 65k` — exactly where the wg ladder leaves 32). Post-fix a direct
`trsm(Right, Lower, ConjTrans, NonUnit)` sweep over {16,32,48,64,77,80,96,109,128,155} at `q=896, batch=256` matches
a host reference to cuBLAS's relative error at every order. **It is kept anyway**: all four shipped `nb` are
multiples of 32 so it is the identity on every default path, and it stops a hand-set `BATCHLAS_POTRF_NB` reaching an
unmeasured V1 block structure.

**2. `beta = 0` read unwritten scratch.** The `W x W` diagonal gemm is issued with `alpha = -1, beta = 0` into
`ws.product`, never written before that gemm; the driver's comment asserted this was safe, citing the *fold's*
lines. `beta == 0` means "C is not read" in the fold and in cuBLAS, and does **not** mean that in any native gemm
here — `LinearEpilogue::apply` is `alpha*accum + beta*prior` with `prior` read unconditionally, and `0 * NaN = NaN`.
Reproduced through the public API (float n=256 batch=8, well-conditioned SPD) with a prior unrelated
`ctx.workspace()` lease leaving poison in the arena bytes potrf then leases:

```
BATCHLAS_POTRF_ROUTE=blocked                            -> 0/8 bad, rel resid 4.724e-07
BATCHLAS_POTRF_ROUTE=blocked BATCHLAS_GEMM_ROUTE=native -> 8/8 bad, rel resid 9.941e-01
```

The second line is the vendor-free build; with cuBLAS injected it survived by luck. **No Phase 2 test saw it**
because every one allocated a fresh `UnifiedVector<std::byte>`, whose `malloc_shared` pages come back **zeroed**.
Fix: one `ctx->fill(ws.product.data(), T(0), ws.product.size())` before the panel loop
(`potrf_blocked.cc:283-285`), fire-and-forget rather than the `fill().wait()` the review also flagged. Guarded by
`PotrfBlockedTest.BlockedDoesNotReadUninitialisedWorkspace` — the only test of 101 that turns red without the fill.

**3. The facade test asserted on a route re-resolution, not on execution.** The previous `FacadeReachesTheCtaKernel`
stayed **green** with the facade's CTA arm removed while every number in it came from cuSOLVER. It is now a
bit-exact comparison against `potrf_cta_dispatch`. The failure message is instructive: the two values *print*
identically (`-0.0428712` vs `-0.0428712`) and differ only in the low bits, which is why a residual check could not
catch it.

**4. Other guards that could not fail:**

| guard | what made it blind | fix |
|---|---|---|
| `PHASE2_BREAK=nofold` residual | computed over the **lower triangle only**, so writing the symmetric product into the upper triangle is invisible by construction | tests poison the opposite triangle and assert it survives bit for bit |
| the A/B harness's residual | computed on **batch item 0 only**, which sits at offset 0 and cannot move when the batch stride is wrong | `max(item 0, item batch-1)`; parent at `ld = n+7`, `stride = ld*n+13` |
| `PHASE2_BREAK=noquench` | a *negative* planted pivot divides to a finite number | plant a **NaN** pivot, in block 1 of 3 not the final block (a final block has `m2 == 0`, nothing propagates): 1045/789/789/533 non-finite words |
| `PackedBatchMatchesSolo` | claimed `G > 1` in a comment, asserted it nowhere | `potrf_cta_debug_launch<T>` plus `ASSERT_GT(packed_ns, 0)` |
| residual bound `40*n*eps` | 40-200x looser than the kernel | now `4*n*eps`; tightening to `0.2*n*eps` turns it red, so the bound brackets |
| tests hardcoding `{128,96,96,64}` | `nb` is clamped by the device SLM ceiling then rounded to a `trsm_cta_max_n<T>()` multiple — three inputs a test cannot compute | `potrf_blocked_debug_params<T>`, a query over the pure function the driver calls |
| `scripts/register_probe.sh`'s summary | counted `Function properties` blocks for non-inlined device functions as well as entry functions, reporting "16 kernels with non-zero spill" when every entry function is clean | prints entry-function and all-function counts separately |

**5. Two design-level wrong answers the kernel shape avoids**, both proven by reintroduction: the **stale pivot**
(reading `S(j+k, j+k)` from the tile gives the *original* diagonal, not the updated Schur diagonal) and the
**scope mismatch** (`group_barrier(sg)` where 2 or 4 sub-groups own a matrix makes (P1)→(P2)→(P3) straight races, no
crash, a plausible wrong factor). `Scope` is *derived* by `potrf_cta_launch_params`, and `WorkGroup` with `G != 1`
throws.

**The stale-pivot break is also this repository's *fourth* blind guard, and the page above would have mis-stated it.**
Reintroduced, it turned 18 of 42 red — every residual test — and `InfoIndexIsExact` and `InfoReportsTheFirstFailure`
**stayed green**, which was not the prediction. Cause: with a plain planted `L0 D L0^H` the *original* diagonal at the
failure column is still negative, so a stale-pivot reader names the same column and the `info` oracle cannot tell the
two apart. `make_planted_ldl` now normalises row `c`'s prefix so the original diagonal there is `+1` and only the
updated Schur diagonal is negative, and the test asserts that property **of its own input**. Re-run with the same
break: 26 of 42 red, `InfoIndexIsExact` among them, reporting `info == 33` where 17 was planted
(`tests/potrf_tests.cc:7-46`). Two of those five breaks turned nothing red and that was checked to be correct rather
than a gap: the `(P3)` forced-real Hermitian diagonal and the load-side real-forcing are each masked by the other two
of three redundant enforcement points, and break 5 (removing both, then scaling the publish) turns
`ComplexDiagonalIsExactlyReal` red at `imag(L(1,1)) = -2.08e-11`.

**6. `info`.** The leaf reports an index **local** to its sub-view and writes it **unconditionally**, so the driver
translates and merges: `if (info[b] == 0 && leaf[b] != 0) info[b] = j + leaf[b]`. Breaking the offset turns three
blocked tests red per type and leaves **the entire CTA suite green**, because there `j` is always 0; breaking the
merge to last-panel-wins leaves the residual green and is visible only to an `info` test. The quench keeps a failed
item finite, and **zeroing the panel alone is not enough** — the spec's naive form (`WP4_POTRF_SPEC.md:387`)
computes `0/0 = NaN`, so the identity diagonal is the load-bearing half. Measured at n=300: planted failures at
global columns 100 and 200 report 101 and 201, both-column items report 101, a NaN pivot reports 101 and stays
finite. Shipped residuals: float 4.4e-07..5.5e-07, double 5.4e-16..1.0e-15, cfloat 2.7e-07..4.1e-07, cdouble
5.3e-16..7.7e-16; `max |imag(diag L)|` is exactly 0.0 up to n=1000.

### Workspace sizing

`potrf_cta_buffer_size` and `potrf_blocked_buffer_size` replay the layout through `BumpAllocator::measuring()`,
**never** hand-summing: `mempool.hh` checks capacity from the *unaligned* cursor while advancing only by the data
extent, so an "exactly computed" figure fails the allocator's own capacity check, and `required_bytes()` rounds to
the coarsest quantum the sequence asked for. `nb` and `W` come from one pure function the driver also calls, so the
query cannot size a layout the call does not build — the Phase 1 failure was exactly that: a raw SLM figure in the
query against a padded one in the launcher, throwing on a call the table had promised.

## What the spec got wrong

`WP4_POTRF_SPEC.md` predates WP0-WP3; `WP4_POTRF_SPEC_CORRECTIONS.md` carries 108 findings against it. Corrections
beat the spec; shipped code beats both.

| spec | shipped | why |
|---|---|---|
| `:273` ceilings {105, 74, 74, 52} from `slm_budget = 45056` | {155, 109, 109, 77} from 97,280 | 45,056 descends from a hardcoded 49,152 that is not a detected property; 105 leaves float `n` 106..155 with no route vendor-free |
| `:267` "48 KB stays the hard per-work-group ceiling" | 101,376 measured, 97,280 used | refuted by direct measurement |
| `:225` blocked leaf at `Scope::SubGroup` | `Scope` derived from `L`, asserted | the spec's own `L` ladder at `:189-195` contradicts `:225` for float; obeying it makes the phase barriers sub-group barriers across a 64-item matrix |
| `:387` quench = zero the panel | identity diagonal **and** zeroed panel | a zero diagonal in `L11` makes the solve compute `0/0 = NaN` |
| `:172-177`/`:184` runtime `nb` resolution (`resolve_potrf_nb<T>(n, hint)`) clamped into a per-type `NB` ladder, default 16/16/16/8 | compile-time per-type `NB`, 8 for every type | an unmeasured knob multiplies instantiations in a device-link-bound build; `NB = 16` was then measured **slower in 18 of 20 cells** |
| `:474` `supports()` gated on the forced route | pure, env-free `supports()` | `resolve_route` never bypasses `supports()`, so env-gated support makes forcing and support mutually recursive |
| `:559`/`:567` batch thresholds, and `:566`'s lower order bound on the **Blocked** arm (`if (n <= potrf_cta_max_n<T>(...)) return false;`) | both belong in `preferred()`, which is all-false | a batch threshold in `supports()` kills the vendor-free fallback and makes every "forced native" test silently run cuSOLVER and pass green. (`route_potrf.hh`'s comment cites this as "spec:567"; the line is `:566`.) |
| `:574` `std::array<Provider,6>` | file-scope array, `sizeof`-computed bounds | removes the hand-counted truncation hazard |
| `W = 128` from a 12.5% waste figure | measured; real waste is 25% at the spec's own shapes | the constant survives for float for an unrelated reason (the register-kernel cliff) and is 32/32/16 elsewhere |
| §2.5 `BATCHLAS_POTRF_UPDATE=herk` oracle swap | host multiply-back residual only | a device `herk` A/B compares the kernel against another BatchLAS path, not an independent one |

Two contract facts found the hard way: the vendor batched `trsm`/`potrf` **require a pointer array on the view** (a
`MatrixView` from the 6-arg constructor has an empty `data_ptrs_` and throws `"data_ptrs target is null"`), and
`data_ptrs(ctx)` launches a kernel **and waits** on every call. That constructor also sizes `data_` as
`stride * batch_size` from the *offset* pointer, so a sub-view at `(j,j)` claims `j*ld + j` elements past the
allocation — inert for `potrf_cta_dispatch`, `trsm` and `gemm`, none of which read `A.data().size()`. The driver
builds every sub-view with the explicit 6-arg constructor and its own stride; the child's default `ld*cols` stride
is RED on all four types (`inf`, 1.99e+266, 9.39e+25, 6.75e+234).

## Open debts

1. **Two windows ship; the rest of the surface is unflipped.** (Was "`preferred()` is all-false" — refuted on
   2026-09-14 by P1's tiny window and P3's LPanel window.) What is routed: `Uplo::Lower`, `float` and
   `complex<float>`, `n <= 32` on `Algorithm::Tiny` and `32 < n <= 256` on CTA/LPanel. What remains unflipped, and
   why, each stated with the right word and split by TYPE, because the two types do not share a verdict here:
   **n > 256 for `float`** is *refuted at the smaller batch* of its bracket (1.049 at n = 384 / batch 512, 1.079
   at n = 512 / batch 512, both under the gate; they clear it only at the doubled batch, so R8a leaves them with
   the vendor). **n > 256 for `complex<float>` is not refuted, because it was never timed.** The two bracket
   cells the grid prints for cfloat read "(does not fit)": `slm_per_matrix(384, 8, 8) = 25,344 B` exceeds
   `occupancy_budget(97280, 4) = 24,320 B`, so `supports()` is false, the pin falls through to `automatic()` and
   the row measures the vendor against itself. Above the cfloat LPanel ceiling of 368 the band is *unreachable* —
   a stronger statement than refuted, and one no timing can change without a wider tier — while the reachable
   strip `257 <= n <= 368` is ***untested***: not one cell of it was run. See
   [the measured LPanel window](#the-measured-lpanel-window) for both bracket rows as recorded.
   `Uplo::Upper` and the two double types are ***untested, not
   refuted*** — see [unmeasured Tiny windows](#unmeasured-tiny-windows-upper-and-the-double-types) and debt 5 for
   the blocked driver's separate Upper gap. Gate evidence, per window, stated as recorded and no wider: LPanel has
   the grid, the bracketing non-winners at n = 384 and 512 (float only — cfloat does not fit there), and an
   end-to-end A/B through the facade
   (`factor_bench --arms=vendor,auto`); Tiny has the grid and the `auto`-arm before-picture (0.999-1.003) read
   beside every cell. **`ortho_benchmark` was not run for potrf at all**, so no window here has a caller-level
   number.
2. **Complex is 0.311-0.509x vendor-free and the cause is outside this driver.** A register-tiled complex GEMM is
   worth ~2.7x on vendor-free cdouble potrf alone — the highest-value follow-on.
3. **Small `n` (<= 256) loses for every type, and it is the Phase 1 leaf.** No blocked-driver tuning addresses it.
4. **The strided-`ld` cost of the trailing gemm has never been isolated.** Every operand is a sub-view at the parent
   `ld`, `OpShape` carries no leading dimension so the router cannot see it, and the cheap probe (pack `L21`
   contiguous once per panel step; ~59 MB, ~0.5 ms of copy at n=1024 b=256) was never run.
5. **`Uplo::Upper` is unimplemented in the blocked driver** and `supports()` refuses it (`route_potrf.hh:55`).
   Routes: mirror (as syev does) or a transposed schedule.
6. **WP3's trsm `preferred()` windows were measured on the racing kernel** above `q*batch ~ 65k` and have not been
   re-run. The barrier costs one `__syncthreads()` per work-group so the timings are approximately still valid, but
   they were not measurements of a correct kernel.
7. **The out-of-order-queue defect could not be made to fail on this box.** Five dependent edges are guarded with
   `if (!ctx.in_order()) ctx.wait()`, free on the default in-order queue, but nothing demonstrates the failure they
   prevent and no test constructs an out-of-order `Queue`.
8. **The trailing update issues three kernels per column panel** where save/restore would issue about one (329 → 126
   launches at n=1024). The fold-free half was measured 11% cheaper and wrong; the save/restore half was never
   built. At `W = 128` the launch count is already 4x below the figure that estimate used, and the GPU is 97-99.7%
   busy, so it is worth at most ~3%.
9. **The buffer-size max is one-sided**, leaving the vendor→native half open. Closing it means computing the blocked
   layout unconditionally and adding `W^2 * batch * sizeof(T)` — megabytes at large batch — to every vendor-present
   `potrf` that will never touch it.
10. **The 48 KB hole has no test and cannot have one** (sticky per-process CUfunction attribute).
11. **Three dispatch guards have no test**: degenerate extents, non-GPU queue, no sub-group 32. Non-square and
    heterogeneous-batch do.
12. **No accuracy test beyond the residual bound.** It is `4*n*eps` while the true worst case is in
    `(0.2, 1] * n * eps`, so a defect degrading accuracy by less than ~4x still passes.
13. **`heterogeneous_batch` is written by potrf's shape builder but not trsm's**, so `route_trsm.hh`'s own
    heterogeneous gate is decorative. Adding it is a strict de-risking but it *is* a route change.
14. **The burn-down instrument cannot see Phase 2**: 26/54 before, 26/54 after, because no unpinned vendor-present
    call reaches the driver by design. An `IDENTICAL` `route_diff` across such a change is not evidence of anything.
    Write the facade-routed over-ceiling test *first*, capture second, and give it an `n` in a `shape_class` bucket
    no CTA-sized call touches.
15. **`Uplo::Upper` coverage in `potrf_tests` is lighter than Lower** for `PackedBatchMatchesSolo`,
    `EmptyInfoSpanStillFactorises`, `ComplexDiagonalIsExactlyReal` and `FacadeReachesTheCtaKernel`; residual and
    `info` sweep both. 112 of 216 cases skip (host backends of a GPU kernel).
16. **The CTA/Blocked overlap below `cta_max_n` is decided by array order, not by a measurement.** `supports()`
    carries no lower order bound on Blocked, so for `Uplo::Lower` at `n <= cta_max_n` both arms are supported and
    the vendor-free walk takes CTA because `kPotrfOrder` lists it first. One data point exists (float n = 128
    b = 512: CTA 0.293 ms against blocked 0.301 ms). This is the item the top of this page points at.
17. **B6 is unfalsifiable on this hardware, and is held by a data-flow argument alone.** The barrier ending
    `potrf_lpanel_body`'s panel loop orders the panel-end global store against the NEXT panel's left-looking
    staging read of those same columns, by different work-items (`src/extensions/potrf_lpanel_device.hh:8-10`;
    the store/`B6` pair is at `:158-167`). Deleting it was planted and measured **GREEN** — nothing red at batch
    1024 in `PanelStoresArePublishedAtSaturatingBatch` (all four types, n to 512), nor at batch 4096/16384 through
    `factor_bench` — apparently because the 8 dependent global loads of the next panel's prefetch cover the window
    on this card. So **no test on this hardware can arm it**, and none claims to; see
    [armed breaks (R9) — LPanel](#armed-breaks-r9--lpanel) row 1 and the comment on that test. Record this as
    *unfalsifiable*, never as *covered*: the dependency is real in the data flow, another card or another
    prefetch depth may expose it, and "no test goes red" is not a licence to delete the barrier. Closing this debt
    means a probe that widens the hazard window on purpose (a shortened prefetch, or a forced warp skew), not
    another batch.

## Raw evidence

Raw data is at tag `perf-evidence/vendor-independence`, retrievable with
`git show perf-evidence/vendor-independence:<path>`.

| topic | path |
|---|---|
| SLM budget, fit ceilings, occupancy ladder, the 48 KB hole | `experiments/wp4_potrf/slm/README.md` |
| the four ceilings launched cold, one process each | `experiments/wp4_potrf/slm/maxn_fitcheck.csv` |
| the hole boundary, located to 8 B at five work-group sizes | `experiments/wp4_potrf/slm/scan_hole_boundary.csv` |
| `local_mem_size` query, bisection, zero-static control, `cudaDeviceProp` | `experiments/wp4_potrf/slm/slm_probe_gpu0.log`, `nostatic_gpu0.log`, `devprop.log` |
| hole order-dependence (cold FAIL, warm OK); occupancy vs bytes per `ncu` run | `experiments/wp4_potrf/slm/slm_hole_order.cpp`, `occ_sweep.csv` |
| register probe generalised to any device-link target; pre-potrf baseline | `experiments/wp4_potrf/regbaseline/regprobe_any.sh`, `batchlas_extensions_cta.tsv` |
| shipped and rejected `(NB, TS)` register reports | `experiments/wp4_potrf/kernel/` |
| Phase 1: CTA vs cuSOLVER, the `L` grid, the `NB=16` refutation, the vendor-free probe | `experiments/wp4_potrf/README.md` |
| Phase 2, authoritative: barrier defect, `beta=0` defect, `W` re-measurement, the performance grid | `experiments/wp4_potrf/phase2/README.md` |
| Phase 2 A/B harness, `nb`/`W` design sweeps, panel-solve study, break list | `experiments/wp4_potrf/phase2_ab/` |
| Phase 2 implementation correctness harness and its break table | `experiments/wp4_potrf/phase2_impl/README.md` |
| first benchmark campaign — **headline superseded**, mechanism work stands; per-cell rows, nsys splits, nb/W sweeps | `experiments/wp4_potrf/phase2_bench/README.md`, `main.csv`, `nsys/`, `nbsweep.csv` |
| the spec and its 108 corrections | `WP4_POTRF_SPEC.md`, `WP4_POTRF_SPEC_CORRECTIONS.md` |

---

## The occupancy rule

**P7, 2026-09-10.** `potrf_cta_max_n_for_slm<T>(budget, min_blocks_per_sm)` divides the
device budget by an occupancy target before walking the footprint. The default target is
4 (`resident::kMinBlocksPerSm`, design rule R1), so the tier **advertises** what fits in
a quarter of the budget and the separate question "can a work-group hold this at all" is
asked with `min_blocks_per_sm = 1`.

At this box's 97,280 B budget:

| type | advertised (target 4) | resident (target 1) |
|---|---|---|
| float | **77** | 155 |
| double | **54** | 109 |
| complex\<float\> | **54** | 109 |
| complex\<double\> | **38** | 77 |

Pinned budget-parameterised in `tests/resident_capacity_tests.cc`, so they hold on any
machine. The walk itself moved to `src/util/resident_capacity.hh`; its `break` (rather
than `continue`) is unchanged and is now armed by a synthetic non-monotone table, because
no budget on this box can reach the case that distinguishes the two.

### native_tier_preferred

`route_potrf.hh` now declares the hook it was missing. The crossover **is** the capacity:
below it the blocked driver at `n <= nb` is the CTA leaf plus a fixup launch and cannot
win; above it the CTA arm is not supported. Declaring it matters because the absent hook
defaulted to `true` for every route, making the choice an accident of the order array.

**Superseded in part by P3.** `preferred()` is no longer all-false: it answers inside one
measured window, and the hook below is what that window resolves the tier with. See
[the measured LPanel window](#the-measured-lpanel-window); the crossover-is-the-capacity
argument still holds for `double` and `complex<double>`, which have no LPanel grid.

#### Every tier is enumerated explicitly

Exactly one arm answers `true` at any shape, and the switch names all four rather than
leaning on `default:`. `default:` returns **true**, so a tier left out of the switch is
handed every shape it supports on the vendor-free walk and on a bare `native` pin -- R8b
(`docs/design/small-n-factorization-plan.md`, "A non-empty `preferred()` pre-empts
`native_tier_preferred`") arriving from the other direction.

`Algorithm::Tiny` therefore carries an EXPLICIT `return false`, the same answer the getrf
and geqrf tables give their own Tiny arms. Tiny leads `kPotrfOrder`, so a true answer here
would hand it every order <= `tiny_max_n` on the vendor-free walk and under a bare
`BATCHLAS_POTRF_ROUTE=native` pin -- before a single cell of the tier had been timed, and
invisibly to a vendor-present build. This arm and `preferred()` flip TOGETHER in the change
that measures the window; until then the tier is reachable only by an explicit
`{Native, Tiny}` pin, which is all a benchmark arm needs.

The other two arms split on the capacity, `cta_holds = (cta_max_n >= 1) && (order <=
cta_max_n)`: CTA takes it, Blocked takes its complement.

### The tiny potrf window

**2026-09-14.** potrf's register tier shipped with P1 and was never routed: `preferred()`
excluded `order <= 32` outright and `native_tier_preferred` answered **false** for Tiny, so
`Auto` took cuSOLVER at every order the tier can hold. It wins at **every one of them**.

Grid `benchmarks/results/p9_potrf_tiny.csv` and `p9_potrf_low.csv`, `t_vendor / t_tiny`,
one process per cell, arms interleaved, 9 reps, `Uplo::Lower`:

| n | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 | 17 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **float** @65536 | 3.92 | 5.02 | — | 7.21 | 9.12 | **10.64** | 3.02 | 3.74 | 3.07 | 2.06 | 2.29 | 2.55 | 2.93 | 2.91 |
| **cfloat** @65536 | 3.43 | 4.34 | 5.17 | 6.02 | 6.40 | **7.43** | 2.10 | 1.57 | 1.62 | 1.23 | 1.30 | 1.26 | 1.32 | 1.30 |

float n = 3 was **discarded and named**: its tiny arm returned `rel_sd` over the 10% gate.
Nothing else in either grid was discarded. The `auto` arm was measured beside every cell
and read **0.999-1.003** — the flip's before-picture, and proof the tier was inert.

The window is therefore the whole tier, `1 <= n <= 32`, float and cfloat. Three properties
of its edges, stated exactly:

* **There is no lower bracket and none is needed.** n = 1 already wins 3.4-3.9x; the tier
  holds nothing smaller.
* **The ceiling is the TIER's, not a measured loss.** `supports()` refuses n = 33 because
  the kernel is square-only to 32, so nothing above the window was measured losing — a
  wider register kernel is *untested, not refuted*. Above 32 the LPanel window takes over.
* **`Uplo::Upper` is excluded although the kernel handles it.** `supports()` carries no
  uplo gate for Tiny (Upper is the same recurrence on `S(i,c) = conj(A(c,i))`), but the
  grid is Lower-only and an unmeasured triangle is not a window. Open cell.

The weakest cells are the ones that fill their register array least — cfloat n = 17 at
1.23x — which is the same bucket-fill effect the getrf and posv windows turn on; see
[the tiny getrf window](lu.md#the-tiny-getrf-window).

#### Arming the tiny window

Three breaks planted; **two red, one unfalsifiable**, recorded rather than counted:

| planted break | observed | why |
|---|---|---|
| fire the window with no tier linked (drop the `tiny_max_n` gate) | **red** | |
| let CTA answer inside the tiny window too (R8b) | **red** | |
| admit `Uplo::Upper`, which has no grid | *stayed green* | `preferred()` refuses Upper before `tiny_window()` is reached, so the property is defended twice |

The duplicated `uplo` gate is not pure redundancy: `native_tier_preferred` calls
`tiny_window` **without** the outer check, so it is what keeps a vendor-free build from
taking the tier for Upper on an unmeasured basis. That path is correct either way -- the
kernel has no uplo gate in `supports()` because Upper is the same recurrence on a
transposed tile -- so the cost of the restriction is an unclaimed win, not a wrong answer.

### Unmeasured Tiny windows: Upper and the double types

Two exclusions from [the tiny potrf window](#the-tiny-potrf-window) are **untested, not refuted**. Neither is a
correctness gate and neither needs new device code — both kernels are already instantiated and linked. What is
missing in each case is a grid. Recorded with the grid that would settle it, so the next pass measures instead of
re-deriving.

#### (a) `Uplo::Upper`, float and `complex<float>`

**The kernel is correct for Upper; this was re-verified, not assumed.** `potrf_tiny.cc` branches on `upper` in
exactly two places, the load and the store, and the recurrence between them is untouched:
`tiny_load_upper` reads `S(i,c) = conj(A(c,i))` (`tiny_device.hh:91-107`) and `tiny_store_upper` writes the result
back through the same transform (`:118-127`). That is why `supports()` carries no `uplo` gate for Tiny. Three
tests exercise both triangles on oracles independent of the kernel: `TinyResidualBothTriangles` sweeps **every**
order 1..cap x {Lower, Upper} x ld in {n, n+5} x batch in {1, per_wg+3} against a host multiply-back residual,
`TinyAgreesWithCtaWithinOnePanel` against the CTA kernel, and `TinyOtherTriangleIsNeitherReadNorWritten` against a
finite sentinel — all four scalar types.

So the exclusion is **purely the missing grid**: `p9_potrf_tiny.csv` and `p9_potrf_low.csv` were run `Uplo::Lower`
only, and an unmeasured triangle is not a window. The restriction is spelled twice on purpose (in `preferred()`
and again inside `tiny_window()`, which `native_tier_preferred` calls without the outer check), which is why the
planted "admit Upper" break stayed green. Cost of the restriction: an unclaimed win, never a wrong answer.

**The grid that would settle it.** The Lower grid re-run with `uplo = Upper` and nothing else changed: the same 14
orders (n in {1,2,3,4,6,8,9,12,16,17,20,24,28,32}), batch 65,536, one process per cell, arms interleaved in the
timed loop, 9 reps, ratio `t_vendor / t_tiny`, the `auto` arm read beside every cell as the before-picture, and
the same `rel_sd` 10% discard rule. The only per-cell difference from the Lower run is the load/store transform,
so what the grid has to establish is whether that transform costs enough to drop the *weakest* cells under the
gate — cfloat n = 17 clears it by 1.23x on Lower, which is the margin at risk. Flipping also means changing the
Upper expectation in `TinyRouteTableAndVocabulary` (`tests/potrf_tiny_cases.inc:576-581`); that expectation is a
deliberate guard on "unmeasured", not an oversight.

#### (b) `double` and `complex<double>`

`potrf_tiny.cc` instantiates both (`BATCHLAS_POTRF_TINY_INSTANTIATE(double)` and
`(std::complex<double>)`) and caps them at `PotrfTinyCap` = 32 and **16**; `supports()` admits them, because
`tiny_max_n` is filled from `potrf_tiny_max_n<T>()` (`src/backends/potrf_route.hh:59`) with no type test. They are
refused one line *before* the window is reached: `preferred()` opens with `if (!lpanel_types()) return false;`,
and `lpanel_types()` is `float || complex<float>`.

**Widening must introduce a separate `tiny_types()` predicate; it must NOT widen `lpanel_types()`.**
`lpanel_types()` has three call sites in `route_potrf.hh` — `preferred()`, `tiny_window()` and the `lpanel_holds`
term of `native_tier_preferred()` — so widening it would drag the `32 < n <= 256` window along with it and route
double and cdouble to CTA/LPanel on no grid at all: P3 measured float and cfloat only, and [open debts](#open-debts)
item 2 records complex running 0.311-0.509x vendor-free with the cause outside this driver. One predicate, two
windows, is exactly how that accident happens.

**The grid that would settle it.** n in {1,2,4,8,9,12,16,17,24,32} for `double` and {1,2,4,8,9,12,16} for
`complex<double>` (its instantiation cap is 16, so there is nothing above it to measure), `Uplo::Lower`, batch
65,536, same protocol as (a). Two things to read off it rather than assume: the padded-bucket effect that makes
cfloat n = 17 the weakest Lower cell should be *stronger* for the wider types, and the register demand is `rA[N]`
per lane, so occupancy at N = 32 is where a loss would appear first. `kTinyWorstProbedRegs` is a probe over the
instantiations that already ship, these two included, so no new launch gate is needed — only a number to quote.

### The posv window is not a no-op

**2026-09-14.** A review flagged `route_posv.hh`'s `tiny_window_max_n()` as vacuous: it
returns 32, which is the tier's own instantiation cap, so a ceiling alone excludes nothing
the tier can hold. Structurally that is true. **The decision it encodes is still correct,
and narrowing it was measured as a 2-5x REGRESSION.**

The trap is the baseline. `posv` has **no vendor arm** -- `resolve_posv_route` passes
`vendor_available = false` -- so "not preferred" does not mean cuSOLVER. It means the
**composed arm** (`potrf` + two `trsm`). Judged against a vendor-PINNED composition, cfloat
looks like it loses at half its orders:

| n | 9 | 12 | 13 | 17 | 20 | 24 | 26 |
|---|---|---|---|---|---|---|---|
| vs vendor-pinned composition (nrhs=1) | 0.79 | 0.96 | 1.20 | 0.78 | 0.91 | 1.17 | 1.28 |
| **vs the incumbent it actually falls back to** | **5.40** | **3.84** | **3.57** | **3.18** | **2.92** | **2.71** | **2.59** |

A three-band window cut from the first row was built, shipped to the tree, and measured
end to end: the excluded cells fell from ~0.79x to **0.146-0.55x**, because they landed on
the composed arm instead. The window was reverted. Every cell the tier holds beats the
arm it replaces by at least **1.93x** (cfloat n=26, nrhs=4).

**The real defect this exposes is in the composed arm, not in posv's window.** P2 already
filed it: the composition under `Auto` is 2-5x slower than the same composition with its
legs pinned to the vendor, at n <= 32 and large batch. Routing `potrf` to its register tier
(above) does not close that gap, so the remaining cost is on the `trsm` leg. Note the
shape: these cells run `nrhs * batch` = 65,536 to 262,144, and `docs/perf` already carries
a warning that trsm's windows above `q * batch` around 65,000 were measured on a racing
kernel. That is the thread to pull, and it is not a posv change.

**The lesson, stated for the next window.** Measure a flip against the arm it actually
replaces. For an op with a vendor arm that is the vendor; for `posv`, `gesv` and any other
composed op it is the composition under `Auto`. P2's own report said this in as many words
and this pass ignored it for one grid.

### The occupancy clamp on nb

`potrf_blocked_params` clamps `nb` by the **advertised** ceiling while `n <=
kPotrfOccupancyNbMaxOrder` (256) and by the **resident** one above it, and launches the
leaf at the same target. The threshold is measured, not assumed: `nb` is also the
trailing update's `k`, so halving it buys leaf occupancy and pays for it in the GEMM.

Native arm, large batch, against `benchmarks/results/factor_baseline_potrf_*.csv`
(ratio = after / before; below 1 is faster):

| type | n | batch | before | after | ratio |
|---|---|---|---|---|---|
| float | 128 | 2048 | 1.4533 | 1.0265 | **0.706** |
| float | 192 | 1024 | 1.4259 | 1.1478 | **0.805** |
| float | 256 | 1024 | 2.3111 | 2.1527 | **0.931** |
| float | 384 | 512 | 2.4585 | 2.4598 | 1.001 |
| float | 512 | 512 | 4.4631 | 4.4580 | 0.999 |
| cfloat | 128 | 2048 | 2.3254 | 1.5470 | **0.665** |
| cfloat | 256 | 1024 | 3.6338 | 3.4442 | **0.948** |
| cfloat | 512 | 256 | 3.7631 | 3.7670 | 1.001 |
| double | 128 | 2048 | 4.7605 | 3.7514 | **0.788** |
| double | 512 | 256 | 11.6444 | 11.6997 | 1.005 |
| cdouble | 512 | 256 | 48.9944 | 48.7266 | 0.995 |

Without the threshold — the occupancy ceiling applied at every order — the wins above are
unchanged but n >= 384 regresses: float 384 1.109x, float 512 1.209x, cfloat 512 1.382x,
double 512 1.097x, cdouble 512 1.050x. That is the negative result the threshold exists
to avoid; do not "simplify" it away.

### The one cell this cost

**float n = 96, batch 2048, native arm: 0.5312 -> 0.7752 ms, 1.46x SLOWER.** Order 96 is
above the new advertised ceiling of 77, so it leaves the single CTA launch for the blocked
driver (two panels, a trsm, a gemm, a fold and a fixup). cuSOLVER serves that cell in
0.2708 ms, so native loses to the vendor either way and `Auto` is unaffected; the loss is
visible only in a vendor-free build or under a forced native route. The blocked arm
overtakes CTA between 96 and 128 (after: 80 -> 0.686, 96 -> 0.775, 112 -> 0.884,
128 -> 1.027 ms, against a CTA baseline of 0.531 at 96 and 1.453 at 128). cfloat moves the
other way at the same order: 1.4272 -> 0.9577 ms, **0.671x**.

### The shared helper: budget, slice, walk and pack

**2026-09-11.** The rule above lives in one place, `src/util/resident_capacity.hh`, because
potrf, getrf and geqrf had each grown their own capacity walk and their own spelling of the
device budget. Everything in it is `constexpr` and free of `<sycl/sycl.hpp>`, so the route
table's shape builders under `src/backends/` — which must not pull SYCL in — can include it.
The prose that used to sit above each helper is here; the header keeps the invariant and a
pointer.

| helper | what it answers |
|---|---|
| `device_slm_budget(local_mem_bytes, reserve_bytes = 4096)` | THE spelling of the runtime budget: the device's `LOCAL_MEM_SIZE` less the reserve the SYCL runtime keeps for itself. A site that inlines its own disagrees with the launcher by exactly the reserve. |
| `kMinBlocksPerSm = 4` | design rule R1's occupancy target, and the value the three CTA tiers advertise with (geqrf overrides it to 2 — [qr.md](qr.md#why-geqrfs-target-is-2-and-not-4)). |
| `occupancy_budget(budget, min_blocks_per_sm)` | the slice of the budget ONE work-group may own. A work-group whose local memory fits in `budget / N` leaves room for N of them per SM. |
| `resident_max_n(bytes_fn, budget, min_blocks_per_sm, n_hi = 4096)` | the largest `n` whose per-matrix footprint fits that slice **and** for which every smaller `n` also fits. |
| `pack_matrices_per_wg(bytes, lanes, wg_budget, max_wg_size, target_wg_size = 128, max_pack = 4)` | how many matrices one work-group should hold when a single sub-group serves a matrix. |

The three potrf test hooks pack two 16-bit fields each, and this is the only place they are
spelled out: `potrf_cta_debug_launch` is **G (matrices per work-group) in the low 16 bits
and L (work-items per matrix) in the high 16**, 0 when the order does not fit;
`potrf_blocked_debug_params` is **nb (diagonal-block order) low, W (trailing-update width)
high**; `potrf_tiny_debug_launch` is **G (partitions per sub-group) low, S (sub-groups per
work-group) high**.

**Why `resident_max_n` is a walk with a `break`, not a closed form or a binary search.**
Each op pads a request landing in the 48 KB hole up past it, so `bytes(n)` is **not
monotone**: 47,200 B becomes 49,920 while 49,700 B stays as it is. `supports()` spells the
capacity as the contiguous `order <= cta_max_n`, so the ceiling must be the largest `n` at
which *every* order up to `n` launches. A `continue` there advertises a range with a hole
in it — reachable only for budgets inside **[49,664, 49,920)**, which is why the armed test
feeds it a synthetic non-monotone table rather than a device budget: no budget on this box
can reach the case that distinguishes the two.

**Why `bytes_per_matrix` must do its own products in `std::size_t` or `int64_t`.** `(m|1)*n`
overflows `int` at m ≈ **46,341**, which is reachable as a blocked panel height.

**Why `pack_matrices_per_wg` is not a free knob.** A one-sub-group work-group caps at the
hardware's **24 resident blocks per SM** — 24 of the 48 warps an SM holds, 50% occupancy
however little local memory it asks for. `G > 1` is correct ONLY where every barrier the
kernel executes is a SUB-GROUP barrier: under a work-group barrier the G matrices
synchronise with each other, which is a race by construction rather than a launch failure —
wrong answers on a suite that stays green. Each caller therefore selects a sub-group-scoped
body when it takes a `G > 1` result, and derives the two together
([lu.md](lu.md#packed-resident-leaves), [qr.md](qr.md#packed-resident-panels)). The result
is a power of two so the slot arithmetic (`matrix = wg_id*G + sg_id`) stays a shift, and a
single matrix that does not fit returns 1: the helper answers "how many", and 1 is the
honest answer when none fit.


### The contiguity rule and the synthetic table

The `break`-not-`continue` rule above is armed in `tests/resident_capacity_tests.cc`
against a synthetic footprint, because no budget this box reports lands in the
`[49,664, 49,920)` band where a `continue`-walk and a `break`-walk disagree.

That footprint is 128 B per unit of `n`, then the library's own 48 KB hole pad
(`kHoleLo = 47,104`, `kHoleHi = 49,664`, `kHolePadTo = 49,920`, byte-identical in
`potrf_cta.cc`, `getrf_cta.cc` and `geqrf_cta.cc`):

| n | raw request | after the pad |
|---|---|---|
| <= 368 | <= 47,104 | unchanged — 47,104 at n = 368 |
| 369..388 | 47,232..49,664 | all in the band, all raised to 49,920 |
| 389 | 49,792 | above the band, NOT padded |

So `bytes(389) < bytes(388)`: the footprint FALLS as `n` rises. That is the library's own
shape — "47,200 pads to 49,920 while 49,700 stays 49,700" — with a step small enough to
land a value inside `(kHoleHi, kHolePadTo)`.

At a budget of **49,850 B**: `n <= 368` fits (47,104), `369..388` are refused (49,920),
and `n = 389` fits again (49,792). The walk must answer **368**. A walk that skipped
misses instead of stopping would answer **389** and advertise twenty orders that cannot
launch. A budget of 49,920 admits the whole prefix and answers **390**; one exactly at the
band's floor, 47,104, answers **368**.

Arithmetic pinned alongside it. `occupancy_budget(97280, 1)` = 97,280 and
`occupancy_budget(97280, 4)` = 24,320; 0 and -3 are not divisors and return 97,280
unchanged. The walk really consumes it: over `bytes(n) = 1024n`, budget 97,280 answers
**95** at target 1 and **23** at target 4. `device_slm_budget` subtracts the 4,096 B
runtime reserve — 101,376 -> 97,280, 49,152 -> 45,056 — and 4,096 and 1,024 both floor at
**0** rather than underflowing.

### The G-packing helper and its three limits

`resident::pack_matrices_per_wg(bytes, lanes, slice, max_wg, target_width = 128,
max_pack = 4)` is bounded by local memory, the work-group width and the cap. The grid
pinned in `tests/resident_capacity_tests.cc` separates all three, at a 24,320 B slice:

| bytes | lanes | max_wg | target | cap | G | what binds |
|---|---|---|---|---|---|---|
| 4,096 | 32 | 1024 | 128 | 4 | 4 | width and cap agree here |
| 8,192 | 32 | 1024 | 128 | 4 | 2 | local memory |
| 20,000 | 32 | 1024 | 128 | 4 | 1 | local memory |
| 4,096 | 32 | 64 | 128 | 4 | 2 | device max work-group size |
| 4,096 | 32 | 32 | 128 | 4 | 1 | device max work-group size |
| 1,024 | 256 | 1024 | 128 | 4 | 1 | a matrix wider than one sub-group is never packed |
| 64 | 8 | 1024 | 128 | 4 | 4 | **the cap alone** — the width admits 16 |
| 64 | 8 | 1024 | 128 | 16 | 16 | cap raised, width admits 16 |
| 64 | 8 | 1024 | 32 | 16 | 4 | **the width alone**, with the cap raised |
| 64 | 8 | 64 | 128 | 16 | 8 | `max_wg`, with the cap raised |
| 6,144 | 8 | 1024 | 128 | 16 | 2 | local memory, below both |

The 8-lane rows are the load-bearing ones: at 32 lanes the cap of 4 and the 128-wide
target agree, so a helper that ignored `max_pack` entirely would answer the same on every
row above them. Degenerate inputs (`bytes = 0`, `lanes = 0`) answer **1** rather than
dividing by zero. `g >= 1` and "g is a power of two" are structural — the walk starts at 1
and only doubles — so the grid asserts the three bounds instead, and only for `g > 1`,
because `G == 1` is the honest answer even when one matrix does not fit.

### The shipped ceilings, pinned in both scales

Pinned budget-parameterised in `tests/resident_capacity_tests.cc` at the reference budget
of **97,280 B** (101,376 B of local memory less the 4,096 B reserve), so they hold on any
machine — and pinned in BOTH scales. The occupancy-scaled figure is what `supports()`
advertises; the unscaled one is what a blocked driver's panel can be held at. Conflating
them is the defect the pin exists to catch.

| type | potrf advertised / resident | getrf advertised / resident | geqrf advertised / resident (elements) |
|---|---|---|---|
| float | **77** / 155 | **77** / 155 | **11,776** / 24,320 |
| double | **54** / 109 | **54** / 109 | **5,888** / 12,160 |
| complex\<float\> | **54** / 109 | **54** / 109 | **5,888** / 12,160 |
| complex\<double\> | **38** / 77 | **38** / 77 | **2,944** / 6,080 |

**Where each number comes from, so it can be re-derived without a build.** Every potrf figure in this section and
in [what ships](#what-ships) is `resident::resident_max_n(n -> potrf_hole_padded(potrf_slm_per_matrix(n, NB, TS,
sizeof(D), sizeof(R))), 97280, min_blocks)` — `potrf_cta_max_n_for_slm<T>` verbatim (`potrf_cta.cc:136-149`), with
`(NB, TS)` from `PotrfCtaConst` (8/4 for float, double and cfloat; 8/2 for cdouble). `min_blocks = 4` is the
declared default (`potrf_native.hh:45-47`, `resident::kMinBlocksPerSm`) and gives the **advertised** row
77/54/54/38 at an `occupancy_budget` of 24,320 B; `min_blocks = 1` gives the **resident** row 155/109/109/77 at
the full 97,280 B. Re-checked against the shipped expressions on 2026-09-15: all eight ceilings, the
bytes-at-ceiling column (96,540 / 95,476 / 95,444 / 95,328) and the first-miss column (98,408 / 98,108 / 98,076 /
99,056) reproduce exactly — **no drift**. The same two numbers reach the blocked driver through
`potrf_blocked_params`, which picks `min_blocks` by order (4 at `n <= 256`, else 1) and clamps
`nb = min(want, ceiling)` (`potrf_blocked.cc:81-86`); that clamp is why no shipped `nb` equals a fit ceiling.

`geqrf`'s occupancy target is 2 rather than 4
([qr.md](qr.md#why-geqrfs-target-is-2-and-not-4)), and its figures are the ones the 48 KB hole clamp touches: half of 97,280 lands
INSIDE the band, so the admissible budget is **47,104** rather than 48,640. Both facts are
why these are pinned rather than re-derived.

Three further properties ride on the same numbers. The advertised ceiling must be
*strictly* inside the resident one, or the occupancy target is not being applied. Each fit
predicate must agree with its own ceiling on both sides, or `supports()` and the
entry-point gate can disagree: `getrf_cta_fits(a)` true and `(a + 1)` false, with
`getrf_leaf_fits(a + 1, a + 1)` still true because the residency predicate is strictly
wider. And the **resident** ceiling gets the same tight pair at its own boundary —
`getrf_leaf_fits(r, r)` true, `(r + 1, r + 1)` false. Probing residency one order above
the *advertised* ceiling is not that test: it passes with the whole occupancy factor as
slack and says nothing about where `leaf_fits` stops, which is the number the blocked
driver's leaf choice and the 48 KB hole clamp both turn on. `geqrf` bounds an AREA rather
than an order, so it gets the pair twice in the same 8-column panel shape: at
`area / 8` rows and at `area_r / 8` rows. Finally, a larger budget must admit more of
everything (97,280 -> 101,376 raises all three ceilings), because the scaling is a
division and not a table.


## The tiny tier

`Algorithm::Tiny`, `src/extensions/potrf_tiny.cc`, the P1 package of
`docs/design/small-n-factorization-plan.md`. One matrix per `SubGroupPartition<N>`,
N in {8, 16, 32}, lane r owning row r in a compile-time `D rA[N]`; right-looking
unblocked Cholesky; every cross-lane value an indexed sub-group shuffle. **Zero local
memory, zero barriers, zero static shared** — `ptxas` reports `used 0 barriers` and no
`smem` line for all 11 kernels, which is what keeps the 48 KB launch hole structurally
unreachable for this tier.

Geometry: work-group 64 (`kTinySubGroups = 2` sub-groups), G = 32/N partitions per
sub-group, so 8/4/2 matrices per work-group at N = 8/16/32. Matrices per work-group
comes from `resident::pack_matrices_per_wg`, with a nominal 1-byte footprint because the
tier owns no local memory; the packing is decided by the lane and work-group terms alone.

`potrf_tiny_debug_launch<T>(ctx, n)` is the only way to see that geometry from outside: it
packs **G (partitions per sub-group) in the low 16 bits and S (sub-groups per work-group)
in the high 16**, and answers 0 when the order is above the tier. The fixture asserts it
against the geometry INVARIANTS — `S*G*N == wg`, `wg % 32 == 0`, `S >= 2` — and never
against a pinned S: S is a tuning constant, and a test that pins it blocks the very A/B the
design calls for (`qr.md` records the same A/B for geqrf at
[the launch shape](qr.md#the-launch-shape-64-work-items-not-128)).

`potrf_tiny_buffer_size` is **not** zero. The kernel needs no algorithmic workspace, but an
empty or short caller `info` span means "not requested" and draws `batch` int32s of pool
scratch, exactly as the CTA tier does; every tier's sizing carries that same term.

`potrf_tiny_max_n<T>()` is a flat compile-time 32, and 16 for `complex<double>` — not a
budget walk. The tier allocates no local memory, so blocks per SM are set by registers
and the hardware's 24-blocks-per-SM cap alone, and there is nothing for
`resident_capacity.hh`'s non-monotone hole-padding walk to walk over.

`preferred()` is still false for every potrf tier, and `native_tier_preferred` answers
**false** for Tiny — the same answer getrf and geqrf give their own Tiny arms. The
vendor-free walk and a bare `native` pin therefore still take CTA at n <= 32, and `Auto`
in a vendor-present build is unaffected: this package changes **no route**. Both
predicates flip together in the change that MEASURES the window, and until then the
build report's own two negative results (cfloat n = 32 holds 12 resident matrices/SM
against potrf_cta's ~20-32; orders 17..31 pay their full padded bucket) stand
unanswered.

### Register probe — the shipped table

`BATCHLAS_BUILD_DIR=$PWD/build/presets/dev-tests scripts/register_probe.sh out.log ''
batchlas_extensions_cta`, sm_89, max of `<name>` and `<name>_with_offset`. Gate:
**stack frame == 0 AND spill == 0 AND regs x 64 <= 65536**, all three.

| type | N | frame | spill | regs | regs x wg (64) |
|---|---|---|---|---|---|
| float | 8 | 0 | 0 | 54 | 3,456 |
| float | 16 | 0 | 0 | 64 | 4,096 |
| float | 32 | 0 | 0 | 78 | 4,992 |
| double | 8 | 0 | 0 | 70 | 4,480 |
| double | 16 | 0 | 0 | 82 | 5,248 |
| double | 32 | 0 | 0 | 104 | 6,656 |
| cfloat | 8 | 0 | 0 | 64 | 4,096 |
| cfloat | 16 | 0 | 0 | 112 | 7,168 |
| cfloat | 32 | 0 | 0 | 166 | 10,624 |
| cdouble | 8 | 0 | 0 | 84 | 5,376 |
| cdouble | 16 | 0 | 0 | 128 | 8,192 |
| cdouble | 32 | — not instantiated; orders 17..32 of this type stay on CTA | | | |

Re-probed at P1 sign-off and the table above reproduced cell for cell, on a 152.2 s link
of 974 entry functions, **0 of which carry any spill**. All 22 potrf entries (11 kernels,
each also `_with_offset`) report `used 0 barriers` and emit **no `smem` line at all** —
that, and not a padding constant, is what keeps the 48 KB launch hole unreachable here.
Worst cell is `complex<float>` N = 32 at 166 registers: 166 x 64 = 10,624 against the
65,536 per-block file, 6.2x of slack.

The launch gate has better than 6x of slack at every cell, so it can never fire; the gate
that actually bites is the frame column. The TU encodes the gate as
`kTinyWorstProbedRegs = 176` against the worst probed cell of 166 (`complex<float>`
N = 32), i.e. a 10-register margin, and `static_assert`s `kTinyWgSize *
kTinyWorstProbedRegs <= 65536`, so widening the work-group fails to compile rather than
aborting an enqueue. Two design predictions were wrong in the same
direction: the plan's "float N=32 ~48 registers (fits 100% occupancy)" and the judged
design's 56 are both under the probed 78, and the complex cells are far under — cfloat
N=32 probed at 166 against a predicted 88.

Occupancy therefore comes out below every prediction. At wg = 64, blocks/SM is
`floor(65536 / (regs_rounded_to_8 * 64))` capped at 24, and no cell reaches the cap:

| cell | regs | blocks/SM | warps/SM | occupancy | resident matrices/SM |
|---|---|---|---|---|---|
| float N=8 | 54 -> 56 | 18 | 36 | 75% | 144 |
| float N=16 | 64 | 16 | 32 | 67% | 64 |
| float N=32 | 78 -> 80 | 12 | 24 | 50% | 24 |
| cfloat N=32 | 166 -> 168 | 6 | 12 | 25% | 12 |

Against `potrf_cta`'s ~20 resident matrices/SM at n = 32 float this is still a gain at
n <= 16 and roughly a wash at n = 32 float; **cfloat n = 32 is worse on residency than
the tier it would replace**, which is the first place to look when the grid is run.

### Two negative results, both found by the register probe and not by a timer

**1. The batch-uniform early exit for the n = 17..31 band relocates `rA[]`.** The design
of record calls for `if (j >= n) break;` on the column loop and `if (k >= n) break;` on
the update loop, both batch-uniform, cutting issued warp-FMAs at n = 17 inside N = 32
from 496 to 136. It was implemented first and **fails the register gate**: the runtime
break defeats the full unroll, `rA[j]` becomes a dynamic index, and ptxas relocates the
whole array to local memory. Probed with the breaks in place:

| type | N | frame | spill | regs |
|---|---|---|---|---|
| float | 8 | 0 | 0 | 39 |
| float | 16 | **64** (= 16 x 4) | 0 | 40 |
| float | 32 | **128** (= 32 x 4) | 0 | 40 |
| double | 16 | **128** | 0 | 40 |
| double | 32 | **256** | 0 | 40 |
| cfloat | 16 | **128** | 0 | 40 |
| cfloat | 32 | **256** | 0 | 40 |
| cdouble | 8 | **160** | 0 | 40 |
| cdouble | 16 | **288** | 0 | 40 |

A frame of exactly `N * sizeof(T)` bytes with a clean spill column and a suspiciously
flat 40-register count in every cell is the signature this page's `#register-gate`
section and `docs/perf/trsm.md:80` both name. The break was removed and replaced by a
predicate (`col_live`, `k < n`), which restores frame 0 everywhere. **The 17..31 band
therefore issues the full N(N-1)/2 shuffles and FMAs of its padded bucket, and the plan's
only stated lever for that band is unavailable at this unroll setting.** It is not worth
re-attempting without a way to keep the array promotable.

**2. `complex<double>` needed a component-wise select before it would stay in
registers.** With the breaks removed, 9 of 11 cells came back at frame 0 but cdouble
N=8 and N=16 still carried 160 and 288 bytes. The cause is the failure path being value
substitution: `rA[k] = cond ? upd : rA[k]` on a 16-byte aggregate is not a `select` in
IR — LLVM emits a branch or a memcpy, SROA then declines to promote the array. Replacing
every such ternary with `tiny_native::tiny_select`, which selects component by component,
took both cells to frame 0 (and raised their register counts from 40/40 to 84/128, which
is the array arriving in registers). `Cx<float>` at 8 bytes was never affected.

### The CTA comparison is elementwise, not bit-exact

At n <= 8 the CTA kernel runs exactly one NB = 8 panel, so its diagonal-block body is
line-for-line this tier's: same `!(akk > 0)`, same `sqrt` then `1/dkk`, same
own-times-`conj`(other) order, same sticky failure, same real-diagonal load. The first
version of `TinyAgreesWithCtaWithinOnePanel` therefore asked for bit-identity, and it
does not hold: **float n = 2, `Uplo::Upper`, one item differed in the last ULP**, because
the compiler contracts `a - b*conj(c)` into an FMA on one side and not the other — CTA's
`c` is a local-memory load and this tier's is a shuffle, and the contraction decision
follows the surrounding code. Editing an unrelated line in the kernel flipped it. The
test now asserts an elementwise `8*eps` relative bound, which is still far tighter than
the `4*n*eps` the residual is allowed.

The exact-bit oracles that ARE stable are tiny-vs-tiny, where both sides are the same
instruction stream: `TinyPaddingIsInert` (a NaN pad must give bit-identical answers to a
zero pad) and `TinyPackedBatchMatchesSolo` (an item in a packed batch must equal the same
item run alone).

### What the geometry test can and cannot check

`TinyLaunchGeometryIsReported` asserts the launch geometry against INVARIANTS and never
against a pinned `S`: `S` is a tuning constant the design calls for A/B-ing, and a test
that pins it turns the constant into a change-detector.

The hook and the launcher compute the geometry from ONE helper
(`potrf_tiny_matrices_per_wg`), so there is no second copy to cross-check the returned
numbers against, and any assertion re-derived from the hook's own answer is an identity
that no break can move. Two of those were written and then removed: `32 % N == 0` over an
`N` the test itself picked, and `wg % 32 == 0` over a `wg` the test rebuilt as `S*G*N`.

What remains are the two statements that CAN move — the work-group against the device's
own `MAX_WORK_GROUP_SIZE`, and R3's two-sub-group floor — plus one real cross-check: the
bucket the hook chose, read back OUT of its answer as `32/G`, against the 8/16/32 ladder
the tier documents. That bucket must hold `n`, must be one of the three instantiated
widths, and must be the TIGHTEST such: a hook that padded every `n` up to 32 would launch
a legal geometry and waste three quarters of every partition.

The geometry is also pinned behaviourally: every partial-work-group case in
`tests/potrf_tiny_cases.inc` derives its batch from `per_wg()`, so a wrong
matrices-per-work-group makes those batches stop straddling a work-group boundary.

### Armed breaks (R9)

Every guard below was planted, observed red, and restored; `src/extensions/potrf_tiny.cc`
was verified byte-identical to its pre-break state afterwards (md5). Breaks 1-3 were
planted in the kernel body, break 4 in the failure-path selects. Re-run recipe: edit,
`cmake --build build/presets/dev-tests --parallel 4`, then the named test.

| # | break planted | guard expected to fire | observed |
|---|---|---|---|
| 1 | the rank-1 update broadcasts the **pre-scale** `rA[j]`, so the update omits one `1/dkk` factor (the plan's break (c), "publish before the scale") | `TinyResidualBothTriangles`, first at n = 2 | RED at **n = 2**, both `Uplo`, both `ld`; residual 4.38e-06 against a 9.54e-07 tolerance. n = 1 passed, as predicted — with one column there is no update to corrupt |
| 2 | `tiny_partition_id` loses its `sg_id *` term, so the two sub-groups of a work-group factorise the same matrices | `TinyPackedBatchMatchesSolo` | RED — `info` for the untouched items came back as the fixture's `-7` poison, i.e. half the batch was never factorised. Fires only at G-packing, which is what the anti-vacuity assertion in that test exists to guarantee |
| 3 | the `info` write drops its `alive &&` guard, so the **last** failing column wins instead of the first | `TinyInfoIsExactAndFirstFailureWins`, two-failure block | RED — failures planted at columns 2 and 7 gave `info = 8` where 3 is required; 7 and 12 gave 13 where 8 is required; 11 and 16 gave 17 where 12 is required |
| 4 | both failure-path `tiny_native::tiny_select` calls replaced by a plain `?:` on the 16-byte aggregate | **no test** — the register probe's frame column | **All 49 tiny cases still PASSED.** The probe caught it: `complex<double>` N = 8 went to a **144-byte frame** and N = 16 to a **272-byte frame**, both with spill still 0, register counts *falling* 84 -> 76 and 128 -> 72 as the array left the register file. `float`, `double` and `complex<float>` stayed at frame 0 — `Cx<float>` is 8 bytes and forms a select either way |

Break 4 is the one that matters for maintenance. The frame is `16 * (N + 1)` bytes here
against the `16 * (N + 2)` recorded in the negative result above, because only the two
selects in the column body were reverted and `tiny_pad_identity`'s stayed; same defect,
same signature. It confirms the gate this page states: **frame 0 AND spill 0**, never
spill alone, and never a green test suite.

### Device link

The attributable figure for THIS tier is `ptxas` compile time from the probe log:
potrf_tiny's 22 entry functions (11 kernels, each also as `_with_offset`) cost
**12.3 s of 174.8 s**, or 7.0% of the library's ptxas total — inside R7's 15% budget.

The 151 s library link this section used to quote is SUPERSEDED and was never
attributable: `geqrf_tiny.cc` landed in the same library concurrently and
`getrf_tiny.cc` was in flight. The re-measure it asked for has been done with all three
tiers present, and the aggregate — which does exceed R7's budget — lives in exactly one
place with its written justification:
`docs/perf/lu.md#r7-the-device-link-all-three-tiny-tiers-landed`. Do not re-derive it
here.

### Not measured yet

**Superseded on 2026-09-14 — the tier IS measured and routed; see
[the tiny potrf window](#the-tiny-potrf-window).** Kept because its predictions are worth
scoring against the grid that arrived: the win at float/cfloat `n <= 16` held (3.02-10.64x and
1.57-7.43x), but the two predicted **losses did not materialise** — cfloat n = 32 measured
1.30x, and across the 17..31 strip (grid cells 17, 20, 24 and 28) **cfloat** measured
1.23-1.32x while **float** measured 2.06-2.93x. Both ranges are type-qualified on purpose:
the 2.93 is FLOAT at n = 28, and no cfloat cell in that strip exceeds 1.32. Weakest of their
rows, but still ahead of cuSOLVER at every cell.
The bucket-fill reasoning was right about which cells would be weakest and wrong about the
sign. Nothing below was re-run; read it as the pre-grid expectation.

No timing at all. The tier is correct and register-resident; the grid
(n in {4,8,9,16,17,24,32} x batch x 4 types x both Uplo, ld = n and ld = n+5 as separate
rows, `resolved_route` checked on every row) and the `preferred()` window are the next
change. From the residency table above, the shapes to expect a win at are float and
cfloat at n <= 16 (144 and 64 resident matrices/SM against CTA's ~20-32), and the ones
to expect a loss at are cfloat n = 32 (12 resident matrices/SM, worse than the tier it
replaces) and n = 17..31 of any type, where the padded bucket issues roughly (32/n)^2
times the useful work and the plan's early-exit lever is unavailable. The kill criterion
is one line: drop a cell from the `preferred()` window and it falls straight through to
CTA with no code change.

### The shared tiny-tier invariants

`src/extensions/tiny_device.hh` is the one header all three tiny tiers
(`potrf_tiny.cc`, `getrf_tiny.cc`, `geqrf_tiny.cc`) share: the geometry constants, the
identity-padded load/store helpers, the partition broadcast and select, and the LU
argmax. Three invariants govern it, and every one of them breaks **silently** — a wrong
answer or a lost register residency, never a compiler diagnostic. The header carries a
one-line statement of each and points here.

**1. `rA[]` must never be dynamically indexed and must never become a by-reference
parameter.** `ptxas` then relocates the whole array to local memory with no diagnostic:
a non-zero stack frame, **ZERO spill** and green tests. Helpers in this header therefore
take `rA` as a plain pointer to a caller-declared array and touch it only under
`#pragma unroll` with compile-time bounds, and every caller declares the array at the top
level of the kernel lambda. The probe rows for both halves of this are
[Register gate](#register-gate), [Two negative
results](#two-negative-results-both-found-by-the-register-probe-and-not-by-a-timer) and
`lu.md`'s [register probe and the unroll that decides
it](lu.md#the-register-probe-and-the-unroll-that-decides-it).

Its sharpest corner is the **component-wise select**. `tiny_select` selects a complex
scalar component by component, and it is not cosmetic: a plain `?:` over a 16-byte
aggregate is not a `select` in IR. LLVM emits a branch or a `memcpy`, SROA then declines
to promote the array, and `ptxas` relocates `rA[]` to local memory — non-zero frame, ZERO
spill, green tests. A tiny kernel's failure path is value substitution, so every write
into `rA[]` under a failure predicate is a select and never control flow. Break 4 of
[Armed breaks (R9)](#armed-breaks-r9) is exactly that break, planted and observed.

**2. `N` must divide the sub-group.** `SubGroupPartition<P>` bases a chunk at
`(lane / P) * P`, so a `P` that does not divide 32 addresses lanes past its end. That is
the whole reason the ladder is `{8, 16, 32}`, and the reason **24 is not on it** — even
though 24 would be the useful bucket for the 17..31 band this page reports as the tier's
worst.

**3. Every lane of a partition must reach every partition collective.** The in-tree tail
idiom `if (prob_id >= nb) return;` (`steqr_cta.cc:90`) is **not** copied into any tiny
kernel: with more than one partition per sub-group it retires part of a sub-group while
the survivors still shuffle, and `sg_compat.hh`'s non-NVPTX branch shuffles across all 32
lanes. Dead partitions instead load the identity from registers and suppress their store.
On sm_89 this invariant has **no numerical guard** — see `qr.md`'s [break
(d)](qr.md#break-sweeps-the-tiny-tier), planted at the exact configuration the rule exists
for and green — so it is guarded by source text instead.

Five further contracts the helpers carry, stated once here rather than at each use:

* **A lane guard belongs INSIDE a collective, never around it.** `tiny_bcast` requires a
  partition-uniform `src` and every lane of the partition to reach it. A complex value is
  shuffled as **two reals**, because `permute`/`select` reject an aggregate and `Cx<R>` is
  one (`gesvdj_cta.cc:94`).
* **The identity pad is what removes the lane guards.** A row that is not live, and every
  row of a dead partition, carries the identity — off-diagonals exactly 0, pivot exactly
  1 — so it adds exactly 0 to every live row and can never raise `info`. For a lane, the
  live columns of a fixed column `c` are the contiguous range `[c, n-1]`, so each column
  is one coalesced segment.
* **`c <= lane` on the triangular load and store is a contract, not a bound.** It is the
  `OtherTriangleIsNeitherReadNorWritten` property `ortho.cc` depends on: the guard on the
  load means the other triangle is not READ, the same guard on the store means it is not
  WRITTEN. Spelling either `c < N` writes past the triangle and, at a padded `ld`, into
  the caller's pad. `real_diag` forces the diagonal to `(real(A(c,c)), 0)`, as LAPACK and
  cuSOLVER do.
* **`Uplo::Upper` is a LOAD TRANSFORM, not a second algorithm.** `A = U^H U` is the same
  recurrence on `S(i, c) = conj(A(c, i))`. Spelling Upper as "lane r owns a row of U"
  would need `rA[lane]` — a dynamic index, and therefore invariant 1 again. `tiny_store_full`
  takes a `row` rather than a `lane` for the mirror-image reason: LU relabels rows instead
  of moving them, so the row a lane stores to is not the row it loaded from.
* **`tiny_partition_id`'s `sg_id *` term is load-bearing** (`steqr_cta.cc:82-86`).
  `part.get_group_linear_id()` is the chunk index within ONE sub-group and repeats across
  the sub-groups of a work-group; without the term, the two sub-groups of a work-group
  factorise the same matrices and half the batch is never touched. The break is armed in
  all three tiers: [break 2 here](#armed-breaks-r9), `lu.md`'s break (d), `qr.md`'s
  break (e).

`kTinySubGroups = 2` is a tuning constant, not a contract: the launch-geometry test
asserts the invariants above and never this value. R3 wants it at 2 or more so the
24-blocks-per-SM cap does not bind.

`tiny_load_pad_identity` — the LU tier's load — is spelled **per element** and not per
row deliberately: it is the one spelling that cannot take the address of the caller's
register array. `tiny_load_full` hands `rA` to a callee as `D*`, which is safe only while
every such callee is inlined and the pointer never escapes, a property no signature
enforces, and the LU body's array sits in the tier's hottest loop. Rows and columns beyond
`n` carry the identity, so `[[A, 0], [0, I]]` factorises to `[[LU(A), 0], [0, I]]`; the
`live` term short-circuits, so a work-item belonging to no matrix issues no global read.

---

## The LPanel tier

WP6/P3. `src/extensions/potrf_lpanel.cc` + `potrf_lpanel_device.hh`. A **left-looking,
thread-per-row panel** kernel: one work-group holds **one `n x NB` panel** in local memory
rather than the whole `n x n` matrix, factors it in place, stores it, and re-reads the
already-factored columns to its left from global memory on the next panel. `NB = 8` for
every type; `NB = 16` is a float-only A/B arm, selected by `nb_hint`.

**It is measured and it now carries potrf's first routed window**, `32 < n <= 256` for
`float` and `complex<float>`, `Uplo::Lower` — see
[the measured LPanel window](#the-measured-lpanel-window). Outside that window `Auto` still
takes the vendor, and `double`/`complex<double>` are untouched. `Algorithm::LPanel` is
enumerator **14**, appended.

`Uplo::Upper` is **refused**, in `supports()` and again in `potrf_lpanel_dispatch`. The
left-looking update reads the already-factored *lower* triangle, so Upper needs the
transformed-tile trick `potrf_cta.cc` uses. No in-tree caller asks: `ortho.cc:93` and
`linalg::cholesky`'s default both pass `Lower`, and the blocked driver already refuses Upper.

### Footprint and the two caps

```
slm_per_matrix(n, NB, sizeof(T)) = (n*NB + NB*NB) * sizeof(T) + 256
```

`sA` is the `n x NB` panel at `ld = n` — no odd padding, because lane `row` reads
`sA[row + i*n]`, already stride 1 across lanes — `sB` the `NB x NB` broadcast block, and the
256 over-covers `*fail` plus alignment slack. The band from
[the 48 KB launch hole](#the-48-kb-launch-hole) is applied to the **total** request, and `G`
is stepped back down if the pad pushes it over.

The ceiling folds **two** independent caps: the local-memory slice, and
`MAX_WORK_GROUP_SIZE`. The second is not slack — `potrf_lpanel_body` requires `L >= n`
because lane `row` carries `rp/rS/rA` across the whole `k` loop, so an order above the
work-group size would be a silent race, not a launch failure.

These are **arithmetic from the formula above**, not timings; the test
`CeilingIsCappedByBothMemoryAndWorkGroupSize` pins them against the budget-parameterised
query at the reference budget 97,280 B with `max_wg = 1024`:

| | float | double | complex\<float\> | complex\<double\> |
|---|---|---|---|---|
| LPanel `NB` | 8 (16 as an A/B) | 8 | 8 | 8 |
| LPanel advertised (4 blocks/SM) | **744** | **368** | **368** | **180** |
| LPanel resident (`min_blocks_per_sm = 1`) | 1024 (wg-capped) | 1024 (wg-capped) | 1024 (wg-capped) | 750 |
| CTA advertised, for comparison | 77 | 54 | 54 | 38 |
| LPanel `NB = 16` advertised (float only) | 360 | — | — | — |

Every advertised figure above lands on exactly 24,320 B, which is
`occupancy_budget(97280, 4)`. The point of the tier is the second row against the fourth: at
the default occupancy target the LPanel arm advertises **the whole 32 < n <= 512 band for
float in a single launch**, where CTA advertises 77 — so the shipped native arm at
`78 <= n <= 256` float is the blocked driver at `nb = 64`, and an 80x80 matrix is factored as
two panels, two leaves, two fixups, a trsm, a gemm and a fold.

### Why LPanel may pack on work-group barriers

`potrf_cta.cc` may only pack `G > 1` matrices into a work-group at `L == 32`, where its
phase barriers are **sub-group** barriers; a work-group barrier there would synchronise the
`G` matrices with each other, which is a race by construction.

LPanel is the opposite case and packs on **work-group** barriers deliberately. Every barrier
in `potrf_lpanel_body` is `group_barrier(it.get_group())`, sits at the top level outside the
`if` that guards its phase, and failure is a predicated skip rather than a `break`. The
barrier schedule is therefore a function of `(n, NB)` alone — and the packed matrices share
both — so the extra matrices are merely synchronised with each other, never raced. Two
consequences a maintainer must preserve:

* **No early return, and no shortened `n`.** A dead slot of the tail work-group runs the
  *whole* body against a live matrix with its stores suppressed (the `store` flag). Giving
  it `n = 0` instead would skip every barrier and hang the group.
* **B6 is not optional.** The store at the end of a panel writes global columns that the
  *next* panel's left-looking staging reads, from different work-items.

### LPanel register gate

The per-lane register arrays are `rp[NB] + rS[NB] + rA[NB]` = `3*NB` scalars before
addressing: 24 for a real type at `NB = 8`, 48 registers for `complex<float>`, 96 for
`complex<double>`. Against the plan's own occupancy line (100% occupancy needs <= 42
registers/thread) that caps cfloat at roughly two thirds and cdouble at roughly one third of
the occupancy float gets — so **the cfloat target in the P3 plan text is not supported by
P3's own arithmetic**. Independently, [open debts](#open-debts) item 2 records that complex
is 0.311-0.509x vendor-free with the cause *outside* this driver (the complex trailing GEMM,
P6's package). The honest gate for this tier is **float first**; cfloat is measured and
reported, not promised. No register probe has been run: the arrays above are a source count,
not a `ptxas` count.

### What P3's plan text got wrong

The section at `docs/design/small-n-factorization-plan.md` around line 600 predates P1 and
P7 and is stale in five places. Recorded so nobody re-derives them:

1. **`Algorithm::LPanel` and `Algorithm::Fused` were never added** (D1 claims they ship).
   Only `Tiny` was. This change appends `LPanel` as enumerator 14; a 15th is free.
2. **The R8b trap is armed and the plan does not mention the hook.** `native_tier_preferred`
   ends `default: return true`, and `tests/CMakeLists.txt:359` runs `potrf_tests` under
   `BATCHLAS_POTRF_ROUTE=native`, so an omitted `LPanel` arm would silently re-point that
   whole suite. The arm is spelled out, and `RouteWindowIsExactlyTheMeasuredOne` counts the arms
   that answer true and asserts the count is exactly 1.
3. **The stated loss mechanism is a route that no longer ships.** The 8.3%
   `sm__warps_active` figure was measured on the CTA arm at n = 96..155; since P7, float CTA
   is advertised only to 77, so `Auto`, the vendor-free walk and a bare `native` pin cannot
   reach the CTA kernel there at all. The right statement is the last paragraph of
   [Footprint and the two caps](#footprint-and-the-two-caps).
4. **"CTA retained for one release then deleted" is not available.** The CTA body is the
   Tiny tier's correctness oracle (`TinyAgreesWithCtaWithinOnePanel`) and the `n <= 77`
   winner. The deletion is struck; CTA stays in `kPotrfOrder` ahead of LPanel.
5. **Open question 4 (Upper) is answerable by grep** and is answered above: refuse it.

### The measured LPanel window

**2026-09-14, integration.** `benchmarks/factor_bench` one process per cell, arms
INTERLEAVED in the timed loop (`--arms=vendor,lpanel,cta,blocked`), 7 reps, medians,
`gpu_guard.sh 1`, host residual on items 0 and batch-1 of every timed row. Raw CSVs:
`benchmarks/results/p3_lpanel_ab.csv` (the grid),
`benchmarks/results/p3_lpanel_edges.csv` (the brackets),
`benchmarks/results/p3_auto_after_flip.csv` (the end-to-end check after the flip).
Ratios are `vendor_ms / arm_ms` — **> 1 means the arm wins** — and every one is a TIME ratio.

| | float | | | complex\<float\> | | |
|---|---|---|---|---|---|---|
| n / batch | lpanel | cta | blocked | lpanel | cta | blocked |
| 33 / 8k-16k | 1.698 / 1.785 | **2.225 / 1.834** | 0.251 / 0.261 | **1.202 / 1.521** | 1.052 / 1.307 | 0.312 / 0.377 |
| 36 / 8k-16k | 1.679 / **1.853** | **1.853** / 1.533 | — | **1.255 / 1.519** | 0.858 / 1.006 | — |
| 40 / 8k-16k | **1.669 / 1.879** | 1.398 / 1.200 | — | **1.290 / 1.537** | 0.754 / 0.887 | — |
| 44 / 8k-16k | **1.562 / 1.854** | 0.917 / 1.022 | — | **1.259 / 1.494** | 0.679 / 0.781 | — |
| 48 / 8k-16k | **1.571 / 1.834** | 0.948 / 1.080 | 0.223 / 0.250 | **1.238 / 1.498** | 0.685 / 0.796 | 0.367 / 0.422 |
| 64 / 8k-16k | **1.888 / 2.226** | 0.833 / 0.944 | 0.820 / 0.933 | **1.604 / 1.866** | (does not fit) | 0.537 / 0.585 |
| 65 / 4k-8k | **1.542 / 1.758** | 1.081 / 1.208 | 0.412 / 0.426 | **1.292 / 1.611** | (does not fit) | 0.450 / 0.508 |
| 96 / 4k-8k | **1.603 / 2.080** | — | 0.383 / 0.473 | **1.496 / 1.850** | — | 0.529 / 0.606 |
| 128 / 4k-8k | **1.568 / 1.920** | — | 0.527 / 0.614 | **1.651 / 1.857** | — | 0.702 / 0.766 |
| 129 / 2k-4k | **1.278 / 1.676** | — | 0.538 / 0.668 | **1.188 / 1.543** | — | 0.655 / 0.694 |
| 192 / 2k-4k | **1.514 / 2.026** | — | 0.631 / 0.783 | **1.359 / 1.579** | — | 0.702 / 0.744 |
| 256 / 2k-4k | **1.617 / 2.121** | — | 0.750 / 0.935 | **1.258 / 1.469** | — | 0.863 / 0.890 |
| 384 / 0.5k-1k | *1.049* / 1.241 | — | 0.838 / 0.929 | (does not fit) | — | 0.797 / 0.919 |
| 512 / 0.5k-1k | *1.079* / 1.322 | — | 1.040 / 1.177 | (does not fit) | — | 0.948 / 1.079 |

**The window that ships: `32 < n <= 256`, `Uplo::Lower`, `float` and `complex<float>`.**
`preferred()` answers for exactly one native tier there (`best_native_tier`, R8b), and the
tier hook splits it: **CTA to n = 35, LPanel above** for float (the only two cells where CTA
is ahead are n = 33 at both batches and n = 36 at the *smaller* batch), **LPanel from 33
up** for cfloat, where CTA loses to the vendor at every order above 33.

**Bracketing non-winners, both edges.**
* Upper, float: **n = 384 at batch 512 is 1.049**, below the 1.11 gate, and n = 512 at batch
  512 is 1.079 — also below. Both clear it at the doubled batch (1.241, 1.322), which is
  exactly the R8a situation, so 257..512 is left with the vendor rather than gated on a
  reading that only holds at one end of the ladder.
* Upper, cfloat: the tier **does not reach** n = 384 — `slm_per_matrix(384, 8, 8) = 25,344 B`
  against an `occupancy_budget(97280, 4)` of 24,320 — so `supports()` is false and the pin
  silently measures the vendor (see the named cells below). The edge is a FIT, not a timing.
* Lower: n = 32 is not a timing edge but a TIER edge. The Tiny tier (P1) is the measured
  winner there (float 2.187 / 2.591 against LPanel's 2.053 / 2.314 at batch 16k / 32k), and
  `n <= 32` stays P1's decision, not P3's. The window's floor is
  `order() > 32` for that reason, and not because native loses below it. (This paragraph as
  first written added "and Tiny is itself unrouted" — true when the grid was read, refuted
  the same day: P1's window ships and `tiny_window()` now owns `n <= 32`.)

**The kill/confirm probe P3's plan asked for, answered.** "Does the blocked driver lose at
n <= 256 because of its leaf?" — **yes, and by a lot.** `blocked` is 0.22-0.94x of the
vendor at every order from 33 to 256 in both types, and only reaches parity at n = 512
(1.04-1.18x float). The premise holds, and the LPanel arm beats `blocked` at every measured
order including 384 and 512.

**Saturation: not reached, and the direction is up.** Every cell was read at two batches
(the ladder value and its double) and every ratio RISES with batch, which is R8a's recorded
"potrf's ratio has no fixed point in batch": the native arm is linear where cuSOLVER is
superlinear. The window is therefore gated on the SMALLER of the two batches in every cell
(minimum in-window ratio: float 1.278 at n = 129 batch 2048, cfloat 1.188 at the same cell),
so it does not depend on the unsaturated tail.

**Cells DISCARDED and named.** No cell was dropped for variance — the worst relative sd in
the whole grid is 0.062 (cfloat n = 24, batch 16384) and the median is under 0.003. Eight
cells are reported but **do not measure the tier they name**: the cfloat `lpanel` arm at
n = 384 and n = 512, and the cfloat `cta` arm at n = 64 and n = 65 (the CTA ceiling for
cfloat is 54), at both batches each. An unsupported pin falls through `supports()` into `automatic()`, so those
`lpanel` rows are the VENDOR — visible both in the time (7.9958 vs 7.9839 ms) and in the
residual column, which matches the vendor's 1.805e-07 rather than the native 9.730e-08.
They are listed as "(does not fit)" above. A ratio of exactly 1.000 against the vendor
is the signature to look for.

**End-to-end after the flip**, `--arms=vendor,auto` through the facade, same guard, 7 reps:
float n = 33 **1.87x**, n = 64 **2.23x**, n = 128 **1.92x**, n = 256 **2.13x**; cfloat n = 64
**1.86x**, n = 256 **1.48x**; and the three cells that must NOT move — float n = 24, float
n = 384, double n = 128 — came back within 0.1% of the vendor arm, which is the window's
floor, ceiling and type gate observed rather than asserted. The n = 33 auto time (0.2642 ms)
matches the `cta` arm (0.2668) and the n = 64 auto time (0.5251) matches `lpanel` (0.5286),
so the tier hook picks the tier the grid chose.

**What could NOT be established.**
* **`NB = 16`, the plan's named float A/B, was not measured.** It is instantiated and
  reachable through `nb_hint`, but `factor_bench` has no flag that reaches the parameter and
  the route carries no `nb` — so the A/B needs either a harness flag or a `PotrfLpanelConst`
  change. `NB = 8` is what ships.
* **`double` and `complex<double>`: no grid.** Both are excluded from the window AND from the
  tier hook, so their routes are bit-identical to pre-P3. R10 does not gate on them, but the
  tier holds them (ceilings 368 and 180) and may well win; that is a measurement, not a
  guess, and it has not been taken.
* **No `ncu`, no register probe, no nsys attribution.** The `sm__warps_active >= 40%` gate in
  P3's plan is unmeasured. The occupancy argument is arithmetic (`slm_per_matrix`), the win
  is a timing, and nothing here says the win comes from occupancy rather than from the
  single-launch schedule.
* **The blocked driver's leaf is still CTA**, and `PotrfBlockedConst` was not re-tuned. The
  grid says the leaf swap is worth trying (LPanel beats `blocked` everywhere measured), but
  that changes numbers on a route that ships, so it wants its own A/B.
* **`lpin` vs `lpout`** (per-panel relaunch with a shrinking work-group) was never built.

### The pivot-cell WAR race

**WP6/P3 integration, 2026-09-14. The kernel as first written returned WRONG ANSWERS at
large batch, and the whole 11-test suite was green.** Found by adding one guard, diagnosed
to a single missing barrier, fixed in one line.

The symptom. At `batch >= 1024` (float `n = 128`) `factor_bench --arms=lpanel` reported
residuals of 1e-04 against a bit-stable 1.3e-07 at `batch <= 512`, and the value *changed
from run to run on identical input* — `fill_spd` is deterministic and gives every item the
same matrix, so a varying residual is proof of nondeterminism, not of a hard bug. The CTA
and Blocked arms were bit-stable at the same shapes, which is what ruled the harness out.

The mechanism, and it is one line. In the panel factorisation every lane READS the pivot
`sA[(j+i) + i*slda]` at the top of column `i`, and the lane that owns row `j+i` then
OVERWRITES that same cell with `sqrt(d)` — with **no barrier between the read and the
write**. It is a write-after-read hazard across warps: a warp that runs ahead publishes the
root before a slower warp has read the pivot, and that whole warp then scales its rows by
`1/sqrt(sqrt(d))`. The fix is `B3b`, one work-group barrier between the `d` read and the
scale; with it, every shape measured is bit-stable again (`n` in {64, 128, 256, 384, 512} x
`batch` in {1024, 2048, 4096, 8192, 16384}).

The signature is worth remembering: **the first wrong row was always a multiple of 32**.
Lanes are rows, so per-warp corruption means the *reader* was a warp, not a lane — which
points at a shared cell read cross-warp, and away from anything per-lane (`rp`, `rA`, the
lane's own global stores).

Why it needed occupancy to appear at all. With a handful of resident work-groups the warps
of one group run near lockstep and the hazard window never opens. The suite's residual sweep
runs `batch <= 3`; `AgreesWithTheCtaKernel`, the poison test and both `info` tests run
`batch <= 2` to 35. **Eleven tests, four types, and not one of them launched enough
work-groups to skew two warps apart.** The new guard
`PanelStoresArePublishedAtSaturatingBatch` runs `n = 128, batch = 1024` with every item the
SAME matrix and demands every item come back BIT-IDENTICAL to item 0 — a cross-item oracle,
deliberately against the sweep's "every item differs" rule, because with identical input a
deterministic kernel has no licence to disagree with itself. `batch = 512` was measured NOT
enough; 1024 is.

### Armed breaks (R9) — LPanel

Planted, built, observed and restored during integration. `src/extensions/potrf_lpanel_device.hh`
and `potrf_lpanel.cc` were checked back to their intended state by md5 afterwards. Three of the
seven predictions were WRONG, and each wrong one is a finding rather than an embarrassment.

| # | the break | expected red | OBSERVED |
|---|---|---|---|
| 1 | delete `B6` (the barrier ending the panel loop) | `ResidualAcrossPanelBoundaries` at every n > NB | **GREEN — NOT ARMED.** Nothing red at `batch` 1024 (all 4 types, n to 512) nor at `batch` 4096/16384 through `factor_bench`. The store -> next-panel-staging dependency is real in the data flow, but the prefetch that precedes the staging read is 8 dependent global loads, and that latency apparently covers the window on this card. Kept on the data-flow argument alone; **no test can see it**. (A first observation of this break "red" was taken before the WAR fix below and was that race, not this one.) |
| 2 | delete `B3` (between publishing `sA` and the panel factorisation) | `ResidualAcrossPanelBoundaries` | **RED**, 19 cases: `double` and `complex<double>` red in 8 tests each (residual sweep, CTA oracle, both `info` tests, padded-ld, packed batch, facade), `float`/`complex<float>` red only in the saturating-batch guard and cfloat's residual sweep. The scalar width decides how far apart the warps drift. |
| 3 | drop `dev_conj` when staging `sB` | complex ONLY red, real green | **RED for complex exactly as predicted** — `complex<float>` and `complex<double>` red in 8 tests each, `float` and `double` green in every test except the saturating-batch guard, which was red for all four types because the WAR race below was still present. The complex/real asymmetry — the anti-vacuity half — held. |
| 4 | prefetch guard `row >= j + i` -> `row >= j` | `OtherTriangleIsNeitherReadNorWritten` | **GREEN — the bound is defended THREE times.** Phase (3) re-applies `row >= j + i` when publishing and the store re-applies it again, so relaxing the prefetch alone changes no output. Only breaking all three (4b) reaches the caller. |
| 4b | relax the prefetch, the publish AND the store guard | as 4 | **RED for all four types**, but only after the test was strengthened — see below. |
| 5 | drop the `ok &&` from `const bool bad = ok && !(d > R(0))` | `info` names the LAST planted failure | **RED**, all four types, 3 tests each: `info = 93` where 1 was planted, `96` where 2 was planted, `96` where 8 was planted. |
| 6 | delete the `case Algorithm::LPanel: return false;` arm from `native_tier_preferred` | `RouteTableIsInertUntilMeasured` (now `RouteWindowIsExactlyTheMeasuredOne`): the arm count becomes 2 | **RED**, all four types, and nothing else moved. The R8b guard works. |
| 8 | `preferred()` returns `true` for every native arm in the window instead of only for `best_native_tier(s)` — the R8b defect, on the NEW window | `RouteWindowIsExactlyTheMeasuredOne`: `pref_hits` must be 1 | **RED**, float and cfloat: `pref_hits` = **3** at n = 33 and 64 (Tiny, CTA and LPanel all answer), 2 at n = 256. `automatic()` would have returned the Tiny tier for every shape in the window. |
| 9 | *(not planted — SHIPPED, and caught by a test in another file)* the window composed `best_native_tier(s)` without asking whether the tier the GRID names actually holds on this device | — | **RED in `route_vocabulary_tests`**, 6 orders x 4 batches x `Uplo::Lower`: its synthetic `PotrfShape` leaves `lpanel_max_n = 0`, so at order 63..156 the walk fell back to CTA — which the grid measures at **0.53-0.94x of the vendor** at those orders. `preferred()` now also requires `best.algo` to be the tier measured for that order (CTA at or below the boundary, LPanel above), and the file needed no edit. |
| 7 | give a dead packed slot `n = 0` instead of `store = false` | `PackedBatchIsIndependentAndTheTailSlotsAreInert` HANGS; expect a ctest timeout | **GREEN, AND THE BREAK WAS ALREADY IN THE TREE.** `potrf_lpanel.cc` shipped `..., Ag, ldg, live ? n : 0);  // BREAK7` — the dead slots were exiting before the first barrier, which is UB in SYCL and which the prediction said would hang. It does not hang: the slots are whole warps (`L` is a multiple of 32), and an exited warp is not counted by a CTA barrier on this hardware. Integration changed the call to pass `n` unconditionally, which is what both files' comments already claimed and what removes the UB; the cost is that at most `G-1` slots of ONE work-group in the grid do full work. |

**The poison that the guard swallowed (the twelfth blind guard).**
`OtherTriangleIsNeitherReadNorWritten` poisoned the upper triangle with a quiet NaN and
checked `isnan` afterwards. That cannot see a WRITE: the value the kernel would store into
the upper triangle is computed *from* the NaN it just read, so the corrupted output is NaN
too and `isnan` passes. Break 4b was green under the original test and red only after the
test was given a **second pass with a finite sentinel** (`-999 + 777i`) and a bit-equality
check. The NaN pass still owns the "not read" half — a stray read makes the residual NaN —
and the finite pass owns the "not written" half. Both halves are needed; neither alone is a
guard. This is the same failure mode `spmm.md` records as "your poison must be something the
code under test will ACCEPT".

### LPanel: what integration measured, and what it could not

**Correctness: measured and green.** 52 of 52 LPanel cases pass on CUDA (the other 48 are
NETLIB rows that skip: it is a GPU kernel), `potrf_tests`, `potrf_tests_native`,
`ortho_tests` and `resident_capacity_tests` all pass with the window in place.
Bit-stability confirmed at `n` in {64, 128, 256, 384, 512} x `batch` to 16384, float and
cfloat, after the `B3b` fix above.

**Performance: measured, with the grid and its limits in
[the measured LPanel window](#the-measured-lpanel-window).** The measurement window on GPU 1
was itself interrupted: another user's process held the card at 100% for the first hour, and
`gpu_guard.sh` refused rather than reporting numbers taken beside it. Everything taken during
that hour was correctness-only on GPU 0 — residuals, bit-equality, `info` — and no time from
that period is quoted anywhere. Every ratio above was taken after the card was exclusively
ours, and `gpu_guard` confirmed "exclusive for the whole run" on each cell.

## The fused posv tier

`src/extensions/posv_tiny.cc`, one launch for `A X = B` with Hermitian
positive-definite A at order `n <= 32` and `nrhs <= 4`: `potrf_tiny.cc`'s unblocked
Cholesky recurrence verbatim, followed in the same kernel by both triangular solves
with L still in registers. Zero local memory, zero barriers, every cross-lane value
a sub-group shuffle, exactly as the tiny tier requires
([the shared tiny-tier invariants](#the-shared-tiny-tier-invariants)).

**It is not routed.** `route_posv.hh`'s `preferred()` is all-false and
`native_tier_preferred` answers false for `Tiny`, so `Auto` takes the composed
`potrf; trsm; trsm` arm. As with `gesv`, there is no batched vendor `posv` anywhere
and no `potrs` op in this library, so `resolve_posv_route` passes
`vendor_available=false` unconditionally — which means the tier hook is consulted on
every call and answering `true` there would ship the tier. That is why it is false.

### posv: the backward solve costs a transpose

**The P2 plan text is wrong on this point, and the kernel does not follow it.** The
plan says the backward solve needs no transpose trick "because every lane holds a
full row". That does not follow. Lane `r` holds *row* `r` of L, so:

* `L y = b` reads `L(r, i) = rA[i]` — the index is the loop counter, so it is
  **local to the lane** and the forward solve is a cheap right-looking sweep whose
  only cross-lane traffic is the pivot row's own values, `NR + 1` shuffles per step.
* `L^H x = y` needs `L(i, r)` — **column** access, which no lane has. Lane `r` needs
  an element that lives on lane `i`, at an offset that differs per lane, and
  `tiny_device.hh`'s first invariant forbids the dynamic index into `rA[]` that a
  direct shuffle would need.

The kernel buys it with an explicit on-the-fly transpose: at step `i`, lane `i`
broadcasts `rA[0..i-1]` and every lane keeps the one element where `c == lane`. That
costs `N(N-1)/2` shuffles — 496 at N = 32 — which is the **same order as the
factorization itself** (also `N(N-1)/2`), so the fused posv kernel is roughly twice
a `potrf_tiny` before the RHS work is counted. It is, however, independent of
`nrhs`, which the alternative is not.

**The alternative, and why it was not taken.** The left-looking form needs
`conj(L(j, i))` on lane `j` at compile-time index `i` — local — and a partition
reduction over lanes `j > i`, i.e. a `log2(N)` butterfly per RHS column:
`NR * N * log2(N)` shuffles, 160 at `N = 32, NR = 1` and 640 at `NR = 4`. It is
cheaper at `NR <= 2` and dearer at `NR = 4`. **This is a named A/B for the
measurement phase, not a settled choice**; the transpose form shipped because it is
`nrhs`-independent and needs no new collective in `tiny_device.hh`.

### P2: the window this tier expects

**Nothing was measured for posv, and unlike `gesv` the bound cannot be computed from
committed data either**: there is no `potrs` op and no `trsm`-composition baseline at
these orders in `benchmarks/results/`, so `t_vendor_potrf + t_two_trsm` is not on
record. What *is* on record is the tiny potrf tier's own standing against cuSOLVER
([the tiny tier](#the-tiny-tier)), and the structural fact above that the fused
kernel costs about 2x a `potrf_tiny`.

Read together those two say the honest prior is: **a win where `potrf_tiny` wins by
more than ~2x and the two `trsm` launches are a real share of the composition** —
which is the small-`n`, large-batch corner, `n <= 16` — and a loss at `n = 32`. That
is a hypothesis with no number behind it; it is written down so the grid that runs
has something to falsify.

The grid P2 owes: `n in {4, 8, 9, 16, 17, 24, 32}` x `nrhs in {1, 2, 4}` x batch
`[4096..65536]` x 4 types, three arms interleaved — `posv` composed, `posv` tiny,
and `potrf_tiny` alone as the floor — with a bracketing non-winner at whichever `n`
the tier stops winning, per R8.

### P2: the measured posv window

**What shipped.** `route_posv.hh::tiny_window_max_n()` = 32 for every type except
`std::complex<double>`, which stops at 16 because P1 instantiates no further (plan
D3). As in `route_gesv.hh` the predicate lives in `native_tier_preferred`, not
`preferred()`: `resolve_posv_route` always passes `vendor_available = false`, so the
vendor-free walk (`supports && native_tier_preferred`, first hit wins) is the
shipping path and the fit is resolved before the hook is asked. `preferred()` stays
all-false, permanently, and `posv_tests.cc` asserts it.

**The arms.** `posv` has no vendor entry either; its `Blocked` route is
`potrf; trsm; trsm`, each leg routing independently. `run_solve_grid.sh` interleaves
four arms in one process: `tiny`; `blocked` (legs under `Auto` — **the incumbent**);
`vendor` (`potrf`=vendor, `trsm`=vendor); `native` (`potrf`=tiny, `trsm`=cta). The
flip is gated on `tiny` vs `blocked`.

#### posv float and cfloat — `t_blocked / t_tiny`, 7 reps, interleaved

Raw: `benchmarks/results/p2_posv_float.csv`, `p2_posv_cfloat.csv`. `!` marks a cell
re-run for `rel_sd` (see the table in `lu.md`); both re-runs agreed.

| n | nrhs | float 8192 | float 16384 | float 32768 | cfloat 8192 | cfloat 16384 | cfloat 32768 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 11.27 | 16.37 | 21.84 | 8.06 | 9.39 | 11.94 |
| 4 | 4 | 9.61! | 11.56 | 15.14 | 5.85 | 6.87 | 7.54 |
| 8 | 1 | 9.33 | 12.57 | 16.62 | 6.11 | 6.72 | 8.19 |
| 8 | 4 | 7.34 | 8.44 | 11.10 | 4.20 | 4.18 | 4.24 |
| 9 | 1 | 10.24 | 12.35 | 14.33 | 5.08 | 5.29 | 5.77 |
| 9 | 4 | 7.73 | 8.56 | 9.52 | 2.99 | 3.00 | 3.14 |
| 16 | 1 | 8.02 | 9.91 | 11.15 | 3.51 | 3.52 | 3.30 |
| 16 | 4 | 6.13 | 6.52! | 6.90 | 2.06 | 1.87 | 1.99 |
| 17 | 1 | 7.06 | 7.69 | 7.60 | 3.29 | 3.20 | 3.16 |
| 17 | 4 | 5.63 | 5.53 | 5.26 | 1.87 | 1.93 | 2.17 |
| 24 | 1 | 6.74 | 6.62 | 6.90 | 2.62 | 2.60 | 2.71 |
| 24 | 4 | 5.10 | 4.52 | 4.76 | 1.55 | 1.72 | 1.92 |
| 32 | 1 | 5.36 | 5.13 | 5.75 | 2.16 | 2.25 | 2.35 |
| 32 | 4 | 3.76 | 3.77 | 4.14 | **1.29** | 1.52 | 1.69 |

Every cell clears the 1.11 gate by a wide margin; the weakest, cfloat n = 32
nrhs = 4 at 1.29, **rises** with batch (1.29 / 1.52 / 1.69) and reads 1.87 at
131072 (`p2_posv_cfloat_bigbatch.csv`). Unlike `gesv`, posv has **no reversing cell
anywhere on the ladder** — checked explicitly at batch 131072 for cfloat n = 4, 16
and 32 (8.20, 2.17, 1.87).

#### posv double and cdouble — routed, unlike gesv's

Raw: `p2_posv_double.csv`, `p2_posv_cdouble.csv`, batches 16384 / 32768:

| n | double nrhs=1 | double nrhs=4 | cdouble nrhs=1 | cdouble nrhs=4 |
|---:|---:|---:|---:|---:|
| 4 | 10.41 / 10.76 | 6.13 / 6.20 | 10.03 / 10.11 | 7.18 / 7.24 |
| 8 | 8.03 / 8.19 | 4.05 / 4.05 | 6.18 / 6.21 | 4.22 / 4.20 |
| 16 | 5.31 / 5.32 | 2.62 / 2.73 | 4.90 / 4.94 | 3.08 / 3.11 |
| 32 | 3.69 / 3.70 | 2.12 / 2.12 | — (capped at 16) | — |

R10 routes double where it wins, and it wins everywhere here, by 2.12x at worst.

#### A SEPARATE DEFECT this grid exposed: posv's composed arm is mis-routed under Auto

The `vendor` arm (`potrf`=vendor, `trsm`=vendor) is **2-5x faster than the same
composition under `Auto`** at these orders. float, `nrhs = 1`, `t_blocked / t_vendor`:

| n | 8192 | 16384 | 32768 |
|---:|---:|---:|---:|
| 4 | 1.82 | 2.66 | 3.61 |
| 8 | 1.51 | 1.96 | 2.46 |
| 9 | 2.95 | 3.63 | 4.58 |
| 16 | 2.27 | 2.89 | 3.23 |
| 17 | 4.45 | 5.17 | 5.56 |
| 32 | 2.74 | 2.86 | 2.86 |

**This is not a posv defect.** `posv`'s `Blocked` arm just calls `potrf` and `trsm`,
so one or both of their `preferred()` windows is choosing a native tier that loses at
n <= 32 and large batch. `trsm` is the first suspect: this composition is
`Side::Left`, which `docs/perf/trsm.md` already records as the one side with a float
cliff. The consequence for P2 is only that the fused tier's measured margin against
the *incumbent* (above) is larger than its margin against the *best reachable*
composition — so `posv`'s window is justified twice over against what ships, and by
less against what `potrf`/`trsm` could route to. **Where the fused tier loses to the
vendor-pinned composition — cfloat n >= 17 at `nrhs = 4`, double n = 4 and n = 32 at
`nrhs = 4`, cdouble n >= 8 — it still beats the arm it actually replaces, so no
routing decision here is wrong; the number to chase is `potrf`/`trsm` at n <= 32.**

#### Route readback after the flip

`benchmarks/results/p2_route_readback.txt`, dispatch coverage under `--arms=auto`:

```
posv   float    n=32  -> native:tiny      posv   float    n=33  -> native:blocked
posv   cfloat   n=32  -> native:tiny      posv   double   n=32  -> native:tiny
posv   cdouble  n=16  -> native:tiny      posv   cdouble  n=17  -> native:blocked
```

#### What P2's integration could NOT establish for posv

- **No measured non-winner at the top edge.** posv's window ends where the tier's
  instantiations end (32, or 16 for cdouble), so there is no reachable cell above it;
  the bracket is `supports()` returning false, which the route test pins but no timer
  can read. The one *measured* bracket P2 owns is `gesv` cfloat at n = 17.
- **The `Uplo::Upper` grid was not timed.** The harness drives `Uplo::Lower` only.
  Upper is correctness-tested at every order in `posv_tests.cc` (and the load
  transform means it runs the same algorithm), but it has no timing row.
- **The register probe was not run** — see the section below.
- **`nrhs` 2 and 3 were not timed**, only 1 and 4.

### P2: the register bound is assumed, not probed

`posv_tiny.cc` carries `kWorstRegsPerThread = 256` in a `static_assert` against the
per-block register file at `kTinyWgSize = 64`. It was **not measured**: it is
`potrf_tiny`'s probed 176 plus room for the `2 * NR` scalars the two solves add.
`scripts/register_probe.sh` must report frame 0 and spill 0 for every
`PosvTinyKernel` instantiation before the tier is timed — a non-zero stack frame
means an unrolled loop was declined and `rA[]` left the register file, which is a
slowdown and never a wrong answer, so only the probe can see it. The same note, and
the 44-instantiation R7 count for both P2 kernels, is in
`docs/perf/lu.md#p2-the-register-bound-is-assumed-not-probed`.

## The probe's own gate could not fail

`scripts/register_probe.sh` is the instrument every register-residency claim in this campaign
rests on. Its spill gate was:

```awk
/Compiling entry function/ {e=1; next}
/Function properties for/  {e=0}
e && /spill stores/ && !/0 bytes spill stores, 0 bytes spill loads/ {n++}
END {print n+0}
```

ptxas emits, per kernel, in this order:

```
ptxas info    : Compiling entry function '_Z...' for 'sm_89'
ptxas info    : Function properties for _Z...
    N bytes stack frame, M bytes spill stores, K bytes spill loads
ptxas info    : Used R registers, ...
```

`Function properties for` arrives **between** the entry-function line and the data, so `e` was
cleared before the spill test could ever run. **The gate printed 0 for every input, always.** A
clean report from a gate that cannot report anything else reads exactly like a clean report.

Fixed, and the fix is what surfaced
[the posv cdouble spiller](#the-posv-tiny-tier-is-not-register-resident-for-cdouble): the same
link now reports **1 entry function with non-zero spill** where the old gate reported 0.

Three further defects fixed in the same pass, each of the same silent-zero family:

* **Stack frame was not reported at all**, and it — not spill — is the gate for a residency
  claim. The link reports **98 of 541 distinct kernels** with a non-zero frame.
* **An entry name that did not match the mangled-name pattern was skipped by the gate while
  still counting toward the denominator.** Now counted, printed, and a non-zero count exits
  non-zero. Zero on every log on disk, so this was latent rather than a live miss.
* **A SIGPIPE abort** in the frame listing (`printf | head` under `set -euo pipefail`) that
  would have killed the script mid-report once the list outgrew the pipe buffer, truncating the
  report it exists to produce. Reproduced at 500 rows (exit 141), fixed with `sed -n '1,20p'`,
  which consumes its whole input.

The script also fails loudly when the named target has no `link.txt`, rather than silently
probing the default library — a real hazard here, since the factorization tiers live in
`batchlas_extensions_cta` and the default is `batchlas_sycl`. Pass
`BATCHLAS_BUILD_DIR` when the build tree is not `<repo>/build`:

```
BATCHLAS_BUILD_DIR=$PWD/build/presets/dev-tests \
  scripts/register_probe.sh out.log '' batchlas_extensions_cta
```

The link takes about 340 s, so this is a deliberate act, not part of a build.

## The posv tiny tier is not register-resident for cdouble

Found by running `scripts/register_probe.sh` after fixing its gate — the gate had been
structurally incapable of reporting a non-zero count, so this was invisible rather than
unnoticed. See [the probe's own gate could not fail](#the-probes-own-gate-could-not-fail).

Probed against `batchlas_extensions_cta` on this tree (sm_89, AOT device link, 1082 entry
functions, 541 distinct kernels):

| kernel | registers | stack frame | spill |
|---|---|---|---|
| `PosvTinyKernel<complex<double>, 16, 4>` | **255** | **96 B** | **184 B** |
| `PosvTinyKernel<complex<float>, 32, 4>` | 149 | **256 B** | 0 |
| `PosvTinyKernel<complex<double>, 16, 1>` | 246 | 0 | 0 |
| `PosvTinyKernel<double, 32, 4>` | 222 | 0 | 0 |
| every `GesvTinyKernel` instantiation | ≤ 182 | 0 | 0 |
| every `GetrfTinyKernel` / `PotrfTinyKernel` / `GeqrfTinyKernel` | ≤ 193 | 0 | 0 |

**Two of the tier's own instantiations are not register-resident**, which is the tier's entire
premise: `complex<double>` at N=16/NR=4 spills 184 bytes over a 96-byte frame, and
`complex<float>` at N=32/NR=4 carries a 256-byte frame with no spill at all. A non-zero stack
frame is the gate for a residency claim and the probe never used to report it.

**Both are on the default `Auto` path, not pin-only.** `route_posv.hh`'s `tiny_window()` admits
every order up to `tiny_window_max_n()`, which is 16 for cdouble and 32 otherwise, so a caller
asking for cdouble `n = 16` with `nrhs` in the 4-bucket reaches the spilling kernel through
`Auto`.

**It is not a launch abort.** The tier launches at `kTinyWgSize = 64`, i.e. 2 warps, so
`ceil(2/4) * 32 * 256 = 8,192` against the sub-partition's 16,384 — comfortably legal. The cost
is spill traffic inside a kernel written specifically to avoid it, not a crash.

**The old bound was a tell.** `posv_tiny.cc` declared `kWorstRegsPerThread = 256`, above the
255-register ISA ceiling, so no kernel could ever have reported it — exactly what an assumed
bound looks like, and the comment said as much ("a BOUND TO RE-PROBE, not a measurement"). It is
now the measured 255. `gesv_tiny.cc`'s assumed 224 turned out to be a safe over-estimate: its
worst real instantiation is 182, clean.

### What is owed

Not fixed here, and deliberately not papered over by lowering a cap:

1. **Why NR=4 costs so much more than NR=1 for cdouble** — 246 → 255-plus-spill is the whole
   budget, and the 2·NR extra scalars the solve arm adds do not obviously account for it.
2. **Whether the spill is measurable end to end.** The cdouble posv window was never timed at
   the shapes this instantiation serves; the `posv` grids on this page are float and cfloat.
   The window may be right anyway — spilling and still beating the composed arm is possible —
   but nobody has looked.
3. **The float N=32/NR=4 frame.** 256 bytes with zero spill usually means a local array that
   did not get promoted; that is a read of the generated code, not of a counter.

Until (2) is measured, the honest statement is that the cdouble posv tier is routed on a
residency claim its own probe refutes.

## The tiny potrf window, extended to Upper and to fp64

Two things this page previously recorded as *untested, not refuted* are now measured, and both
turned out to be wins — one of them the widest margin in the campaign. Both grids are at batch
**32768**, R8, interleaved arms in one process, ratios in time against the **vendor** arm, and
both were taken with the degraded guard described in
[the NVML caveat](qr.md#a-measurement-caveat-that-is-part-of-the-result).

### Uplo::Upper is the wider window, by a long way

`factor_bench` could not express `Uplo::Upper` at all — it hardcoded `Uplo::Lower` at four call
sites — which is the entire reason the grid was Lower-only and the doc said "an unmeasured
triangle is not a window". It now takes `--uplo=upper|lower`, with a residual path that forms
`Uᴴ U` from the upper triangle rather than `L Lᴴ` from the lower one. Reusing the Lower sweep on
an Upper factor reads the untouched triangle and scores a correct answer as garbage; the two
paths agree to the digit on a symmetric input (both `9.555e-08` at float n=16), which is what a
correct pair must do.

| n | float | cfloat | double | cdouble |
|---|---|---|---|---|
| 1 | 21.539x | 20.214x | 3.891x | 2.949x |
| 2 | 27.065x | 21.875x | 4.297x | 3.540x |
| 3 | 22.462x | 21.825x | — | — |
| 4 | 28.600x | 21.418x | 5.235x | 4.690x |
| 8 | **28.727x** | 13.251x | 6.987x | 6.310x |
| 10 | 9.991x | 6.643x | 1.621x | 1.417x |
| 12 | 11.154x | 5.978x | 1.808x | 1.684x |
| 16 | 11.018x | 3.462x | 2.317x | 2.127x |
| 17 | 3.192x | 1.738x | — | — |
| 20 | 3.352x | 1.952x | — | — |
| 24 | 3.544x | 2.072x | — | — |
| 28 | 3.626x | 2.346x | — | — |
| 32 | 3.314x | 2.118x | — | — |

**Every measured cell wins, for every type, with no holes.** The mechanism is not that our
kernel is better at Upper — it is that cuSOLVER is much worse at it. A direct pair at float
n=16, batch 4096: vendor **0.0360 ms** Lower against **0.0730 ms** Upper, while the tier moves
only 0.01096 → 0.01263. The vendor roughly doubles; we barely notice the triangle. That gap is
wide enough to swallow the N-bucket fill penalty that splits the *Lower* band below, which is
why the Upper window is the whole tier and the Lower one is not.

The window is therefore `1 <= n <= tiny_max_n` for all four types on Upper. LPanel is untouched:
its grid is still Lower-only and `supports()` still refuses Upper outright, so one order past
the register tier Upper returns to the vendor. That pair is asserted in both test files so the
`uplo` test cannot be deleted wholesale now that one window no longer needs it.

### fp64 wins on Lower too, in two bands

`lpanel_types()` is float/cfloat, and `preferred()` tested it before anything else, so neither
fp64 type could reach the tiny window at all. That gate was about the **LPanel** grid; it was
never a claim about the register tier. Measured:

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 20 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **double** | 0.715 | 1.158 | 1.629 | 2.099 | 2.558 | 3.014 | 3.494 | **3.956** | 0.865 | 0.957 | 1.049 | 1.145 | 1.233 | 1.322 | 1.411 | 1.505 | 0.548 | 0.590 | 0.768 |
| **cdouble** | 0.709 | 1.291 | 1.885 | 2.486 | 3.080 | 3.687 | 4.279 | **4.892** | 0.882 | 0.976 | 1.071 | 1.167 | 1.263 | 1.354 | 1.446 | 1.545 | — | — | — |

Window: **2..8 and 12..16**, both types. The bands are the N-bucket structure exactly — n=1 is
overhead-bound, 2..8 fills the N=8 bucket, 9..11 fills N=16 badly and loses, 12..16 fills it
well, and 17+ enters N=32 where double collapses to 0.548-0.768x. Two independent types trace
the same curve to within a few percent, with `rel_sd` around 0.001, which is why this window
uses the standard 1.11 gate rather than the widened margin the geqrf complex grid needed: the
structure corroborates itself across types and matches a known mechanism, where that one did not.

cdouble has no rows above 16 because the tier caps there, and **the four cells that looked like
`1.000x` in the first pass were vacuous, not neutral**: at cdouble n=20 the `tiny` arm read
4.410 ms against the vendor's 4.413 ms, i.e. `supports()` refused the pin and the route fell
through to the vendor, so both arms measured the same code. `pin_parsed` says the value parsed,
never that the route ran — the trap this harness documents and still caught nobody's eye until
the ratio came out at exactly 1.000.

### End to end

The flip moves `Auto`, which is the only thing that matters to a caller. Vendor against `Auto`,
batch 32768, after the change:

| | n=6 | n=10 | n=14 | n=16 |
|---|---|---|---|---|
| double | 3.017x | **1.000x** | 1.323x | 1.505x |
| cdouble | 3.687x | **1.001x** | 1.353x | 1.546x |

Inside the window `Auto` tracks the pinned tier to three digits (pinned: 3.014, 1.322, 1.505 /
3.687, 1.354, 1.545). At n=10 — the hole the window deliberately excludes — `Auto` reads exactly
1.000x, which is the vendor. The window does what it says and nothing else.

### Arming both windows

R9, four breaks, each rebuilt and run against `potrf_tests` and `route_vocabulary_tests`:

| break | expected | observed |
|---|---|---|
| admit the fp64 Lower hole n=9..11, a measured loss | red | **RED** |
| refuse `Uplo::Upper` again | red | **RED** |
| let Upper reach the LPanel window, which is Lower-only | red | **RED** |
| fire the window with `tiny_max_n == 0` (no tier linked) | red | green |

The fourth is recorded as **unfalsifiable, not as coverage**: `supports()` independently returns
false when `tiny_max_n < 1`, so deleting the same test from `tiny_window()` changes nothing a
test can observe. Same shape as the corresponding break on the `geqrf` window. The first three
are the load-bearing ones and all went red — in particular the third, which is what stops the
`uplo` test being deleted from `preferred()` now that the tiny window no longer needs it.

## The fused potrs solve

**2026-09-26.** posv's composed arm was `potrf; trsm; trsm`, and at nrhs = 1 the two
trsm launches cost 4.6x the potrf they follow (float n = 64, batch 16384: potrf 0.53 ms,
posv 2.96 ms against cuSOLVER's 1.74 ms). A new tier, `{Native, CTA}`, runs the routed
potrf and then **one** kernel for both triangular solves: `sycl_getrs::potrs_fused_dispatch`
in `getrs_fused.cc`, the fused narrow-RHS getrs body with no pivots and a non-unit
diagonal on both phases. Lower solves L in the axpy form and L^H in the dot form; Upper
is the mirror image. It shares getrs's resident-RHS capacity and `kGetrsFusedMaxRhs`.

The arm order is Tiny, CTA, Blocked; above the tiny window CTA takes every shape it can
hold and Blocked is its capacity fallback. Ratios are `t_vendor / t_native`, Lower,
nrhs = 1, factor_bench, 5 reps, interleaved; batch 32768 to n = 32, 16384 at 64,
4096 at 128, 1024 at 256.

| n | float before | float after | cfloat before | cfloat after |
|---:|---:|---:|---:|---:|
| 64 | 0.59 | **2.07** | 1.18 | **4.56** |
| 128 | 0.79 | **1.62** | 3.35 | **7.03** |
| 256 | 1.02 | **1.46** | 11.3 | **15.4** |

Upper, batch 16384: float 1.39 / 1.10, cfloat 2.27 / 2.24 at n = 64 / 128. double and
cdouble, batch 8192, CTA against Blocked: 4.2 / 3.9 / 2.5 / 1.6x (double) and
4.5 / 4.1 / 2.5 / 1.5x (cdouble) at n = 33 / 64 / 128 / 256, and ahead of cuSOLVER at
every one of those cells.

### The tiny window moved

CTA also beats the fused tiny kernel inside part of its old window. Batch 32768,
`t_vendor / t_arm`:

| type | nrhs | n = 12 | 16 | 17 | 24 | 32 |
|---|---:|---|---|---|---|---|
| float | 1 | tiny 3.31, cta 1.92 | 3.41 / 2.07 | 1.35 / **1.46** | 1.59 / **1.70** | 2.01 / **2.22** |
| float | 2 | 2.35 / 2.10 | 2.51 / 2.33 | 0.88 / **1.38** | 1.10 / **1.73** | 1.18 / **1.86** |
| float | 4 | 3.74 / 1.54 | 3.54 / 1.45 | **1.19** / 1.14 | **1.30** / 1.15 | **1.45** / 1.22 |
| cfloat | 1 | 0.91 / **1.15** | 1.13 / **1.30** | 0.47 / **1.07** | 0.88 / **1.94** | 1.40 / **3.13** |
| cfloat | 2 | 0.46 / **0.68** | 0.63 / **0.84** | 0.33 / **0.94** | 0.63 / **1.62** | 0.98 / **2.52** |
| cfloat | 4 | **0.75** / 0.57 | **1.11** / 0.83 | 0.44 / **0.86** | 0.76 / **1.33** | 1.23 / **1.98** |

So `tiny_window` is now: float `n <= 16 || nrhs >= 3`, cfloat `n <= 8 || (n <= 16 &&
nrhs >= 3)`. cfloat n = 8 stays tiny at every width (2.93 / 1.29 / 1.74 against CTA's
1.27 / 0.72 / 0.47). double and cdouble keep the tier ceiling: tiny against CTA is
unmeasured for them. nrhs = 3 is interpolated from 2 and 4.

**Still losing:** cfloat 9..17 at nrhs >= 2 loses to cuSOLVER on both arms (best 0.68-0.94).

### The poisoned triangle is only poison if potrf leaves it alone

`FusedSolveArmSolvesOnBothTriangles` pins potrf `native`. Unpinned, Upper at n >= 33
resolves potrf to cuSOLVER, which **overwrites the unreferenced lower triangle**, so
the solve's wrong-triangle read (the armed break: swap `ld_a`/`ld_h` in the backward
staging) returned the right answer and stayed green on Upper at n = 33, 64 and 100.
Pinned, the break is red on Lower at every n > 1 and on Upper wherever potrf ran native
(n = 17 and 33 for every type, n = 64 where CTA fits). The test asserts the poison
survived only where potrf resolved native.

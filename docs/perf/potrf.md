# POTRF: the CTA kernel, the blocked driver, and the 48 KB SLM launch hole (WP4)

Two native tiers ship, both linked and correct; **neither is preferred**, so a vendor-present build still resolves
every `potrf` to cuSOLVER. They exist so a `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` build stops throwing `NoRouteError`.

All numbers: RTX 4090 (sm_89, 128 SM), one card held per campaign, under `experiments/gpu_guard.sh`. Ratios are
`vendor / native` — **> 1 means native wins** — unless the table says otherwise.

## What ships

| tier | file | orders | Uplo |
|---|---|---|---|
| `{Native, Tiny}` | `src/extensions/potrf_tiny.cc`, `tiny_device.hh` | `<= potrf_tiny_max_n<T>()` = 32, and 16 for cdouble | both |
| `{Native, CTA}` | `src/extensions/potrf_cta.cc`, `potrf_cta_device.hh` | `<= potrf_cta_max_n_for_slm<T>(local_mem - 4096)` = **77/54/54/38** here (advertised, at the default target of 4 blocks/SM); 155/109/109/77 is the *resident* ceiling, `min_blocks_per_sm = 1` | both |
| `{Native, Blocked}` | `src/extensions/potrf_blocked.cc` | any | **Lower only** |
| `{Vendor, Auto}` | cuSOLVER | any | both |

All four scalar types on every arm (cdouble n = 17..32 excepted on Tiny). Tiny is FIRST in the candidate order,
but `native_tier_preferred` answers **false** for it and `preferred()` is false everywhere, so **no route moves**:
a vendor-free build still takes CTA at n <= 32 and `Auto` in a vendor-present build still takes the vendor. Tiny is
reachable by an explicit pin only. See [the-tiny-tier](#the-tiny-tier) — it has a register table and two negative
results but **no timing**, which is precisely why the tier is not yet preferred.
The rest of the candidate order (`route_potrf.hh`) is *mostly* a capability ladder:
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

### `preferred()` is false everywhere

`preferred()` returns `false` unconditionally (`route_potrf.hh:65-69`). Un-preferred is not unroutable:
`route_resolve.hh:38-49` still hands a **vendor-free** caller any supported native route, while `Origin::Auto` in a
vendor-present build keeps taking cuSOLVER. `route_diff.sh` across both landings shows zero changed non-potrf rows;
what changed is potrf's `native_route_supported` column, 0 → 1 (`route_potrf.hh:3-4`), and the Phase 2 capture is
+28 additions / 0 removals. **The Phase 2 triage delta's route diff was never re-run** — the capture needs
`-DBATCHLAS_ENABLE_COVERAGE=ON`, i.e. two device-link-bound rebuilds — and was argued instead from the diff touching
no routing predicate (one `group_barrier`, two `fill`s, four in-order guards, one tuning constant, comments).

The grids below are the *input* to flipping a cell, not the decision. The gate is three-part:
`t_native <= 0.90 * t_vendor` at saturation, **and** no accuracy regression, **and** an end-to-end
`ortho_benchmark` win. None has been run. (This repo once turned a 2.16x kernel win into an 11% gesvd loss.)

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

1. **`preferred()` is all-false.** No cell flipped; the three-part gate has not been run.
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

`preferred()` stays all-false, so `Auto` still takes cuSOLVER at every shape and none of
the below is reachable except in a vendor-free build or under `BATCHLAS_POTRF_ROUTE`.

#### Every tier is enumerated explicitly

Exactly one arm answers `true` at any shape, and the switch names all three rather than
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

# WP6 — LU: `getrf` / `getrs` / `getri`

The work-package summary. The four subdirectories are the evidence and each has
its own README; this file is the synthesis, the verdicts, and — deliberately at
the same weight — the list of things WP6 **did not** settle and the things that
were tried and **measured worse**.

| directory | what is in it |
|---|---|
| `baseline/` | the pre-kernel survey: the vendor's own cost, the 48 KB launch hole re-measured from scratch, the pivot-format and pivot-metric probes, the work-group sweep |
| `kernels/` | the kernels themselves: A/B grids, tier sweep, cross-build check, the kernel-side breaks |
| `tests/` | `tests/getrf_tests.cc` and the 14-break record that proves it can fail |
| `bench/` | the measure phase: 982 timed rows through the public API, one program linked against both builds |

Everything below was produced on this box: 2× RTX 4090 (128 SMs, 97,280 B of
usable local memory), one GPU pinned per run.

---

## 1. What WP6 built

Before WP6 all three LU ops were **pure vendor facades with no route resolution
at all**. WP6 gave each one the full WP5 structure and then filled it:

| file | what |
|---|---|
| `include/batchlas/blas/dispatch/route_{getrf,getrs,getri}.hh` | the three route tables — pure headers, `supports()` / `preferred()` / (getrf) `native_tier_preferred()` |
| `src/backends/{getrf,getrs,getri}_route.hh` | the shape builders and the env read. All three set `s.backend = B` |
| `src/extensions/getrf_cta_device.hh` | one `?GETF2` device body, shared by both residencies and by the blocked driver's panel |
| `src/extensions/getrf_cta.cc` | the CTA tier: the resident leaf, the global leaf, the fit predicate, the capacity |
| `src/extensions/getrf_blocked.cc` | the blocked driver — panel, two interchanges, routed `trsm`, routed `gemm` |
| `src/extensions/lu_laswp.hh` | the shared row-interchange kernel |
| `src/extensions/getrs_native.cc` | interchange + two triangular solves, all three `transA` |
| `src/extensions/getri_blocked.cc` | `C := P` written directly, then two triangular solves — no permutation buffer at all |
| `tests/getrf_tests.cc` | 17 typed cases × 4 scalar types × 2 backends |

**Two native `getrf` tiers.** The CTA tier holds the whole matrix in local
memory; its per-type ceiling is measured from the **runtime** local-memory budget
and is **155 / 109 / 109 / 77** for float / double / cfloat / cdouble here. Above
that the blocked driver splits into panels and the CTA device body becomes the
panel leaf. Choosing between them is `native_tier_preferred()` — the third
predicate WP5 added — because `preferred()` runs above the vendor-free walk and
cannot express "among the native routes, which tier".

**The trailing update and the panel solve go through the ROUTER**, injected by
the facade as `std::function`s, never by calling `sycl_gemm::gemm_custom` from a
kernel TU (the WP3-step-16 defect). Proven live rather than merely present: the
same blocked `getrf` at n=256 runs `GemmRegister128x128Kernel` in the vendor-free
build and `ampere_sgemm_128x128_nn` in the vendor build.

---

## 2. Route state: WP6 ships ROUTE-NEUTRAL, on purpose

`preferred()` is **false everywhere in all three tables**, so a vendor-present
build still resolves `{Vendor, Auto}` for every shape — measured, 96 of 96 route
cells across 4 types × 8 orders × 3 ops, and confirmed at execution (an unpinned
vendor-build profile contains no `batchlas::sycl_getrf/getri` kernel at all). A
vendor-free build resolves **native for 96 of 96**, every residual `ok`.

`route_diff` over the whole `ctest -LE slow` corpus, before WP6 versus after:

| | vendor `build/` | `build-novendor/` |
|---|---|---|
| decisions | 3373 → 3549 | 3177 → 3368 |
| ADDED | 176 | 191 |
| **REMOVED** | **0** | **0** |

Every addition is attributed by per-binary capture, not by argument: 140/131 are
LU rows that could not have existed before (the ops had no `resolve_route` call);
the remaining 36/60 are `trsm` rows, all of them the native `getrs`/`getri`
drivers' own injected routed `trsm`, reached only because LU stopped throwing.

---

## 3. Burn-down

```
build-novendor:  ctest -LE slow
    before WP6   30 passed / 25 failed of 55
    after  WP6   33 passed / 23 failed of 56
```

Failing-set diff — **two suites left, none joined:**

* **closed:** `inverse_tests`, `linalg_layer_tests`
* **joined:** *(none)*
* **added test:** `getrf_tests` (passes)
* still failing (23): `options_api syevx lanczos gemv trsm ortho cond ormqr
  ormqr_cta ormqr_blocked orgqr iluk symm hemm herk her2k syrk syr2k syev trmm
  sytrd_blocked syev_cta syev_blocked`

`cond_tests` and `iluk_tests` remain, exactly as the ground brief predicted:
`cond` also needs `gemv` (WP7) and `syev`; `iluk` is blocked by `syev` alone and
was never an LU suite.

`inverse_tests` was checked for **reached, not merely linked** — the recorded
trap — with a coverage capture:

```
reached,getrf,float,CUDA,1281,40,40,40,2,native,cta,...
reached,getri,float,CUDA,1281,40,40,40,2,native,blocked,...
```

Vendor build: full `ctest` **59 of 61**, the two failures (`lanczos_tests`,
`steqr_tests`) pre-existing NETLIB-double ones unrelated to LU;
`ctest -L "blas|ortho"` **23 of 23**.

---

## 4. The measured picture, including the losses

Full record in `bench/README.md`: 982 timed rows, 491 per arm, 0 discarded, max
relative sd 7.2 %, one program linked against both builds so "vendor-free" is the
BUILD and never a forced route, correctness checked in process on every row.

**The headline is a correction to the naive reading.** `cublas{S,D,C,Z}getrfBatched`
is genuinely batched, but at large `n` it is **batch-parallel only**:
`cublasZgetrfBatched` at n=2048 takes **2587 / 2589 / 2591 / 2595 / 2657 / 2801 ms
for batch 4 / 8 / 16 / 32 / 64 / 128** — 32× the work for 0.3 % more time. Every
`n ≥ 512` ratio therefore divides by a routine that is not saturated.

| | geo at the grid's batch | geo at each arm's OWN best batch |
|---|---|---|
| `getrf` (28 cells) | 0.885× | **0.805×** |
| `getri` (28 cells) | 1.463× | **1.284×** |
| n=2048 row (8 cells) | 7.098× | **2.954×** |

`getrf` float n=2048 collapses 7.31 → 2.33×; `getri` float n=2048 33.84 → 8.09×;
`getrf` cdouble n=2048 3.74 → **1.05×**.

**`getrs` is not a loss, it is a crossover on `nrhs`.** Geomean by nrhs:
0.323 / 0.586 / 0.484 / 0.848 / 1.088 / 1.362 / 1.261 for nrhs = 1/2/8/32/64/128/512
— monotone, crossing 1.0 between 32 and 64. All-cell geomean 0.617× over 72 cells.
`linalg::solve` only ever issues nrhs = 1, which is the losing end.

**What the roofline says.** cuBLAS runs cdouble `getrf` at **90–91 % of this
card's FP64 peak** at n=512–1024 — there is no 2× to find. For FP32 *both* arms
sit at **1–10 % of peak**, and the reason is a decomposition rather than a slow
kernel: at `getrf` double n=128 the native arm spends 48.5 % panel + 33.6 % laswp
+ 9.2 % gemm + 8.8 % trsm across four kernels per block step, where cuBLAS does
the whole factorisation in **one fused kernel at 99.9 %**.

**Also established:** the axes must be separated — the same order sweep gives
geomean 1.438×/3.571× at batch 32 and 0.668×/0.907× at batch 1024, so a
`preferred()` window fitted to an order sweep alone would be wrong at both ends.
That is why `preferred()` is still empty; it is a deliberate hold, not an absence
of kernel.

The complex-Tiled16 prediction WP6 inherited is **refuted for LU**: both complex
trailing updates reach `GemmRegister64x64K16WideKernel` because `nb = 32` exactly
meets the wide-scalar `min_dim ≥ 32` gate. `double` is the type that lands on the
`GemmTiledGeneralKernel` fallback.

---

## 5. Every deliberate break, and whether it turned red

**Test-suite breaks — 14 applied, rebuilt and run, all 14 red** (per-type detail
in `tests/breaks.txt`; `break.py` also carries a 15th, `getrs_perm_first2`, which
is a variant anchor of `getrs_perm_first`):

| break | what it corrupts | outcome |
|---|---|---|
| `piv_base_zero` | `ipiv` 1-based → 0-based | RED, 12–13 of 16 per type |
| `getrs_forward` | transposed interchange walked forwards | RED, all 4 — **see finding T1** |
| `info_block_local` | `info` offset made block-local | RED, all 4 |
| `short_final` | drop the short final panel | RED + SIGSEGV (139), all 4 |
| `subview_ld` | sub-view built with rows, not the parent `ld` | RED, 9 of 16 per type |
| `getrs_perm_first` | permutation applied on the wrong side | RED, all 4 |
| `hole_pad` | remove the 48 KB pad | RED (**arithmetic layer only** — finding T3) |
| `pivot_metric` | `cabs1` → modulus | RED cfloat (5) / cdouble (4); correctly nothing for the real types |
| `laswp_left` | drop the left-hand interchange | RED, 9 of 16 per type |
| `getri_forward` | permutation trace walked forwards | RED, all 4 |
| `leaf_swap_right` | swap only columns ≥ k in the leaf | RED, 12 of 16 per type |
| `info_epsilon_floor` | flag `|pivot| < eps` | RED, all 4 |
| `piv_stride_nb` | pivot stride `nb` instead of the order | RED + SIGABRT (134), all 4 |
| `getri_perm_t` | transpose the permutation | RED, all 4 |

**Two more breaks written in the repair pass**, for the two wrong answers fixed
here — each applied, the `.so` rebuilt, and confirmed red:

| break | outcome |
|---|---|
| delete `if (!ctx.in_order()) ctx.wait();` from both `getrf` tiers | RED — `4682` of 1,638,400 CTA items and `4370` of 491,520 blocked items returned the caller's own `-12345` |
| delete `if (s.backend == Backend::NETLIB) return false;` from all three tables | RED — 5 assertions across `getrf`(both tiers), `getrs`, `getri` |

**Two breaks were re-run after the repair** to confirm the record still holds
with the edited files: `getri_perm_t` (63 → 55 passed, 8 failed) and `laswp_left`
(63 → 27 passed, 36 failed). All 15 break anchors were verified to still match
exactly once in both directions.

### Findings the breaks produced (not confirmations)

* **T1 — the test file shipped its own blind guard, and only a break found it.**
  `getrs_forward` first turned **NOTHING** red. The fixture permuted rows by a
  **reversal**, which is its own inverse, so `F = F⁻¹` and the transposed arm
  (“the same list walked backwards”) returns the identical answer walked
  forwards. Three direction tests were unfalsifiable on every scalar type while
  reading as the file's strongest. Fixed with a cyclic shift plus an
  `interchange_is_involution()` assertion at every direction-sensitive use. The
  general rule: *a test of an inverse operation is vacuous on any self-inverse
  instance, and self-inverse instances are exactly the tidy ones an author
  reaches for.*
* **T2 — `max |L| ≤ 1` is the WRONG partial-pivoting oracle for complex.** LAPACK
  selects on `cabs1`, and `cabs1(z) ≤ √2·|z|`, so a correct `zgetrf` returns
  `|L|` up to √2 — measured at 1.051 on the first random cfloat matrix. The
  metric-aware form is strictly stronger and turns the *ordinary* complex sweeps
  red, where the kernel campaign had needed an adversarial probe matrix.
* **T3 — the 48 KB hole does not reproduce for this kernel, and the test says
  which half knows it.** Removing the pad turns the **arithmetic** layer red for
  every in-band row and every type and leaves the **launch** layer green: the
  49,152 B resident launch succeeds without the pad. That agrees with the
  attribution (`sycl::reduce_over_group` alone reopens the hole; this body uses
  only `permute_group_by_xor`). The pad is defensive; the arithmetic assertion is
  the guard with teeth. Both layers are `EXPECT`, not `ASSERT`, precisely so the
  first cannot mask the second.

---

## 6. The repair pass: what was confirmed, what was refuted

Eleven review findings were triaged. **Three wrong-answer/crash findings were
confirmed and fixed; two performance findings were confirmed as descriptions of
the code and then REFUTED as improvements by measurement.**

### Fixed (wrong answer / crash)

1. **The `info` zero-fill raced the panel that reads it, in BOTH native `getrf`
   tiers.** `getf2_panel_device` *reads* `info[b]` to keep first-failure-wins
   across panels, so the fill is a read-after-write dependence, not a pure
   output; on an out-of-order queue (public API) the panel read the caller's
   pre-call garbage, never recorded the real failure, and wrote the garbage back.
   6,979 of 1,638,400 items on the CTA tier and 3,743 of 983,040 on the blocked
   tier returned the caller's own `-12345`. Fixed with the guard every other
   dependent boundary in the family already carried; re-measured **0 wrong of
   1,638,400 and 0 of 983,040**, and guarded by a new test in `getrf_tests.cc`.
2. **`supports()` never gated on `s.backend`, so `Backend::NETLIB` on a GPU queue
   could select the native arm.** The native kernels write/read **packed int32**
   in the caller's `int64` pivot span; netlib writes and reads **genuine int64**.
   Measured before the gate: `‖A·C − I‖_F / n = 5.32e-01` with `info == 0`,
   against `5.15e-07` when both arms agree — silent, and invisible to the suite
   because its NETLIB rows run on a CPU queue. One predicate in each of the three
   tables, plus a new `RouteLuPivotFormat` test with a backend axis (the axis the
   route tests did not have). ROCm is explicitly still admitted, because it packs
   int32 like CUDA.
3. **PRE-EXISTING, in `cublas.cc`: the `batch_size <= 1` `getrs` arm passed the
   raw `int64` pivot pointer to `cusolverDnXgetrs`** while every `getrf` in the
   tree writes packed int32. `getrf` then `getrs` at batch 1 — the exact sequence
   `linalg::solve` performs — aborted with `CUDA_ERROR_ILLEGAL_ADDRESS` (exit
   134). The two-arm split bought nothing; `cublas?getrsBatched` is correct at
   `batchCount = 1` and reads the format that was actually written. Now
   `‖A·X − B‖/‖B‖ = 1.20e-07` at batch 1.

### Confirmed as descriptions, REFUTED as improvements

4. **The rank-1 update's two runtime integer divisions.** The description is
   exact: `e % mm` / `e / mm` with a loop-variant `mm` really are runtime 32-bit
   divisions in the innermost loop of both residencies. The proposed fix — an
   equivalent power-of-two (row, column) split, division-free, still coalesced,
   still saturating, with `U(k,j)` hoisted into a register — was implemented and
   swept: **geomean 0.936× for float `getrf`, 0.976× for cdouble, worst cells
   0.829–0.833×** (float n=128: 3.938 → 4.746 ms at batch 2048), 0 discarded, 0
   route changes. The reason is trip counts, not arithmetic: at the resident
   tier's shape `mm ≈ wg`, so the split gives every work-item an inner loop of
   trip count **one**. Reverted; the negative result is recorded at the site. The
   same transform applied to the resident stage/store loops (64-bit divisions
   there) was part of the same regression and was reverted with it.
5. **The `getrf_leaf_wg` cap of 512 against a device maximum of 1024.** Again the
   description is exact, and the original sweep really did stop at its own
   boundary. Raised to 1024 and re-swept over all 156 saturation cells:
   **geomean 0.974×, `getrf` cdouble 0.939× and float 0.960×, worst 0.814–0.837×**.
   The prediction was *right about the one shape it aimed at* — the blocked
   driver's global panel leaf at n=2048 improves, by 0.6–1.2 % — and wrong about
   the knob, because the same function serves the resident tier, where 1024
   work-items on an order-256 tile lose 16–19 %. Reverted and recorded.
6. **`getri`'s `range<2>(batch, n²)` zero-fill and its two 64-bit divisions.**
   Replaced with `range<3>(batch, n, n)` — division-free, identical coalescing —
   and measured **1.000 geomean over the 78-cell `getri` sweep, 0.874× at the
   largest fill in it**. Reverted and recorded.
7. **`getri`'s hardcoded `wg = 256`.** Derived from `n` instead. It is **neutral**
   (0.9999 geomean, spread 0.982–1.023 = noise), and the comment at the site says
   so. Kept because it costs nothing and because a device with a small
   `MAX_WORK_GROUP_SIZE` should not be handed a width chosen for this one — *not*
   kept as a performance claim.

### Fixed (hygiene)

8. **Twelve scaffolding-era STATUS comments asserted the exact opposite of the
   shipped state**, in the files a reader consults to reason about routing —
   `factorization.cc` (“NO NATIVE LU KERNEL IS LINKED… returns 0… all return
   false”, “NEITHER NATIVE ARM CAN FIRE TODAY”, two “Unreachable today”),
   `route_getrf.hh`, `route_getrs.hh`, `route_getri.hh`, the three
   `src/backends/*_route.hh` builders, the three `src/extensions/*_native.hh`
   headers, and one comment in `route_vocabulary_tests.cc`. All rewritten to the
   measured state.
9. **A mis-costed decision note.** `getrs_native.cc` declined the collapsed
   gather on the ground that it costs “67,371,008 B at n=2048, nrhs=64,
   batch=32” — but nrhs=64 is the case the gather *wins*; at the nrhs that
   decides (1) the same buffer is 262,144 B. The conclusion stands for the other
   reason the file gives; the number that carried it did not.

### Confirmed and NOT fixed (recorded)

10. **`lu_laswp`'s nd_range is `range<2>(batch, ncols)`, i.e. batch-only at
    nrhs = 1** — 32 work-items at n=2048, batch=32, on a 128-SM device. This is
    the recurring BatchLAS defect in a shipped kernel. It is *structural*: for a
    fixed column the interchanges are a serial dependent chain, and the only
    parallel alternative is the collapsed gather, which needs an out-of-place
    RHS. Priced at 26 % of the cell at float n=512 and 11 % at n=2048, and
    13–63 % of native `getrf` by profile. Left as the routing step's problem,
    with the cost stated at the site.
11. **`getrf_panel_factorize` re-queries `LOCAL_MEM_SIZE` and
    `MAX_WORK_GROUP_SIZE` once per block step** — 128 device queries per n=2048
    `getrf`. Real, unmeasured, and fixing it changes an exported signature.
    Recorded rather than fixed.
12. **`MatrixView::data_ptrs(ctx)` re-runs `init_data_ptr_array` unconditionally**,
    which is a submit plus a blocking `.wait()`, so a vendor-routed `trsm`/`gemm`
    inside the blocked driver costs two host drains per panel. The root cause is
    in `matrix.hh` and is a known open bug the campaign works around rather than
    fixes (WP3–WP5 did the same).

---

## 7. What WP6 did NOT settle

* **`preferred()` has no window for any of the three ops.** That is the whole
  routing question, and it is open. `getrf_tests`'s
  `RouteTableAndTheVendorFreeFallback` currently asserts the *opposite* — that a
  vendor-present build routes to the vendor at every shape — which is exactly
  what would fire if a window landed without a grid. **That assertion must be
  replaced by the routing step, not deleted.**
* **The highest-value open measurement:** cuBLAS does the whole small-n
  factorisation in one fused kernel; native has a one-kernel arm (the CTA tier)
  and `double`'s tier hook routes n > 32 *away* from it. Whether widening the
  double CTA window closes the 0.27× at n=128 is unmeasured.
* **`native_tier_preferred()` is covered synthetically** in
  `route_vocabulary_tests.cc`, not against the real device. `getrf_tests` asserts
  only that the real builder reports non-zero `cta_max_n` and
  `blocked_available`; the tier split is *visible* in a coverage capture but not
  asserted.
* **`Backend::NETLIB` on a GPU queue is now gated but not exercised end to end**
  by `getrf_tests` — that file's fixture skips every NETLIB row because its queue
  is a CPU queue. The gate is guarded by a synthetic route test and by a
  standalone probe, not by a device test.
* **Two of the fourteen breaks are red by crash** (SIGSEGV / SIGABRT), so they do
  not demonstrate *which* assertion would have caught them.
* **Residual tolerances are `c·n·eps` with `c ∈ [200, 800]`**, not tightened
  against a measured error distribution. No break in the record was caught by a
  tolerance — every one was caught by an equality or structural assertion.
* **The complex GEMM deficit is untouched**, by design: `route_gemm.hh` returns
  false for complex and the register ladder is float-only. WP6 recorded it and
  did not fix GEMM.
* **Comment staleness of the same class survives in WP5's files**
  (`src/backends/geqrf_route.hh`, `include/batchlas/blas/dispatch/route_geqrf.hh`)
  — identical wording, identical falsity, left alone as out of WP6's scope.

---

## 8. Reproducing

```bash
# both builds
cmake --build build -j 32
cmake --build build-novendor -j 32

# the LU suites
CUDA_VISIBLE_DEVICES=0 ./build/tests/getrf_tests
CUDA_VISIBLE_DEVICES=0 ./build-novendor/tests/getrf_tests

# the gates
cd build && ctest -L "blas|ortho"
cd build-novendor && ctest -LE slow

# one break, end to end (rebuilds the .so)
experiments/wp6_lu/tests/run_break.sh piv_base_zero

# the route gate
scripts/route_diff.sh capture build   wp6-v
scripts/route_diff.sh compare wp6-before-v wp6-v

# the register gate (both libraries; the default target does not contain these kernels)
scripts/register_probe.sh out.log '' batchlas_extensions_cta
scripts/register_probe.sh out.log '' batchlas_extensions_factorization

# the benchmark
experiments/wp6_lu/bench/build_v.sh && experiments/wp6_lu/bench/build_nv.sh
CELLFILE=experiments/wp6_lu/bench/sat_cells.txt \
  experiments/wp6_lu/bench/run_cells.sh out.csv lubench6_nv none
```

Nothing in this tree commits a profiler capture, a trace JSON or a compiled
harness: `experiments/wp6_lu/.gitignore` and one per subdirectory keep the
classes out, and the committed set is 182 files totalling 1.6 MB with no file
above 200 KB.

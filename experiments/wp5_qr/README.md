# WP5 — QR (`geqrf` + `orgqr`): what shipped, what it costs, and what is still open

This is the top-level record for WP5. It indexes the three phase write-ups and
then states the things a reader needs that none of them can state alone: the
final measured verdicts, the repair pass applied on top of them, every
deliberate break and its outcome, and the questions WP5 did **not** settle.

| directory | phase | what is in it |
|---|---|---|
| [`baseline/`](baseline/README.md) | 0 | the vendor target, and the two design questions settled before a line of kernel was written |
| [`kernels/`](kernels/README.md) | 1 | what was built, the correctness harness, 5 reference breaks + 7 kernel breaks |
| [`bench/`](bench/README.md) | 2 | the measured grid against cuSOLVER/cuBLAS, the nsys splits, the native-internal tier sweep |

Everything timed here is **GPU 1 of this box** (2× RTX 4090, sm_89, 128 SMs),
`CUDA_VISIBLE_DEVICES=1`, `WARM_S=1.5`, medians of interleaved A/B, cells with
relative sd > 10% discarded. Nothing was ever timed under
`BATCHLAS_KERNEL_TRACE`.

---

## 1. What ships

**Two native `geqrf` tiers and one native `orgqr`**, plus the routing scaffolding
that did not exist for either op before WP5 (`geqrf` and `orgqr` were pure vendor
facades with no route resolution at all).

- `src/extensions/geqrf_cta_device.hh` — **all** of geqrf's device code. A
  LAPACK-faithful `larfg` and one `geqr2_panel_device` body written against a
  `Tile` abstraction, instantiated twice: against a `local_accessor` (the CTA
  tier, whole matrix resident) and against a raw global pointer (the blocked
  tier's panel leaf). One algorithm, two residencies.
- `src/extensions/geqrf_cta.cc`, `geqrf_blocked.cc` — the two drivers. The
  blocked one is a right-looking WY schedule whose trailing GEMMs go through the
  **injected routed** `gemm`, never `gemm_custom` from the kernel TU.
- `src/extensions/orgqr_blocked.cc` — `orgqr` is `ormqr` applied to an identity,
  through an injected routed apply.
- `src/extensions/larft_wy.hh` — `larft` + `pack_v`, lifted verbatim out of
  `ormqr_blocked.cc` so both ops share one copy.
- `include/batchlas/blas/dispatch/route_{geqrf,orgqr}.hh`,
  `src/backends/{geqrf,orgqr}_route.hh` — the route tables and shape builders.
- `tests/geqrf_tests.cc` (new, `blas` label) and an extension to
  `tests/orgqr_tests.cc`.

### Route neutrality, precisely

`preferred()` is **false everywhere** for both ops, so a **vendor-present build
takes cuSOLVER for every shape**. Only a vendor-free build, an explicit
`BATCHLAS_GEQRF_ROUTE` / `BATCHLAS_ORGQR_ROUTE`, or a direct entry point reaches
the kernels. The repair pass (§3) added one exception that is *not* a
neutrality break: `RouteTable<Op::geqrf,T>::native_tier_preferred` chooses
between the two **native** tiers and is consulted **only** on the vendor-free
walk, so it moves nothing in a vendor-present build. The route-diff gate proves
it: one substitution and one addition, both inside the new test, zero
vendor-present decisions moved.

---

## 2. The measured verdict, losses included

Public API, same program linked against `build/` and `build-novendor/`, so
"vendor-free" is the **build** and not a forced route.

**Headline: `geqrf` geomean 3.24x (25/36 order cells won), `orgqr` geomean 7.85x
(31/36).** The geomean is the least useful number in this document, because the
two ops move in *opposite* directions with order.

| | small n | large n |
|---|---|---|
| `geqrf` | **0.21–0.78x (loss)** at n=32 | 15–181x at n=2048 |
| `orgqr` | 12–123x at n=32 | **0.31–0.46x (loss)** at n=2048 |

**Do not quote the 181x as "faster than cuBLAS".** `geqrfBatched` is
*unsaturated* at n ≥ 512 — float n=1024 costs 1204 ms at batch 8 and 2276 ms at
batch 256 — so that ratio is mostly the vendor's launch behaviour. The honest
ceiling-to-ceiling number is native **3564 / 1079 / 1683 / 132 GFLOP/s**
(float/double/cfloat/cdouble) against cuBLAS's ~380 / 200 / 205 / 108, i.e.
**9.2x / 5.4x / 8.2x / 1.2x**.

### The losses, stated plainly

- `geqrf` cdouble n ≤ 256 (0.33–0.84x), double n ≤ 96 (0.21–0.79x),
  float/cfloat n = 32 (0.71–0.78x).
- `orgqr` n = 2048 float/cfloat/cdouble (0.31–0.47x — **genuine**), and
  cfloat/cdouble n = 1024 (0.78–0.88x, still climbing with batch).
- **Batch is the dominant axis for FP64, not order.** `geqrf` double and cdouble
  at n=64 go from a 4–5x *win* at batch ≤ 128 to a **flat 0.53x loss from batch
  2048 to 16384**. Both arms are launch-bound below batch ≈ 512, so the 5x is
  overhead-vs-overhead and **0.53x is the saturated number**. Quote the 0.53x.

### Where the time goes

nsys on winning *and* losing cells, because a split from a winner does not
explain a loss:

| | float n=1024 (41x) | cdouble n=1024 (4.3x) | cdouble n=256 (**0.84x loss**) | double n=64 (**0.53x loss**) |
|---|---|---|---|---|
| transposed GEMM (Tiled16) | **46.6%** | **69.7%** | **51.3%** | — |
| NN GEMM (128x128 / 64x64-wide) | 24.3% | 21.4% | 15.6% | — |
| `larft` + `pack_v` | 19.7% | 6.5% | 22.3% | — |
| panel factorisation | 9.3% | 2.5% | 10.6% | **100.0%** |

Two inherited claims are **corrected** by this table:

1. **"WP5 will be decided by the panel factorisation" is wrong for the shipped
   blocked driver.** The panel is 9.3% (float) / 2.5% (cdouble) at n=1024 and
   10.6% at the losing cdouble n=256 cell; the trailing update is ~71%. The
   63x-headroom estimate behind that claim was made against a *routed* trailing
   update, and the shipped one lands on **Tiled16** for its largest GEMM. The one
   place the panel *is* the whole cost is the CTA tier — 100% one kernel — and
   that is exactly where FP64 loses.
2. **The complex deficit is outside WP5, as briefed.** `route_gemm.hh:113-114`
   returns false for complex and `gemm_kernels.cc:471` keeps the register ladder
   float-only, so every complex trailing GEMM lands on Tiled16. Closing cdouble
   needs a **transposed wide-scalar GEMM** — WP2 territory. Recorded, not
   attempted.

---

## 3. The repair pass

Applied on top of the three phases above, after a review. Each item states what
was measured, not what was expected.

### 3a. The vendor-free build was taking the *slower* of its own two native tiers

`kGeqrfOrder` lists `{Native, CTA}` first, and with `preferred()` all-false the
vendor-free walk (`route_resolve.hh`) returned the first **supported** native
route — i.e. CTA anywhere the tile fits SLM, which is square n ≤ 155 for float
and n ≤ 110 for double on this box. The tier sweep had already measured the
blocked driver *ahead* of CTA from n ≈ 104 (float) and n ≈ 48 (double). So the
one build this work package exists for shipped a pure loss with the better route
already linked in.

**This could not be fixed with `preferred()`**, and that is the interesting part.
`preferred()` is consulted by the loop *above* the vendor-free walk, which runs
regardless of `vendor_available` — so a window written to fix the vendor-free
tier choice necessarily also moves vendor-**present** traffic, including at
shapes where cuSOLVER beats both natives. The two questions genuinely differ, so
there is now a third, **optional** predicate:

```
preferred(r,s)            -> "r is the best route available, vendor included".
                             Flipping it moves the default in EVERY build.
native_tier_preferred(r,s)-> "among the NATIVE routes that can serve s, r is the
                             better one". Consulted only where there is no vendor.
```

`route_resolve.hh`'s vendor-free walk is now two passes; the hook is detected
with a `requires` expression and **defaults to `true`**, which makes the first
pass identical to the second for every table that does not declare it. That is
why this is not a flag day for gemm, trsm, potrf or gesvd — and the route diff
confirms it empirically.

**Measured, same binary, same session, interleaved, three reps, vendor-free,
with the forced arm's resolved route printed and verified to read `native:cta`
on every row** (this is the shipped default vs. what the default *was*):

| type | n | default (blocked) | forced `cta` | gain |
|---|---|---|---|---|
| float | 112 | 9.15 ms | 11.98 ms | **1.31x** |
| float | 128 | 9.97 ms | 15.39 ms | **1.54x** |
| float | 155 | 16.39 ms | 22.73 ms | **1.39x** |
| double | 64 | 27.42 ms | 29.72 ms | **1.08x** |
| double | 80 | 18.49 ms | 24.07 ms | **1.30x** |
| double | 96 | 23.86 ms | 32.75 ms | **1.37x** |
| double | 110 | 30.55 ms | 42.52 ms | **1.39x** |

And the check that the window did **not overshoot** — below the crossover,
forcing `blocked` must be slower:

| type | n | default (cta) | forced `blocked` |
|---|---|---|---|
| float | 64 | 2.95 ms | 5.70 ms |
| float | 96 | 5.00 ms | 6.02 ms |
| double | 32 | 10.87 ms | 11.42 ms |
| double | 48 | 19.34 ms | **19.02 ms** ← a tie, see below |
| cfloat | 96 | 10.64 ms | **10.72 ms** ← a tie |
| cdouble | 64 | 59.18 ms | 113.79 ms |

**Two cells are honest ties and are resolved in CTA's favour deliberately.**
double n=48 (blocked ahead 1.7%) and cfloat n=96 (blocked ahead 0.8%) are inside
this harness's run-to-run resolution. They go to CTA for a reason that is not
timing: **CTA's workspace is zero** — the tile is local memory and `tau` is the
caller's span — while the blocked driver allocates `m*nb*batch` of V plus T plus
the WY scratch.

### 3b. Two uncoalesced fills, and one needlessly strided operand

`OrgqrIdentityKernel`, `OrgqrCopyBackKernel` and `pack_v_panel_batched` all
launched `sycl::range<3>(batch, rows, cols)` and read `idx[2]` as the **column**.
`sycl::id<3>` makes dim 2 the fastest-varying index and every operand here is
column-major, so a 32-lane warp touched 32 sectors instead of 4 — on both sides
for the copy-back and the pack. The repo's own fast kernels already use the
opposite convention (`src/matrix.cc:400`, `register_128x128.hh:127`: dim 2 is the
**row**). Swapped.

Separately, the blocked driver handed `V` to both trailing GEMMs with the
**parent** leading dimension `m` although `V` lives in scratch the driver owns
outright. Late panels therefore gave the GEMM a short operand with a long stride
— at j0=992 of a 1024-column factorisation, a 32-row panel whose columns were
4 KB apart — which is the recorded "native GEMM collapses on strided `ld`"
shape, for no reason. `V` is now packed contiguously at `ld = mp`.

**Measured as a clean same-session A/B** (the fixes reverted, rebuilt, measured,
restored — not against a prior agent's table; all relsd ≤ 0.11%):

| op | cell | before | after | gain |
|---|---|---|---|---|
| `orgqr` | float n=512 b=512 | 45.03 ms | 37.65 ms | **1.196x** |
| `orgqr` | float n=1024 b=128 | 73.75 ms | 65.94 ms | **1.118x** |
| `orgqr` | float n=2048 b=32 | 140.08 ms | 132.26 ms | 1.059x |
| `orgqr` | double n=512 b=512 | 135.68 ms | 131.15 ms | 1.035x |
| `orgqr` | double n=1024 b=128 | 243.93 ms | 239.92 ms | 1.017x |
| `orgqr` | cdouble n=1024 b=128 | 2043.19 ms | 2032.49 ms | 1.005x |
| `geqrf` | float n=512 b=512 | 30.23 ms | 28.68 ms | 1.054x |
| `geqrf` | float n=1024 b=128 | 50.99 ms | 49.57 ms | 1.029x |
| `geqrf` | double n=1024 b=128 | 169.37 ms | 168.84 ms | 1.003x |
| `geqrf` | cdouble n=1024 b=128 | 1386.81 ms | 1384.39 ms | 1.002x |

16 of 16 cells improved or were neutral; nothing regressed. The gain is
concentrated exactly where the profile predicted it (the two orgqr movers were
20.7% of GPU kernel time at float n=1024) and is small everywhere the cell is
GEMM-bound — which is most of the FP64 and complex grid.

### 3c. 32 device entry functions that could never launch

`larft_forward_columnwise_batched` took `use_device` as a **runtime bool**, so it
instantiated *both* implementations for every `(Tag, T, WG)`. `geqrf` passes a
literal `false`, so `larft_forward_columnwise_wg_device<GeqrfWyTag, …>` was 32
entry functions — 4 types × 4 work-group rungs × 2 (base and `_with_offset`) —
compiled, ptxas'd and device-linked into `batchlas_extensions_cta`, the
slowest-linking library in the tree, and never launched (nsys confirms: no
`(bool)1` variant appears in any WP5 run). They included the highest-register
kernel in the whole WP5 set (cdouble, 90 registers, 208 B stack frame).

`UseDevice` is now a template parameter with a runtime-selecting wrapper retained
for `ormqr`, whose choice really is a getenv. Measured:
**880 → 848 entry functions, device link 125.45 s → 116.63 s.**

### 3d. One latent consistency defect

`orgqr_buffer_size` gated "did a native tier fire?" on `native_need == 0` — the
exact defect the same change deliberately removed from `geqrf_buffer_size` 170
lines above, with a 12-line comment explaining why a zero workspace is a
legitimate answer. Unreachable today only because `orgqr_blocked_layout`
unconditionally allocates `m*n*batch`; reachable the moment a specialised
in-place `orgqr` lands, which both `orgqr_native.hh` and `orgqr_blocked.cc`
explicitly contemplate. Now uses `native_fired`, matching its sibling.

---

## 4. Every deliberate break, and its outcome

Breaks from phases 0–2 are in `kernels/README.md` (5 reference + 7 kernel) and at
the bottom of `tests/geqrf_tests.cc` (9, including the two that turned nothing
red). The repair pass added four more. Each was applied to the **source**, the
affected target **rebuilt**, and both suites re-run.

| break | what was deleted | outcome |
|---|---|---|
| **BR1** | LAPACK's beta sign choice in `geqrf_larfg_scalars` (`(alphr>=0) ? -nrm : nrm` → `? nrm : -nrm`) | **RED**, `ConventionMatchesReferenceLapackWithoutAVendor`, **all four types**, in **both** builds |
| **BR3** | the division arm of the reciprocal guard (`vfactor = use_mul ? r : d` → always `r`) | **RED**, `SubnormalScaleColumnsTakeTheDivisionPath`, all four types, and **nothing else** |
| **BR4** | `RouteTable<Op::geqrf>::native_tier_preferred` entirely | **RED**, `NativeTierTieBreakPicksTheFasterNativeVendorFree`, float and double (the two types with a measured crossover), and nothing else |
| **BR4b** | moved the same window **into `supports()`** — the four-times-shipped defect | **RED**, and on the *intended* assertion: `Tbl::supports(cta, *sh_hi)` was false. Proves the test's second half is not vacuous |

**BR1 is the one worth carrying forward.** It is the same break as the earlier
kernel break K3, whose recorded outcome was "qr, orth, qrQ ALL GREEN for every
type" — a residual test cannot see a convention. Before the repair pass the
*only* test that could see it was `NativeFactorMatchesTheVendorElementwise`,
which opens with `GTEST_SKIP` in a vendor-free build. So **in the build this
whole work package exists for, the real-scalar half of geqrf's drop-in contract
had no guard at all.** The new test compares against an independent host `xGEQR2`
written from the LAPACK reference and closes it: BR1 is red for all four types in
`build-novendor`.

Two secondary findings from running BR1:

- Under BR1 a *few* residual tests did go red (`BlockedResidualAndOrthogonality`
  float/double, `ShortFinalPanel` double) — because dropping the sign choice
  causes cancellation in `alpha - beta` on some data. That is **data-dependent
  and type-dependent**: the CTA tier's residuals stayed green, and both complex
  types stayed green. A residual test catches this break *sometimes*; the
  convention test catches it deterministically.
- Writing the reference exposed a second asymmetry worth knowing:
  **`zgeqr2` applies `conj(tau)`, not `tau`** (`CALL ZLARF(..., DCONJG(TAU(I)), ...)`),
  because reducing from the left applies `H^H`. The first version of the
  reference used `tau` and disagreed with the kernel by 1–4% for cfloat/cdouble
  while being exact for float/double — the same signature as kernel break KE. For
  a real `T` the conjugate is the identity, which is why this class of defect is
  invisible to half the type list by construction.

---

## 5. What WP5 did NOT settle

Recorded so the next reader does not have to re-derive them, and so nothing here
is mistaken for a claim.

1. **The vendor-present default is still cuSOLVER, everywhere.** `preferred()` is
   all-false for both ops. The measured 3.24x / 7.85x geomeans are therefore
   **unrealised** in the default build. Flipping a cell is gated on more than a
   kernel-level win — this tree has turned a 2.16x kernel win into an 11% gesvd
   loss — and needs an end-to-end harness (`ortho_benchmark`, a `syev` path) that
   WP5 did not run. **That is the single largest piece of value left on the
   table by this work package.**
2. **The tier crossover is measured on SQUARE shapes only.** `native_tier_preferred`
   gates on `n`, which is a *mechanism* argument (CTA's serial cost is its
   per-reflector chain, `k = min(m,n)` long, and `geqrf_panel_wg` derives the
   work-group from `n` alone), not a measured one. A tall skinny panel — m=512,
   n=32, float — is CTA-eligible and has **no measured cell**. It is left on CTA
   deliberately.
3. **cfloat 97..110 is extrapolated, not measured.** CTA's margin there is
   collapsing (3.171 → 2.093 → 1.253 → 1.079 → ~1.008 at n=96) and the CTA
   capacity ends at 110. It is the first place to look if this is re-measured.
4. **`geqrf_cta`'s capacity has no blocks-per-SM term.** It is a pure byte budget
   (the whole ~97 KB allowance), so above ~50 KB the tile forces one work-group
   per SM — 256 of 1536 threads — and the per-reflector barrier chain has nothing
   to overlap with. The float crossover lands *exactly* where that happens
   (n=96 → 36,864 B → 2 blocks/SM, CTA ahead 1.294; n=112 → 50,176 B → 1
   block/SM, CTA behind 0.821). `native_tier_preferred` now routes around this;
   it does not fix it. A `m*n*sizeof(T) <= local_mem/2` shape for the *capacity*
   would track the measured crossover almost exactly and is untried.
5. **`geqrf_panel_wg(n, max_wg)` never looks at `m`.** It returns wg=256 for a
   column that may hold 64 elements, so at double n=64 at least 193 of 256
   work-items execute zero loop iterations and still pay a full 256-wide
   `reduce_over_group`. That kernel is **100.0% of GPU time** in the campaign's
   worst measured `geqrf` loss (double n=64, 0.53x, 187 GFLOP/s = 14.5% of FP64
   peak). The file's own comment says the number "is NOT a tuned number"; it is
   the number the shipped losing cell runs on. **Untouched by the repair pass** —
   it is a hot-kernel change needing its own correctness and measurement pass.
6. **`larft_forward_columnwise_wg_legacy` is 15.2% of `geqrf` float n=1024 and
   20.3% of the losing cdouble n=256 cell**, doing ~1/60 of the arithmetic of the
   GEMM next to it. Two mechanisms: the `(j,col)` loop re-reads two full V columns
   per pair (ib-fold redundant traffic, `O(ib²/2)` work-group barriers per panel),
   and the T-update triangular recurrence runs serially on `lid == 0` with 255
   lanes idle. For cdouble n=256 it costs **twice** what the panel factorisation
   it exists to accelerate costs. This body is *inherited* (moved verbatim from
   `ormqr_blocked.cc`), but WP5 is what put it on geqrf's critical path.
   Recorded, not rewritten.
7. **The `!dev_is_zero(r)` half of the reciprocal guard is not tested and may not
   be correct.** `r == 0` needs `|alpha - beta| > 2e323`, i.e. `alpha - beta` must
   itself have overflowed to `inf` — reachable at input ~1e308. There `vfactor`
   becomes `inf` and `v` becomes exactly zero: finite, but not a correct
   reflector. The new test covers only the overflow-of-reciprocal half.
8. **Reaching the reciprocal guard at all requires subnormal input**, which is
   itself a finding: `|alpha - beta| >= s`, so the reciprocal misbehaves only
   below the smallest normal. At that magnitude the *input* carries few mantissa
   bits, so no tight residual is possible there and the new test asserts
   finiteness plus orthogonality at a loose, explicitly justified bound.
9. **`geqrf_buffer_size` builds its shape twice and does 6 uncached SYCL
   `get_info` calls per API call** (once inside `geqrf_route`, once explicitly for
   the `supports()` tests; `geqrf_op_shape` reads `LOCAL_MEM_SIZE` and enumerates
   sub-group sizes, one of which heap-allocates). This lands on
   `band_reduction.cc:595` and `sytrd_sy2sb.cc:504`, which call `geqrf(...).wait()`
   once per step — `O(n²/kd²)` steps, ~500 for n=1024. Pure host overhead, no
   wrong answer, and **not measured**. Fix is to hoist the shape out of
   `geqrf_route` or memoize `Device::supports_sub_group_size`, which
   `queue-impl.cc:333-338` already invites. Measure before acting.
10. **CORRECTED BY THE ORCHESTRATOR -- three suites closed, not zero.** The
    claim originally recorded here ("no suite closed, and none will until WP9")
    came from reading `25 tests failed out of 55` as a PASS count. It is a
    failure count: the burn-down is **30 of 55**, against **26 of 54** before
    WP5. Diffing the two failing sets:

    | | |
    |---|---|
    | now passing, was failing | `backend_dispatch_tests`, `syev_two_stage_tests`, `sytrd_sy2sb_tests` |
    | newly failing | none |
    | plus | the new `geqrf_tests`, passing vendor-free (72/0) |

    The half that WAS right: the four QR suites themselves are still red, and
    they stay red until WP9, because every one carries `Backend::NETLIB` rows
    that no CUDA kernel can fix. So the brief's "worth up to FOUR suites" was
    wrong about WHICH suites -- the beneficiaries are three DOWNSTREAM algorithm
    suites that call `geqrf` on CUDA and could not run vendor-free at all
    before. Verified by re-running each directly in `build-novendor/`:
    `backend_dispatch_tests` 13/13, `syev_two_stage_tests` 20/20,
    `sytrd_sy2sb_tests` 2/2.

---

## 6. Gate results after the repair pass

| gate | result |
|---|---|
| `cmake --build build -j 32` | **exit 0** |
| `cmake --build build-novendor -j 32` | **exit 0** |
| `build/tests/geqrf_tests` | exit 0 — 168 ran, **80 passed, 0 failed** |
| `build/tests/orgqr_tests` | exit 0 — 24 ran, **24 passed** |
| `build-novendor/tests/geqrf_tests` | exit 0 — 168 ran, **72 passed, 0 failed** |
| `build-novendor/tests/orgqr_tests` | exit 1 — 12 passed / 12 failed, **all 12 Backend 6 (NETLIB, WP9)**; every CUDA row green |
| `ctest -L "blas\|ortho"` in `build/` | **100% passed, 0 failed of 22** |
| `ctest -L "blas\|ortho"` in `build-novendor/` | 16 failed of 22 — unchanged set |
| `ctest -LE slow` in `build/` | 1 failed of 55 — `lanczos_tests`, pre-existing and proven independent of WP5 |
| **`ctest -LE slow` in `build-novendor/` (burn-down)** | **25 FAILED of 55, i.e. 30 PASSING** — unchanged by the repair pass, but **up from 26 of 54 before WP5**. See §5.10: that 25 is a failure count and was misread as a pass count when this file was first written |
| register probe, `batchlas_extensions_cta` | 848 entry functions, **spill(entry) = 0**, spill(all) = 16 (all pre-existing gesvdj / ormqx / syev_cta complex-double) |
| register probe, `batchlas_extensions_factorization` | 424 entry functions, **spill(entry) = 0, spill(all) = 0** |
| `route_diff.sh compare wp5-structural wp5-repair` | 1 substitution + 1 addition, **both attributed to the new test**, zero vendor-present decisions moved |

Note that a single `-L "blas|ortho"` flag is used, not two `-L` flags: repeated
`-L` flags AND together and select **zero** tests while exiting 0.

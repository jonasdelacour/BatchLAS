# WP4 — native batched POTRF (Phase 1, the CTA kernel)

State of the work after the adversarial review and the repair pass. Everything
below was **executed on this box** unless it says otherwise; where a number is
arithmetic rather than a measurement it says so.

Sub-directories: `slm/` (step 0.2, the local-memory ceiling), `regbaseline/`
(step 0.3, the register probe and its baselines), `kernel/` (step 1.x notes).

---

## 1. What exists, in one paragraph

A native batched Cholesky that factorises a whole batch of order-`n` matrices in
one work-group each, entirely in local memory, with no vendor call. It serves
`n <= 155/109/109/77` (float/double/complex\<float\>/complex\<double\>) and both
triangles. Above that ceiling there is **still no native route** — the blocked
driver is Phase 2 and is not written. `preferred()` returns false everywhere, so
in a vendor-present build **nothing routes here**; the kernel is reachable only
by forcing `BATCHLAS_POTRF_ROUTE=cta`, by calling
`sycl_potrf::potrf_cta_dispatch<T>` directly, or by building without a vendor.

---

## 2. The local-memory ceiling (step 0.2), and the correction the repair made

`sycl::info::device::local_mem_size` reports **101,376 B** on this box. The
`49152` in `build/include/batchlas/device_limits.hh` is hardcoded by
`cmake/BatchLASDetectSYCL.cmake:44-45` for any `nvidia_gpu_sm_*` pattern and is
wrong here by 2.06× — W1 confirmed. The budget the kernel sizes from is
`local_mem_size - 4096` = **97,280 B**.

| `T` | ceiling `n` | bytes at the ceiling | first miss |
|---|---|---|---|
| `float` | **155** | 96,540 | 156 → 98,236 |
| `double` | **109** | 95,476 | 110 → 97,680 |
| `complex<float>` | **109** | 95,444 | 110 → 97,680 |
| `complex<double>` | **77** | 95,328 | 78 → 98,592 |

`tests/potrf_tests.cc:MeasuredFitCeilings` pins these against the
budget-parameterised query, so the assertion holds on any machine.

**The repair changed the formula and did not change the ceilings.** The slack
term was 64 B/matrix and **it was too small — measured, not argued**. At float
`n = 155` the raw accessor sum is 96,288 B, the old formula said 96,348, and
`ncu` reports `launch__shared_mem_per_block_dynamic = 96,408`
(`launch__shared_mem_per_block_static = 0`). So the launch asked for **60 B more
than the number `supports()`, `potrf_cta_max_n_for_slm()` and `p.fits` are all
computed from** — the stated invariant, inverted. The real runtime overhead for
these four accessors is 120 B. The term is now 256 B, which covers it with
margin and leaves all four ceilings unchanged.

On this box nothing failed, because the unrelated 4,096 B reserve left 872 B of
headroom. On a device whose budget lands within ~120 B above the formula value at
its ceiling, `supports()` would have advertised an order whose launch requests
more local memory than the device allows, arriving as a
`CUDA_ERROR_INVALID_VALUE` at enqueue rather than as the documented throw.

**Second correction, same family:** the advertised ceiling walked the *raw* size
while the launcher gates on the *hole-padded* size. The two are now the same
predicate. The `break` in that walk is load-bearing and documented in place:
`potrf_hole_padded` is **not monotone**, and `supports()` spells the capacity as
`order <= cta_max_n`, i.e. a contiguous range, so the ceiling must be the largest
`n` for which *every* order up to `n` launches.

### The 48 KB launch hole

Measured cold (`slm/scan_hole_boundary.csv`): a dynamic request in
`(49152 − static_shared, 49152]` fails with `CUDA_ERROR_INVALID_VALUE`. The pad
that steps over it is **currently inert and that was measured, not assumed** —
ptxas reports *no* `smem` field for any of the eight potrf instantiations, so
their static shared is 0 and the interval is empty. It stays because one
`reduce_over_group` added anywhere in the body reintroduces it, and the failure
mode is a cold-start error a warm test suite cannot see. **No automated test can
cover it**: the CUfunction attribute is sticky per process, so any earlier
>48 KB launch masks it.

---

## 3. Registers — the gate, passed

Probed on `batchlas_extensions_cta` via `regbaseline/regprobe_any.sh`. The stock
`scripts/register_probe.sh` hardcodes `batchlas_sycl.dir/link.txt` and would
report clean for code it never compiled.

| `T` | NB/TS | Scope | regs | frame | spill st/ld | callee spill | max `regs × WG` |
|---|---|---|---|---|---|---|---|
| `float` | 8/4 | SubGroup | 64 | 0 | 0/0 | 0/0 | 8,192 |
| `float` | 8/4 | WorkGroup | 56 | 0 | 0/0 | 0/0 | 14,336 |
| `double` | 8/4 | SubGroup | 94 | 0 | 0/0 | 0/0 | 12,032 |
| `double` | 8/4 | WorkGroup | 80 | 0 | 0/0 | 0/0 | 20,480 |
| `complex<float>` | 8/4 | SubGroup | 102 | 0 | 0/0 | 0/0 | 13,056 |
| `complex<float>` | 8/4 | WorkGroup | 92 | 0 | 0/0 | 0/0 | 23,552 |
| `complex<double>` | 8/2 | SubGroup | 128 | 0 | 0/0 | 0/0 | 16,384 |
| `complex<double>` | 8/2 | WorkGroup | 109 | 0 | 0/0 | 0/0 | **27,904** |

Three-condition gate (`frame == 0` **and** `0 spill` **and** `regs × WG <= 65536`)
passes everywhere with **2.35× headroom** at the worst cell. SubGroup runs at
`L = 32`, `G <= 4`, so `WG <= 128`; WorkGroup runs at `G = 1`, `WG = L <= 256`.

Diffed against the FOUNDATION 2 baseline (`regbaseline/batchlas_extensions_cta.tsv`):
**16 rows added, all potrf; zero non-potrf rows changed.** The unit's 16
pre-existing `complex<double>` spillers (gesvdj / ormqr / syev) are untouched,
which is why the gate is scoped to the potrf entries rather than asserted over
the whole unit.

---

## 4. Performance — the honest table

**Read this before planning Phase 2 or a `preferred()` window.** Measured at
`batch = 4096` (saturation), `experiments/gpu_guard.sh 0`, JIT-warmed, two
passes agreeing within a few percent. `vendor/native > 1` means native wins.

| `n` | float | double | complex\<float\> | complex\<double\> |
|---|---|---|---|---|
| 8 | **2.75×** | 1.07× | **1.79×** | **1.47×** |
| 16 | **1.88×** | 0.64× | 1.05× | 0.76× |
| 32 | **1.85×** | 0.87× | 0.91× | 1.00× |
| 48 | **1.26×** | 0.83× | 0.67× | 0.87× |
| 64 | 1.00× | 0.58× | 0.40× | 0.57× |
| 96 | 0.51× | 0.49× | 0.37× | — |
| 128 | 0.36× | — | — | — |
| 155 | 0.63× | — | — | — |

**The kernel loses to cuSOLVER over most of the range `supports()` advertises.**
It wins for float up to `n ≈ 64`, and at `n = 8` for every type. Above that it is
2–3× slower. Absolute efficiency at float `n = 155`, `batch = 4096`: 5.08 GFLOP in
4,046 µs = **1.26 TFLOP/s, about 2.7% of this card's ~47 TFLOP/s FP32**. `ncu`
gives the reason: `sm__warps_active` is **8.3%** at `n = 96/128/155` — shared
memory allows 1–2 resident blocks per SM and the occupancy limiters are
`shared_mem` 1–2 against `registers` 9–18, i.e. shared memory binds by 9–18×.

In a vendor-present build **none of this reaches a user**, because `preferred()`
is false everywhere. In a **vendor-free** build — the build WP4 exists for —
every `potrf` with `n` under the ceiling runs this kernel, so a vendor-free user
at float `n = 128` pays ~2.9× the runtime of the cuSOLVER build they replaced.

### The one performance change the repair made

The `L` ladder (work-items per matrix) was `Ntiles_0 <= 64 ? 32 : <= 256 ? 64 : 128`,
justified by a thread-limit argument that `ncu` refutes — the thread limit never
binds. It is now derived from the **elements** the first trailing update touches,
`Ntiles_0 · TS²`, at 24 elements per work-item, capped at 256.

A tile count is the wrong unit because a tile is `TS × TS` and `TS` is not
constant across the type ladder: `complex<double>` runs `TS = 2`, so for the same
order it has 4× the tiles at ¼ the work each, and a tile-count rule over-shoots
it by two rungs. The element count is `TS`-independent.

Measured, native µs, old ladder → new ladder:

| cell | old | new | gain |
|---|---|---|---|
| float n=48 | 131.9 | 117.2 | 1.13× |
| float n=64 | 274.5 | 242.5 | 1.13× |
| float n=96 | 1482 | 1162 | **1.28×** |
| float n=128 | 3224 | 3052 | 1.06× |
| float n=155 | 4838 | 4046 | **1.20×** |
| double n=48 | 639.4 | 602.1 | 1.06× |
| double n=96 | 5976 | 4938 | **1.21×** |
| cfloat n=48 | 309.9 | 259.7 | 1.19× |
| cfloat n=96 | 3959 | 3001 | **1.32×** |
| cdouble n=32 | 642.4 | 680.0 | **0.94× (a loss)** |

Across 21 cells the rule picks the measured best or a cell within 1% of it in 19,
and is within 5.5% in the other two (`complex<float>` n=64, `complex<double>`
n=32). The `complex<double>` n=32 cell is a genuine ~5% regression, accepted for a
rule that is 1.06–1.32× better in eleven others. **24 is a fitted constant** —
the one number on that line that is not derived — pinned by this grid and nothing
else. `route_diff` confirms this moved **zero** routing decisions.

### A measured negative result: `NB = 16` is worse

The review argued from `ncu` that the register cost of `NB = 16` is free above
`n ≈ 32` (correct — registers do not bind there) and predicted a time win from
halving the panel count. **Built it and timed it: `NB = 16` is slower in 18 of 20
cells**, by 2.70× at float n=8, 2.04× at float n=96, 1.05–1.31× elsewhere. The
only improvement is double n=16 (1.08×). The proposal to instantiate both `NB`
and choose at launch is therefore **declined**: it would double potrf's device
instantiations from 8 to 16 in a build this repository documents as
device-link-bound, in order to select between a winner and a loser.

---

## 4a. The vendor-free build — the deliverable, verified by execution

Every prior report on this work package said "no vendor-free build was configured
or run", and the vendor-free fallback was argued only at the pure-table layer.
**It has now been built and run.**

`cmake -S . -B build-novendor` (`BATCHLAS_ENABLE_VENDOR_BLAS=OFF`, CUDA still on)
configures, compiles and links cleanly with the potrf sources in place, and:

    ctest target potrf_tests, vendor-free build:  104 tests ran, 50 PASSED, 0 FAILED

That includes `FacadeReachesTheCtaKernel`, which in this build has no vendor to
fall through to — so the facade demonstrably runs the CTA kernel.

The claim WP4 exists to establish is stronger than "the tests pass": it is that an
**ordinary, unforced** `potrf` call now works where it used to throw. Probe
(`BATCHLAS_POTRF_ROUTE` explicitly unset, plain `potrf<Backend::CUDA, float>`
through the facade, tridiagonal SPD input whose exact `L(0,0)` is `sqrt(4) = 2`):

| `n` | vendor-free result |
|---|---|
| 8 | **OK**, `info = 0`, `L(0,0) = 2.000000` |
| 48 | **OK**, `info = 0`, `L(0,0) = 2.000000` |
| 155 | **OK**, `info = 0`, `L(0,0) = 2.000000` |
| 156 | **throws** `NoRouteError: no route for potrf<float> ... built without cuSOLVER` |

That is exactly the intended shape. Below the CTA ceiling a vendor-free build now
factorises natively and correctly; one order above it, where the blocked driver
would be needed, it still throws — because **Phase 2 does not exist**. The
`n = 156` row is not a defect; it is the honest boundary of what shipped.

---

## 5. Tests — what they cover, and what they do not

`tests/potrf_tests.cc`, registered in **both** `TEST_TARGETS` and
`BATCHLAS_TEST_LABELS_blas`. 104 tests, **50 pass**, 54 skipped (the NETLIB
backends by the GPU-only `SetUp`, plus the complex-only test on real types).
0.35 s.

**The oracle is never the vendor.** Every numerical test either computes a host
multiply-back residual `‖L Lᴴ − A‖_F / ‖A‖_F` in the test file, compares two
native launches bit-for-bit, or uses a planted `L₀ D L₀ᴴ` whose exact failure
column is known analytically. Ten of thirteen tests call
`potrf_cta_dispatch<T>` directly — a call no vendor can serve.

### Covered

- Residual, **both triangles**, all four types, `n ∈ {1,2,3,7,8,9,15,16,17,31,32,33,47,63,64,65,108..111,cap−1,cap}`.
- The ceiling is hard: `cap+1` is refused by `supports()` and throws at dispatch.
- The other triangle is neither written (bit-exact poison compare) nor read (NaN poison), at **both an even and an odd `n`** — the pad-row parity.
- `G > 1` packed launch agrees bit-for-bit with the same matrices launched solo, and the test now **asks** whether `G > 1` was reached rather than claiming it in a comment.
- Exact `info` index across panel boundaries, **both triangles**; first-failure-wins; `info` at `batch = 64` with three non-PD items at different columns; failed items stay finite.
- Complex: asserted non-trivial imaginary part, exactly-real factor diagonal, conjugation actually changes the factor, `imag(diag(A))` ignored.
- Empty `info` span.
- **Padded `ld` and a stride that is not `ld*cols`** (new).
- **The direct entry point's non-square and heterogeneous-batch guards throw** (new).
- The facade reaches the CTA kernel, verified **bit-exactly against the direct entry point** (fixed; see below).
- The four fit ceilings, pinned against the budget-parameterised query.

### NOT covered — read this before trusting a green run

- **The blocked driver does not exist.** `potrf_blocked_available<T>()` is false for every type; the facade's Blocked arm is unreachable and untested at runtime.
- ~~No vendor-free build was ever configured or run.~~ **DONE — see §4a.** This was the largest unverified claim in the work package and it is now closed by execution.
- **The 48 KB launch hole cannot be tested** (sticky per-process CUfunction attribute).
- **The remaining three dispatch guards** (degenerate extents, non-GPU queue, no sub-group 32) have no test. Non-square and heterogeneous do.
- **`Uplo::Upper` is still lighter than Lower** for `PackedBatchMatchesSolo`, `EmptyInfoSpanStillFactorises`, `ComplexDiagonalIsExactlyReal` and `FacadeReachesTheCtaKernel`. Residual and `info` now sweep both.
- **No accuracy test beyond the residual bound.** The bound is now `4·n·eps`; the true worst case is in `(0.2, 1]·n·eps`, so a defect that degrades accuracy by less than ~4× still passes.
- **The `heterogeneous_batch` writer was not added to `trsm_route.hh`** — trsm's gate remains decorative. It is a route change and needs its own `route_diff` run.

### The deliberate breaks — re-run in this pass, observed red

Every one of these was executed on this tree: broken, rebuilt, run, restored.

| # | break | expected | **observed** |
|---|---|---|---|
| 1 | Facade's CTA arm removed, so `is_native(route)` falls through to `backend::potrf_vendor` | `FacadeReachesTheCtaKernel` red | **RED, all 4 types, and nothing else.** The bit-exact comparison fires; the route assertion still passes, which is the point |
| 2 | `stride_a = ldg * n` in the launcher | `PaddedLeadingDimension…` red | **RED, exactly those 4, nothing else** |
| 3 | Heterogeneous-batch guard deleted | `DirectEntryPointRefuses…` red | **RED, all 4 types** |
| 4 | `G` forced to 1 | `PackedBatchMatchesSolo` red (vacuity guard) | **RED, all 4 types** — which also proves every type does pack at some `n` in the sweep |
| 5 | Residual bound tightened to `0.2·n·eps` | `ResidualBothTriangles` red | **RED, all 4 types** — so the bound brackets the true residual rather than being unfalsifiable |

`grep -rn "DELIBERATE BREAK" src/ tests/ include/` → **0**. Full rebuild exit 0,
suite green, five consecutive runs all 50/50.

**Break 1 is the important one.** The review demonstrated that the *previous*
version of that test stayed **green** with the facade's CTA arm removed, while
every number in it came from cuSOLVER — this repository's fifth recorded blind
guard. The failure message is worth reading: the two values *print* identically
(`-0.0428712` vs `-0.0428712`) and differ only in the low bits, which is exactly
why a residual check could not have caught it and bit-exactness can.

---

## 6. Review findings — triage

### Fixed

| finding | what was done |
|---|---|
| **T1 (blocker)** facade test asserted on a route re-resolution, not on execution | Replaced with a bit-exact comparison against `potrf_cta_dispatch`. Break 1 confirms red |
| **T2 (major)** no test could distinguish `ld`/`stride` from their defaults | New `PaddedLeadingDimensionAndNonDefaultStride`, both triangles, non-PD poison around the window. Breaks 2 confirms red |
| **R2 (major)** `L` ladder derived from tile count; 8.3% occupancy | Rederived from trailing-update **elements**, cap raised to 256. Measured 1.06–1.32× in 11 cells |
| **R4** the `+64` SLM slack under-counted the real launch request by 60 B | Raised to 256 after confirming the 96,408 B request with my own `ncu` run. Ceilings unchanged |
| **C2** advertised ceiling skipped the hole pad the launcher applies | Both now use the same predicate; the non-monotonicity and the required `break` documented in place |
| **C3 / T5** fixture `ceiling()` used the hardcoded reference budget | Now asks the device, as `supports()` and the launcher do |
| **T3** `PackedBatchMatchesSolo` claimed to assert `G > 1` and did not | New `potrf_cta_debug_launch<T>` + an `ASSERT_GT(packed_ns, 0)` vacuity guard |
| **T4** residual bound `40·n·eps` was 40–200× looser than the kernel | Tightened to `4·n·eps`; break 5 proves it brackets |
| **T7** five of six dispatch guards untested | Added non-square and heterogeneous-batch cases; break 3 confirms red |
| **T8** untouched-triangle test ran only at odd `n` | Now `{36, 37}` — both pad-row parities |
| **C4 / T6** failure-path tests were Lower-only | `InfoIndexIsExact` now sweeps both triangles |

### Declined, with reasons

| finding | verdict |
|---|---|
| **C1** "a second agent is editing this worktree concurrently" | **Not a code defect; resolved.** I verified the tree byte-matches the reviewer's captured `.orig` copies, `grep DELIBERATE` is clean, and no build was running. Every number in this document was taken on a quiesced tree with a completed build |
| **R1 (major)** native is 2–3× slower than cuSOLVER above `n≈64` | **Accepted as measured, no code change** — and the reviewer agrees none is required. `preferred()` is false, so no traffic moves. I reproduced the table and it is §4 above. This is now the step-3.1 baseline rather than something to re-measure. **The honest conclusion is that the vendor-free story for `n > 64` is Phase 2's blocked driver, not this kernel stretched to its fit ceiling** |
| **R3 (major, SUSPECTED)** make `NB` a launch-time choice, `NB=16` where shared memory binds | **REFUTED BY MEASUREMENT.** Built and timed `NB=16`: slower in 18 of 20 cells, up to 2.70×. The register-limiter reasoning was sound and the time prediction was wrong. Declining the dual instantiation |
| **R5** `G>1` divides the grid, so small-`n` cells need `batch ≥ G·SMs·blocks` | **Accepted, no code change** — it is a measurement-protocol constraint. All timings here are at `batch = 4096` |
| **C2's second half** `supports()` does not know `max_work_group_size` | **WON'T FIX.** The launcher clamps `L` down to `max_wg`, so the only residual gap is a device with `max_wg < 32` — which cannot offer sub-group size 32 either, and `has_sg32` already gates that. Closing it would mean plumbing a second device property into the shape for an unreachable case |
| **T8 (SUSPECTED)** the even-`n` pad row | **Fixed anyway**, though I confirmed the reviewer's reading that the store-back is correctly `i < n`: it was a coverage gap, not a defect |

---

## 7. Verification, this pass

| gate | result |
|---|---|
| `cmake --build build -j 32` | **exit 0** |
| `ctest -R '^potrf_tests$'` | **passed**; 104 tests, **50 passed**, 54 skipped, 0 failed; stable over 5 consecutive runs |
| `ctest -L "blas\|ortho" -LE slow` | **21/21 passed** |
| `ctest -R '^(route_vocabulary\|options_api\|linalg_layer\|ortho\|cond\|inverse)_tests$'` | **6/6 passed** |
| register probe vs FOUNDATION 2 baseline | 16 rows added, all potrf, frame 0 / spill 0 / callee 0; **zero non-potrf rows changed** |
| `route_diff compare wp4-s1 wp4-repair-final` | **IDENTICAL — 3135 decisions** |
| `route_diff compare wp4-complex-gate wp4-repair-final` | 78 additions, **0 removals, 0 non-potrf rows** |
| **vendor-free build** (`build-novendor`, `BATCHLAS_ENABLE_VENDOR_BLAS=OFF`) | configures, builds, links; `potrf_tests` **50/50 passed**; unforced facade `potrf` correct at `n = 8/48/155`, `NoRouteError` at 156 |

**On the route diff against `wp4-complex-gate`:** it is not empty and *cannot*
be. potrf had **zero** `reached` rows at that label because it never called
`resolve_route` before WP4; every potrf row is therefore an addition by
construction. WP3 did the same to trsm (0 rows → 35). The meaningful statements
are the ones above: **no decision was removed, and no non-potrf decision moved.**
Of the added rows, the only four `CUDA,native,cta` decisions in the entire suite
come from `potrf_tests`' own `FacadeReachesTheCtaKernel` with
`BATCHLAS_POTRF_ROUTE=cta` forced (verified by running that binary alone under
`BATCHLAS_COVERAGE_OUT`); three more `native` rows are `Backend::AUTO` synthetic
rows from the pure-table unit tests. **Every real, unforced library call resolves
to `vendor,auto`.** The `IDENTICAL` result against `wp4-s1` is the stronger
statement: the repair itself moved nothing at all.

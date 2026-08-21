# WP4 Phase 2 — the blocked POTRF driver

What Phase 2 built, what it measured, what it got wrong and had to repair, and
what it did **not** settle.

Read this before `phase2_ab/`, `phase2_bench/` or `phase2_impl/`. Where this
file and those disagree, **this file wins**: several of their numbers were taken
on a panel solve that was returning wrong answers, and are superseded below.

Sibling records:

* `../../../WP4_POTRF_SPEC_CORRECTIONS.md` — the 108 findings the phase was
  written against.
* `phase2_ab/` — the implementation-time A/B harness and its break list.
* `phase2_bench/` — the first benchmark campaign. **Its headline is superseded**
  (see §5); its localisation of the correctness defect was right.
* `phase2_impl/` — implementation notes.

---

## 1. What ships

| file | what |
|---|---|
| `src/extensions/potrf_blocked.cc` | the driver (new) |
| `src/extensions/potrf_native.hh` | `PotrfTrailingGemm`, `PotrfPanelSolve`, `potrf_blocked_buffer_size`, `potrf_blocked_debug_params`, `potrf_blocked_dispatch`, `potrf_blocked_available` |
| `src/dispatch/entry_points/factorization.cc` | the Blocked arm of the facade, both seams injected, and buffer-size max over supported native tiers |
| `include/batchlas/blas/dispatch/route_potrf.hh` | `supports(Blocked)` — presence + `Uplo::Lower`, **no** order bound |
| `src/sycl/trsm_native.cc` | one added `group_barrier` — a wrong-answer fix, §3 |
| `tests/potrf_tests.cc` | 14 typed blocked tests |
| `tests/trsm_tests.cc` | 1 test, the guard for §3 |

The schedule is right-looking, `Uplo::Lower` only, `j = 0, nb, 2nb, …`:

1. **leaf** — `A(j:j+ib, j:j+ib)` by the Phase 1 CTA kernel on a sub-view;
2. **fixup** — one kernel: local→global `info` merge (first failure wins) plus
   the failed-item quench that keeps a failed item finite;
3. **panel** — `L21 = A21 · L11^{-H}` through the **injected, routed** trsm;
4. **trailing** — `A22 -= L21 L21^H` through the **injected, routed** gemm, cut
   into `W`-wide column panels, each a `W×W` diagonal block into scratch +
   triangular fold, plus a direct rectangle gemm. A square gemm over `A22`
   would write the upper triangle, which LAPACK `potrf(Lower)` must not touch.

`preferred()` is still **all-false**, so this phase moves zero traffic in a
vendor-present build. It is reachable by pinning `BATCHLAS_POTRF_ROUTE`, and in
a vendor-free build via `route_resolve.hh:60-63`.

### Shipped constants

| type | `nb` | `W` |
|---|---|---|
| `float` | 128 | **128** |
| `double` | 96 | 32 |
| `complex<float>` | 96 | 32 |
| `complex<double>` | 64 | 16 |

`float`'s `W` was **32** as first shipped; §4 is why it is now 128.

---

## 2. The blocking correctness defect — vendor-free potrf returned wrong answers

**This is the most important thing in this phase.** The first benchmark campaign
reported it; triage root-caused and fixed it.

### The symptom

`build-novendor`, no environment set, the genuine default path. Float and double,
`n = 1024`, `batch = 256`, an input with condition number < 1.05 that cuSOLVER
factors to 1e-8 / 1e-16 in the same process:

```
float  n=1024 batch=256 rep=0  bad=69/256  cols: 897 129 641 769 129 769 769 257 385 770 769 897
float  n=1024 batch=256 rep=1  bad=71/256
float  n=1024 batch=256 rep=2  bad=75/256
double n=1024 batch=256 rep=0  bad=15/256  cols: 577 577 673 577 289 97 577 385 577 673 674 482
```

Non-deterministic. Every reported failing column is `≡ 1 (mod nb)` — the *first*
column of a panel, i.e. a diagonal block that the previous panel's bad `L21` had
already destroyed. Clean at `batch ≤ 96` and at `n ≤ 384`, which is why the
implementation-time proof runs (batch 3–32) missed it entirely.

### Localisation, by measurement

Holding one seam at a time (`n=1024, b=256`, vendor-present build so both arms
exist):

| trsm seam | gemm seam | result |
|---|---|---|
| vendor | vendor | **clean**, 0/256 |
| vendor | native | **clean**, 0/256 |
| native | vendor | 19–29 / 256 bad |
| native | native | 61–65 / 256 bad (float) |

So it is the **panel trsm**, and nothing else. Not the leaf: factoring the same
`ib × ib` block as a sub-view of an `n = 1024` parent at `batch = 256` is
**bit-identical across all 256 items** and reports `info == 0` for every one.

### The mechanism

`src/sycl/trsm_native.cc`, the V1 CTA kernel. It stages the canonical triangle
into local memory with a lane-strided loop, then reads the **diagonal** back to
form reciprocals — with **no barrier in between**:

```
for (size_t idx = lane; idx < tri_elems; idx += wg) { ... sLc[idx] = v; }
                       <-- nothing here
for (int s = lane; s < N; s += wg) { ... const D d = sLc[tri_idx(s, s)]; ... }
```

Element `idx` is written by lane `idx % wg`; lane `s` reads `sLc[s(s+1)/2 + s]`,
a *different* lane's write for nearly every `s`. A garbage diagonal gives a
garbage reciprocal gives a wrong solve. The same missing barrier also lets lane
0's `sDiv[0] = 0` land *after* another lane's atomic store of 1, discarding the
revert-to-division flag.

**Why WP3's entire test suite and every WP3 benchmark were green.** The launcher
picks the work-group width from `{256,128,64,32}`, taking the first candidate
with `batch·ceil(q/wg) ≥ 4·CU` (512 on this box). Every trsm test in the tree
uses `batch ≤ 3` and `q ≤ 257`, so **every one of them runs at `wg = 32`** — a
single sub-group, executing the two loops in lock step, where the race cannot
express itself. Nothing in the suite ever left one sub-group, and nothing said
so. The blocked POTRF panel solve is the first caller that does: at `n = 1024`,
`batch = 256` the first panel has `q = 896` and `wg = 256`, eight sub-groups.

### The fix

One `sycl::group_barrier(it.get_group())` immediately after the staging loop.
It is the only functional line changed in that file (`git diff` confirms).

Post-fix, over every configuration above, every rep: **0 bad**, float and double,
up to `n = 512 batch = 1024` and `n = 1024 batch = 256`.

### It also explains the `nb` rounding workaround

`potrf_blocked_params` rounds `nb` down to a multiple of `trsm_cta_max_n<T>()`
(32). That line was shipped as a *correctness containment* for a recorded
observation that the panel solve returned host residuals of 1e+04…1e+20 at
triangular orders 48, 77, 80, 109 but not 32, 64, 96, 128, and only above
roughly `q·batch = 65k`. Both conditions are the barrier race: `65k` is exactly
where the wg ladder leaves 32, and the "bad orders" are the ones whose final V1
block lands in a bucket whose diagonal indices collide differently.

Re-measured post-fix — direct `trsm(Right, Lower, ConjTrans, NonUnit)` against a
host reference, orders `{16,32,48,64,77,80,96,109,128,155}`, `q = 896`,
`batch = 256`, float and double — the native answer agrees with the reference to
the **same relative error as cuBLAS at every order**. Nothing resembling 1e+04
appears anywhere.

**The rounding is kept anyway**, and deliberately: all four shipped `nb` values
are already multiples of 32, so it is the identity on every default path, and
what it now buys is that a hand-set `BATCHLAS_POTRF_NB` cannot wander into an
order whose V1 block structure has never been measured.

### Consequence for WP3's recorded trsm numbers

Every WP3 trsm A/B cell above `q·batch ≈ 65k` was timing a kernel that computed
the wrong answer. The barrier costs one `__syncthreads()` per work-group, so the
*timings* are approximately still valid — but they were not measurements of a
correct kernel, and `trsm`'s `preferred()` windows inherit that caveat. Nobody
has re-run the WP3 grid. **Open.**

---

## 3. The second wrong answer — an unwritten scratch read through `beta = 0`

The `W×W` diagonal-block gemm is issued with `alpha = -1, beta = 0` into
`ws.product`, which was never written before that gemm. The driver's own comment
asserted this was safe, citing `symmetric_product_fold.hh:49,:68`.

**Those are the fold's lines, not the gemm's.** `beta == 0` means "C is not read"
in the fold, and in cuBLAS. It does **not** mean that in any native gemm in this
tree: `LinearEpilogue::apply` is `alpha*accum + beta*prior` with `prior` read
unconditionally (`gemm/epilogue_linear.hh:7-9`, `tiled_general.hh:79-81`,
`register_tiled_common.hh:598,613`). `0 * NaN = NaN`.

Reproduced through the ordinary public API, no artificial buffer: a prior
unrelated `ctx.workspace()` lease leaves poison in the arena bytes potrf then
leases (`options.hh:550-551`). Float, `n = 256`, `batch = 8`, well-conditioned SPD:

```
BATCHLAS_POTRF_ROUTE=blocked                            -> 0/8 bad, rel resid 4.724e-07
BATCHLAS_POTRF_ROUTE=blocked BATCHLAS_GEMM_ROUTE=native -> 8/8 bad, rel resid 9.941e-01
```

The second line is the vendor-free build this work package exists for. With
cuBLAS injected it survived by luck.

**Why no Phase 2 test saw it:** every one allocates a fresh
`UnifiedVector<std::byte>`, and `sycl-util-impl.cc:37-46` is a bare
`malloc_shared` whose pages the CUDA driver hands back **zeroed**. This is the
repository's recurring blind-guard shape, fourth-and-a-half occurrence.

**Fix:** one `ctx->fill(ws.product, T(0))` before the panel loop — one launch per
call, not per column panel. After the first diagonal gemm the scratch holds a
finite product and `0 · finite == 0`; the only way a later read is non-finite is
if the product itself overflowed, in which case the rectangle gemm has already
written the same magnitudes into `A` and the answer is garbage either way.
`src/extensions/syrk.cc:51`, the only other caller of the same fold helper,
allocates its scratch with `Matrix::Zeros` for exactly this reason.

---

## 4. `W` for float: 32 → 128, and why the "per-route W" remedy is not needed

Two prior reports disagreed. The implementation recorded `W = 32` from
trailing-*stage* timings at `n = 512` on the native gemm, with a caveat that the
vendor gemm wanted 96–128 and that "one number cannot be right for both". The
review turned that into a request for a per-route `W`. The benchmark campaign
said `W = 128` was better on both.

Re-measured end to end, at the shipped `nb`, on a **correct** factorisation
(which none of the earlier float cells at large batch were), interleaved,
3 passes × 2 reps, worst rel sd 4.9%. Whole-potrf ms by `W = 16/32/64/96/128`:

```
float, both seams NATIVE (the vendor-free build):
  n=512  b=256    4.422 /  3.449 /  3.774 /  4.271 /  3.403     -> 128
  n=1024 b=256   28.286 / 17.863 / 18.654 / 20.495 / 16.785     -> 128 (1.06x over 32)
  n=2048 b=128  107.046 / 52.937 / 50.879 / 54.936 / 46.510     -> 128 (1.14x over 32)
float, both seams VENDOR:
  n=1024 b=256   29.330 / 19.657 / 15.619 / 15.124 / 15.122     -> 128 (1.30x over 32)
```

`W = 128` wins on **both** routes, by more as `n` grows. **So the per-route `W`
the review asked for is not needed** — one constant beats the old one everywhere,
and no dispatch fact enters the kernel TU. The review's *observation* was right;
its proposed *remedy* is refuted.

Mechanism: the diagonal-block gemm has `m = n = W`, `k = nb`, and
`gemm_kernels.cc:472-480` gives float's transposed register kernel only at
`m ≥ 128 && n ≥ 32 && k ≥ 128`. At `W ∈ {32,64,96}` it can never reach it and
lands on `Tiled16`; at `W = 128` it does. The curve is **non-monotonic** (96 is
worse than 64), which is the signature of a kernel-selection cliff rather than of
the linear wasted-work term — so do not interpolate this table.

The other three types were re-measured at the same point (`n=1024 b=256`, native):

```
double  77.73 / 78.58 / 81.94 / 85.11 / 89.17    (16 beats the shipped 32 by 1.1%)
cfloat  54.63 / 54.00 / 57.40 / 59.95 / 63.45    -> 32, as shipped
```

`double`'s 1.1% is inside the noise of the `n=512` table its 32 came from; left
alone. `cdouble` was not re-swept: its `k = nb = 64` cannot reach any register
kernel at any `W`, and both its own `n=512` table and the double/cfloat trend say
smaller wins.

**And the shipped `(nb, W)` pair is now measured together**, which it never was
before (the two were swept independently at each other's non-shipped values).
Float, `n=1024 b=256`, native, at the shipped `W = 128`, `nb = 32/64/96/128`:
`46.918 / 36.484 / 31.781 / 16.452` — `nb = 128` by 1.93x.

**Reproduction trap, recorded because the `nb` table will mislead the next
reader:** the multiple-of-32 rounding applies to `BATCHLAS_POTRF_NB` too, so a
sweep of `48/80/109/155` collapses onto `32/64/96/128` and produces four cells
identical to their neighbours. Read the recorded `nb` table with those labels
substituted.

---

## 5. The performance picture, including the losses

Whole-`potrf` wall time, `Uplo::Lower`, GPU 1 idle, JIT and clocks warmed and
discarded, host↔device page migration excluded from the timer (a host refill
loop inflated float `n=1024 b=256` from ~17 ms to ~93 ms — a 5x fabricated
result if it had gone unnoticed), 3 interleaved passes × 2 reps, medians.
`bad = 0` in every arm of every cell below.

* **cuSOLVER** = `BATCHLAS_POTRF_ROUTE=vendor`
* **nn** = blocked with **both seams native** — this is the vendor-free build
* **routed** = blocked with both seams choosing their own route (vendor-present)

Ratios are `cuSOLVER / ours`; **> 1 means we win**.

| type | n | batch | cuSOLVER | nn | routed | nn/vend | routed/vend |
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

Worst rel sd 5.3%; two cells first measured at `n=256` with rel sd 8.7–12.5%
were **discarded and re-measured at a saturating batch** (float 512→2048,
cfloat 256→1024), which is what the table shows.

### What to take from it

* **Vendor-free `float` is now FASTER than cuSOLVER at `n ≥ 1024`** — 1.11x at
  1024, 1.40x at 2048. `double` is at parity across `n = 512…2048`
  (0.99–1.06x). That is the WP4 goal met for the real types at the orders this
  driver exists for.
* **Complex is not there.** Vendor-free `cfloat`/`cdouble` are 0.31–0.51x, and
  the gap *widens* with `n`. The cause was identified by the benchmark campaign
  and is not in this driver: `route_gemm.hh:113-114` returns false for complex
  and `gemm_kernels.cc:471` keeps the register ladder float-only, so every
  complex trailing gemm lands on the `Tiled16` fallback. At `cdouble n=1024` the
  trailing gemm is ~97% of the call and 2.95x slower than cuBLAS. **A
  register-tiled complex GEMM is worth roughly 2.7x on vendor-free cdouble potrf
  on its own**, and is the single highest-value follow-on.
* **Small `n` loses for every type** (0.43–0.83x at `n = 256`). That deficit is
  Phase 1's, not Phase 2's: at `float n=128` the CTA leaf alone is 0.293 ms
  against cuSOLVER's 0.140, and the blocked driver at small `n` is that leaf plus
  one launch.
* **The crossover is in ORDER, not batch.** Real types converge on and then pass
  cuSOLVER as `n` grows; complex diverges.

### This supersedes `phase2_bench/`

That campaign concluded "no, it is never faster", geomean 0.74 (routed) / 0.52
(vendor-free), best reliably-correct cell 0.996x. Both causes are now fixed: its
large-batch cells were racing (§2), and float ran at `W = 32` (§4). Its
*mechanism* work — the nsys splits, the complex-GEMM diagnosis, the
launch-overhead negative result, the `BATCHLAS_POTRF_ROUTE` typo trap — stands.

---

## 6. The panel-solve verdict (open question 5), re-settled

Injected **routed trsm**, not a bespoke panel kernel. The original evidence
("46 of 48 panel cells") was taken on the racing kernel and is not evidence of
anything. Re-measured post-fix, end to end, with the **gemm seam held at the
vendor** so the panel solve is the only variable (whole-potrf ms,
`vendor trsm / native trsm`, > 1 = native faster, `bad = 0` everywhere):

| type | n | batch | trsm=vendor | trsm=native | ratio |
|---|---|---|---|---|---|
| float | 512 | 256 | 3.174 | 2.256 | 1.407 |
| float | 1024 | 256 | 15.144 | 10.663 | 1.420 |
| float | 2048 | 128 | 38.674 | 29.569 | 1.308 |
| double | 512 | 256 | 15.503 | 13.003 | 1.192 |
| double | 1024 | 256 | 99.482 | 87.893 | 1.132 |
| cfloat | 512 | 256 | 7.212 | 3.781 | 1.907 |
| cdouble | 512 | 128 | 69.349 | 25.770 | 2.691 |

The verdict stands and is stronger than before: the routed trsm wins in every
cell tried, now on correct answers. A bespoke kernel would also be aimed at the
wrong stage — the panel is 5–22% of a vendor-free blocked potrf against 65–95%
for the trailing update.

---

## 7. Every deliberate break, and whether it turned red

### Implementation phase (13 tests, 11 breaks) — recorded, not re-run here

| break | result |
|---|---|
| `noffset` — `info = li` instead of `j + li` | RED: B3, B4, B5 × 4 types. **The entire CTA suite stayed green**, because there `j` is always 0 |
| `nofold` — diagonal gemm aimed straight at `A` | RED: **B2 only**. Every residual test, both info suites, B7 and B13 stayed green — and it is 11% *cheaper* |
| `noshort` — short final block never factored | RED: 34 tests |
| `ldrows` — `A.ld()` → `A.rows()` | RED: **B7 only** |
| `nozero` — delete the info zero pre-pass | RED: 42 tests |
| `nomerge` — last-panel-wins | RED: 12 tests |
| `noquench` | RED: B5 only, and **only the NaN item** |
| `transb` — `Trans` for complex too | RED: complex types only, correctly a no-op for real |
| `noinject` — facade passes empty seams | RED: **B13 only** |
| `fallvendor` — facade silently runs cuSOLVER | RED: **B13 only** |
| `chosen-only` buffer sizing | RED: **B12 only** |

One break did not fail the way its author first expected and **the test was
fixed rather than the report**: `noquench` with the NaN planted in the *final*
block produced exactly one non-finite word, because a final block has `m2 == 0`
and nothing propagates. The NaN was moved to block 1 of 3, where a panel solve
divides by the failed pivot and a trailing update smears it; 1045/789/789/533
non-finite words. The placement and the reason are recorded in the test.

### Triage phase (2 tests, 2 breaks) — run here

Both applied to the shipped source in one rebuild, whole suites run, then
reverted.

| break | what | result |
|---|---|---|
| **A** — delete the trsm `group_barrier` | the §2 fix | **RED: exactly one test in the 92-test trsm suite** — `TrsmNativeCta.MultiSubGroupWorkGroupStagesItsTriangleCorrectly`. The other 91 stayed green. **The whole 216-test potrf suite also stayed green** (104/104) |
| **B** — delete the `ws.product` zero fill | the §3 fix | **RED: exactly `PotrfBlockedTest.BlockedDoesNotReadUninitialisedWorkspace`, all four types.** The other 100 passing tests, including all 13 of the phase-2 suite, stayed green |

Break A is the sharpest statement in this document: a genuine, reproducible
wrong-answer race in a shipped kernel was **invisible to 91 of 92 tests written
for that kernel and to every one of the 216 tests written for its only new
caller**, because all of them ran at one sub-group and none of them said so. The
new test asserts its own non-vacuity by replaying the launcher's work-group
ladder from the device properties and failing if it does not clear 32.

---

## 8. Gate results after the repairs

Verbatim.

```
cmake --build build -j 32                      -> exit 0
cmake --build build-novendor -j 32             -> exit 0

build/tests/potrf_tests                        -> 216 tests ran. [ PASSED ] 104. 0 failed.
build-novendor/tests/potrf_tests               -> 216 tests ran. [ PASSED ] 104. 0 failed.
build/tests/trsm_tests                         -> [ PASSED ] 92. 0 failed.
build-novendor/tests/trsm_tests                -> [ PASSED ] 60, [ FAILED ] 32
                                                  (pre-existing vendor-free baseline;
                                                   the new test passes)

build:           ctest -L "blas|ortho"         -> 100% tests passed, 0 failed out of 21
build-novendor:  ctest -LE slow                -> 48% tests passed, 28 failed out of 54
```

`28/54` is **byte-identical to the recorded pre-phase baseline**, and so is the
failing set: `backend_dispatch options_api linalg_layer syevx lanczos gemv trsm
ortho inverse cond ormqr ormqr_cta ormqr_blocked orgqr iluk symm hemm herk her2k
syrk syr2k syev trmm sytrd_sy2sb sytrd_blocked syev_cta syev_blocked
syev_two_stage`. Nothing joined it and nothing left it.

**Register gate:** `scripts/register_probe.sh <log> '' batchlas_sycl` — **424
entry functions, 0 with non-zero spill**. Worst trsm cell
`TrsmCtaKernel<complex<double>, 32, Right>` at **226 registers**, exactly the
figure the file's own `static_assert` documents, so the added barrier cost no
registers; 226 × 256 = 57,856 against the 65,536-per-block limit.

**Route diff: not re-run, and here is precisely why.** The capture requires
`-DBATCHLAS_ENABLE_COVERAGE=ON`, which neither existing build dir has, so it
costs two full device-link-bound rebuilds. The triage delta touches **no routing
predicate at all**: one `group_barrier`, two `fill`s, four `if (!ctx.in_order())
ctx.wait()`s, one tuning constant, and comments. `git diff` on
`route_potrf.hh` is unchanged from the implementers' version and
`factorization.cc`'s delta is comment-only. The implementers' capture stands:
**+28 additions, 0 removals**, every addition a call shape that did not exist
before Phase 2. What a fresh capture *would* show differently is the trailing
gemm's shape rows moving with `W` — and note that the *kernel variant* that
changes with `W` (`Tiled16` → `Tiled128x32RegisterK32NT`) is invisible to
`route_diff` in any case, since `coverage::record` stores the resolver `Route`
and never the `KernelVariant`.

---

## 9. Review findings — confirmed, refuted, and what was done

| # | severity as filed | verdict | action |
|---|---|---|---|
| 1 | WRONG_ANSWER — `beta=0` reads unwritten scratch | **CONFIRMED**, reproduced independently through the public API | **FIXED** (§3) + test B14 + break B |
| 2 / 6 | WRONG_ANSWER — out-of-order queue: only 2 of 5 dependent edges guarded | **CONFIRMED by code reading**; not observable on this box (the CUDA/UR backend appears to serialise anyway), so it is a code-level defect, not a measured one | **FIXED** — a guard on every one of the five edges; zero cost on the default in-order queue |
| 3 | HYGIENE — buffer-size max leaves the vendor→native half open | **CONFIRMED as stated**: the code does not implement the invariant its comment claims | **Comment corrected, code left alone.** Closing it means computing the blocked layout unconditionally, adding `W²·batch·sizeof(T)` — megabytes at large batch — to every vendor-present `potrf` that will never touch it, to defend against a `getenv` that changes inside a single API call. A throw is also the benign end of the class. Revisit if `preferred()` comes off all-false |
| 4 | HYGIENE — B2 runs at an `n` where the rectangle gemm takes a different kernel than production | **CONFIRMED** | **FIXED** — B2 now also runs an `n` with `n - nb - W ≥ 128`, asserted, so it exercises the register-tiled store path |
| 5 / 8 | PERF — `fill(...).wait()` drains the queue every call | **CONFIRMED**; it is the only `fill().wait()` in `src/` | **FIXED** — fire-and-forget plus the out-of-order guard |
| 7 | PERF — `W` should be per-route; float loses 1.6x with the vendor gemm | **Observation CONFIRMED, remedy REFUTED.** `W = 128` wins on *both* routes, so one constant suffices and no dispatch fact enters the TU | **FIXED** by changing the constant (§4) |
| 9 | PERF — the float `Trans` flip lands on the one ld-sensitive kernel, cost unrecorded | **Partly refuted as framed.** The concern is real and the shipped rectangle shape still has no isolated measurement. But the claimed "0.5–2 ms of unrecorded cost" is not visible end to end: with that flip *and* `W = 128`, vendor-free float is 1.11–1.40x **faster** than cuSOLVER. Whatever the ld penalty is, it does not stop the driver winning | **RECORDED as open** (§10). The pack-`L21`-contiguous probe remains unmeasured |
| 10 | HYGIENE — the `nb` table cannot be reproduced; no shipped `(nb, W)` pair ever measured | **CONFIRMED, both halves** | Comment corrected to state that the rounding applies to the env override; and the shipped `(nb=128, W=128)` pair is now measured end to end (§4) |
| — | benchmark campaign: "vendor-free returns wrong factors at large batch, it is the panel trsm, it is in the V1 CTA kernel" | **CONFIRMED** and root-caused | **FIXED** (§2) + test + break A |
| — | benchmark campaign: "no, it is never faster; geomean 0.74/0.52" | **SUPERSEDED** — it was measuring a racing kernel at a bad `W` | Re-measured (§5) |

---

## 10. What Phase 2 did NOT settle

1. **`preferred()` is still all-false.** No cell has been flipped, so a
   vendor-present build still never chooses this driver. The measured windows in
   §5 are the input to that decision, not the decision. Spec 10.3's three-part
   gate (kernel win ≥ 1.11x at saturation, no accuracy regression, an end-to-end
   `ortho_benchmark` win) has not been run.
2. **Complex is 0.31–0.51x vendor-free and the cause is outside this driver.**
   A register-tiled complex GEMM is worth ~2.7x on vendor-free `cdouble potrf`
   alone. Highest-value follow-on.
3. **Small `n` (≤ 256) loses for every type, and it is Phase 1's leaf.**
4. **The strided-`ld` cost of the trailing gemm has never been isolated.** Every
   operand is a sub-view at the parent `ld`, `OpShape` carries no leading
   dimension so the router cannot see it, and the cheap probe (pack `L21` into a
   contiguous `ld == rows` buffer once per panel step; ~59 MB and ~0.5 ms of copy
   traffic at `n=1024 b=256`) was not run.
5. **`Uplo::Upper` is unimplemented** and `supports()` refuses it. The two known
   routes are a mirror (as `syev` does) or a transposed schedule.
6. **The out-of-order defect could not be made to fail on this box.** The guards
   are correct by construction and free on the default queue, but nothing
   demonstrates the failure they prevent, and no test constructs an
   out-of-order `Queue`.
7. **WP3's trsm `preferred()` windows were measured on the racing kernel** above
   `q·batch ≈ 65k` and have not been re-run (§2).
8. **The trailing update issues three kernels per column panel** where a
   save/restore formulation would issue roughly one, cutting launches from 329 to
   126 at `n=1024`. The implementer measured the fold-free variant at 11% cheaper
   and wrong only in the upper triangle; the save/restore half was never built.
   Note this interacts with §4 — at `W = 128` the launch count is already 4x
   lower than the figure that estimate was made at.
9. **`potrf_blocked` is not reached by any test above the CTA ceiling through the
   unpinned facade in a vendor-present build**, by design (`preferred()` is
   all-false). The burn-down instrument therefore cannot see this phase: 26/54
   before, 26/54 after. The evidence that Phase 2 works is that `potrf_tests`
   contains 14 tests reaching orders that threw `NoRouteError` before it, and
   they pass unpinned in `build-novendor`.

---

## 11. Independent re-verification by the orchestrator

Everything above was produced by the Phase 2 agents. The four claims that decide
whether this phase ships were re-run from scratch afterwards, against the
SHIPPED code in the SHIPPED builds. All four hold; two of the instruments used
to establish them did not.

### 11.1 The missing barrier — A/B in both directions

`src/sycl/trsm_native.cc`, the one functional line of this phase. Deleted it,
rebuilt the library (the .so relink is the AOT device-compile, so this is a real
rebuild and not a stale binary), and re-ran the same command:

| `phase2 trsmdiff float 1024 48 0 128 sub 0 0` | max rel diff vs vendor | items wrong | native residual |
|---|---|---|---|
| barrier deleted | 6.05e+16 | 127 / 128 | 8.0e+05 |
| barrier restored | 4.27e-07 | 0 / 128 | 2.38e-07 (= vendor) |

The mechanism is also visible by reading: the staging loop at `:355-376` writes
`sLc[idx]` strided by `lane`, and the loop immediately after has lane `s` read
`sLc[tri_idx(s,s)]` — a different lane's write — with nothing in between.

### 11.2 The regression test was VACUOUS, and is now not

The test shipped by the test agent as
`TrsmNativeCta.MultiSubGroupWorkGroupStagesItsTriangleCorrectly` called V1
**directly** at `n=16, q=1024, bs=128`. That clears the work-group ladder, and
its anti-vacuity assertion (`wg > 32`) passes — but with the barrier deleted and
the library rebuilt **it still came back green**. Clearing the ladder is
necessary and not sufficient.

It is now `TrsmNativeBlocked.MultiSubGroupWorkGroupStagesItsTriangleCorrectly`
and drives the configuration that actually reproduces: order 48 through V2, so
the FINAL V1 block is order 16, at `q=976, batch=128`, checked by
`RunTrsmBlocked`'s independent multiply-back oracle. Verified RED with the
barrier deleted and GREEN with it restored, both on a full rebuild.

This is the **fifth** recorded instance in this repository of a guard that could
not fail, and the first one where the vacuous test was written *in the same
change as the fix it guards*.

### 11.3 The performance headline — re-measured on the public API

§5's table was reproduced independently with `phase2_ab/realpotrf.cpp`, which
calls the **shipped `potrf` public API** with no forced route, built twice: once
against `build/` and once against `build-novendor/`. So "vendor-free" here means
the vendor-free BUILD, not a forced route inside a build that still links
cuSOLVER.

| type | n | batch | vendor build | vendor-free build | ratio | §5 claim |
|---|---|---|---|---|---|---|
| float | 256 | 2048 | 3.419 | 5.727 | 0.597 | 0.593 |
| float | 1024 | 256 | 17.812 | 15.784 | **1.129** | 1.108 |
| float | 2048 | 128 | 65.813 | 46.971 | **1.401** | 1.396 |
| double | 1024 | 256 | 81.202 | 79.219 | **1.025** | 1.014 |

Residuals identical between the two builds in every cell, `info == 0`
everywhere. The loss at `n=256` reproduces as faithfully as the wins.

Also confirmed: blocked with routed seams in the vendor-present build is
**10.386 ms** (1.71x over cuSOLVER), and blocked under forced
`BATCHLAS_GEMM_ROUTE=native BATCHLAS_TRSM_ROUTE=native` is **15.787 ms**, which
matches the vendor-free build's 15.784 to 0.02%. The forced-vs-resolved trap
does not bite this driver.

### 11.4 Two instruments that misreported, and were fixed

**`phase2_ab/phase2`'s `blocked` mode does not time the shipped driver.** It
times a `Blocked<T>` class defined inside the harness. Measured on the same
shape, the harness's re-implementation is ~2x slower than the shipped driver
(30.9 ms against 15.8 ms, float `n=1024 b=256`, both seams native), which is why
an early orchestrator re-check appeared to contradict §5 by a factor of two. The
harness was the right instrument for the design study, which ran before the
driver existed; it is not an instrument for "how fast is `potrf`".
`realpotrf.cpp` is, and is committed beside it.

**`scripts/register_probe.sh` reported a spill regression that does not exist.**
Its summary line counted `Function properties` blocks for non-inlined DEVICE
functions as well as entry functions, so on `batchlas_extensions_cta` it printed
"16 kernels with non-zero spill" when **every entry function is clean** and all
16 belong to `gesvdj_cta_impl<complex<double>>`, a pre-existing 255-register
kernel in a file this phase does not touch. The script now prints the
entry-function count (the gate) and the all-functions count separately. Verified
on the same log: entry-function spills **0**, potrf entry functions **24**, all
zero spill.

### 11.5 Gates, re-run by the orchestrator

```
cmake --build build -j 32                     -> exit 0
build/tests/trsm_tests                        -> 92/92 passed
build/tests/potrf_tests                       -> 216 ran, 104 passed, 0 failed, 112 skipped
build-novendor/tests/potrf_tests              -> 216 ran, 104 passed, 0 failed, 112 skipped
build-novendor/tests/trsm_tests               -> 92 ran, 60 passed, 32 failed
build:          ctest -L "blas|ortho"         -> 21/21, 0 failed
build-novendor: ctest -LE slow                -> 48% passed, 28 failed of 54  (26/54, baseline, same set)
```

The 112 potrf skips are the host-backend instantiations ("potrf_cta is a GPU
kernel"), not silent omissions. All 32 vendor-free `trsm_tests` failures are
Backend 6 (NETLIB, host) — the recorded pre-existing set, zero CUDA failures;
the suite gained one passing CUDA test, which is §11.2's.

# The vendor-free burn-down baseline

Recorded at WP0 S6, on `worktree-vendor-independence-plan`.

```
cmake -S . -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF
cmake --build build-novendor -j
ctest --test-dir build-novendor -LE slow
```

**This configuration now configures, compiles, links, loads and runs.** That is
WP0's deliverable. It is emphatically **not** green, and no dispatch mechanism
could make it so: the gap is missing *kernels*, not missing *routing*. Closing it
is WP1–WP8.

## Result at S6

`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` on this machine yields
`BATCHLAS_HAS_CUDA_BACKEND 1` with every CUDA math library at 0 — a CUDA device
with no CUDA math libraries, a state the pre-WP0 scheme could not express and
could not link.

    ctest -LE slow: 20 / 53 pass (38%)

Failures are `NoRouteError`, not crashes and not link errors. The message names
the op, the scalar type and the switch that would restore it:

    BatchLAS: no route for getri<float> on this backend (built without cuBLAS).
      This build has no vendor library for that op, and BatchLAS has no
      native kernel for it yet. If you configured with
      -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF, re-enabling it restores this op;
      otherwise the vendor library was not found at configure time.

## The 33 failing suites

Any change to this SET is a reviewable diff: a suite leaving it means a native
kernel now covers that op, and a suite joining it is a regression.

```
backend_dispatch_tests   bdsdc_tests            cond_tests
gemm_tests               gemv_tests             hemm_tests
her2k_tests              herk_tests             iluk_tests
inverse_tests            lanczos_tests          linalg_layer_tests
options_api_tests        orgqr_tests            ormqr_blocked_tests
ormqr_cta_tests          ormqr_tests            ortho_tests
ritz_values_tests        syev_blocked_tests     syev_cta_tests
syev_tests               syev_two_stage_tests   syevx_tests
symm_tests               syr2k_tests            syrk_tests
sytrd_blocked_tests      sytrd_cta_tests        sytrd_sy2sb_tests
transpose_tests          trmm_tests             trsm_tests
```

Two of those — `lanczos_tests` and part of `steqr_tests` — fail in the
vendor-PRESENT build too, for reasons unrelated to this work; see
`VENDOR_INDEPENDENCE_PLAN.md`.

## After WP1

The numbers below are the **pre-WP1** baseline and are kept as the historical record. WP1
moved them: the suite is now **24/53**, `gemm_tests` is 167 passing / 17 failing (was
48/136), and the `symm`/`syrk`/`syr2k`/`trmm` tile kernels are reached rather than merely
linked. Four suites recovered — `bdsdc_tests`, `ritz_values_tests`, `sytrd_cta_tests`,
`transpose_tests` — with none newly failing. See `WP1_LEVEL3_SPEC.md`.

## After WP2's correctness track

`gemm_tests` is now **184/184 vendor-free** (was 48/184 pre-WP1, 167/184 after WP1 S5), and
the suite is **25/53**. `gemm_tests` left the failing set and nothing joined it.

The gap that closed was heterogeneous batch, and it was never a kernel gap: the per-item loop
— including the `m==0`/`n==0` skips and the `k==0 → scale(beta)` substitution — lived inside
cuBLAS-gated code, so a vendor-free build simply did not have those semantics. See
`WP2_GEMM_SPEC.md`.

One named gap remains in this area:

- **Level-3 non-float.** `syrk`'s gram branch and `trmm`'s tile branch for double/complex
  are still reachable only from `cublas.cc`, and `syr2k` has no non-float tile route at all.

And one that WP2's envelope track owns: `select_kernel_variant` is **float-only**, so a
vendor-free `double` or `complex` GEMM reaches `Tiled16`, not a register kernel.

## After WP2 E2

That last gap is now closed for the aligned square NN bucket.
`src/sycl/gemm/register_64x64_k16_wide.hh` is the first register-tiled kernel in this tree to
serve a non-float scalar, and vendor-free `gemm_tests` goes **184 → 200 passing** (the 16 new
forced-variant tests, all four scalar types, aligned and ragged). The suite stays **25/53**
with a byte-identical failing set, and the vendor-present route diff moves **zero** existing
decisions.

Note what it does *not* close, measured rather than assumed: on real demand (with
`route_gemm_equivalence_tests`'s 2312 synthetic probe rows removed) the kernel's routing gate
fires on **46 of 7223 non-float gemm calls, 0.64%**. BatchLAS's own GEMM is dominated by panel
updates — large m, large n, small k, usually transposed — and `min_dim >= 256` takes the min
over k, which is a blocking constant. See `WP2_GEMM_SPEC.md`.

## After WP3 — and why the suite number did not move

The suite is **still 25/53**, and `trsm_tests` is still in the failing set. That is not a
disappointing result; it is the burn-down instrument failing to measure what changed.

Vendor-free `trsm_tests` is **59 passing / 32 failing**, and **all 32 failures are the host
(NETLIB) backend**:

    BatchLAS: no route for trsm<float> on this backend
      (built without netlib CBLAS/LAPACKE).

Not one CUDA-backend case fails. On the GPU, vendor-free `trsm` is complete — every order, both
sides, all four scalar types — which is exactly what WP3 set out to deliver, and the suite-level
pass count cannot show it, because `ctest` runs each level-3 suite against the host backend as
well and a vendor-free build has no netlib LAPACK either.

Classifying every vendor-free failure by which library its `NoRouteError` names:

| suite | host (netlib) | GPU (cuBLAS/cuSOLVER/cuSPARSE) |
|---|---|---|
| `trsm_tests` | **32** | **0** |
| `sytrd_blocked_tests` | 12 | 0 |
| `syev_tests` | 4 | 0 |
| `ormqr_cta_tests` | 2 | 0 |
| `symm_tests` | 2 | 5 |
| `hemm_tests` | 6 | 6 |
| `herk_tests` | 8 | 8 |
| `her2k_tests` | 6 | 6 |
| `syrk_tests` | 6 | 6 |
| `syr2k_tests` | 4 | 6 |
| `trmm_tests` | 8 | 8 |
| `gemv_tests` | 20 | 20 |
| `ortho_tests` | 8 | 8 |
| `orgqr_tests` | 8 | 8 |
| `ormqr_tests` | 12 | 12 |
| `ormqr_blocked_tests` | 10 | 20 |
| `cond_tests` | 20 | 22 |
| `syev_blocked_tests` | 36 | 24 |
| `syevx_tests` | 0 | 67 |
| `syev_two_stage_tests` | 0 | 16 |
| `linalg_layer_tests` | 0 | 7 |
| `options_api_tests` | 0 | 8 |
| `iluk_tests` | 0 | 4 |
| `inverse_tests`, `sytrd_sy2sb_tests`, `lanczos_tests`, `backend_dispatch_tests`, `syev_cta_tests` | 0 | 1–2 each |

**Four suites now fail only because of the host path.** That is WP9 (the CPU story), not a
missing GPU kernel, and BatchLAS's stated purpose is batched GPU work.

**Action for whoever picks this up: split the burn-down by backend.** A single pass count over
a suite that exercises two backends with different coverage cannot distinguish "we shipped a
kernel" from "we shipped nothing". This is the same lesson as `linked` vs `reachable` above,
one level up.

## After WP7 (and its repair pass) — 34 / 56, and the failing SET is the reviewable artefact

    ctest --test-dir build-novendor -LE slow
    61% tests passed, 22 tests failed out of 56          <- i.e. 34 PASSED

**Read that line carefully.** `N tests failed out of M` is a FAILURE count. The pass count is
`M - N`. This has been misread in this campaign before.

`gemv_tests` is the suite that left the failing set, and **it is the only one**. Vendor-free it
went **40 FAILED → 0 FAILED**, including the 20 `Backend::NETLIB` rows that run on a
`native_cpu` `Device("cpu")` queue — which is the whole reason `RouteTable<Op::gemv>`'s
`Direct` arm carries **no `is_gpu` clause**, the only native tier in this campaign that does
not. Proved by RESOLVED ROUTE rather than by symbols (`BATCHLAS_COVERAGE_OUT`, 160 reached
`gemv` rows, vendor-free):

```
     24 CUDA   native:cta     transA=1        40 NETLIB native:direct transA=0
     16 CUDA   native:cta     transA=2        24 NETLIB native:direct transA=1
     40 CUDA   native:direct  transA=0        16 NETLIB native:direct transA=2
```

The CPU half takes `native:direct` for **all three** `transA` values; the GPU half takes
`native:cta` for the transposed ones. `transA` appears as its own column, so `GemvShape` is
not shadowing `OpShape::transA` and gemv's two arms — which are different KERNELS, not
different flags — did not collapse into one first-writer-wins row.

### The ninth blind guard, found after the suite was already rewritten

WP7 replaced `gemv_tests`' blind fixture with 192 new cases — and every one of them still used
the **natural batch stride** (`a_stride == ld*n`, `x_stride == size*inc`, `y_stride == size*inc`).
A kernel that *derived* each batch stride instead of reading it from the view therefore passed
all 232 cases. That is a live property, not a hypothetical one: `src/extensions/ortho.cc:218-222`
hands the native path `A.stride() == m*A.cols()` against a view whose `ld*cols` is `m*i`, on every
CGS iteration — so until now the only thing guarding stride handling was `ortho_tests`, a
different suite, by accident.

Four `stride_pad` cases, one per kernel body, take the suite to **264**. Break `padstride` (all
four bodies compute `stride_a = ld*cols` and `stride_x/y = size*inc`) turns **exactly 32 tests
RED — the four new cases across all eight typed suites, and nothing else.** Both halves of that
number matter: the new cases are armed, and the 232 that preceded them are proven blind.

The lesson is the one the campaign keeps re-learning in a new costume: *a rewritten test suite is
not automatically an armed one.* The rewrite was driven by a list of degrees of freedom, and
batch stride was not on the list, so it inherited the original fixture's single blind spot
unchanged through 192 new cases.

The same run in the **vendor-present** build gives **160 reached rows, all `vendor:auto`**.
`preferred()` ships all-false, so WP7 is route-neutral there, and `scripts/route_diff.sh`
agrees: **0 removed decisions** in both builds.

### The 22 failing suites, after WP7

Recorded as a SET, because a suite leaving it means a native kernel now covers that op and a
suite joining it is a regression. Reproduced identically on two full runs:

```
options_api_tests   syevx_tests         lanczos_tests       trsm_tests
ortho_tests         cond_tests          ormqr_tests         ormqr_cta_tests
ormqr_blocked_tests orgqr_tests         iluk_tests          symm_tests
hemm_tests          herk_tests          her2k_tests         syrk_tests
syr2k_tests         syev_tests          trmm_tests          sytrd_blocked_tests
syev_cta_tests      syev_blocked_tests
```

**The pre-WP7 set is these 22 plus `gemv_tests` — 23 names.** It is written down here
explicitly so the next work package's auditor can diff SETS rather than infer from failure
text, which is what WP7's own auditor had to do.

**NOTHING JOINED, and that is measured rather than inferred.**
`ctest -LE slow --rerun-failed --output-on-failure` over all 22 suites produces 3,957 lines,
and `grep -ci gemv` on it returns **0**. Every failure is a `NoRouteError`, and the ops they
name are:

| op | occurrences | op | occurrences |
|---|---|---|---|
| `syev` | 87 | `syrk` | 12 |
| `geqrf` | 44 | `her2k` | 12 |
| `trsm` | 32 | `hemm` | 12 |
| `ormqr` | 24 | `syr2k` | 10 |
| `trmm` | 16 | `symm` | 8 |
| `herk` | 16 | `spmm` | 2 |
| `getri` | 16 | | |

No suite fails for a `gemv` reason, and `gemv` is the only op WP7 touched.

### Two suites that improved without leaving the set, and one that is not WP7's

* `ortho_tests`: **16 FAILED → 8**. All 8 remaining are `Backend` 6 = NETLIB, naming `geqrf` —
  the host path, i.e. WP9, not a missing GPU kernel.
* `cond_tests`: **30 → 24**. Does not close, exactly as forecast: the residue is
  `NETLIB`/`getri` (WP9) plus `src/extra/cond.cc`'s `syev_vendor_or_throw` bypass, which is
  filed in `WP7_FILED_DEFECTS.md`.
* `lanczos_tests` fails in **both** builds and is **not WP7's**. Verified rather than assumed:
  its coverage dump contains only `linked,gemv` rows and **zero `reached` rows** — it never
  calls `gemv` at all — and it fails identically vendor-present (where the whole suite is
  55/56) and vendor-free.

### The instrument caveat this pass added

**A `gemv` coverage row cannot confirm that a particular SHAPE ran.** `src/dispatch/coverage.cc`
keys rows on a power-of-two `shape_class` and is first-writer-wins, so the m/n/batch columns
can report a *different* call's shape. Two `gemv` coverage tests at m=41, n=76, batch=5
produce no row of their own — they collapse into the m=70, n=48, batch=6 row. To prove a
specific shape ran, use a break that is red only for that shape. The `linked` vs `reachable`
lesson, one level further down.

## After WP8 — 35 / 57, and the suite count is the WRONG METRIC here

    ctest --test-dir build-novendor -LE slow
    61% tests passed, 22 tests failed out of 57          <- i.e. 35 PASSED

**Neither half of that line records what WP8 did.** The 22 failing names are byte-identical to
the post-WP7 set — nothing left it, nothing joined it — and the one suite that joined the *run*
is `spmm_tests` itself, which passes **368 / 368** vendor-free. So `34/56 → 35/57` is the new
suite counting itself, and read alone it says WP8 did nothing.

The plan predicted exactly this, and the metric that actually moved is the per-op
`NoRouteError` census over `ctest -LE slow --rerun-failed --output-on-failure`:

| op | after WP7 | after WP8 | op | after WP7 | after WP8 |
|---|---|---|---|---|---|
| `syev` | 87 | 87 | `syrk` | 12 | 12 |
| `geqrf` | 44 | 44 | `her2k` | 12 | 12 |
| `trsm` | 32 | 32 | `hemm` | 12 | 12 |
| `ormqr` | 24 | 24 | `syr2k` | 10 | 10 |
| `trmm` | 16 | 16 | `symm` | 8 | 8 |
| `herk` | 16 | 16 | **`spmm`** | **2** | **0** |
| `getri` | 16 | 16 | | | |

`grep -c 'no route for spmm'` is **0**. Every other count matches the recorded baseline digit
for digit, which is both the deliverable and the evidence that nothing else moved.

**Why the suite count cannot show it** — the same lesson as WP3's, one op further on. The
suites that still fail vendor-free fail on *other* ops, and the two suites that consume `spmm`
are the clearest cases:

* `iluk_tests` is **17/17 vendor-present**; its four vendor-free failures are all named
  `Syevx*`, i.e. downstream of `syev`'s 87 `NoRouteError`s. It is a `syev` row, not an `spmm`
  row.
* `lanczos_tests` fails in **both** builds and is not WP8's. Re-checked rather than assumed:
  `lanczos` does consume the moved `spmm` gather, so it was re-run with
  `BATCHLAS_SPMM_ROUTE=vendor` and produced the **same two failing cases**
  (`LanczosTestBase.LanczosTest`, `LanczosTestBase.ToeplitzEigenpairs`).

Vendor-present, for the record: `ctest -LE slow` **56/57**, and `spmm_tests` unpinned goes
**276 passed / 92 skipped / 0 failed → 282 / 86 / 0** with the clause. Those six recovered cases
are the type-conditional showing up as a test-count delta: `transA = NoTrans` with
`transB = Trans`/`ConjTrans` on `Backend::NETLIB` now takes the native gather instead of
netlib's blanket transpose refusal — for `float`, `double` and `complex<double>` only.
`complex<float>` keeps all 23 of its skips, because the clause refuses exactly that family.

### `BATCHLAS_SPMM_ROUTE=vendor` turns 92 skips into 92 FAILURES, and never did anything else

Worth writing down before someone runs it as a control and reports a regression. Pinning the
vendor gives **276 passed / 92 FAILED**, and:

* all 92 are `Backend::NETLIB`; the same pinned run on `Backend::CUDA` is **184/184 passed**;
* the cause is two pre-existing lines — `netlib_lapack.cc:259` hard-throws on any transpose,
  and `tests/spmm_tests.cc:499` disables its refusal-skip whenever `BATCHLAS_SPMM_ROUTE` is
  set (`transposed && !route_pinned`). That gate reads the environment, never `preferred()`;
* 92 is **precisely the old unpinned skip count**, i.e. the pin converts pre-existing skips
  into failures and nothing else;
* the clause provably does not participate: `BATCHLAS_COVERAGE_OUT` on that exact run shows
  **144 `spmm` `reached` rows, all `vendor:auto`** — a forced vendor route returns before
  `preferred()` is ever consulted.

"Pin the vendor to prove the vendor arm still works" does not do what it sounds like in this
suite.

### The eleventh blind guard: a bound that nothing in a 352-case suite could see

`tests/spmm_tests.cc` shipped a `PaddingAboveNnzIsNotReadTrans` case built to prove the
scatter body bounds its nonzero loop by **each item's own row offsets** rather than by
`A.nnz()`, which is the per-item *capacity* (the batch maximum). Breaking that bound left the
**entire 352-case suite green**. A kernel reading uninitialised NaN padding on every transposed
call would have shipped.

**The reason is that the poison was unreachable, not that the axis was uninteresting.** The
case poisoned the padding with a `NaN` value at an **out-of-range column index** (2^30). The
scatter's own defensive range guard (`spmm_native.cc:541-542`) discards an out-of-range column
*before* the value is ever multiplied — so both halves of the poison went in the bin together,
and the over-read left no trace in `C`.

The guard is **correct and it stays**: in the gather a bad column index is an out-of-range
read; in the scatter it is an out-of-range **atomic write**, i.e. heap corruption. The *test*
is what had to change. The padding now carries an **in-range** column — an output row the
scatter will happily accumulate into — and a **large finite sentinel** rather than a NaN
(`8192` against live data of magnitude ≤ 1), so an over-read lands in `C` where the
backward-error comparison names it, instead of turning into a NaN that any of several unrelated
bugs could also produce. With that poison the break is red, and its named controls stay green.

Two meta-findings from the same exercise, both worth more than the fix:

* **A test can be defeated by the kernel's own defensive code.** The poison must be a value the
  kernel will *accept and use*, not one it is entitled to discard. Nothing about the case's
  wording revealed that; only running the break did.
* **`git diff` cannot verify the revert of an UNTRACKED file.** `src/sycl/spmm_native.cc` is
  new in this package, so after a deliberate break `git diff` reports **nothing** — an
  assertion that cannot fail, which is the same defect class as the blind guard it was being
  used to police. The recipe recorded in the file is an `md5sum` of the pristine source taken
  **before** the first break and compared after the last.

That is the eleventh blind guard recorded in this campaign. (This document last numbered the
ninth, at WP7; the counters in `tests/gemv_tests.cc` and `src/sycl/gemv_native.cc` ran ahead of
it during the WP7 repair pass. The count is approximate; the pattern is not.)

### The instrument caveat this pass added: `route_diff.sh compare` needs a `backend != AUTO` filter

The raw before/after capture diff shows **65 removed and 175 added `reached` lines**, which
looks like far more churn than the 65 decisions that actually moved. **110 of those additions
are fabricated pure-layer shapes** resolved by this pass's seven new `route_vocabulary_tests`
cases, recorded with `backend = AUTO`. `route_vocabulary_tests.cc` carries its own standing note
to filter those out before concluding anything about what the *library* routes — but
`route_diff.sh`'s `compare` does not apply the filter, so `compare` alone makes a clean
65-decision move look like 240 lines of churn.

`experiments/sparse_spmm/route_census.py` keys on the decision tuple
(op + scalar + backend + shape_class + uplo/side/diag/transA/transB) and splits `AUTO` from real
backends. After that filter: **65 moved decisions, all `spmm`, all `vendor:auto` →
`native:direct`, all `transA = 0`; zero non-`spmm` decisions moved, zero disappeared, and the
added-key set is empty.** This is arguably a fifth defect of the coverage instrument, in the
same family as the four already recorded.

### One comment that was false, and the break that caught it

Recorded because it is the campaign's own lesson in a new costume. This pass first wrote — in
both `route_spmm.hh` and `route_vocabulary_tests.cc` — that now the clause has landed, a
reversed `kSpmmOrder` would silently send every admitted shape back to cuSPARSE. **It does
not.** `automatic()`'s first walk skips any route whose `preferred()` is false, and
`preferred()`'s first line refuses the vendor (`if (!is_native(r) ...)`), so the vendor entry is
skipped wherever it sits; the vendor-free walk filters on `is_native()` and has one candidate.
The reversal was applied, rebuilt and run: **exactly one case goes red, `OrderIsExactlyTwoEntries`,
and structurally rather than through any routing decision.** Both comments now record the
correction and the break result. A plausible mechanism asserted instead of measured — caught
only by running the break.

## The same gap, per op

`cmake --build build-novendor --target batchlas_coverage` writes `coverage.csv`, whose
`linked` rows answer this exactly and without running anything:

| native kernel linked | ops |
|---|---|
| yes | `gemm`, `ormqr`, `syev`, `gesvd` |
| **no** | `gemv`, `trsm`, `trmm`, `symm`, `syrk`, `syr2k`, `hemm`, `herk`, `her2k`, `geqrf`, `orgqr`, `getrf`, `getrs`, `getri`, `potrf`, `spmm` |

> **This table is the WP0 reading and it is NOT maintained — do not use it as a status
> board.** WP3–WP8 have since landed native `trsm`, `potrf`, `geqrf`/`orgqr`, `getrf`/`getrs`,
> `gemv` and `spmm`, none of which is reflected above — the `no` column is now empty in fact,
> and `spmm` was the last entry to leave it. It is left as recorded so the WP0 → today diff
> stays legible.
>
> It is also **not trustworthy op by op**, which is worth knowing before anyone regenerates
> it. In the post-WP7 vendor-free capture the `linked` rows report
> `native_route_existed = 0` for `trsm` — after WP3 shipped a native `trsm` — and `= 1` for
> `getri`, for which WP9 has not started. The `linked` half of the coverage instrument answers
> "does this build have a native route registered for this (op, scalar, backend)", which is
> not the same question as "is there a native kernel", and it is stale for both. **Read the
> `reached` rows and the resolved route.** A kernel being linked is not evidence it runs; a
> `linked` row saying 0 is not evidence it does not exist.

`miss` rows add what a run actually reached, e.g.

    miss,getri,float,CUDA,,,,,,,,1,0,0,cuBLAS

## Why so many, and what closes them

The count is dominated by transitive dependence rather than by breadth of
missing ops. `ortho` alone needs `potrf`, `trsm`, `geqrf`, `orgqr` and `gemv`,
and `ortho` sits under `syevx`, `lobpcg` and `lanczos` — so one missing native
`trsm` fails a dozen suites.

`gemm` is the interesting entry in the list, and the first version of this
paragraph got it **wrong** in a way worth keeping visible.

It claimed the native register-tiled kernel "exists and is vendor-free, so
`gemm_tests` fails only on the shapes outside `gemm_custom_problem_supported`".
The kernel does exist and is vendor-free. It is also **unreachable**: the facade
(`src/dispatch/entry_points/level3.cc:60-64`) is

```cpp
if constexpr (!dispatch::level3_vendor_available<Back>) { throw_no_vendor_route(...); }
else { return backend::gemm_vendor<Back, T>(...); }
```

and the three-way route that would pick the native kernel lives in
`backend::gemm_vendor` at `cublas.cc:153-178` — inside the cuBLAS-gated TU. So a
vendor-free `gemm` throws on *every* call. Measured, not argued:
`build-novendor/tests/gemm_tests` is 48 passed / 136 failed, and every one of the
48 is a pure route-resolution test (`GemmDispatchPolicyTest.*`,
`GemmTest/*.RouteAdapter*`, all 0 ms) that never executes a GEMM. The failures
are `NoRouteError: no route for gemm<float> on this backend`.

**`linked` is not `reachable`, and the table above reports `linked`.** That is
the distinction the whole burn-down turns on, so read the `yes` column as "the
kernel is in the build", never as "the op works". Closing the gap for `gemm` is
WP1 S5; until then `gemm`'s `yes` is a statement about the linker.

The four level-3 tile routes (`symm`/`syrk`/`syr2k`/`trmm`) are portable SYCL
already, but their dispatch still terminates in `*_vendor_cuda_raw`, so they are
compiled only with cuBLAS. WP1 frees them — but see `WP1_LEVEL3_SPEC.md`, which
corrects three claims made here and in the plan:

- `symm` carries **no tile kernel**; its only portable kernel is the mirrored
  expansion, and everything after it is a GEMM.
- The four routes are reachable **only from `cublas.cc`** (`:269`, `:763`,
  `:882`, `:1005`). Compiling the TUs everywhere makes them linked everywhere
  and called nowhere.
- Routing the terminal at the public `gemm` is the right destination but is not
  sufficient, per the `gemm` correction above: it converts a `NoRouteError`
  naming `symm` into one naming `gemm`.

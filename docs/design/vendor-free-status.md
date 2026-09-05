# Vendor-free status: where it stands and what is left

BatchLAS is meant to build, link, load, run and perform without cuBLAS, cuSOLVER, cuSPARSE,
rocBLAS, rocSOLVER, rocSPARSE, oneMKL or netlib LAPACK — while still *using* any of them when
they are present and genuinely faster. Work packages WP0–WP8 delivered the dispatch machinery
and a native SYCL kernel for every public dense op plus sparse `spmm`. WP9 has not started.

**M1 is not reached.** The vendor-free build configures, compiles, links, loads and runs, and
that is real; the full suite is not green, and the remaining gap is an enumerated list of
missing kernels and unrouted host paths rather than an unknown.

This page is the status board. It does not carry per-op performance evidence — that lives in
[`../perf/`](../perf/README.md), one page per op, and every ratio quoted here links there. How
dispatch itself works is [`vendor-independence.md`](vendor-independence.md); located, unfixed
bugs are [`known-defects.md`](known-defects.md).

## The two build configurations

| | vendor-present | vendor-free |
|---|---|---|
| configure | default | `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` |
| result | `BATCHLAS_HAS_CUBLAS` / `CUSOLVER` / `CUSPARSE` = 1 | `BATCHLAS_HAS_CUDA_BACKEND 1` with **every CUDA math library at 0** — a CUDA device with no CUDA math libraries, a state the pre-WP0 scheme could not express and could not link |
| links | yes | yes |
| `ctest -LE slow`, last recorded (end of WP8) | **56 / 57** | **35 / 57** |

The one vendor-present failure is `lanczos_tests`, pre-existing and reproduced by rebuilding
with the campaign's changes reverted; its coverage dump holds only `linked` rows and **zero
`reached` rows** for `gemv`, i.e. it never calls the op it was once blamed on.

`BATCHLAS_ENABLE_VENDOR_BLAS` is a master switch over per-library options
(`cmake/BatchLASOptions.cmake:180-198`), and the vendor list deliberately includes
**cuBLASDx/cuSOLVERDx**: they are third-party NVIDIA source that ships only for NVIDIA, so a
vendor-independence measurement that let them through would be measuring the wrong thing
(`:185-189`). MathDx is absent on this box (`BATCHLAS_HAS_CUBLASDX 0`), so every "cublasdx"
route in the tree is silently its fallback.

Both pass counts are the last **recorded** runs, not re-run for this document. Treat them as
provenance and re-derive before quoting — and read `N tests failed out of M` as a *failure*
count. That line has been misread in this campaign before.

## Why the ctest pass count is the wrong instrument

`ctest` runs each level-3 and factorization suite against the **host (`Backend::NETLIB`)
backend as well as CUDA**, and a vendor-free build has no netlib LAPACK either. A suite can
therefore fail entirely on host rows while every CUDA case in it passes. Two demonstrations:

* **`trsm` after WP3.** The suite number did not move at all, and vendor-free `trsm_tests` was
  59 passing / 32 failing with **all 32 failures on the host backend and not one on CUDA**. On
  the GPU, vendor-free `trsm` was complete — every order, both sides, all four scalar types —
  and the suite-level count could not show it.
* **`spmm` after WP8.** `34/56 → 35/57` is the **new suite counting itself**: `spmm_tests`
  joined the run and passes 368/368 vendor-free, while the 22 failing names stayed
  byte-identical to the post-WP7 set. Read alone, the number says WP8 did nothing.

The structural reason every native tier in this campaign is invisible to the host half:
`supports()` carries `if (!s.is_gpu) return false;` for `geqrf` (`route_geqrf.hh:48`), `orgqr`
(`route_orgqr.hh:32`), `ormqr` (`route_ormqr.hh:59`), `getrf` (`route_getrf.hh:41`), `getrs`
(`route_getrs.hh:52`), `getri` (`route_getri.hh:39`), `potrf` (`route_potrf.hh:38`), `trsm`
(`route_trsm.hh:39`) and `gemm` (`route_gemm.hh:34-67`). **`gemv`'s `Direct` arm and `spmm`'s
gather are the only two exceptions in the tree** — both run on a `native_cpu` `Device("cpu")`
queue, which is exactly why `gemv_tests` went 40 failed → 0 vendor-free when nothing else did.

### The honest metric: the per-op `NoRouteError` census

Every vendor-free failure is a `NoRouteError` naming an op, a scalar type and the switch that
would restore it (`no_route.hh:36-53`, `:62-72`). Counting those over

```
ctest --test-dir build-novendor -LE slow --rerun-failed --output-on-failure
```

gives a number that moves when a kernel lands. As recorded after WP8:

| op | count | op | count |
|---|---|---|---|
| `syev` | 87 | `getri` | 16 |
| `geqrf` | 44 | `syrk` | 12 |
| `trsm` | 32 | `her2k` | 12 |
| `ormqr` | 24 | `hemm` | 12 |
| `trmm` | 16 | `syr2k` | 10 |
| `herk` | 16 | `symm` | 8 |
| | | **`spmm`** | **0** (was 2) |

WP8's deliverable is the `spmm` row going to zero **with every other digit unchanged** — that
is both the result and the evidence that nothing else moved. Diff the *census*, and diff the
failing *set* of suite names; do not diff the pass count.

The static `linked` half of the coverage instrument is not a substitute. It answers "does this
build have a native route *registered* for this (op, scalar, backend)", not "is there a native
kernel", and it is stale in both directions: `src/dispatch/coverage.cc:168` still reports
`trsm` as having no native route, two work packages after WP3 shipped one. Read the `reached`
rows and the resolved route.

## What has a native kernel, and what routes to it by default

Every public dense op and `spmm` now has a native SYCL kernel. What differs is whether
`preferred()` sends traffic to it in a **vendor-present** build. Vendor-free, an un-preferred
native route still runs: `automatic()` accepts a merely *supported* native route when
`vendor_available == false` (`route_resolve.hh:34-51`).

| op | native arms (`order` sequence) | `preferred()` in a vendor-present build | where |
|---|---|---|---|
| `gemm` | `RegisterTiled` | GPU, homogeneous, `batch >= 64`; **`double` at `k >= 2`**; `float` NN square `max_dim <= 32`; **complex never** | `route_gemm.hh:34-67` |
| `gemv` | `CTA`, `Direct` | one window: `complex<double>`, transposed, `64 <= red_len <= 352`, `out_len >= 256`, `batch >= 320` | `route_gemv.hh:60-71` |
| `trsm` | `CTA`, `Blocked` | native from `batch >= 8`; `float`/`Side::Right` additionally needs `batch >= 128 \|\| order <= 32`; everything else true | `route_trsm.hh:64-80` |
| `potrf` | `CTA`, `Blocked` | **false everywhere** | `route_potrf.hh:65-69` |
| `geqrf` | `CTA`, `Blocked` | **false everywhere**; tier choice via `native_tier_preferred` (`:443`) | `route_geqrf.hh:73-77` |
| `orgqr` | `Blocked` | **false everywhere** | `route_orgqr.hh:60-64` |
| `ormqr` | `Blocked` | `is_native(r) && supports(r, s)` — native-first, and predates WP5 | `route_ormqr.hh:77-79` |
| `getrf` | `CTA`, `Blocked` | `float` order ≥ 256, `cfloat` order ≥ 512 | `route_getrf.hh:67-74` |
| `getrs` | `CTA`, `Blocked` | CTA at `nrhs <= 2` (all types) and `nrhs <= 4` (`float`); Blocked at `batch >= 128` with `float nrhs >= 64` / `double nrhs >= 128` | `route_getrs.hh:79-98` |
| `getri` | `Blocked` | `float` order ≥ 128, `cfloat` order ≥ 256 | `route_getri.hh:65-72` |
| `spmm` | `Direct` | CSR and `transA == NoTrans`, minus `complex<float>` with `transB != NoTrans` | `route_spmm.hh:65-75` |
| `syev` | `CTA`, `Blocked`, `TwoStage` | a measured per-`n` grid, `Backend::CUDA` only | `syev.hh:357-385` |
| `gesvd` | `Jacobi`, `CTA`, `Blocked` | the wide-band rule | `route_gesvd.hh:100` |

Two families sit outside this table and must not be read from it:

* **`symm`, `syrk`, `syr2k`, `trmm` have no `RouteTable` and never call `resolve_route`.** Their
  thresholds are hand-rolled `if`-chains in the facade, guarded
  `Back == Backend::CUDA && std::is_same_v<T, float>` (`entry_points/level3.cc:189`, `:380`,
  `:418`, `:457`), and they run **before** the vendor-available test — so anything below that
  gate is unreachable vendor-free. `symm` has no tile kernel at all; its portable arm is a
  mirrored expansion feeding the public `gemm`.
* **`hemm`, `herk`, `her2k` have no native arm in the facade whatsoever** — vendor or throw
  (`entry_points/level3.cc:210-361`). Their expansion routes are reachable only from inside
  `cublas.cc`.

### Vendor-first by measurement, not by absence

This distinction is the campaign's main product and the easiest thing to get wrong. The
following are vendor-first because the vendor **won**, and re-doing the kernel work will not
change that:

* **`gemv`, almost everywhere.** cuBLAS `gemvStridedBatched` runs at 94–105% of the achievable
  DRAM roof on 90 of 92 reproducing cells. There is nothing to take —
  [`gemv.md`](../perf/gemv.md).
* **`spmm`'s transposed scatter.** 169 of 458 saturated cells lose, worst 3.011, and **no
  shape-expressible clause recovers a window**. It also has zero in-tree C++ callers —
  [`spmm.md`](../perf/spmm.md).
* **`gemm` for complex.** Not merely unpreferred — there is **no register kernel**: the entire
  register ladder in `select_kernel_variant` sits inside `if constexpr (is_same_v<T,float>)`, so
  complex falls to `Direct`/`Tiled16`, measured 3.2–7.1× slower than cuBLAS. Widening the
  predicate first is a regression. Order: port the tile → wire the selector → move the
  predicate — [`gemm.md`](../perf/gemm.md).
* **`getrf`/`getri` in `double` and `complex<double>`.** They earn nothing at any order; the
  windows are `float`/`cfloat`-leaning on purpose — [`lu.md`](../perf/lu.md).
* **`potrf` in complex (0.311–0.509×) and at `n <= 256` for every type.** The complex cause is
  outside the driver — it is the missing complex GEMM above —
  [`potrf.md`](../perf/potrf.md).

And these are vendor-first because **no decision was taken**, which is a different debt:

* **`geqrf` and `orgqr`.** `preferred()` is false everywhere while the measured geomeans are
  **3.24×** and **7.85×**. That value is unrealised in the default build and is the single
  largest piece left on the table. Flipping it is gated on an end-to-end harness, not a kernel
  ratio: this tree has turned a 2.16× kernel win into an 11% `gesvd` loss.
* **`potrf`.** `preferred()` all-false, though vendor-free `potrf` works at every order and
  `float` at `n >= 1024` is **1.13–1.40× faster than cuSOLVER**. `potrf` also still has not
  declared `native_tier_preferred`, although it has the same two native tiers and the same
  all-false `preferred()` as `getrf` — so its vendor-free tier choice is a static order walk
  that cannot follow a crossover. Whether that choice is wrong is unmeasured; measuring it is
  the first step, not fixing it.

## M1 — self-sufficient. Not reached.

**Definition.** `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` configures, builds and passes the full
`ctest` suite. No performance claim whatsoever; the vendor stays first in every Auto order.

**What it still needs**, in descending census order:

1. **The host (`Backend::NETLIB`) path — this is WP9, and it is most of the residue.** Every
   native tier except `gemv`'s `Direct` and `spmm`'s gather refuses a non-GPU queue in
   `supports()`. All 32 vendor-free `trsm` failures, all 12 `sytrd_blocked`, all 8 remaining
   `ortho_tests` (naming `geqrf`) and 20 of `cond_tests`' failures (naming `getri`) are this and
   nothing else.
2. **`syev`, 87 — the largest single entry, and part of it is a routing-vocabulary defect rather
   than a missing kernel.** Four call sites reach into `dispatch::detail` and demand the
   *vendor* `syev` instead of resolving a route, so they throw vendor-free by construction
   regardless of what `syev`'s three native tiers support: `src/extra/cond.cc:52`,
   `src/extra/norm.cc:45`, `src/extensions/syevx_lobpcg.cc:532` and `:1282`. The recorded
   measurement attributes 6 of `cond_tests`' 30 vendor-free failures to the `cond.cc` one. The
   fix is to call the routed `syev` and let `resolve_route` decide.
3. **`hemm` 12, `herk` 16, `her2k` 12 — no native arm exists.** The facade is vendor-or-throw for
   all three.
4. **Level-3 non-float: `trmm` 16, `syrk` 12, `syr2k` 10, `symm` 8.** `syrk`'s gram branch and
   `trmm`'s tile branch for `double`/complex are reachable only from `cublas.cc`, and **`syr2k`
   has no non-float tile route at all** — `syr2k_triangular_tiles` has exactly one call site in
   the tree, inside the float-only dispatcher. `double` `symm` has no expansion route at all.
5. **`geqrf` 44, `ormqr` 24, `getri` 16** — host rows as above, plus the shapes each table's
   `supports()` refuses.
6. **`potrf` refuses `Uplo::Upper`** in the blocked driver (`route_potrf.hh:55`), and that is
   the correctness kind of false, not the slower kind. `syev` shows the cheap route: mirror the
   upper triangle and run the Lower pipeline.

## M2 — vendor-free by default, cell by cell

**Definition.** For each (op, type, shape class) where the native path meets

```
t_native <= 1.10 * t_vendor    at saturated, large-batch shapes
```

and stays inside the op's accuracy tolerance, native moves ahead of Vendor in the Auto order.
A cell that fails the gate stays vendor-first and is **published** rather than quietly papered
over. That is a legitimate outcome, not a failure — it preserves M1 while being honest about M2.

Keeping M1 and M2 separate is the single most important structural decision here, and it paid:
**M2 is reached for several ops while M1 is still open.** `gemm`'s unset default went `Vendor` →
`Auto` (WP2 E6) and `trsm` prefers native in 167 of 168 measured cells — both shipped, both
measured — while the suite is nowhere near green because other ops had no kernel at all. Under
a conflated milestone, none of that would have shipped.

Reached as measured windows: `getrf`, `getrs`, `getri`, `gemv` and `spmm`'s gather. Not
reached, by measurement: `potrf`, `spmm`'s transposed scatter, `gemm` for complex. Not reached
for want of an end-to-end measurement: `geqrf`, `orgqr`.

## Remaining work

### WP9 — the CPU story

Not started. The decision it needs is explicit: does a CPU SYCL device have to be *fast*, or
merely *correct*?

**The standing recommendation is correct and not embarrassing, nothing more.** The CPU BLAS
market is well served by MKL and OpenBLAS, both still reachable through the existing backends,
and BatchLAS's purpose is batched GPU work. Spending WP2-grade tuning effort on CPU SYCL
kernels would be the worst value in this campaign. Mechanically it is small: `gemv` and `spmm`
already demonstrate that dropping the `is_gpu` clause from `supports()` is all a tier needs to
serve the host queue, and it moved the vendor-present burn-down by zero both times.

Two traps for whoever picks it up: `Backend::INTEL` is hard-wired false and oneMKL cannot be
tested on this box; and a CUDA-off `ctest` shows roughly 30 failures that are artefacts of the
CPU-only verification build, not regressions.

### Measured but unrouted — levers with a number already attached

Each of these has a measured win and no route. Ordered by the size of the recorded win, not by
effort.

| lever | measured | where |
|---|---|---|
| `geqrf` / `orgqr` default flip | 3.24× / 7.85× geomean, unrealised | [`qr.md`](../perf/qr.md) |
| `getri` at `batch <= 32`, **every** type | 1.7–28× over cuBLAS, whose batched `getri` is a per-item loop there | [`lu.md`](../perf/lu.md) |
| `potrf` `float` at `n >= 1024` | 1.13–1.40× over cuSOLVER | [`potrf.md`](../perf/potrf.md) |
| `getrs` cells handed to the vendor by clauses A and B | 84 winning cells, largest 3.944× | [`lu.md`](../perf/lu.md) |
| `gemv` `out_len >= 768 && batch >= 128` | ~18 cells at 2.26–2.91×, declined only because the batch floor is the edge of the sampled range | [`gemv.md`](../perf/gemv.md) |
| `gemv` batch floor 320 → ~288 | six measured wins, 1.27–4.45×; 288 was never in the threshold list the search enumerated | [`gemv.md`](../perf/gemv.md) |
| `getrs` clause-C batch floor 128 → 32 | float 3.87–5.96×, double 3.56–4.31× at `nrhs = 128` | [`lu.md`](../perf/lu.md) |
| `spmm` gather narrowed to `nrhs >= 16` | 183 cells instead of 176 at worst 0.968; refused because its axis is the column pattern | [`spmm.md`](../perf/spmm.md) |
| a complex register-tiled GEMM | ~2.7× on vendor-free `cdouble potrf` alone, and it is what unblocks complex `gemm`, `potrf` and the level-3 complex arms | [`gemm.md`](../perf/gemm.md), [`potrf.md`](../perf/potrf.md) |

### Known wrong, deliberately deferred

Preserved rather than fixed in passing, because each is a route change that needs its own
measurement:

* **`BATCHLAS_SYRK_ROUTE=native` returns a wrong answer.** `{Native, Auto}` passes
  `syrk_use_cuda_custom`, fails every arm inside `syrk_cuda_custom`, and lands on the
  `DiagFullGemm` fallback — **which writes both triangles**, clobbering the one the caller did
  not name. **No test in the tree sets `BATCHLAS_SYRK_ROUTE`.**
* **`BATCHLAS_SYR2K_ROUTE=native` throws a cuBLASDx message it did not ask for**; the throw is
  not guarded by `forced`.
* **`symm` has no `expansion_fits()` ceiling** where `hemm`/`herk`/`her2k` all have one, so a
  large enough `symm` hits the 2³¹-element SYCL range failure instead of falling back.
* **`trsm`'s heterogeneous-batch correctness gate can never fire.** `supports()` rejects the
  field at `route_trsm.hh:43`, but `trsm_op_shape` never *writes* it, so it keeps `OpShape`'s
  default `false`. A documented intention, not an enforced one.
* **`resolve_ormqr_route` is called with two arguments** (`ormqr.hh:209`), taking
  `vendor_available = true`, so `ormqr` never reaches the vendor-free fallback. It gets away
  with it only because its `preferred()` is native-first. Do not inherit the omission.
* **`cublas.cc`'s `getrs` sits in a TU gated on `BATCHLAS_HAS_CUBLAS`**, so a
  cuBLAS-present / cuSOLVER-absent configure claims a vendor it cannot link. The fix belongs in
  `vendor_available.hh`.
* Three call-site defects located and left alone by decision, with their reasoning, in
  [`known-defects.md`](known-defects.md): `ortho.cc`'s transposed `gemv` view, the `syev`
  resolver bypasses above, and `lanczos.cc`'s two-column `gemm` whose second column is
  discarded.

### Do not re-attempt these

The campaign's negative results are its most expensive knowledge. Each cost as much to
establish as a win, and each is the obvious next idea.

| idea | verdict |
|---|---|
| `syrk`/`herk` for `ortho`'s Gram matrix | **73–96× slower** at the shapes `ortho` issues; its callers all pass `k` = the block size, and the winning column is `k >= 512` and square-ish |
| `trmm` for the WY block factor | loses at every shape; and the re-measurement after the tile kernel landed splits **per type, not per precision** — complex still takes the GEMM |
| complex Gram tiles (`herk`) | loses to the GEMM-plus-Hermitian-fold everywhere; a complex multiply is four real ones, so `herk` is compute bound where real `syrk` is bandwidth bound |
| `syr2k` for the `sytrd_blocked` trailing update in `double` | 7.7× slower in the regime that matters; it wins only where the batch is small enough that per-item launch cost amortises. The route stays CUDA + float |
| the cooperative TRSM solve (W work-items per solve) | passes the register gate at order 128 in fewer registers than the shipped kernel needs at order 32 — and still measures 0.39× at order 64. The traffic model missed the serial recurrence |
| transcribing the level-3 gate thresholds into `RouteTable::preferred` (the "split-tu" WP1 design) | the live thresholds are **gate-only**, so a faithful transcription sends `129 <= n <= 383` to a route that writes both triangles |
| taking "the first merely supported route" unconditionally in `automatic()` | inverts GEMM's default for small shapes, because the order arrays list natives first |
| a compile-time coverage gate | `resolve_route` is an inline function template; a TU compiled without the macro interposes its uninstrumented copy by weak-symbol resolution and recording silently stops. `cmake/BatchLASOptions.cmake:109` records that the option was deliberately never added |
| `potrf`'s fold-free trailing update | measured 11% cheaper **and wrong** |

## How to re-derive this page

```
cmake -S . -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON
cmake --build build-novendor -j"$(nproc)"
ctest --test-dir build-novendor -LE slow
ctest --test-dir build-novendor -LE slow --rerun-failed --output-on-failure \
  | grep -o 'no route for [a-z0-9_]*' | sort | uniq -c | sort -rn
```

The last line is the census. For what *routes* rather than what throws, set
`BATCHLAS_COVERAGE_OUT` and read the `reached` rows — never the `linked` rows, and never a
symbol table. A kernel being linked has never been evidence that it runs, and that misreading
is how the vendor-free build was once recorded as having a working `gemm` while every
vendor-free `gemm` call threw.

The superseded root-level documents this page replaces — `VENDOR_FREE_BASELINE.md`,
`VENDOR_INDEPENDENCE_PLAN.md`, the five `WP*_SPEC.md` files and the two corrections files that
supersede two of them — are preserved verbatim at the tag `perf-evidence/vendor-independence`:

```
git show perf-evidence/vendor-independence:VENDOR_INDEPENDENCE_PLAN.md
```

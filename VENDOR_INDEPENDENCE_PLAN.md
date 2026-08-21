# Vendor Independence Plan

**Goal.** BatchLAS should build, run and perform without cuBLAS, cuSOLVER, cuSPARSE,
rocBLAS, rocSOLVER, rocSPARSE, MKL or netlib LAPACK present — while still *using* any of
them when they are available and genuinely faster.

Every factual claim about the tree was read out of the source and is cited by file and line
so it can be re-checked rather than believed. Several claims in the first draft turned out
to be wrong; each has been corrected in place with the correction called out, rather than
quietly edited.

## Status

**WP0, WP1, WP2 and WP3 are complete. WP4 is complete (Phases 1 and 2). WP5–WP9 are not started.**

| | state |
|---|---|
| Vendor-free build (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) | configures, compiles, links, loads and runs; `ctest -LE slow` **26/54** |
| Vendor-present build | **52/53**, the one failure pre-existing and unrelated |
| `gemm` | vendor-free **complete** (184/184); default flipped `Vendor` → **`Auto`**; complex still vendor-preferred |
| `trsm` | native, both tiers, all four scalar types; beats the vendor in **167 of 168** measured cells; vendor-free failures are **host-backend only** |
| level-3 tile ops (`symm`/`syrk`/`syr2k`/`trmm`) | free of the CUDA object library, and reached rather than merely linked |
| `potrf` | native CTA kernel **and** a blocked driver above its ceiling, all four types, both triangles; **vendor-free potrf works at every order** and, for `float`, is **1.13–1.40× FASTER than cuSOLVER** at n≥1024. Still route-neutral: `preferred()` false everywhere, because complex is 0.31–0.51× and small `n` loses |
| M1 (vendor-free `ctest` green) | **not reached** — the gap is missing kernels, and it is now an enumerated list |
| M2 (native-by-default per cell) | reached **for `gemm` and `trsm`**; every other op is still vendor-first |

**Companion specifications** (each the output of a multi-agent design pass with adversarial
critique, and each superseding the sketch in §5 of this document):

| Document | Covers | Agents |
|---|---|---|
| `WP0_DISPATCH_SPEC.md` | the dispatch axes, the vendor gate, the coverage instrument | 14 |
| `WP1_LEVEL3_SPEC.md` | freeing the four level-3 dispatchers from the CUDA backend | 12 |
| `WP2_GEMM_SPEC.md` | closing the GEMM envelope; the two-track split | multi-pass |
| `WP3_TRSM_SPEC.md` | native batched `trsm` — **read with `WP3_TRSM_SPEC_CORRECTIONS.md`, which supersedes it** | 19 (shared) |
| `WP4_POTRF_SPEC.md` | native batched `potrf` | 19 (shared) |

**Implemented and verified** (each step built clean and passed the tests named):

| Step | What | Verification |
|---|---|---|
| WP1 prep | `route_common.hh` split out of the CUDA-only `cublasdx_dispatch_common.hh` | `triangular_expand.hh` + the four `*_tiles.hh` compile at `-fsycl-targets=spir64_x86_64`, with a negative control |
| WP1 prep | missing `<complex>` in `triangular_tiles.hh` | same standalone compile |
| WP0 S1 | per-library probes, `BATCHLAS_HAS_<LIB>` in the generated header | configure clean; build exit 0; 7/7 smoke |
| WP0 S2a | vendor *includes* keyed on library, not family | `-DBATCHLAS_ENABLE_CUBLAS=OFF` ⇒ 0 cuBLAS/cuSOLVER includes, vs 2 in the normal build |
| WP0 S3 | each vendor TU gated on the library it calls | identical 17-TU object set to baseline; `gemm`/`symm`/`syrk`/`trmm`/`backend_dispatch` tests pass |
| WP0 S2b | vendor *types and handles* keyed on the library axis | `ortho.cc` compiles clean with `-DBATCHLAS_ENABLE_CUBLAS=OFF`, where it previously produced 20+ errors |
| WP0 S4a | the `Route` vocabulary (`Origin` × `Algorithm`) and the legacy env alias table, added additively | 17 new tests pinning every legacy spelling |
| WP0 S4b | GEMM's three-way split: env read / `supports` / `preferred` | route diff vs a transcribed replica over 10 env spellings × ~2,300 shapes × 4 scalar types; `ReplicaIsFaithful` guards the transcription |
| WP0 S4c | `gemm_use_sycl_custom` becomes an adapter over `resolve_gemm_route` — GEMM is wired | 6 typed adapter tests over live `MatrixView`s; full build; `ctest -LE slow` 52/53 |
| WP0 S4d | `ormqr` onto `Route` | 2 regression tests; full build; `ctest -LE slow` 52/53 |
| WP0 S4e | `gesvd` onto `Route`; the wide-band rule becomes `preferred` | translated routing test asserts both the default *and* that Jacobi still `supports` the shape |
| WP0 S4f | `syev` onto `Route`; `provider.hh`/`env.hh`/`context.hh` deleted | full `ctest` 56/58 — both failures reproduced with the change reverted |
| WP0 S4g | the four level-3 dispatchers onto `Route`; the last env parser deleted | 30 (variable, value) pairs swept before and after: byte-identical failure sets |
| WP0 S5 prep | `scripts/rocm_syntax_check.sh` — the ROCm TUs are checkable after all | all three PASS; a deliberately undeclared symbol makes it FAIL |
| WP0 S5a | `gemm`'s public definition leaves the vendor TUs; `mkl.cc` deleted | symbol check: absent from the cuBLAS component, present in the facade |
| WP0 S5b | the other nine level-3 entry points | `scripts/facade_symbol_check.sh` 10/10; ROCm check caught a bad `trsm` instantiation |
| WP0 S5c | the twelve factorization entry points (each with its buffer-size query) | signature divergence surveyed *first*; build clean first attempt; ctest 56/58 |
| WP0 S5d | `spmm`, and `syev`/`ormqr`'s instantiations | ROCm check caught 4 orphaned lines invisible to the CUDA build |
| WP0 S6 | the vendor-free build **configures, links, loads and runs** | `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`: build exit 0, ctest 20/53 with `NoRouteError` diagnostics; vendor-present ctest unchanged at 56/58 |
| WP0 S7 | the coverage instrument (`batchlas_coverage`) | static table names the 16 ops with no native kernel; `miss` rows verified against a known gap |
| WP0 S8 | six sites stop using the `Backend` enum to mean something else | vendor-present ctest 56/58 and vendor-free 20/53, both unchanged — no site silently flipped |
| WP1 S0 | the four level-3 routes become measurable | a 4-suite run goes from 96 rows for one op to 312 across five |
| WP1 S1 | a portable vendor-fallback seam replaces 10 `*_vendor_cuda_raw` calls | route diff 3016 decisions identical; capture CSVs byte-identical |
| WP1 S2 | the expansions' terminal GEMM becomes the public entry point | routes identical; timings within noise, measured at saturating batch |
| WP1 S3 | the cuBLASDx fused tails leave the dispatchers | `nm -C` finds **no** CUDA symbol in any of the four .o files |
| WP1 S4 | the four TUs leave the CUDA object library | vendor-free build links them with **no** CUDA object library present |
| WP1 S5 | the facade's `gemm` gains a native arm | vendor-free `gemm_tests` 48/184 → 167/184; suite 20/53 → 24/53 |
| WP1 S6 | the level-3 gates move to the facade | tile kernels **reached** vendor-free for the first time (measured, 41 native rows) |
| WP1 S7 | the tile predicate gains a scalar parameter instead of flipping | vendor-present unchanged by construction; failing set byte-identical |
| WP2 prep | measure what GEMM shapes the library actually **issues**, before tuning any of them | 23 134 non-float calls captured; with probe rows removed, 7 223 are real demand — and the wide-scalar gate fires on 0.64% of them, not 3.56% |
| WP2 C1–C3 | heterogeneous batch: the per-item loop becomes portable and the facade gains a vendor-free arm | vendor-free `gemm_tests` 167/184 → **184/184**; suite 24/53 → 25/53; vendor-present route diff moves **zero** decisions |
| WP2 E1–E2 | three in-tree claims corrected; the 64×64×16 t4×4 wide-scalar tile ported into `src/` | first register-tiled GEMM kernel for a non-float scalar; two of the three claims would have caused a wrong edit |
| WP2 E3 | double's window, settled first because it is 85% of the flip | native 1.05–4.51× at n=4..512; one misplaced `Direct`/`Tiled16` boundary fixed |
| WP2 E4 | float's window **narrows** on measurement | two regions `preferred()` claimed measure 0.34–0.97× and are removed; the predicated path it was gating on is worth 2–4.4× |
| WP2 E6 | **the flip** — GEMM's unset default goes `Vendor` → `Auto` | route diff field by field: 262 decisions moved, **0** regressions, **0** complex; the pre-flip prediction caught an unmeasured transposed-double window |
| WP2 E5 | double widened to non-square, any size, `k >= 2` | the demand table's own shapes, 36/36 cells 1.10–1.41×; 81 decisions moved, zero regressions; `k=1` is the single losing double shape in the package |
| WP3 S1–S2 | `RouteTable<Op::trsm, T>` and the native TRSM translation unit | correctness split only — nothing `preferred()` yet, so no route moves |
| WP3 S3–S6 | V1, the one-work-item-per-solve CTA kernel, both sides; the facade hook | the register gate (`scripts/register_probe.sh`) rejected the first bucket ladder; **vendor-free `trsm` runs for the first time** |
| WP3 S7–S8 | V2, the blocked driver, and complex support | native `trsm` covers **every** order; the conjugation test that was blind by construction is replaced |
| WP3 S9–S11 | the benchmark grid the spec asked for, and `preferred()` flipped onto what it says | 9 of the spec's 54 cells are not physical on 24 GB and are dropped explicitly; complex, double and float-Right all flip together |
| WP3 S12 | the `Side::Left` SLM staging tile | float/Left window moves `order <= 16` → `order <= 128`; staging **gated to real types** after it cost complex a 464 B stack frame |
| WP3 S13 | win at every order 8..512, both sides, float and `complex<float>` | 3 of 4 (type, side) combinations win at every order; `complex<float>`/Right at order 8–16 shown to be a roofline tie at 88.5–90.5% of DRAM peak |
| WP3 S14 | the cooperative CTA solve (W work-items per solve) | **built, measured and rejected** — passes the register gate at N=128 in fewer registers than V1 needs at N=32, and still measures 0.39× at order 64, 0.80× at 128 |
| WP3 S16 | the trailing-update GEMM goes through the **route table** instead of the native kernel unconditionally | **167 of 168 cells win**; the n=512 solve 18.8 ms → **11.19 ms** against a 14.28 ms vendor; `trsm_tests` 91/91, vendor-free failing set byte-identical |

The milestone the S1-S3 steps reach: `-DBATCHLAS_ENABLE_CUBLAS=OFF` now configures to
`BATCHLAS_HAS_CUDA_BACKEND 1` with `CUBLAS 0` and `CUSOLVER 0` — **a CUDA device with no
CUDA math libraries**, a state the old scheme could not express at all. It does not yet
*link*, and cannot until the public op definitions leave the vendor TUs (`WP0_DISPATCH_SPEC.md`
step S5).

### What S4 turned up

The split was specified because getting it wrong silently changes the default route of the
hottest op. It also found four real defects, none visible by reading:

1. **The order-walk fallback inverted GEMM's default.** Taking "the first merely supported
   route" picks Native, because the order lists it first — moving an 8×8×8 batch-1 GEMM from
   vendor to native.
2. **`BATCHLAS_GEMM_VARIANT=native` means the opposite of canonical `native`.** It is
   `gemm_variant.hh`'s alias for `cuda-native`/`direct-cuda` — the raw CUDA path — consumed
   purely as an *exclusion*. Routing it through the generic parser flips GEMM from vendor to
   native for anyone who had set it.
3. **`ormqr`'s buffer size and call disagreed by 108×.** `cta`, `two_stage` and `jacobi` all
   parse but match no branch, so `ormqr_dispatch` ran on the vendor while `ormqr_buffer_size`
   returned the *blocked* size — 2560 bytes against the 276480 the call then demanded. Sizing
   a workspace with the public API and passing it to the public call threw, deterministically,
   on every GPU type.
4. **`{Vendor, FusedDevice}` satisfies `is_vendor`, but is not "the plain vendor call".** The
   level-3 dispatchers' `request == Vendor` tests meant `cublasSsyrk` specifically; rendering
   them as `is_vendor()` makes a forced cuBLASDx request answer yes. `is_plain_vendor` now
   names the distinction.

Defects 1, 3 and 4 are the same shape as `Provider` itself: two different questions sharing
one value, so checking one looks like checking the other. Defect 2 is the vocabulary
collision the user flagged at the outset, and it recurred twice more — `custom` means the
fused cuBLASDx kernel in the level-3 ops but the register-tiled GEMM family in GEMM.

### What S5 turned up

The facade move is the step the plan called *the* obstacle: `gemm<Backend::CUDA, float>`
was **defined** at `cublas.cc:1568`, so dropping cuBLAS dropped `batchlas::gemm` itself.
That is now fixed for all 21 entry points. Three things are worth recording.

1. **The spec's top-ranked risk was answerable here.** It says `rocblas.cc`/`rocsolver.cc`/
   `rocsparse.cc` "cannot be compiled on this machine" and proposes a container CI job.
   `/opt/rocm-6.2.4` has all three vendor headers, under `include/roc*/roc*.h` — a
   subdirectory, which is why they read as absent. `scripts/rocm_syntax_check.sh` gates on
   "exactly one expected error" (a `get_native<ext_oneapi_hip>` overload this CUDA-only
   DPC++ lacks). **It then caught two real defects that nothing else could see:** a `trsm`
   instantiation left in the old parameter order, and four orphaned macro-continuation
   lines. Both were in files the normal build never compiles.
2. **Divergence between vendor TUs was invisible until one declaration served them all.**
   `trsm`'s vendor form takes `alpha` last while the public form takes it third;
   `symm`/`syrk`/`syr2k` were `RealScalar`-constrained everywhere except cuBLAS. Generating
   the facade bodies from the public declarations — the obvious approach — would have
   silently passed `alpha` where `side` was expected on every backend. The bodies are
   therefore lifted verbatim from the forwarders being deleted.
3. **An instantiation binds as hard as a definition.** `syev` and `ormqr` were already
   defined in headers, so it looked like there was nothing to move. Their *instantiations*
   were in `cusolver.cc`/`cublas.cc`, which is enough to make them vanish from a build
   without those libraries.

Verification is by **symbol**, not by diff (`scripts/facade_symbol_check.sh`): a forwarder
left behind, or an instantiation aimed at the wrong template, still compiles and links.

### M1 reached, and what it does and does not mean

`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` now yields `BATCHLAS_HAS_CUDA_BACKEND 1` with every
CUDA math library at 0 — a CUDA device with no CUDA math libraries — and **that build
compiles, links, loads and runs**. The pre-WP0 scheme could not express that state, let
alone link it.

It is **not green**: `ctest -LE slow` passes 20 of 53. That is the expected and honest
outcome, and no dispatch mechanism could improve it — the gap is missing *kernels*, not
missing routing. The failing set is recorded in `VENDOR_FREE_BASELINE.md` as the WP1–WP8
burn-down baseline, so any change to it is a reviewable diff. Failures are `NoRouteError`
naming the op, the scalar type and the switch that would restore it — not crashes, and
not link errors.

S6 also declined the spec's design. It proposes seven `src/dispatch/absent/*.cc` stub TUs
defining a throwing `backend::<op>_vendor` for every absent library; that restates all 26
vendor signatures a second time, and S5b's two real bugs were both signature divergence
between restated copies. An `if constexpr` in the facade is the same gate with no
signature duplicated: the vendor call is not compiled at all when the library is absent,
so there is no symbol to satisfy.

Building it turned up **four more family-vs-library guards** of exactly the kind S2 was
written to remove — in `linalg-impl.hh`'s host arms, `backend_handle_impl.hh`'s cuSPARSE
descriptors, and three test files — plus **nineteen call sites in extensions, tests and
benchmarks that reached `backend::*_vendor` directly**, bypassing the public entry point
and therefore every gate. None of these was findable without actually building without a
vendor.

### WP0 is complete

S7 turns the burn-down from a list of failing *suites* into a list of missing *ops*, which
is the difference between "33 suites fail" and this, from the vendor-free CUDA build:

| native kernel linked | ops |
|---|---|
| yes | `gemm` (register-tiled), `ormqr` (blocked), `syev` (cta/blocked/two-stage), `gesvd` (jacobi/cta/blocked) |
| **no** | `gemv`, `trsm`, `trmm`, `symm`, `syrk`, `syr2k`, `hemm`, `herk`, `her2k`, `geqrf`, `orgqr`, `getrf`, `getrs`, `getri`, `potrf`, `spmm` |

Four of those sixteen — `trmm`, `symm`, `syrk`, `syr2k` — already *have* portable SYCL tile
kernels; they are simply compiled only alongside cuBLAS, because their dispatch still ends
in `*_vendor_cuda_raw`. That is WP1, and it should move four rows without writing a kernel.

S8 removed the last places where the `Backend` enum stood in for something else. Three sites
used `B == Backend::CUDA` to mean "the tile kernel is wired on this route" — one of them
said so in its own comment while the code said otherwise — and two used
`B == Backend::NETLIB` to mean "this is a host device". Both questions are now asked
directly. This matters beyond tidiness: in the vendor-free build the backend is *still*
`Backend::CUDA` while the tile TUs are absent, so all three sites would have claimed a
kernel that is not linked.

Deliberately left alone: `syev.hh`'s measured-grid guard, which the spec also lists.
`s.backend == Backend::CUDA` there is measurement *provenance*, not wiring — the routes are
compiled elsewhere, they have simply not been measured there. Rewriting it as
`route_compiled` would assert something false; changing it needs a measurement.

### WP1 is complete, and it corrected the plan in five places

The full design and its corrections are in `WP1_LEVEL3_SPEC.md` (12-agent pass: 5 mapping,
3 designs, 3 adversarial judges). What WP1 delivered:

- **The four level-3 dispatchers are genuinely CUDA-free.** `nm -C` over
  `{symm,syrk,syr2k,trmm}_custom_dispatch.cc.o` finds no CUDA symbol at all; the sources
  carry no CUDA include and no preprocessor directive. They compile in every configuration.
- **Vendor-free went 20/53 → 24/53**, and vendor-free `gemm_tests` went 48 passing → 167.
- **The tile kernels are reached, not merely linked** — proven by coverage rows, since a
  symbol being present was never evidence it runs.
- **Vendor-present did not move.** All eight steps report the same 3016 distinct routing
  decisions, diffed per step with `scripts/route_diff.sh`.

Five plan claims were wrong and are corrected in the spec:

1. `batchlas::gemm` was **not** "the already-routing public entry point" — it threw or
   forwarded, with the routing inside cuBLAS-gated `cublas.cc`. Retargeting at it alone
   would have converted a `NoRouteError` naming `symm` into one naming `gemm`.
2. **`symm` has no tile kernel.** Its only portable kernel is the mirrored expansion; three
   ops carry tile kernels, not four.
3. The four routes were **reachable only from `cublas.cc`**, so compiling them everywhere
   made them linked everywhere and callable nowhere.
4. The crossover constants are **4 and 256**, not 2 and 128.
5. The level-3 ops have **no `RouteTable`** and never call `resolve_route`.

And `route_compiled.hh`'s own prediction — that the flag would simply flip `true` — was
wrong in two directions; it took a scalar parameter instead (S7).

### WP2 is complete, and it flipped GEMM's default

Full design, measurements and corrections in `WP2_GEMM_SPEC.md`. WP2 ran as **two tracks that
do not share commits**, because they have different acceptance criteria: a correctness track
(C1–C3) closing the vendor-free gap, and an envelope track (E1–E6) moving the default.

- **Vendor-free `gemm` is complete.** `gemm_tests` 167/184 → **184/184**. The 17 remaining
  failures were all heterogeneous batch, and `gemm_heterogeneous_vendor_impl` — which also
  carries the `m==0`/`n==0` skip and the `k==0 → scale(beta)` substitution — lived only inside
  cuBLAS-gated code. Vendor-freedom costs **nothing measurable** there (C3): both paths pay the
  same dominant cost, one launch per batch member.
- **GEMM's default is now `Auto`.** This was the one op whose native kernel never ran by
  default. 262 decisions moved, **zero** regressions, verified field by field.
- **The measurement narrowed the window as often as it widened it.** E4 found `preferred()`
  claiming two float regions it *loses* — all 30 transposed cells at 0.34–0.55×. Had that not
  been measured first, E6 would have moved float in all nine transpose forms instead of NN only.

Three things WP2 established that later work depends on:

1. **Measure demand before tuning.** The E2-prep capture found that 16 000 of 23 134 non-float
   GEMM calls were the test suite's own probe rows. On real demand the wide-scalar gate fires on
   **0.64%**, not 3.56% — which reframed every step after it.
2. **Predict the flip before making it.** Enumerating the intended moves found `preferred()`'s
   `double` branch was a bare `max_dim <= 512` with *no transpose test at all*, which would have
   shipped an unmeasured transposed-double window.
3. **Complex is what is still vendor-dependent**, and that is the honest headline: `preferred()`
   refuses it, because the complex register ladder is reachable only through a gate needing
   `min_dim >= 256` and an aligned NN shape. The panel-update population that dominates real
   demand needs a **transposed and predicated wide-scalar kernel** — a new kernel, not a routing
   change.

### WP3 is complete: `trsm` is native, and faster than the vendor almost everywhere

`WP3_TRSM_SPEC.md` is the design; **`WP3_TRSM_SPEC_CORRECTIONS.md` supersedes it** — the spec
was written against a pre-WP1/WP2 tree and a verification pass found six edits that would have
produced incorrect code, including three hook points that no longer exist and an SLM size
formula that writes 127 elements past the end of its allocation.

`trsm` was the only genuine hole in level 3 and had no native implementation of any kind. It now
has two tiers — V1, a CTA kernel with one work-item per solve, and V2, a blocked driver — and
across the measured grid (orders 8–512, both sides, `float` and `complex<float>`, at saturating
batch) **native beats the vendor in 167 of 168 cells**:

| vendor_ms / native_ms | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| float, Left | 1.60 | 1.72 | 1.69 | 1.65 | 1.42 | 1.25 | 1.21 |
| float, Right | 1.62 | 2.37 | 2.23 | 2.01 | 1.60 | 1.23 | 1.00 |
| `complex<float>`, Left | 1.05 | 1.41 | 2.64 | 4.69 | 11.21 | 16.62 | 51.70 |
| `complex<float>`, Right | 1.01 | 1.03 | 1.35 | 1.90 | 8.49 | 15.31 | 19.64 |

The single cell that does not clear parity is float / `Side::Right` / order 512 / q=256 /
batch=128 at **0.978–0.983×** over three repeats — the smallest-work cell at that order, whose
neighbours win 1.30–1.38×. It is published rather than papered over, and no router clause was
fitted to it: the clause would be narrower than the noise floor of most of that table.

Four results from WP3 that generalise beyond `trsm`:

1. **The accuracy risk the plan flagged never materialised, because the design was rejected.**
   §2.4 of the spec rejects diagonal-block inversion at every tier on its own argument, so the
   substitution-based backward error bound is preserved and Risk 4 below is retired for `trsm`.
2. **A cooperative CTA solve was built, measured and rejected** (step 14). It passes the
   register gate at N=128 in *fewer* registers than V1 needs at N=32 and still measures 0.39× at
   order 64. The traffic model that motivated it was right about DRAM and silent about the
   serial recurrence it introduced. Kept as a patch, not as code.
3. **The residual gap was never in the triangular solve.** V2's trailing update called
   `sycl_gemm::gemm_custom`, the *native kernel entry point*, which bypasses
   `RouteTable<Op::gemm>` entirely — so every update took the native GEMM whether or not it was
   better. Routing it took the n=512 solve from 18.8 ms to **11.19 ms** against a 14.28 ms
   vendor, with no change to the kernel.
4. **The leading dimension is the whole effect, and no square benchmark can see it.** Every
   operand `trsm` hands GEMM is a sub-view carrying its parent's `ld` — a 128-row `C` with
   `ld = 512`. The same six shapes measure 0.86–0.98× at `ld == rows` and **0.43–0.62×** at the
   real `ld`; cuBLAS barely moves. This is a **defect in the native GEMM that every panel-update
   caller in the tree pays**. *(The mechanism WP3 originally named for it was **wrong** — it
   blamed `register_tiled_common.hh`, which those shapes never execute. See "The strided-`ld`
   defect, re-diagnosed" below.)*

Item 4 is the next work item and is **not** a `trsm` item.

### The strided-`ld` defect, re-diagnosed — and a routing fix worth 1.74×

A 17-agent measurement pass (commit `3f0afbd`) re-examined the defect WP3 left open. The
measurement survived; **the explanation did not**, and correcting it changed which fix was worth
building.

WP3 blamed `register_tiled_common.hh` — odd tile strides, `[n][k]` B staging, a read-modify-write
epilogue, and a contiguity predicate every sub-view fails. **Those shapes never execute that
file.** `select_kernel_variant` (`gemm_kernels.cc:509-511`) routes them to
`Tiled128x128RegisterK8` with `AlignedFastPath = true` in *both* the packed and strided columns;
`can_use_128x128_fast_path` never tests contiguity, only `ld%4` and a 16-byte base, which a
strided sub-view satisfies.

What ncu measures instead: **every transaction counter is byte-identical** between the two
configurations — 16.00 load sectors/request (the ideal), identical DRAM sectors, identical
instructions, 119 registers, zero spill. The DRAM does the same work and then idles 45% of the
time. The entire regression is exposed global-load latency at the k-loop barrier (barrier stall
1.552 → 7.703). It is attributable to **one operand, B** (A alone 1.003×, C alone 1.056×, B alone
1.552×), it is a **slope** rather than a cliff, and it is **beta-independent** — which refutes the
epilogue story directly.

Two candidate fixes were **built and measured dead**, and are recorded so they are not
re-proposed: double-buffering the k-loop (127 registers, zero spill, barriers halved, and it
incidentally fixed a split-`LDG` defect to *exactly* cuBLAS's sector count — for **zero** time
recovered), and packing B into contiguous scratch (paid at the same roofline the kernel already
achieves; loses harder as `m` grows).

**What did work was routing.** `can_use_128x128_fast_path` is a *leg* predicate — the dispatcher
evaluates it again and chooses the leg itself — but `:509` used it as a *routing* gate, where
failing it did not demote the call to the predicated leg but handed it to a different, much slower
kernel. Routing by what the kernel can run is worth geomean **1.74×**, moving native from 0.58× →
0.99× of cuBLAS packed and 0.54× → 0.93× strided.

**With cuBLAS present it changes no runtime at all**, and that is stated in the code: the float NN
`preferred()` window requires `m==n==k`, so every shape the new gate captures resolves to the
vendor (coverage: 79 native float `gemm` calls against 102,791 vendor). It is a vendor-free and
ROCm win, and it makes a future `preferred()` flip *arguable* rather than winning one — at 0.93×
it is not yet arguable.

Two process points this pass established, both of which cost real time before they were learned:

1. **Confirm which kernel runs before theorising about why it is slow.** This is the second time
   in this campaign a named mechanism belonged to code that was not executing (the first was
   WP3 step 14's inner blocking level).
2. **A benchmark's own hygiene is part of the measurement.** The padded operands were allocated
   *uninitialized* while the unpadded ones used `::Random`, so every cross-`ld` ratio compared
   data content as well as leading dimension. Fixed, and the reference cell moved 0.34% — so the
   defect is real, but nobody knew that until it was checked.

### WP4 Phase 2: the blocked driver, and the wrong answer it uncovered in WP3

The blocked driver (`src/extensions/potrf_blocked.cc`) factorises above the CTA
ceiling: the Phase 1 leaf on each diagonal block, the **routed** `trsm` for the
panel solve, and the **routed** `gemm` plus an explicit triangular fold for the
trailing update, both injected through a seam modelled on `TrsmTrailingGemm`
(`trsm_native.hh:105-111`) so the kernel TU never sees the dispatch layer.

**Two design questions were settled by measurement before any code was written.**

*The panel solve needed no new kernel.* The spec's step 2.1 lists a
`PotrfPanelSolveKernel`; WP3's routed `trsm(Right, Lower, ConjTrans, NonUnit)`
already serves that exact shape and wins every cell tried. It would also have
been aimed at the wrong stage: the panel is 5–22% of a vendor-free blocked
potrf against 65–95% for the trailing update.

*The trailing update cannot use a rank-k update.* `herk` (`level3.cc:295-306`)
has **no native arm at all** and calls `throw_no_vendor_route` vendor-free, so
routing potrf's hot loop through it would throw on exactly the shapes WP4 exists
to serve; and `syrk`'s custom fallback **writes both triangles**, which is a
wrong answer for a `Lower` factorisation rather than a slowdown. Hence gemm plus
`fold_symmetric_product_into_triangle`, with the diagonal block at
`alpha = -1, beta = 0` into scratch and the sub-diagonal rectangle straight into
`A` at `alpha = -1, beta = 1`.

**The result, re-measured by the orchestrator on the public API in both builds
(no forced routes):** vendor-free `float` is **1.13×** cuSOLVER at n=1024 and
**1.40×** at n=2048; `double` is at parity (1.03×); `float n=256` loses at
0.60×, and complex loses 0.31–0.51× with the gap widening in `n`. The complex
cause is outside this driver and is now the highest-value follow-on in the whole
campaign: `route_gemm.hh:113-114` returns false for complex and
`gemm_kernels.cc:471` keeps the register ladder float-only, so every complex
trailing gemm lands on the `Tiled16` fallback — 97% of a `cdouble` call, 2.95×
slower than cuBLAS. **A register-tiled complex GEMM is worth ~2.7× on
vendor-free `cdouble` potrf on its own.**

It still ships route-neutral, because `preferred()` is a *measured* window and
two of four types lose.

#### The finding that outranks the phase: WP3's trsm returned wrong answers

Phase 2 was not looking for this. The V1 CTA kernel stages its canonical
triangle into local memory with a loop strided by `lane`, then reads the
diagonal back from a **different lane's** write with no barrier in between — an
unsynchronised cross-sub-group read-after-write, plus the same problem on the
`sDiv[0]` revert flag. One `sycl::group_barrier` is the entire functional fix.

It survived WP3's suite and every WP3 benchmark because the launcher's ladder
picks `wg = 32` — one sub-group, in lock step, the single width where the race
cannot express itself — for every shape below roughly `q·batch = 65k`, **and
every trsm test and A/B cell in the tree sat there.** The blocked potrf panel
solve does not: at n=1024, batch=256 the first panel has q=896 and wg=256.
A/B, deleting the barrier and rebuilding: worst relative difference against the
vendor **6.05e+16** with 127 of 128 items wrong, host residual 8.0e+05 against
the vendor's 2.4e-07; restored, 4.27e-07 and 0 items.

**Consequence for the record: WP3's `preferred()` windows above `q·batch ≈ 65k`
were measured on a racing kernel and have not been re-run.** Nothing above that
threshold in the WP3 tables should be trusted until it is.

And the guard shipped with the fix **was itself vacuous** — it called V1
directly, cleared the work-group ladder, asserted that it had, and still passed
with the barrier deleted. It now drives the configuration that reproduces
(order 48 through V2, so the final V1 block is order 16, q=976, batch=128) and
was verified red. That is the fifth recorded blind guard in this repository, and
the first written in the same change as the fix it guards.

### WP4 Phase 1 has landed, and the spec's foundation was wrong

`WP4_POTRF_SPEC_CORRECTIONS.md` (108 findings surviving adversarial refutation) supersedes
`WP4_POTRF_SPEC.md`. The single most consequential correction was not a dispatch item — it was
the number every capacity in the spec is derived from.

The spec sizes its SLM budget from "49152 / 45056 on this box", presented as re-verified device
facts. `cmake/BatchLASDetectSYCL.cmake:44-45` **hardcodes** 49152 for any `nvidia_gpu_sm_*`
pattern, and the detection routine never queries `local_mem_size` at all — while
`src/extensions/gesvdj_cta.cc` already allocates **71,744 B in production**. Shipping the spec's
ceilings would have left float `n` in 106..155 with no route at all in a vendor-free build. The
implementation measured the ceiling instead of inheriting it.

**`potrf` now factorises with no vendor present** — `potrf_tests` 50/50 in the vendor-free build,
byte-identical to the vendor-present run — and the burn-down moves 25/53 → 26/54 with the failing
set unchanged.

It ships **route-neutral**: `supports()` is correctness only and `preferred()` returns false
everywhere, so nothing routes here by default. That is not caution, it is measurement: **the CTA
kernel is 2–3× slower than cuSOLVER above n≈64, at 8.3% achieved occupancy.** The vendor-free
story for larger `n` is Phase 2's blocked driver calling a small CTA leaf, not this kernel
stretched to its fit ceiling — so planning the routing step as "cut the preferred window" would be
planning around the wrong problem.

Three things this phase established that generalise:

1. **A test can assert the route table and still prove nothing.** The routing test originally
   asserted on a *re-resolution* of the table, which is not evidence that the facade executed the
   kernel. It now asserts **bit-exact** agreement with the direct entry point — cuSOLVER does not
   reproduce this kernel's reduction order, so a residual check would have been satisfied by
   either arm. Removing the facade's native arm turns it red while the two answers *print
   identically*.
2. **The register probe was blind to half the tree.** `scripts/register_probe.sh` replayed only
   `batchlas_sycl.dir/link.txt`, which links two objects — so it could report "424 entry
   functions, 0 with spill" over a set containing no potrf kernel at all. A clean report from the
   wrong library is indistinguishable from a clean one from the right library. It now takes a
   target and fails loudly on an unknown one.
3. **Five deliberate breaks were run and two turned nothing red** — recorded rather than hidden,
   because it located a *fourth* instance of this repository's blind-guard pattern: a planted
   factorisation left the original diagonal negative at the failure column, so a stale-pivot
   reader named the same column and the `info` tests stayed green.

### The burn-down number, and why it stopped moving

The vendor-free suite is **25/53**, unchanged by WP3 — and the reason is worth stating, because
the headline number now conflates two different gaps.

`ctest` runs each level-3 suite against the **host (NETLIB) backend as well as CUDA**, and a
vendor-free build has no netlib LAPACK either. Classifying every vendor-free failure by which
library its `NoRouteError` names:

| suite | failures on host (netlib) | failures on GPU (cuBLAS/cuSOLVER) |
|---|---|---|
| `trsm_tests` | **32** | **0** (59 pass) |
| `sytrd_blocked_tests` | 12 | 0 |
| `syev_tests` | 4 | 0 |
| `ormqr_cta_tests` | 2 | 0 |
| `symm_tests` / `hemm_tests` / `herk_tests` / `her2k_tests` / `syrk_tests` / `syr2k_tests` / `trmm_tests` | 2–8 each | 5–8 each |
| `syevx_tests` / `syev_two_stage_tests` / `iluk_tests` / `inverse_tests` / `linalg_layer_tests` | 0 | 1–67 each |

**Every vendor-free `trsm` failure is the host backend.** On the GPU, vendor-free `trsm` is
complete — which is exactly what WP3 set out to deliver, and the suite-level number cannot show
it. Four suites now fail *only* because of the host path, which is WP9 (the CPU story) and not
a missing GPU kernel.

The lesson is the same one S7 taught with `linked` vs `reachable`: **a suite-level pass count is
the wrong instrument**, and the coverage table is the right one. A future step should split the
burn-down by backend so a GPU kernel landing is visible in the number.

**Not implemented:** WP4–WP9.

The vendor-present suite stands at **52/53**. Its one failure, `lanczos_tests`, is **not** from
this work and was reproduced by rebuilding with the relevant change reverted. (At the time WP1
closed there were two: the other was `steqr_tests`, 4 host-backend cases, 3 of them `double`,
matching the known bad OpenBLAS kernel on this machine. It has since been fixed.)

---

## Table of contents

1. [Where we actually stand](#1-where-we-actually-stand)
2. [The four classes of dependency](#2-the-four-classes-of-dependency)
3. [Target architecture](#3-target-architecture)
4. [Two milestones, deliberately separated](#4-two-milestones)
5. [Work packages](#5-work-packages)
6. [Performance strategy and acceptance gates](#6-performance-strategy-and-acceptance-gates)
7. [Risks, ranked](#7-risks-ranked)
8. [Non-goals](#8-non-goals)
9. [Sequencing and first step](#9-sequencing-and-first-step)

---

## 1. Where we actually stand

The premise behind this project is sound and, unusually, already measured. From the GEMM
head-to-head in `experiments/sycl_vs_cuda/`:

- One SGEMM body compiled by both nvcc and DPC++ produces an **identical SASS inner loop**
  (512 FFMA, 32 `LDS.128`, 2 `BAR.SYNC`, 0 spills, 113 vs 115 registers) and runtimes
  within 1.3% at every shape. Portable SYCL is not handicapped against CUDA.
- `Tiled128x128RegisterK8` (`src/sycl/gemm/register_128x128.hh`) reaches **41.5 TFLOP/s** at
  512³ × batch 512, i.e. **88–102% of cuBLAS across shapes**.
- cuBLAS's own strict-FP32 SGEMM only reaches ~46–48 TFLOP/s on the RTX 4090, against a
  ~81.5 TFLOP/s FFMA ceiling. The 78–87 figure everyone quotes is TF32, a different
  precision. **~47 is the real parity target, and we are already at ~0.9× of it.**

So the hard part — "can a portable kernel match a vendor kernel at all" — is answered. What
remains is *coverage*: the native GEMM is a narrow island, most of the rest of the library
has no native path at all, and the pieces that do exist are wired so that they can only be
reached through a vendor backend.

Three numbers frame the size of the job:

| | count |
|---|---|
| Public dense ops with **no** native implementation anywhere | **9** at the time of writing (`gemv`, `trsm`, `potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `spmm`) — **now 8**: WP3 delivered `trsm` |
| Portable level-3 kernels compiled **only** into the CUDA object library | **4** files (`symm`, `syrk`, `syr2k`, `trmm` custom dispatch) — **now 0**: WP1 freed all four |
| Backends `with_backend` can dispatch to | **4** (CUDA, ROCM, MKL, NETLIB) — `Backend::SYCL` is in the enum and throws |

---

## 2. The four classes of dependency

Naming these separately matters, because three of the four are removable without writing a
single new numerical kernel.

### Class A — ops with no native implementation at all

These reach a vendor library and there is nothing else to reach.

| Op | Vendor sources | Native? | Who needs it internally |
|---|---|---|---|
| `gemv` | `cublas.cc`, `rocblas.cc`, `netlib_lapack.cc:353` | no | `ortho.cc:219`, `ormqr_blocked.cc` |
| `trsm` | `cublas.cc`, `rocblas.cc:138`, `netlib_lapack.cc:405` | no | `ortho.cc:194,281` (Cholesky-QR) |
| `potrf` | `cusolver.cc:42`, `rocsolver.cc:32`, `netlib_lapack.cc:944` | no | `ortho.cc:192,280`, `syevx_lobpcg.cc` |
| `geqrf` | `cusolver.cc`, `rocsolver.cc:52`, `netlib_lapack.cc:1290` | no | `ortho.cc:369`, `band_reduction.cc`, `sytrd_sy2sb.cc`, `matrix.cc` |
| `orgqr` | `cusolver.cc`, `rocsolver.cc:152`, `netlib_lapack.cc:1324` | no | `ortho.cc:370` |
| `getrf` | `cusolver.cc`, `rocsolver.cc:188`, `netlib_lapack.cc:1201` | no | `inv.cc:48` |
| `getrs` | `cublas.cc`, `netlib_lapack.cc:1147` | no | public API only |
| `getri` | `cublas.cc`, `netlib_lapack.cc:1248` | no | `inv.cc:49` |
| `spmm` | `cusparse.cc`, `rocsparse.cc`, `netlib_lapack.cc:218` | no | `lanczos.cc`, `syevx*.cc`, `ritz_values.cc` |

Note the second column of consumers. `ortho` alone pulls in **five** of these, and `ortho`
sits under `syevx`, `lobpcg` and `lanczos`. There is no route to a vendor-free eigensolver
stack that does not go through `potrf`, `trsm`, `geqrf` and `orgqr`.

### Class B — portable kernels imprisoned inside the CUDA backend

`src/backends/{symm,syrk,syr2k,trmm}_custom_dispatch.cc` and `src/backends/triangular_expand.hh`
implement the expand-then-gemm strategy that produced the measured 6.7–8.8× on symm/trmm.
That logic is portable — it is workspace management plus a batched GEMM — but:

- it is listed under `BACKEND_CUDA_SOURCES` in `src/backends/CMakeLists.txt`, so it is
  compiled only when `BATCHLAS_HAS_CUDA_BACKEND`;
- it is reachable only from `cublas.cc:20-25`, which is the only file that includes it;
- its terminal GEMM is `gemm_cublasdx(...)` (`symm_custom_dispatch.cc:111,122`), and per
  prior investigation the cuBLASDx header is never actually defined in this build, so every
  "cublasdx" route is silently its fallback.

One asymmetry worth knowing before scheduling WP1 against WP2. Unlike GEMM, these four are
**already the default where their heuristics fire**: `parse_cublasdx_variant_request`
(`cublasdx_dispatch_common.hh:22-30`) returns `auto_variant` when its env var is unset, and
`syrk_route_request` (`syrk_custom_dispatch.cc:45-48`) likewise returns `SyrkRoute::Auto`.
So `syrk`'s triangular-tile and gram-tile kernels, and the symm/hemm/trmm expansions, are
exercised in production today. Only GEMM is vendor-by-default. That makes WP1 a relocation
of *already-trusted* code, and it means the genuine default-vendor gap is WP2.

Type coverage splits in two here, and the split matters:

- `triangular_expand.hh` — the expand-then-gemm machinery behind symm/hemm/trmm — **is**
  templated on `T` (`triangular_expand.hh:85,163`) and serves every scalar type. This is the
  part with the measured 6.7–8.8×.
- The **tile-masked kernels** and their routing — `syrk_triangular_tiles.hh`,
  `syrk_gram_tiles.hh`, `trmm_triangular_tiles.hh`, `syr2k_triangular_tiles.hh` and all four
  `*_custom_dispatch.hh` — are declared on `MatrixView<float, ...>` and are **float-only**.
  Double and complex `syrk`/`syr2k`/`trmm` therefore reach the vendor regardless.

WP1 relocates both; extending the tile kernels to the other three scalar types is a separate
item, and it inherits WP2's register-budget problem for wide scalars.

Consequence: on ROCm, `symm`, `hemm`, `herk` and `her2k` **do not exist at all** —
`rocblas.cc` instantiates only `gemm`, `gemv`, `trsm`, `syrk`, `syr2k`, `trmm`. The Class B
work is therefore not merely a vendor-independence item; it is the fix for a backend that is
currently missing half of level 3.

### Class C — native algorithms that call vendor ops underneath

Everything in `src/extensions/` — `syev` (CTA / blocked / two-stage / Jacobi), `gesvd`,
`sytrd`, `stedc`, `steqr`, `stebz`, `stein`, `ormqr`, `ortho`, `syevx`, `lanczos` — is
portable SYCL. But they call the *public* entry points (`gemm<B>`, `gemv<B>`, `potrf<B>`,
`trsm<B>`, `geqrf<B>`, `orgqr<B>`), and at `B == Backend::CUDA` those land in `cublas.cc` /
`cusolver.cc`. A "BatchLAS_CTA" provider is a vendor-dependent code path today.

The one op with a native alternative — GEMM — is **opt-in and off by default**:

```cpp
// src/backends/gemm_variant.hh:54
inline GemmVariantRequest gemm_variant_request() {
    const char* raw = std::getenv("BATCHLAS_GEMM_VARIANT");
    if (!raw) return GemmVariantRequest::Vendor;   // <-- default
    ...
}
```

and even under `BATCHLAS_GEMM_VARIANT=auto` the envelope in `gemm_use_sycl_custom`
(`gemm_variant.hh:135-198`) is narrow: GPU only, `ComputePrecision::Default` only, no
heterogeneous batch, **complex excluded outright**, square-only (`m == n && n == k`),
`batch_size >= 64`, and then per type — float NN: `max_dim <= 32` or `128 <= max_dim <= 512`;
float with a transpose: `batch >= 128 && 128 <= max_dim <= 512`, ConjTrans rejected;
double: `max_dim <= 512`.

### Class D — structural

- `Backend::SYCL` is declared (`enums.hh:84`) and has no implementation; `with_backend`
  falls through to `throw` for it (`queue-dispatch.hh:52-58`), and
  `backend_dispatch_tests.cc:72` asserts exactly that.
- `with_backend`'s `static_assert` requires at least one of CUDA / ROCM / MKL / HOST. There
  is presently **no configuration of BatchLAS that builds with zero vendor backends.**
- netlib is already soft: `BatchLASDependencies.cmake:238` downgrades a missing LAPACKE/CBLAS
  to a `WARNING` and disables the host backend. Good.
- oneDPL is a hard `FATAL_ERROR` dependency (`BatchLASDependencies.cmake:258`). It is
  header-only and is not a BLAS, so it does not violate the goal, but it should be noted in
  any "no dependencies" claim. Five files use it, all for `dpl::random` / `dpl::algorithm`.
- **Five** parallel, non-communicating dispatch axes exist, not three:

  | # | Mechanism | Granularity | Bound at |
  |---|---|---|---|
  | 1 | `enum class Backend` (template parameter `B`) | whole library | compile time, chosen at runtime by `Queue::backend()` |
  | 2 | `Provider` + `DispatchPolicy` | 3 ops only (`syev`, `gesvd`, `ormqr`) | runtime, per call |
  | 3 | `BATCHLAS_GEMM_VARIANT` | `gemm` | runtime, `getenv` per call |
  | 4 | `BATCHLAS_{SYMM,SYRK,SYR2K,TRMM}_VARIANT` | 4 ops | runtime, `getenv` per call |
  | 5 | ad-hoc per-op knobs — `BATCHLAS_ORTHO_GRAM`, `BATCHLAS_ORMQR_IMPL`, `BATCHLAS_SYEVX_ALGORITHM`, `BATCHLAS_SYTRD_FUSE_PANEL_UPDATE` | one site each | runtime |

  None of them share a vocabulary. Unifying them is a prerequisite for being able to
  *state*, let alone enforce, vendor independence.

- **`Backend` carries four distinct meanings**, and only the first is what the name says:
  1. *Which device / SYCL runtime* — `queue-impl.cc:92-107`, the only place a device becomes
     a `Backend`. Note it maps device vendor Intel → `Backend::MKL`, i.e. a hardware property
     selecting a math library.
  2. *Which vendor math library* — `linalg-impl.hh:876-880` keys the cuBLAS/cuSPARSE/cuSOLVER
     handle triple off it.
  3. *Hardware errata* — `steqr.cc:21-30` disables the CTA path for `Backend::ROCM` because
     chunked sub-group ops give wrong eigenvalues on gfx1200. That is a statement about one
     GPU model, not about a backend.
  4. *Measurement provenance* — `syev.hh:778-788` gates a routing grid on `Backend::CUDA`
     with the comment that CUDA "is the only backend the grid above was measured on".

  Meanings 3 and 4 are the ones that make a mechanical refactor dangerous: they look like
  backend logic and are not, so moving them to a new axis silently changes behaviour.

- **The second axis already exists and is simply unwired.** `enums.hh:102-113` declares
  `enum class BackendLibrary { CUBLAS, CUSPARSE, CUSOLVER, ROCBLAS, ROCSPARSE, ROCSOLVER,
  MAGMA, MKL, CBLAS, LAPACKE }` — exactly the vendor-library axis this plan needs. It is used
  only inside `linalg-impl.hh` for handle/scalar conversion. **The `Backend → BackendLibrary`
  mapping exists only in comments, never in code.**

- Inside `cublas.cc`, every `if constexpr (Back == Backend::CUDA)` guard (`:162, :267, :530,
  :761, :873, :996`) is tautologically true — the file is instantiated only for
  `Backend::CUDA` (`cublas.cc:1771`). Those guards are documentation, not selection.

---

## 3. Target architecture

### 3.1 Do **not** add `Backend::SYCL`

*(This section reverses the plan's first draft. The correction is the point.)*

The first draft proposed making `Backend::SYCL` a real backend. That is wrong, and it is
wrong in exactly the way §2 Class D describes: it would express "no vendor library is
installed" as "we are on a different device". On an NVIDIA GPU with no cuBLAS, the device
family is still CUDA — that is what the SYCL runtime is targeting, what the queue submits
to, and what the errata in `steqr.cc:21-30` are keyed on. A build with no vendor library
must not change the answer to "what am I running on".

Instead:

- **`Backend` narrows to its meaning (1)**: which device / SYCL runtime family. It keeps its
  current spellings so no call site churns.
- **Native implementations are instantiated for every `Backend`.** They are portable SYCL;
  they run wherever the queue runs. This is what makes `Backend::SYCL` unnecessary — there is
  no backend on which the native path is unavailable.
- **`BATCHLAS_HAS_<X>_BACKEND` is decoupled from "the vendor library was found".** Today
  these are the same condition: `BatchLASDependencies.cmake` sets `BATCHLAS_HAS_CUDA_BACKEND
  TRUE` when `CUBLAS_LIBRARY` is found. After the split, "can dispatch a CUDA queue" depends
  only on there being a CUDA SYCL target, and separate `BATCHLAS_HAS_CUBLAS` /
  `BATCHLAS_HAS_CUSOLVER` / `BATCHLAS_HAS_CUSPARSE` record the libraries.

This removes an entire work item: `with_backend` needs no new case, `Queue::backend_available`
keeps its meaning, and `backend_dispatch_tests.cc:72` — which asserts `Backend::SYCL` is
unavailable — stays true and unmodified.

The vendor axis gets the enum that already exists for it, `BackendLibrary` (`enums.hh:102-113`),
whose `Backend → BackendLibrary` mapping is currently comments-only. Wiring that mapping in
code is the actual second axis, not a new `Backend` enumerator.

### 3.2 One dispatch mechanism

Extend `blas::dispatch::Provider` to cover every op, and add:

```cpp
enum class Provider {
    Auto,
    Vendor,
    BatchLAS,           // NEW: this op's native implementation, algorithm chosen by the op.
                        // Deliberately NOT "BatchLAS_SYCL": every BatchLAS provider is SYCL,
                        // so the suffix carries no information and collides with the
                        // Backend axis.
    BatchLAS_CTA,       // algorithm-qualified spellings, for ops that have several
    BatchLAS_Blocked,
    BatchLAS_TwoStage,
    BatchLAS_Jacobi,
    Netlib,
};
```

`Provider` still mixes origin with algorithm, which is not ideal — but the origin question is
the one the vendor-independence gate has to answer, and it can be answered by a predicate
rather than by splitting the enum:

```cpp
inline constexpr bool is_vendor(Provider p) {
    return p == Provider::Vendor || p == Provider::Netlib;
}
```

Expressing the gate as `is_vendor(...)` rather than by enumerating names means adding a
future algorithm spelling cannot silently escape it.

Fold `BATCHLAS_GEMM_VARIANT` and the four `BATCHLAS_*_VARIANT` knobs into
`BATCHLAS_<OP>_PROVIDER`, keeping the old spellings as deprecated aliases (they appear in
benchmark scripts and in `output/` result provenance; breaking them silently invalidates
recorded measurements).

`DispatchPolicy::order` grows to hold the new entry. Per-op orders keep working exactly as
`default_order_gesvd` does today.

### 3.3 The enforcement knob — this is the load-bearing piece

Two switches, and they are what turn "we have native paths" into a property that cannot
silently regress:

- **Build:** `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` compiles no vendor backend source at all.
  If the library links and the tests pass, independence is proven by construction.
- **Runtime:** `BATCHLAS_NO_VENDOR=1` makes any dispatch that resolves to `Provider::Vendor`
  throw, naming the op and the shape.

The runtime knob is the more useful of the two day to day, and it has an obvious home:
`include/batchlas/blas/dispatch/op.hh` already contains

```cpp
// Lightweight tag for operations that are pure wrappers around external libraries.
// This is currently a no-op, but provides a single place to add tracing/
// instrumentation later.
template <class F> decltype(auto) op_external(const char* name, F&& f);
```

That hook was put there for exactly this. Instrument it to count and optionally reject.

### 3.4 The coverage table becomes a build artifact

Run the full test suite under `BATCHLAS_NO_VENDOR=1`; every throw is a work item. Emit the
result as a generated table — op × scalar type × shape class → {native default, native
available, vendor only}. This is the burn-down chart for the whole project and it can exist
in week one, before any kernel is written.

---

## 4. Two milestones

Keeping these separate is the single most important structural decision in this plan,
because conflating them means nothing ships until everything is fast.

**M1 — Self-sufficient.** `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` configures, builds, and passes
the full `ctest` suite. No performance claim whatsoever. Vendor remains first in every Auto
order. The library *can* run alone.

**M2 — Vendor-free by default.** For each (op, type, shape class) cell where the native path
meets the acceptance gate at saturated batched shapes, native moves ahead of Vendor in the
Auto order. Cells that do not meet the gate stay vendor-first and are **published** in the
coverage table rather than quietly papered over.

M1 is a correctness and packaging milestone. M2 is a performance campaign that can then
proceed cell by cell, indefinitely, without ever regressing the guarantee M1 established.

**Status after WP3.** Separating them was the right call, and the reason is visible in the
numbers: **M2 has been reached for two ops while M1 is still open.** `gemm`'s default is `Auto`
and `trsm` is preferred over the vendor in 167 of 168 cells — both shipped and both measured —
while the vendor-free suite is 25/53 because `potrf`, `getrf`, `geqrf` and the rest still have no
native kernel at all. Under the conflated version of these milestones, none of that would have
shipped yet.

---

## 5. Work packages

Ordered by (value ÷ risk), not by dependency. WP0 and WP1 unblock measurement; WP2 is the
linchpin; WP3–WP7 are the genuinely new numerics.

### WP0 — Unify dispatch, add the gate — **COMPLETE**

*No kernels. Pure enabling work.*

1. Add `Provider::BatchLAS` and the `is_vendor()` predicate; widen `DispatchPolicy::order`.
2. Give `gemm`, `gemv`, `trsm`, `trmm`, `symm`, `hemm`, `syrk`, `herk`, `syr2k`, `her2k`,
   `potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `spmm` a `choose_*_provider`
   following the `choose_ormqr_provider` pattern (`ormqr.hh:161`).
3. Map the legacy `BATCHLAS_*_VARIANT` env vars onto the new knob, with aliases.
4. Instrument `op_external`: a per-op counter, and a throw under `BATCHLAS_NO_VENDOR=1`.
5. Add `BATCHLAS_ENABLE_VENDOR_BLAS` (default ON) and `BATCHLAS_HAS_SYCL_BACKEND`; add the
   `Backend::SYCL` case to `with_backend`; update `backend_dispatch_tests.cc:72`.
6. Generate the coverage table from a `BATCHLAS_NO_VENDOR=1` test run.

**Deliverable:** an exact, mechanically-produced list of what is missing. Every later
estimate in this document should be re-derived from that list rather than from this one.

**Effort:** small. **Risk:** low — the one real hazard is the widened `std::array<Provider, 6>`,
which is a fixed-size array in four places (`provider.hh:26`, `env.hh:58,99,111`) and will
not fail to compile if one is missed — it will silently truncate an order. Introduce a
single `kProviderCount` constant rather than bumping four literals.

### WP1 — Free the level-3 kernels from the CUDA backend — **COMPLETE**

Move `symm_custom_dispatch`, `syrk_custom_dispatch`, `syr2k_custom_dispatch`,
`trmm_custom_dispatch`, `triangular_expand.hh`, and the `*_triangular_tiles.hh` /
`syrk_gram_tiles.hh` family from `src/backends/` to `src/sycl/level3/`, and instantiate them
for every `Backend` rather than only CUDA.

**The terminal GEMM is a design decision, not a rename — this is why WP1 is not mechanical.**
The first draft said "retarget `gemm_cublasdx(...)` to `sycl_gemm::gemm_custom(...)`". That is
wrong in both directions:

- `gemm_cublasdx(...)` is not a cuBLASDx call in this build. MathDx is not found
  (`configure`: *"MathDx package not found; cuBLASDx/cuSolverDx wrappers will remain
  disabled"*), so `cublasdx_gemm_variant_available()` is false and every path falls through to
  `gemm_vendor_cuda_raw(...)` (`gemm_cublasdx_dispatch.cc:300-305,348`). The expansions
  currently terminate in **raw cuBLAS**.
- Hardcoding `sycl_gemm::gemm_custom(...)` instead would be a genuine regression. That
  function is the *unrouted* native kernel; its fast 128×128 path is float-NN-square-aligned
  only, so symm/trmm would lose cuBLAS for every shape outside that envelope even on a machine
  that has cuBLAS installed.

The correct target is the **public, already-routing entry point** `gemm<Back, T>(...)`
(`cublas.cc:158-179`), which selects cuBLASDx → heterogeneous-vendor → native SYCL → vendor in
that order. Calling it means the expansions inherit whatever WP0 decides, per shape, with no
duplicated routing logic and no hardcoded vendor assumption.

Three things to preserve carefully:

- The measured crossovers. `expand+gemm` loses to a per-batch vendor loop for
  `batch <= 2 && n <= 128` on symm/hemm; trmm wins everywhere because `cublas?trmm` has a
  flat ~110 µs floor. Those thresholds were derived from independent float and complex
  sweeps. Under `BATCHLAS_NO_VENDOR` there is no loop to fall back to, so the small-batch
  cell needs either a direct single-CTA path or an accepted regression — and per standing
  policy, batch ≤ 2 is not a regime we optimise for.
- The `trmm` uplo/diag correctness constraint. There is a prior incident here where the
  tempting 8× "fix" was the wrong-answer one and the guarding test could not fail by
  construction. Re-check the test actually discriminates before touching that file.

A fourth item, small but blocking: `cublasdx_dispatch_common.hh` includes
`<cuda_runtime_api.h>` (line 6) purely so that `cuda_stream_from_queue` can name
`cudaStream_t`. The other five helpers in it — `ceil_div`, `parse_cublasdx_variant_request`,
`is_gpu_queue`, `should_use_cublasdx`, `throw_forced_cublasdx_unavailable` — are portable and
are what `triangular_expand.hh` actually needs. Split those into a backend-neutral header;
that is the only genuine CUDA coupling in the whole family. (The `*_tiles.hh` kernels
reference CUDA nowhere but in a `BATCHLAS_KERNEL_TRACE_SCOPE` string literal.)

**Deliverable:** symm/hemm/herk/her2k/syrk/syr2k/trmm available with zero vendor libraries,
and available on ROCm for the first time.

**Effort:** medium. **Risk:** medium — the terminal-GEMM retarget above is a behaviour
change on a path with measured crossovers, not a code move. **Depends on WP0**, because the
routing it should defer to is what WP0 defines.

### WP2 — GEMM: close the envelope — **COMPLETE**

> **Superseded by `WP2_GEMM_SPEC.md`.** The sketch below is the original estimate and is kept
> for the record; where it and the spec disagree, the spec wins. Delivered: vendor-free `gemm`
> complete (184/184), the default flipped to `Auto`, double widened to non-square real-demand
> shapes. **Complex remains vendor-dependent** and needs a transposed, predicated wide-scalar
> kernel — the largest single remaining item in this area.

Everything downstream is expand-then-gemm or blocked-panel-plus-gemm, so every gap in the
GEMM envelope propagates into every op above it. The float-NN-large-square cell is already
at 88–102% of cuBLAS; the work is the *other* cells, in rough priority order:

| Gap | Current state | Approach |
|---|---|---|
| **complex float / complex double** | rejected outright by `gemm_use_sycl_custom` | 64 accumulators spill for wider scalars. Shrink the thread tile as the scalar widens — but the register-residency work says not too far, and an out-parameter reference alone cost 43%. Use an explicit complex multiply in the inner loop, not `std::complex operator*`, which emits an isnan branch and a `__mulsc3` call in device code (worth 1.2–1.3× in hot loops). |
| **transposes** | float only, `batch >= 128`, `128 <= max_dim <= 512`, ConjTrans rejected | The TN/NT/TT variants exist across the `register_tiled_common` family. Needs the 128×128 treatment (aligned shared strides, `[k][n]` B staging) per orientation, then a routing sweep. |
| **non-square / ragged / misaligned** | predicated path is correct and tested but **unbenchmarked**; routing is gated on the unpredicated fast path | Benchmark first. This may be a routing change, not a kernel change. |
| **heterogeneous batch** | rejected | Needed for API completeness; low frequency. Per-group launch over homogeneous sub-ranges. |
| **k-dominant / skinny** | `split_k.hh` exists, experimental-gated | Ungate, benchmark, route. |
| **`ComputePrecision != Default`** | rejected | TF32 via `joint_matrix` + `precision::tf32` verifiably emits real `mma.sync.m16n16k8` on sm_89. Untuned and unmeasured — this is the path to the 78–87 TFLOP/s numbers and is a **separate track**, not an M1 blocker. |

Two traps that must be in the acceptance criteria:

- **Always confirm a new GEMM kernel at `beta = 1`.** A first version scored 26 instead of
  41 TFLOP/s with an identical inner loop, purely because the epilogue had the `m` index
  slow-varying and the `beta != 0` read of C became one scattered transaction per lane. The
  standalone harness defaults to `beta = 0` and cannot see this.
- **Warm the JIT.** A first-run SYCL JIT once fabricated an entire 3.7× regression.

**Deliverable:** `gemm_use_sycl_custom` accepts the shapes the library actually issues, and
`BATCHLAS_GEMM_VARIANT`'s default flips from `Vendor` to `Auto`.

**Effort:** large. **Risk:** medium — this is tuning-heavy and the retune cycle is ~12 min,
with a known trap that the CMake tuning-header target is a no-op.

### WP3 — `trsm` — **COMPLETE**

> **Superseded by `WP3_TRSM_SPEC.md`, itself corrected by `WP3_TRSM_SPEC_CORRECTIONS.md`.**
> The sketch below is the original estimate. Two of its three design guesses did not survive
> measurement: **the diagonal-block-inverse formulation was rejected outright** (so the accuracy
> caveat below is moot, and Risk 4 is retired for `trsm`), and the delivered large-`n` tier is a
> two-level blocked driver whose inner blocks are solved by substitution, not inverted.
> Delivered: native `trsm` beating the vendor in **167 of 168** measured cells.

The only genuine hole in level 3, and `ortho`'s Cholesky-QR path needs it
(`ortho.cc:194,281`). There is no device-level `trsv`/`trsm` in `group_blas` either, so this
is new from the ground up.

Design, for the batched regime:

- **Small n (the common case — a k×k Gram factor, k ≲ 256):** single-CTA blocked
  forward/back substitution over `group_blas` primitives, one work-group per matrix,
  triangle handled at thread-tile granularity. The existing triangular kernel design rules
  apply: tile to n, respect the thread-tile triangle granularity, and avoid the band-split
  trap.
- **Larger n:** invert the diagonal blocks (small, resident in SLM/registers) and turn the
  off-diagonal updates into GEMM — i.e. the same expand-then-gemm shape as WP1, which means
  it inherits WP2's kernel automatically.

**Accuracy caveat, stated up front:** the diagonal-block-inverse formulation changes the
backward error bound relative to substitution. For BatchLAS's actual use (a well-conditioned
Cholesky factor of a Gram matrix) this is standard practice and acceptable, but it must be
verified with `benchmarks/orthogonality_accuracy.cc` and `orthogonality_miniacc.cc` before
it becomes the default, not after.

**Effort:** medium. **Risk:** medium (accuracy).

### WP4 — `potrf`

Needed by `ortho` and `syevx_lobpcg`. Batched, small-to-medium n. A right-looking CTA-resident
Cholesky built on `group_blas_rankk` plus the WP3 in-SLM triangular solve is a well-understood
kernel and should beat cuSOLVER's batched potrf comfortably at large batch, where the vendor
is launch-bound.

Two contract details to preserve exactly: the `info`/failure convention for non-positive-definite
input as `cusolver.cc:42` implements it, and the `PotrfOptions{}` overload behaviour — there is
a known trap where a bare `{}` picks the positional overload and silently returns wrong numbers.

**Effort:** medium. **Risk:** low.

### WP5 — QR: `geqrf` + `orgqr`

The largest genuinely-new numerical build in this plan, and the one with the most existing
scaffolding to reuse: `ormqr` already has native CTA (`ormqr_cta.cc`) and blocked
(`ormqr_blocked.cc`) paths with a WY representation and a tuned block width
(`resolve_ormqr_block_size`, `ormqr.hh:184`).

- **`geqrf`:** blocked Householder QR — panel factorization plus WY-form trailing update.
  The panel machinery in `latrd_lower_panel.cc` and `sytrd_cta.cc` is the closest existing
  analogue; the trailing update is a GEMM pair, so again it inherits WP2. A CTA variant for
  n ≲ 128 follows `sytrd_cta`'s structure directly.
- **`orgqr`:** accumulate Q from the reflectors — structurally `ormqr` applied to an
  identity, so most of it is already written. Consider implementing it *as* that first, and
  specialising only if measurement demands it.

Watch for the two recurring defects in this family: the short-final-panel bug that produced
a silent numerical failure in `sy2sb` stage 1, and batch-only parallelism starvation — check
the `nd_range` before believing a disappointing number.

**Effort:** large. **Risk:** medium-high (correctness surface is wide; the failure mode is
silent).

### WP6 — LU: `getrf` / `getrs` / `getri`

Lowest internal urgency — only `inv.cc` consumes them — but they are public API and M1 needs
them. Standard batched partial-pivoting LU: CTA-resident for small n, right-looking blocked
above. `getrs` is then two triangular solves (WP3); `getri` is `getrs` against an identity,
or `trtri` + `trsm`.

Pivoting is the interesting part in a batched setting: the pivot search is a work-group
reduction per column and the row swap is a strided exchange. Both are cheap; the risk is
that they serialise the whole factorization at small n. Measure the un-pivoted variant as a
lower bound to know how much the pivoting is costing.

**Effort:** medium. **Risk:** low-medium.

### WP7 — `gemv` and the level-2 gap

`gemv` is vendor-only at host level, and this is simultaneously a self-sufficiency item and
a known *performance opportunity*: the panel `symv` inside `latrd` is bound by 12–16× L1
over-fetch and a double triangle read, with roughly 2.7× of headroom, and it is on the
critical path of `syev`.

Device-level `group_blas_gemv` and `group_blas_symv` already exist. The work is a host-level
batched launcher with a correctly-shaped `nd_range` — and this is precisely the family where
"4 kernels parallel over batch **only**" has bitten repeatedly. Check the `nd_range` first,
and note that the grid-`latrd` path is dead at batch ≥ 128 because its cap is `SMs/batch`,
which makes any A/B there vacuous.

**Effort:** medium. **Risk:** low, with a real chance of a performance *win*, not just
parity.

### WP8 — sparse: `spmm`

cuSPARSE / rocSPARSE are the last dense-adjacent dependency. Consumers: `lanczos`, `syevx`,
`syevx_filtered`, `syevx_lobpcg`, `ritz_values`, `iluk`. A batched CSR SpMM (and the `iluk`
triangular solves) is a different specialty from the dense work above and does not share the
GEMM foundation.

Recommendation: schedule this **last**, and consider whether M1 should be declared over the
dense API with sparse tracked separately. Vendor sparse is a much smaller moat than vendor
dense — but it is also the least-shared code in the plan, so it buys the least.

**Effort:** medium-large. **Risk:** medium.

### WP9 — the CPU story

Once `Backend::SYCL` exists it runs on a CPU SYCL device, so "no BLAS installed anywhere"
becomes a buildable, runnable configuration for the first time. Decide explicitly whether
CPU `Backend::SYCL` needs to be *fast* or merely *correct*.

**Recommendation: correct and not embarrassing, nothing more.** The CPU BLAS market is well
served by MKL and OpenBLAS, both of which remain available through the existing backends,
and BatchLAS's purpose is batched GPU work. Spending WP2-grade tuning effort on CPU SYCL
kernels would be the worst value in this document.

(Related, worth knowing before anyone benchmarks on this machine: double-precision CPU
numerical failures here are usually the broken OpenBLAS Cooperlake `dgemm` kernel, not
BatchLAS. And a CUDA-off `ctest` shows ~30 failures that are artefacts of the CPU-only
verification build, not real regressions.)

---

## 6. Performance strategy and acceptance gates

### The gate

For each (op, scalar type, shape class) cell, native becomes the Auto default when:

```
t_native <= 1.10 * t_vendor    at saturated, large-batch shapes
```

and accuracy is within the op's existing test tolerance. A cell that fails the gate stays
vendor-first and is recorded as *available but not default*. That is a legitimate outcome,
not a failure — it preserves M1 while being honest about M2.

### Measurement rules (non-negotiable, all previously established)

1. **Compare only at saturation.** Numbers below saturation are ratios of overheads and
   routinely rank the worse algorithm first. State the saturation level alongside any ratio.
2. **Batch ≥ 128**, pairing small n with larger batch (n=256/batch=2048, n=2048/batch=32+).
   A result that only holds at batch = 1 is not a result.
3. **But still profile across the range.** Benchmarking only at saturation is exactly what
   concealed the batch-only-parallelism defect for so long. Compare at saturation; hunt bugs
   everywhere.
4. **Warm the JIT** before the first timed iteration.
5. **Confirm every GEMM-family kernel at `beta = 1`.**
6. Watch for GPU contention (this box has two RTX 4090s) and cold clocks. Note that the
   `output/gemm_*` vendor numbers carry a fixed ~0.36 ms event-timer overhead, making them
   ~12% pessimistic — it penalises fast kernels and flatters slow ones.

### Build-time budget

The `.so` is device-link-bound, and the seven standard fixes for that have already been
measured dead. This plan adds a whole backend's worth of kernels across four scalar types,
which multiplies template instantiations. **Budget for it explicitly:** measure link time
after WP1 and again after WP2, and if it grows unacceptably, the lever is fewer instantiated
shape variants (route more shapes through fewer kernels), not more parallel link jobs.

---

## 7. Risks, ranked

1. **Register pressure for complex and double (WP2).** Already measured: the 64-accumulator
   tile spills for anything wider than float. If the mitigations do not hold, complex GEMM
   stays vendor-preferred and complex `syev`/`gesvd` inherit that — a large fraction of the
   library. *Mitigate:* prototype the complex tile early, in WP2's first week, before
   committing to the WP3–WP6 schedule that assumes it.
   **Outcome after WP2: this risk partly materialised and is the package's honest headline.**
   The wide-scalar tile landed and wins for `double` (1.01–4.51×), but complex GEMM **stays
   vendor-preferred** — `preferred()` refuses it, because the complex ladder is reachable only
   through a gate needing `min_dim >= 256` and an aligned NN shape, which fires on 0.64% of real
   demand. Closing it needs a transposed, predicated wide-scalar kernel. Note this did **not**
   propagate to `trsm` as feared: WP3's complex column wins 1.01–51.70×, because the complex
   "vendor" baseline there is not cuBLAS at all but a hand-written substitution kernel
   (`cublas.cc:1111`).
2. **Silent numerical failure in the QR/panel work (WP5).** This family has produced exactly
   this failure mode before, and the guarding tests did not catch it. *Mitigate:* write the
   discriminating test first and confirm it *can* fail; use the `-UNDEBUG` device-assert
   recipe for out-of-bounds hunting.
3. **Build time (WP1, WP2).** See above.
4. **Accuracy regression from inversion-based `trsm` and Cholesky-QR (WP3, WP4).**
   *Mitigate:* gate on the existing orthogonality accuracy benchmarks before defaulting.
   **Retired for `trsm`:** the design was rejected before it was built. `WP3_TRSM_SPEC.md` §2.4
   declines diagonal-block inversion at every tier on its own argument, and the delivered kernels
   solve by substitution throughout, so the backward-error bound is unchanged. Still live for
   WP4's Cholesky-QR.
5. **Tuning surface explosion.** Every new kernel family adds a tuning axis; the retune cycle
   is ~12 min and previously a 2.16× kernel win turned into an 11% `gesvd` loss. *Mitigate:*
   validate every tuning change end-to-end at the algorithm level, never at the kernel level
   alone. And note the prior finding that the routing grid was float-only for a long time —
   generate buckets for every scalar type from the start.
6. **Sunk-cost drift.** There is precedent here: a research document's top-ranked item was
   implemented and measured 85–211× *slower* than the path it replaced. Implementation cost
   already spent is not evidence of value. Every WP in this document should be killed
   without ceremony if its first measurement says so.
   **Exercised once so far, successfully:** WP3's cooperative CTA solve was designed, built,
   proved correct, shown to pass the register gate in fewer registers than the kernel it would
   replace — and then deleted, because it measured 0.39× at order 64. The traffic model that
   motivated it was right about DRAM and silent about the serial recurrence it introduced.

---

## 8. Non-goals

- **Removing the vendor backends.** They stay, they stay first in Auto wherever they win,
  and `Provider::Vendor` remains reachable. The goal is *no requirement*, not *no use*.
- **Beating cuBLAS at un-batched single large GEMM.** Not our regime.
- **Tensor-core / TF32 parity as an M1 requirement.** Reachable from portable SYCL and worth
  a separate track, but it is a different precision and a different project.
- **A fast CPU SYCL backend.** See WP9.
- **Removing oneDPL.** Header-only, not a BLAS, five files.

---

## 9. Sequencing and first step

```
WP0 (gate + coverage table)  ──┬──> WP1 (level-3 unchained)  ──┐
                               │                               ├──> M1
                               └──> WP2 (GEMM envelope) ───────┤
                                        │                      │
                                        ├──> WP3 trsm ─────────┤
                                        ├──> WP4 potrf ────────┤
                                        ├──> WP5 geqrf/orgqr ──┤
                                        ├──> WP6 getrf/getrs/getri
                                        ├──> WP7 gemv/symv ────┤
                                        └──> WP8 spmm ─────────┘
```

WP3–WP8 all consume WP2's GEMM, which is why WP2 is the linchpin and why its complex-tile
prototype should be the first *numerical* thing attempted.

**The first step was WP0, and specifically the coverage table.** That is done, and it did what
it was supposed to: it converted this document from an argued estimate into a mechanically
verified list, and every claim since has been re-derived from it rather than from the prose above.

**Where the next step is now.** WP0–WP3 are complete. The candidates, in the order the
measurements argue for:

1. **The native GEMM's strided-`ld` collapse.** Not a work package in the original list, because
   it was not known. It is a defect, not a gap: the kernel is at parity when `ld == rows` and
   0.43–0.62× at the `ld` a sub-view actually carries, and **every panel-update caller in the
   tree passes sub-views**. It is what a vendor-free build still pays, everywhere, and the
   mechanism is already localised to `register_tiled_common.hh`. Highest value per unit of risk.
2. **WP4 (`potrf`).** The listed successor, small and low-risk, and it now has WP3's in-SLM
   triangular solve to build on. `ortho` and `syevx_lobpcg` both need it.
3. **Complex GEMM** — the transposed, predicated wide-scalar kernel WP2 identified. Larger, and
   it is a new kernel rather than a routing change.
4. **Splitting the burn-down by backend**, so a landing GPU kernel is visible in the number. Four
   suites now fail vendor-free *only* on the host path; the suite-level count cannot show that
   `trsm` is finished on the GPU.

The original definition of "done" for the first iteration —

```bash
cmake -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF .
cmake --build build-novendor -j"$(nproc)"
ctest --test-dir build-novendor
```

green, with no performance claim attached — is **M1, and it is not yet reached**: the build
configures, compiles, links, loads and runs, but the suite is 25/53. What has changed is that
the remaining gap is now a named list of missing kernels rather than an unknown.

# Vendor Independence Plan

**Goal.** BatchLAS should build, run and perform without cuBLAS, cuSOLVER, cuSPARSE,
rocBLAS, rocSOLVER, rocSPARSE, MKL or netlib LAPACK present — while still *using* any of
them when they are available and genuinely faster.

Every factual claim about the tree was read out of the source and is cited by file and line
so it can be re-checked rather than believed. Several claims in the first draft turned out
to be wrong; each has been corrected in place with the correction called out, rather than
quietly edited.

## Status

**Companion specifications** (each the output of a multi-agent design pass with adversarial
critique, and each superseding the sketch in §5 of this document):

| Document | Covers | Agents |
|---|---|---|
| `WP0_DISPATCH_SPEC.md` | the dispatch axes, the vendor gate, the coverage instrument | 14 |
| `WP3_TRSM_SPEC.md` | native batched `trsm` | 19 (shared) |
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

**Not implemented:** WP0 steps S7–S8, the rest of WP1, and all of WP2–WP4.

Next is S7 (the coverage instrument, which turns the burn-down list into a per-shape
table) and S8 (retiring the four `B == Backend::CUDA` sites that mean "our kernel is wired
here"). After that, WP1 — routing the four level-3 tile routes' terminal `*_vendor_cuda_raw`
call through the public `gemm` entry point — should remove four suites from the baseline on
its own.

Two failures are present in the suite and are **not** from this work — both were reproduced
by rebuilding with the relevant change reverted: `lanczos_tests` (2 cases) and `steqr_tests`
(4 cases, all host-backend, 3 of them double, matching the known bad OpenBLAS kernel).

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
| Public dense ops with **no** native implementation anywhere | **9** (`gemv`, `trsm`, `potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `spmm`) |
| Portable level-3 kernels compiled **only** into the CUDA object library | **4** files (`symm`, `syrk`, `syr2k`, `trmm` custom dispatch) |
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

---

## 5. Work packages

Ordered by (value ÷ risk), not by dependency. WP0 and WP1 unblock measurement; WP2 is the
linchpin; WP3–WP7 are the genuinely new numerics.

### WP0 — Unify dispatch, add the gate

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

### WP1 — Free the level-3 kernels from the CUDA backend

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

### WP2 — GEMM: close the envelope

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

### WP3 — `trsm`

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
2. **Silent numerical failure in the QR/panel work (WP5).** This family has produced exactly
   this failure mode before, and the guarding tests did not catch it. *Mitigate:* write the
   discriminating test first and confirm it *can* fail; use the `-UNDEBUG` device-assert
   recipe for out-of-bounds hunting.
3. **Build time (WP1, WP2).** See above.
4. **Accuracy regression from inversion-based `trsm` and Cholesky-QR (WP3, WP4).**
   *Mitigate:* gate on the existing orthogonality accuracy benchmarks before defaulting.
5. **Tuning surface explosion.** Every new kernel family adds a tuning axis; the retune cycle
   is ~12 min and previously a 2.16× kernel win turned into an 11% `gesvd` loss. *Mitigate:*
   validate every tuning change end-to-end at the algorithm level, never at the kernel level
   alone. And note the prior finding that the routing grid was float-only for a long time —
   generate buckets for every scalar type from the start.
6. **Sunk-cost drift.** There is precedent here: a research document's top-ranked item was
   implemented and measured 85–211× *slower* than the path it replaced. Implementation cost
   already spent is not evidence of value. Every WP in this document should be killed
   without ceremony if its first measurement says so.

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

**The first step is WP0, and specifically the coverage table.** It costs little, it is pure
enabling work, and it converts this document from an argued estimate into a mechanically
verified list. Every effort figure above should be re-derived from that table before any
schedule is committed to.

A reasonable definition of "done" for the first iteration:

```bash
cmake -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF .
cmake --build build-novendor -j"$(nproc)"
ctest --test-dir build-novendor
```

green, with no performance claim attached. That is M1, and it is worth shipping on its own.

# Vendor independence: how dispatch works

> **Covers:** the vendor-free build, the per-library vendor gate, headers keyed on the library
> axis, the contracts every public entry point keeps (positional validators, per-item `info`
> spans), the coverage instrument and the verification tooling; and, as history, the
> `RouteTable` layer that carried all of this before flat kernel selection.
> **Status:** current, except the section [History: the RouteTable layer](#history-the-routetable-layer),
> which describes code deleted by flat kernel selection phase 5. How an op chooses its kernel now:
> @ref design_flat_selection (design), @ref selection (the `src/select/` code) and
> @ref selection_tables (every op's families and tuned tables).

BatchLAS configures, compiles, links, loads and runs with no vendor math library. `cmake -B
build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` yields
`BATCHLAS_HAS_CUDA_BACKEND 1` with every CUDA math library at `0` — a CUDA device with no cuBLAS,
cuSOLVER or cuSPARSE. That is not a special build mode; it is the ordinary build with one axis
switched off, and every mechanism on this page exists to keep it expressible.

This is the architecture document for that property. It contains no performance numbers — every
measured window, every rejected design and every open perf debt lives under
[`docs/perf/`](../perf/README.md), one page per op, starting at
[`docs/perf/dispatch.md`](../perf/dispatch.md). Where the vendor-free build stands today is
[`vendor-free-status.md`](vendor-free-status.md).

## Vendor independence today

Two independent questions decide whether a call can reach a vendor library, and they are spelled
independently:

| axis | type | question it answers | where |
|---|---|---|---|
| device family | `Backend` | which SYCL runtime family is this call compiled for | `include/batchlas/blas/enums.hh` |
| library present | `BATCHLAS_HAS_<LIB>` | which third-party math library exists in this build | `cmake/BatchLASOptions.cmake` → `batchlas/backend_config.h` |

"NVIDIA GPU with no cuBLAS" is therefore `Backend::CUDA` + `BATCHLAS_HAS_CUBLAS == 0`, not a new
device family: the SYCL runtime still targets CUDA, and a build with no vendor library must not change
the answer to "what am I running on". The third question, *whose code runs for this call*, used to be
a `Route{Origin, Algorithm}` (see the history section); it is now an op's **choice**, a value of the
op's family variant, and the vendor library is simply one family, `vendor`.

How the pieces fit:

* **Every public definition lives in `src/ops/<op>/<op>.cc`**, compiled unconditionally into
  `batchlas_dispatch_obj` and instantiated on the device family. The vendor TUs (`cublas.cc`,
  `cusolver.cc`, `rocblas.cc`, `netlib_lapack.cc`, …) define and instantiate only
  `backend::<op>_vendor<B, T>`. The public symbol therefore exists in every build that has the
  device; this was WP0's change, recorded under [The entry-point facade](#the-entry-point-facade).
* **The `vendor` family's `can_run` is `d.has_vendor`**, which `select::device_of` fills from
  `select::has_library<B>(spec.vendor)`, the library group the op's `OpSpec` names. In a
  vendor-free build the vendor entry of every table row is never runnable, so the first runnable
  *native* entry of the row runs (`can_run` is correctness only; the table order is speed).
* **The vendor arm is an `if constexpr`** on the same predicate, so with the library absent the
  vendor call is not compiled at all and there is no symbol to satisfy; its `else` is
  `select::no_vendor<B, T>(spec)`, which records a coverage `miss` and throws `NoRouteError`
  ([The vendor gate](#the-vendor-gate)).
* **Nothing runnable** in a build without the op's library is reported by `select::pick` as
  `NoRouteError`, the per-op census [`vendor-free-status.md`](vendor-free-status.md) counts.
* **Ops with no `choice.hh` still select by hand.** Since the level-3 flat-selection wave (#147)
  these are only `hemm`, `herk` and `her2k` (entry points in `src/ops/level3/level3.cc`), whose
  expand-or-vendor choice is written in `src/backends/cublas.cc`; it is recorded on
  [`docs/perf/level3.md`](../perf/level3.md). The generated @ref selection_tables page lists
  exactly the ops that select from tables.

The MathDx device libraries (cuBLASDx, cuSolverDx) counted as vendor even though their kernels
compiled into our `.so`: the source is NVIDIA's and ships only for NVIDIA, so vendor independence had
to be measurable without them. The level-3 flat-selection wave (#147) deleted every cuBLASDx path and
the MathDx probe (MathDx was absent on both boxes, so none of it ever ran); the rule stands for any
header-only vendor library added later.

## The vendor gate

`src/select/vendor.hh` asks per **library**, not per device family, because the map is not uniform:
on NVIDIA `getrf`/`getri` come from cuBLAS while `geqrf`/`orgqr`/`ormqr`/`getrs` and `potrf`/`syev`
come from cuSOLVER; on AMD all of them come from rocSOLVER. An op names its group once, as
`OpSpec::vendor` (`select::Lib`), and `select::has_library<B>(lib)` answers for it.

| predicate | `select::Lib` | ops | CUDA | ROCM | NETLIB |
|---|---|---|---|---|---|
| `level3_vendor_available<B>` | `level3` | `gemm` `gemv` `trsm` `trmm` `symm` `hemm` `syrk` `herk` `syr2k` `her2k` | `BATCHLAS_HAS_CUBLAS` | `ROCBLAS` | `kHasNetlib` |
| `factorization_vendor_available<B>` | `factorization` | `geqrf` `orgqr` `getrf` `getrs` `getri` `ormqr` | `CUBLAS` **and** `CUSOLVER` | `ROCSOLVER` | `kHasNetlib` |
| `solver_vendor_available<B>` | `solver` | `potrf` `syev` `gesvd` | `CUSOLVER` | `ROCSOLVER` | `kHasNetlib` |
| `sparse_vendor_available<B>` | `sparse` | `spmm` | `CUSPARSE` | `ROCSPARSE` | `kHasNetlib` |
| — | `none` | `gesv`, `posv`: no vendor family | — | — | — |

`kHasNetlib` is `BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS`, tested together because
`netlib_lapack.cc` calls both and is compiled only when both were found. Parallel `k<Group>Library<B>`
constants, reached through `select::library_name<B>(lib)`, give the library name a diagnostic should
quote.

The gate is an `if constexpr` in the op file, so **the vendor call is not compiled at all** when the
library is absent and there is no symbol to satisfy. The alternative design, a stub TU per absent
library defining a throwing `backend::<op>_vendor`, was declined because it restates all 26 vendor
signatures a second time, and signature divergence between restated copies is a defect class this tree
has already shipped. Where a *header* or a non-op caller needs the same gate, `*_vendor_or_throw`
shims do it inline (for example `syev_vendor_or_throw` in `src/ops/syev/vendor.hh`).

When nothing serves a call, `select::throw_no_vendor_route<T>` records a coverage miss and throws
`NoRouteError` (`<batchlas/no_route.hh>`), whose message names the op, the scalar type and the build
switch that would restore it, and deliberately not the backend, which `NoRouteError` carries but
discards when formatting.

A separate question is whether the native kernel is **linked**: `select::level3_tile_route_available<B, T>`,
which is `B == Backend::CUDA && (std::is_same_v<T, float> || bool(BATCHLAS_HAS_CUBLAS))`. Four sites
in `src/extensions/` and one in the coverage table used to spell this `B == Backend::CUDA`, which is
wrong in the vendor-free build — the backend is still `Backend::CUDA` and the tile TUs are not
compiled. It takes a **scalar** parameter because the answer varies per `(backend, scalar)`; the next
section has why.

### The vendor gate: why the tile route predicate is per backend and scalar

Four sites in `src/extensions/` used to ask "is the tile kernel linked" as `B == Backend::CUDA`, one
with the comment "Not a statement about CUDA the vendor -- it is where the kernel is wired." The
comment was right and the expression wrong: the tile kernels are portable SYCL (verified by compiling
`triangular_expand.hh` and the `*_tiles.hh` family standalone at `-fsycl-targets=spir64_x86_64`);
at the time they lived in `{symm,syrk,syr2k,trmm}_custom_dispatch.cc`, which `src/backends/CMakeLists.txt`
compiled only with cuBLAS because their dispatch terminated in `*_vendor_cuda_raw`. With
`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` the backend is still `Backend::CUDA`, the tile TUs were not
compiled, and every such site claimed a kernel that was not linked. (The level-3 flat-selection
wave, #147, deleted those dispatchers: the kernels are now launched from
`src/ops/{symm,syrk,syr2k,trmm}/<op>.cc`, built unconditionally into `batchlas_dispatch_obj`, and
each native `can_run` carries a `kWired = B == Backend::CUDA` term, e.g. `src/ops/trmm/trmm.cc:54`.)

WP1 S7 recorded that the header's own prediction was wrong. It said that once WP1 freed the four TUs
the predicate "becomes true for every backend -- and that is the only edit needed". A bare `true`
would have been wrong in two directions:

* **Too wide in type.** The float routes became reachable everywhere once S6 put their gate in the
  public entry point; the double and complex ones did not: syrk's non-float gram branch and trmm's
  non-float tile branch stayed in `cublas.cc`, and syr2k has no non-float tile route at all
  (`syr2k_triangular_tiles` has one call site, in the float-only dispatcher). A bare `true` tells
  `ortho.cc`'s `gram_via_syrk` (which admits double) that a kernel exists, and vendor-free that call
  throws.
* **Too wide in backend.** The entry-point gate is guarded on `Backend::CUDA`, so nothing is wired for
  ROCM or NETLIB; claiming otherwise re-introduces the defect WP0 S8 removed (the Backend enum
  standing in for a question it does not answer).

Hence the second template parameter: the answer varies per (backend, scalar). Vendor-present
behaviour is unchanged by construction (with `BATCHLAS_HAS_CUBLAS` true it is true for every T on
CUDA); only the vendor-free float case moved, which was the WP1 gain. The level-3 four have since
moved to flat selection (#147) and their own `can_run` decides their public entry points; the
predicate survives for the internal callers that ask "is a tile kernel linked for this (backend,
scalar)" before calling one directly (`ortho.cc`, `ormqr_blocked.cc`, `sytrd_blocked.cc`, and the
coverage table in `src/select/coverage.cc`), and that contract is what they rely on.

Code site: `select::level3_tile_route_available` in `src/select/vendor.hh` (formerly
`include/batchlas/blas/dispatch/route_compiled.hh`).

### The vendor gate: history of the per-library predicates

Until WP0 S5 "is a vendor implementation compiled in" could not be asked at all: the public entry
point was defined inside the vendor TU, so "no vendor" and "no entry point" were the same condition
and the answer was trivially yes. The spec proposed seven stub TUs under `src/dispatch/absent/*.cc`
defining a throwing `backend::<op>_vendor` per absent library; it was declined because it restates all
26 vendor signatures, and S5b's bugs were precisely signature divergence between restated copies.

`factorization_vendor_available` was first keyed on cuBLAS alone. On NVIDIA the group spans two
libraries (getrf and getri are `cublas<t>getrfBatched` / `getriBatched`; geqrf, orgqr, ormqr and
getrs's batch <= 1 arm are cuSOLVER: `cusolverDnXgeqrf`, `cusolverDnSormqr`, ...), so the predicate
claimed a geqrf vendor route whenever cuBLAS was on and told a vendor-free user to "re-enable cuBLAS"
to get geqrf back, which does not restore it. It now requires both. A finer per-op split would have
to move every call site and is open debt (see [What is still open](#what-is-still-open-architecturally)).

Until phase 5 the predicates lived in `include/batchlas/blas/dispatch/vendor_available.hh`, and
`select::Device` carried two flags, `has_vendor_solver` and `has_vendor_blas`, for four library
groups; `OpSpec::vendor` and the single `Device::has_vendor` replaced them
(@ref design_flat_selection, section 12, "After phase 5, one run() per op").

## Vendor independence: headers keyed on the library axis

The same rule applies to includes and types in the private `src/linalg-impl.hh` (WP0 S1/S2). Its
vendor includes (`<cublas_v2.h>`, `<cusolverDn.h>`, `<cusparse.h>` and the ROCm equivalents) are
guarded by `BATCHLAS_HAS_<LIB>`, not by `BATCHLAS_HAS_<FAMILY>_BACKEND`. They used to be guarded by
the family flag, which conflated "can a queue target this device family" with "is this vendor's
math library installed": a CUDA SYCL device with no CUDA math libraries is a coherent configuration,
exactly the one vendor independence aims at, and under the old guard it still tried to include
`<cublas_v2.h>`. `cuda_runtime.h` deliberately stays on the family flag: it is the CUDA *runtime*,
needed for streams and device queries by anything targeting an NVIDIA device, and not a math
library. `<cuComplex.h>` is included explicitly on the same flag: the pointer-cast helpers use
`cuComplex`/`cuDoubleComplex`, which used to arrive transitively through `cublas_v2.h` and so made
those helpers silently depend on cuBLAS.

The CUDA handle types follow suit. `cublasComputeType_t` comes from `cublas_v2.h`, and the handle
triple needs cuBLAS, cuSPARSE and cuSOLVER all three, so those declarations sit under
`BATCHLAS_HAS_CUBLAS` (and the other library flags): under `BATCHLAS_HAS_CUDA_BACKEND` alone, which
can be true with any of them absent, naming those types does not compile. The CUDA device with one or
more math libraries absent still gets a `LinalgHandle<Backend::CUDA>` specialisation with no vendor
handles to own, because the type must be **complete**: a *native* TU declares one
(`src/extensions/ortho.cc` has `static LinalgHandle<B> handle;` with `B` deduced from the queue), and
the primary template is declared without a definition, so leaving the specialisation out makes an
`ortho` build fail on an incomplete type in a file that calls no vendor code at all.

The counterpart for symbols: public instantiations (in `src/ops/<op>/<op>.cc`) are keyed on the
**device family** (the vendor arm compiles to a throw when the library is absent), while the vendor
TUs instantiate only their `backend::*_vendor` symbols; see
[the runtime-internals note](runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols).

## Positional validators: reject only what no route can serve

Each positional (workspace-taking) entry point of `geqrf`, `orgqr`, `getrf`, `getrs`, `getri` and
`potrf` has an `<op>_validate_params` in its public header. It runs first in the op's public
definition in `src/ops/<op>/<op>.cc`, **before the table key is built**, because the key reads
`A.rows()` / `A.cols()` and must not describe a non-conforming view. `trsm`'s `trsm_validate_params`
sits in the same place for the same reason.

**The rule: validate only what no kernel could serve.** A native family's `can_run` that says "this
driver cannot serve this shape" steers the choice to another family, usually `vendor`; it does not
say the call is invalid. A validator that threw on such a shape would turn a working positional call
into an error, which is a user-visible behaviour change and belongs in its own commit with its own
test. The option-struct and arena overloads in `include/batchlas/blas/options.hh` are where the
stricter checks (`require_square`, `require_span_at_least`, `require_info_span`) live.

Per op, what is deliberately *not* checked on the positional path:

* **potrf.** Checks negative extents, squareness and `uplo`. History: there was no
  `potrf_validate_params` anywhere in the tree (the analogous `functions/trsm.hh` one existed).
  `require_square` / `require_info_span` were attached only to the option overloads; the
  workspace-taking `<Backend B>` overload — the spelling `src/extensions/ortho.cc` uses — had neither,
  so a non-square view reached the backend and cuSOLVER factorised `A.rows() x A.rows()` out of it. It
  does **not** check the length of a non-empty `info` span: a short non-empty span silently becomes
  pool scratch (`detail::info_target` in `src/linalg-impl.hh`, whose test is `>= count`), documented as
  by design; turning that into a throw is a behaviour change.
* **geqrf.** One check (negative extents). No squareness check: rectangular A is the point of geqrf
  and the library's own callers pass tall panels (`band_reduction.cc`, `sytrd_sy2sb.cc`); copying
  potrf's `A.rows() != A.cols()` there would be a wrong edit. No `m >= n` check, although the native
  families' `can_run` (`src/ops/geqrf/can_run.hh`) carries one: that leaves a wide view to the vendor,
  which serves it. No `tau` length check: the arena spellings already `require_span_at_least`.
* **orgqr.** Negative extents only. No `n <= m` check although the native `can_run` has one: Q's
  columns live in R^m, so n > m is meaningless, but every backend currently accepts such a view and
  hands it to a vendor. No `tau` length check (arena spellings do it).
* **getrf** (WP6). Negative extents only. No squareness check, although the native `can_run` and the
  arena spellings (`require_square`) carry one. No pivots length check; the arena spellings do
  `require_span_at_least`.
* **getrs** (WP6). Negative extents only. Squareness of A, `A.rows() == B.rows()`, equal batch and the
  pivot span's length are all checked on the arena spellings; a non-conforming pair is refused by the
  native families' `can_run`, which leaves the vendor.
* **getri** (WP6). **Two arities, forced by the signatures:** `getri_buffer_size` takes A alone while
  the call takes A and C, and the query must validate exactly the view its choice is computed from —
  `ops::getri::key_of` is a function of A alone. The two arities check A identically; the second adds
  C's extents, which nothing else on the positional path looks at. Neither checks squareness of A or C,
  their agreement in order and batch, or the pivot span's length (all checked on the arena spellings);
  a non-square A is refused by the native `can_run`, which leaves the vendor.

**The contrast: gesv and posv reject more, structurally.** They have no vendor family
(`OpSpec{..., Lib::none}`), so a non-conforming pair would not be served by anyone; it would reach
"nothing runnable" and report the wrong cause. Their validators therefore check squareness,
`B.rows() == A.rows()` and equal batch (posv also `uplo`).

Until phase 5 the validator ran in the facade (`src/dispatch/entry_points/factorization.cc`) ahead of
the per-op shape builder, which the native "route" predicates read; the rule was then phrased as
"a `supports()` that says the native drivers cannot serve this routes the call to the vendor".

## Per-item info spans for potrf, getrf and getri

`info` on `potrf`, `getrf` and `getri` (and, through them, `posv` and `gesv`) is the LAPACK per-item
status: one int32 per batch item, 0 on success, and > 0 for the leading minor that is not positive
definite (potrf), the column at which U became exactly singular (getrf), or the zero diagonal of U
that leaves the item without an inverse (getri).

History (issue #73): it used to be unreachable. Every backend allocated the array the vendor call
needs, passed it, and dropped it, so a caller could not tell a batch that factorised from one where
item 37 is rank-deficient (potrf), singular with every downstream `getrs`/`getri` producing noise
(getrf), or left holding a matrix of infinities (getri).

Design points that are part of the contract:

* An **empty** span means "not requested" and is exactly the previous behaviour: the backend falls
  back to its own scratch allocation.
* The **workspace size is deliberately the same either way**, so `*_buffer_size` stays correct whether
  or not a caller asks for status.
* A **short non-empty** span is silently treated as "not requested" on the positional path
  (`detail::info_target`, `src/linalg-impl.hh`); the option overloads reject it with
  `require_info_span`. See the potrf item under
  [Positional validators](#positional-validators-reject-only-what-no-route-can-serve) for why the
  positional path does not throw.
* **Old-arity forwarders, not a defaulted parameter.** `info` cannot be a defaulted trailing
  parameter: the `sig::` aliases used for explicit instantiation (`BATCHLAS_INSTANTIATE`,
  `src/util/template-instantiations.hh`) are function *types*, and function types cannot carry default
  arguments, so a default would not be part of the instantiated signature. A separate inline overload
  keeps every existing call site of the old arity compiling unchanged. `gesv` and `posv` follow the
  same pattern.

## Info spans on syev, gesvd and steqr: forwarder or default

Five routines (`syev`, `syevx`, `gesvd`, `steqr`, `stedc`) and every tier below them take a trailing
`Span<int32_t> info`, and two different declaration shapes carry it. Both keep every pre-existing call
site compiling unchanged; which one an entry point uses is forced by how it is explicitly
instantiated. The user-facing contract of `info` is in
[the C++ API page](../cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc).

**Public `syev` and `gesvd`: an old-arity forwarder, not a defaulted parameter** (the same shape as
`potrf.hh`). What forces it is the `sig::syev` / `sig::gesvd_vendor` alias used by
`BATCHLAS_INSTANTIATE` (`src/util/template-instantiations.hh`): it is a function *type*, and function
types cannot carry default arguments, so `info` has to be spelled out in the alias whichever way the
declaration is written. Leaving the declaration default-free too keeps alias and declaration
parameter-for-parameter identical, which is the invariant `BATCHLAS_INSTANTIATE` reads. The inline
forwarder is then what keeps the six-argument `syev` and the eight- and nine-argument `gesvd` call
sites (the `SyevOptions` / `GesvdOptions` spellings in `blas/options.hh` among them) compiling. For
`gesvd`, arity plus the `Uplo`/`Span` type difference at parameter 8 keeps all four overloads
unambiguous.

**`backend::syev_vendor` and `backend::gesvd_vendor`: a defaulted `info_out`.** A default argument is a
property of the declaration, not of the function type, so `sig::syev_vendor` still names the full
seven-parameter signature (nine for `gesvd_vendor`) and the explicit instantiations in the vendor TUs
still match. That default is what keeps the internal six-argument `syev_vendor` callers
(`src/extra/norm.cc`, `src/extra/cond.cc` and `src/extensions/syevx_lobpcg.cc`, all through
`syev_vendor_or_throw` in `src/ops/syev/vendor.hh`; before the flat-selection merge also
`src/backends/cusolverdx.cc`) compiling with no extra overload; none of them is public API, so none
needs a forwarder.

**`steqr`, `steqr_cta`, `stedc`, `syevx*`, the `syev_*` tiers and the SVD tiers (`extensions.hh`): a
defaulted trailing parameter.** They are instantiated from hand-written macros that spell the
parameter list out, not from a `sig::` alias, and they already default `jobz`, `params` and
`eigvects`, so an old-arity forwarder would be *ambiguous* with the primary rather than additive.

History: before the `info` span existed, every vendor already allocated a per-item status array
because the vendor call demands somewhere to write, and it was pool scratch that nobody read.
cuSOLVER's `gesvdjBatched` returned an info array that the library dropped; the netlib path captured
LAPACKE's scalar `info` per item and threw it away in a batch-wide exception. `backend::gesvd_vendor`
also used to be *defined* in `functions/gesvd.hh` (a NETLIB LAPACKE loop plus a throw for every other
backend), which made a CUDA definition in `src/backends/cusolver.cc` a redefinition error rather than
an override — the reason there was never a cuSOLVER SVD binding until the header became
declaration-only (the LAPACKE body now lives in `src/backends/netlib_lapack.cc`; see
[the gesvd design page](gesvd.md#gesvd-design-vendor-binding-and-dispatch)).

## Vendor independence: the coverage instrument

`src/select/coverage.hh` writes three kinds of row, and reading one as another is how a working
vendor-free `gemm` came to be claimed at a point when every such call threw.

| row kind | question | how produced | cost |
|---|---|---|---|
| `linked` | is the kernel **in this build** — the planning question | `coverage::static_table()`: per `(op, backend)`, the vendor gate and a native-linked flag, no kernel run | exact, instant, no GPU needed |
| `reached` | did a call **get there** — the burn-down question | one row per `(op, scalar, backend, shape_class, variant, choice)`, recorded by `select::TraceScope` (and by the hand-selected ops at each terminal) | one predicted branch per op invocation |
| `miss` | nothing served this call at all | recorded by `throw_no_vendor_route` | rare by construction |

**`linked` is not `reached`, and a symbol being present is never evidence it runs.** An op can report
`native = 1` in the `linked` table while its tables never rank a native family first on this device,
so a vendor-present build sends it nothing and only a vendor-free build reaches it. The `linked`
column is reported **for `float`**, because the level-3 tile routes are float-only outside a cuBLAS
build and a type-blind column would restate exactly the overclaim it exists to prevent.

`native_route_supported` on a `reached` row is a **tri-state**: `1` yes, `0` no, `-1` the call site
could not tell. For a table-selected op `select::native_facts` computes it from the candidates'
`can_run`. The third value is load-bearing for the hand-selected level-3 ops: a declining gate never
enters the tile dispatcher, so it cannot distinguish "nothing native serves this shape" from
"something does but the heuristic preferred the vendor", and recording either as definite would be a
claim the call site cannot support. The gate-declined half is recorded explicitly, beside each
`return` and never in place of one — a shape moving *off* a native kernel is otherwise invisible.

Four properties of the row key:

* `uplo`/`side`/`diag`/`transA`/`transB` are part of the **key**, not decoration
  (`variant_key` in `src/select/coverage.cc`). They select which triangle or which operand an op
  touches, so two calls differing only in `uplo` must not collapse into one first-writer-wins row.
* `shape_class` buckets `max(m,n,k)` and `batch` by power of two (`coverage::Shape::shape_class`), so
  a 10,000-iteration test collapses to a handful of rows rather than 10,000.
* Rows are **first-writer-wins**, so the `m`/`n`/`k`/`batch` columns can report a *different* call's
  exact shape. A coverage row cannot confirm that a particular shape ran; prove that with a deliberate
  break that is red only for it.
* A table-selected op's `reached` rows carry the real backend: `select::run` fills `shape.backend`
  from its template parameter. In the route era the shape builder never learned the backend, and
  every such row read `AUTO`.

The instrument is gated at **runtime** on `$BATCHLAS_COVERAGE_OUT` — the same variable `emit()` reads,
so recording and emission cannot disagree about whether coverage is on. Two further failure modes
are fixed in place and worth not reintroducing: the tables are deliberately **leaked** so an `atexit`
handler cannot walk a destroyed container, and `emit()` writes **one file per pid** because a `ctest`
run is dozens of binaries and a shared `"w"` handle meant each truncated the last.

### Coverage instrument: why the compile-time gate was rejected

Why two tables: [`vendor-free-status.md`](vendor-free-status.md) records which test SUITES fail, which
is the right unit for spotting regressions and the wrong unit for planning: "ortho_tests fails" does
not say whether the gap is potrf, trsm, geqrf, orgqr or gemv, and one missing kernel fails a dozen
suites. The static table names the ops with no native kernel; the dynamic one says which shapes real
callers hit, so WP3-WP8 could cover those first.

The gate was a compile-time macro first, and that was wrong twice over. The stated reason ("the
counters are cheap per call, but gemm is called in inner loops") did not survive reading the code: the
route resolver ran once per op invocation, not per element, and already walked the order calling the
predicates, so a predicted branch on a global bool is far cheaper than the work beside it. The failure
that bit: with `-DBATCHLAS_ENABLE_COVERAGE=ON`, `libbatchlas_backends.so` referenced
`coverage::record`, yet `gemm_tests` produced a file with a correct header and zero `reached` rows,
because `gemm_tests` carried its own `resolve_route_uninstrumented<Op::gemm, float>` weak copy that
ELF bound ahead of the library's. A compile-time switch on an inline function in a header is sound
only if every TU in the process agrees, which a library cannot enforce on its consumers.
`cmake/BatchLASOptions.cmake` records that the option was deliberately never added.

## Verification tooling

| script | what it is for | the property that makes it trustworthy |
|---|---|---|
| `scripts/route_diff.sh capture\|compare` | prove a change moved **no selection decision** | treats a capture with **zero `reached` rows as a hard error**, not as "nothing changed" — the instrument has produced a correct header with no rows twice, for unrelated reasons, and both times it looked clean |
| `scripts/coverage_merge.sh` | collapse the per-pid shards a `ctest` run produces | sums `calls`, de-duplicates the identical `linked` block every process emits |
| `scripts/facade_symbol_check.sh` | prove the public entry points are **not** defined in a vendor component | matches **Itanium mangling directly**: `nm -C` silently fails to demangle concept-constrained templates and would report `symm`/`herk` as missing when present |
| `scripts/rocm_syntax_check.sh` | `-fsyntax-only` the three ROCm vendor TUs this machine never compiles | the gate is **exactly one** expected error (a `get_native<ext_oneapi_hip>` overload this CUDA-only DPC++ lacks); anything else is a real defect. It forces the CUDA macros off, exercising the per-library `#if` structure nothing else here builds. The ROCm headers live under `/opt/rocm/include/roc*/`, a subdirectory, which is why a naive probe reads them as absent |
| `scripts/register_probe.sh` | register/spill residency of the SYCL device link | replays a target's `link.txt` verbatim, so the flags stay exactly the real build's; **fails loudly** when the named target has no `link.txt` rather than silently probing the default library |
| `BATCHLAS_SELECT_TRACE=1` | print each decision, its table and the runner-up | one line per call, indented under its parent op; the tag names a borrowed table, an override or the last resort |

`route_diff.sh` is the only tool that can see a **vendor-to-vendor** change: the kernel trace cannot
(its `Record` holds a `sycl::event`) and timing cannot (an unsaturated ratio is overhead, and routing a
shape to the vendor may well be faster, so a perf gate cannot flag a wrong choice). Its comparison key
is `(kind, op, scalar, backend, shape_class, origin, algo, native flags, uplo, side, diag, transA,
transB)` — `m`/`n`/`k`/`batch` and the call count are dropped on purpose, since counts vary with test
scheduling and are not part of the decision. It applies **no `backend != AUTO` filter**, so pure-layer
test shapes recorded with `backend = AUTO` can make a small real move look like hundreds of lines of
churn.

`register_probe.sh`'s gate is **not** "stack frame == 0": on this tree 220 of 376 entry functions carry
a non-zero stack frame with zero spills, so gating on it rejects healthy kernels. Use `0 bytes spill
stores, 0 bytes spill loads` on the kernel's lines, and `Used N registers × work-group size <= 65536`,
the per-block limit whose failure mode is a launch abort rather than a slowdown. Each kernel appears
twice, as `<name>` and `<name>_with_offset`; take the max.

Adding an op is described in [Adding entry points](../extending.md) and the @ref selection group;
the vendor-specific part is one `OpSpec::vendor` value and the `if constexpr` arm above.

## What is still open, architecturally

Per-op performance debts live on the `docs/perf/` pages. These belong to the vendor seam itself:

1. **`route_diff.sh compare` applies no backend filter**, so `AUTO` rows from pure-layer tests inflate
   every diff.
2. **`Backend::INTEL` is hard-wired false and oneMKL cannot be tested here**; only the dead branch that
   produced undefined references was removed. ROCm is reachable only through `rocm_syntax_check.sh`;
   statements about it are stated, not verified. (MathDx-present boxes were the other untestable
   case until #147 deleted cuBLASDx.)
3. **The runtime `BATCHLAS_NO_VENDOR=1` enforcement knob was never built.** Call sites that reach
   `backend::*_vendor` or a `*_vendor_or_throw` shim directly (for example the `syev` calls in
   `src/extra/cond.cc`, `src/extra/norm.cc` and `src/extensions/syevx_lobpcg.cc`) bypass the public
   entry point and therefore its selection, and throw vendor-free by construction. The build-time
   switch `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` is what enforces independence today.
4. **`factorization_vendor_available` is one predicate for six ops over two NVIDIA libraries**; a
   per-op split (getrf/getri on cuBLAS, the rest on cuSOLVER) would have to move every call site.
5. **Ops without a `choice.hh` still select by hand** and keep their windows in code; see
   [`docs/perf/dispatch.md`](../perf/dispatch.md) and @ref selection_tables for which ops these are.
6. **The vendor-free suite is not green, and no selection mechanism can make it so** — the remaining
   gap is missing kernels, not routing. The failing **set** is the reviewable artefact, not its count;
   it is tracked in [`vendor-free-status.md`](vendor-free-status.md).

Resolved since the route era: `Route::library` (an output the resolver never wrote) went with
`Route`; the static coverage table's `trsm` row reads `true` (WP3); the claims of a
`BATCHLAS_ENABLE_COVERAGE` build option went with `route_resolve.hh`; the two known-wrong forced
level-3 routes (`BATCHLAS_SYRK_ROUTE=native`, `BATCHLAS_SYR2K_ROUTE=native`) were fixed by the level-3
pin words and are now ordinary flat-selection pins ([`docs/perf/dispatch.md`](../perf/dispatch.md#the-level-3-pin-words)).

## History: the RouteTable layer

Everything in this section describes code that **no longer exists**: `include/batchlas/blas/dispatch/`
(`route.hh`, `route_resolve.hh`, `route_env.hh`, `vendor_available.hh`, `route_compiled.hh`,
`op.hh` and the per-op `route_<op>.hh` tables), the `src/backends/<op>_route.hh` shape builders and
`src/dispatch/entry_points/`, all deleted by flat kernel selection phase 5
(@ref design_flat_selection, section 12, "Phase 5 rip, the old layer removed"). File:line citations
below refer to that deleted code. It is kept because its reasoning carried over — `supports()` became
`can_run` (correctness only), `preferred()` became the table ranking, and "tables are pure" became
"sizing and running call the same `choose()`" — and because code still cites
[The entry-point facade](#the-entry-point-facade).

### The three axes

Three questions were once answered by one enum (`Provider`, deleted by WP0). WP0 spelled them
independently: device family (`Backend`), library present (`BATCHLAS_HAS_<LIB>`), and **route**,
`Route{Origin, Algorithm}` (`route.hh:43-53`).

`Origin ∈ {Auto, Native, Vendor}` answered *whose code* (`route.hh:16-20`); `Algorithm` answered
*which strategy* (`route.hh:22-39`). There was deliberately no `Origin::SYCL`: every route in the tree
is SYCL, so the value would name nothing and would collide with the device-family axis
(`route.hh:3-5`). The MathDx libraries were `Origin::Vendor` for the reason given at the top of this
page.

Two predicates sat on `Origin`, and confusing them was a shipped-and-fixed defect. `is_vendor(r)` was
the gate question (`route.hh:56`). `is_plain_vendor(r)` — `Vendor` **and** `Algorithm::Auto` — was
"the ordinary library call" (`route.hh:61-63`); the level-3 dispatchers' `request == Vendor` tests
meant `cublasSsyrk` specifically, and rendering them as `is_vendor()` made a forced cuBLASDx request
answer yes to "did the caller ask for the vendor?". The level-3 pin words keep `vendor` and
`cublasdx` apart today.

**Known wrong when deleted.** `Route::library` and `Route::library_valid` were declared as resolver
outputs and excluded from `operator==` (`route.hh:46-51`), but nothing in the tree ever wrote them: no
resolver, no table, no facade. Every resolved `Route` carried the default `BackendLibrary::CBLAS` with
`library_valid == false`. The library name a coverage `miss` row carries came, and comes, from
`throw_no_vendor_route`'s own `library` argument.

### The three routing predicates

Each `RouteTable<Op, T>` supplied two required static predicates and one optional third, answering
genuinely different questions.

| predicate | question | `false` means | consulted |
|---|---|---|---|
| `supports(r, s)` | can `r` produce the **correct answer** for shape `s` | the kernel would compute a **wrong answer** or index out of bounds | always, including for a forced route |
| `preferred(r, s)` | is `r` the best route available, **vendor included** | merely **slower** | on the automatic walk, in every build |
| `native_tier_preferred(r, s)` | among the **native** routes that serve `s`, is `r` the better one | another native tier is better here | only on the vendor-free walk |

Three rules followed, and each cost this codebase something; the first and third survive as flat
selection's R3 and R5:

* **Never put a speed threshold in `supports()`.** A forced route bypassed `preferred()` — that is what
  forcing is for — but never `supports()` (`route_resolve.hh:76`). A speed cutoff there made a pinned
  route fall through to `automatic()`, so the test that pinned it silently measured something else.
  Conversely, moving a measured window into `supports()` left a working shape with **no supported
  route at all** the moment the vendor went away. `can_run` follows the same rule.
* **Never fix a vendor-free tier choice in `preferred()`.** `preferred()` was consulted by the loop
  above the vendor-free walk, which ran regardless of `vendor_available`. A window written to pick the
  right native tier therefore also moved vendor-present traffic onto that tier — including at shapes
  where the vendor beat both natives. That is what `native_tier_preferred` existed for
  (`route_resolve.hh:18-25`). In a table the vendor-free order is simply the ranking with the vendor
  entry skipped.
* **Tables are pure.** Everything a table read came from its arguments: no `getenv`, no SYCL query, no
  dereference of operand data. That is what made an op and its `*_buffer_size` query reach the same
  route *by construction* (`route_resolve.hh:3-4`) — the `ormqr` defect where the sizing query returned
  2560 bytes and the call then demanded 276480 became structurally unreachable.

`native_tier_preferred` was optional and defaulted to `true` (`native_tier_preferred_or_default`),
which made the two vendor-free passes identical for a table that did not declare it — `true` rather
than `false` because a table that had not thought about the question had to keep its old answer.
Three tables declared it: `geqrf`, `getrf` and `getrs`.

### The resolver

`dispatch::resolve_route<Op, T>` (`route_resolve.hh:85-103`) was the instrumented entry point and the
only one ops called; it wrapped a pure `resolve_route_uninstrumented` (`:89-176`):

1. `forced.origin == Auto` → `automatic()` (`:132-134`).
2. `automatic()` walked `Table::order_begin()..order_end()` and took the first route that was both
   `supports` and `preferred` (`:110-112`).
3. Only if `vendor_available == false` did it then accept a merely **supported** native route — in
   two passes, the first honouring `native_tier_preferred`, the second the raw order (`:113-128`).
   Taking "first merely supported" unconditionally inverts the default for small shapes, because the
   orders listed natives first.
4. Falling all the way through returned `{Vendor, Auto}` (`:129`), the honest "this needs a vendor and
   there isn't one" signal; the caller turned it into a diagnostic, not a wrong answer.
5. A forced **vendor** was honoured only when `vendor_available` **and** `supports()` held (the
   `is_vendor(forced)` branch); otherwise it fell back to `automatic()` (`:142-144`).
6. A forced **bare origin** (`native`, no algorithm) walked the order restricted to that origin,
   preference first then mere support (`:153-163`). Returning `{Native, Auto}` verbatim would have
   handed the caller a route no dispatch tail could map to a kernel.
7. A forced route that `supports()` the shape was returned (`:165`); one that did not fell back to
   the **ordinary automatic choice**, not to the vendor (`:175`).

**The silent trap in rule 7, which this campaign paid for repeatedly.** Pinning a route the shape
could not take did not fail and did not warn — it resolved to `automatic()`, which in a vendor-present
build *is* the vendor. `BATCHLAS_SPMM_ROUTE=cta` resolved to `{Native, CTA}`, `supports()` rejected it
because no CTA body existed, and the run silently measured cuSPARSE. A **misspelled** value: originally
`parse_route_value` failed, the `ParsedRouteEnv::unparsed` flag was discarded at every adapter's
`parsed.found ? parsed.route : legacy_unset_default(...)`, and every decision went to the vendor with
no message; later the unset default became `{Auto, Auto}` for every op and `warn_unparsed_route_env`
printed a one-time warning, so a typo resolved to Auto *with* a warning. Flat selection made both
errors (rule R6): a pin that does not parse, names no compiled candidate, or fails `can_run` throws
`std::invalid_argument`.

### Route tables and shape structs

Thirteen ops had a `RouteTable<Op, T>` specialisation: `gemm`, `gemv`, `trsm`, `potrf`, `getrf`,
`getrs`, `getri`, `geqrf`, `orgqr`, `ormqr`, `gesvd` and `spmm` one header each under
`include/batchlas/blas/dispatch/`, and `syev` in `include/batchlas/blas/functions/syev.hh`. Flat
selection deleted them one op at a time (potrf in phase 2, posv/trsm/gemm in phase 3, the rest in
phase 5).

Each table was paired with a **shape builder** — `src/backends/<op>_route.hh`, or the op header for
`gemm`/`gesvd`/`syev`/`ormqr` — where everything impure happened: the `getenv`, the SYCL device query,
the operand-agreement checks, and the calls into kernel TUs that asked what the build actually
contained. A builder returned `std::optional<Shape>`, and `nullopt` meant "these views do not describe
one call of this op", which resolved to the vendor. These headers could include only public headers
plus at most one private kernel header, which is what let the vendor-free facade include them at all.
Today `key_of` and `can_run` in the op file do that work, and `select::Device` carries the device
facts.

An op whose routing read something `OpShape` had no field for **extended** it rather than growing
`OpShape` into a union of every op's arguments (`TrsmShape`, `GesvdShape`, `SpmmShape`, `GeqrfShape`,
`GetrfShape`, `GetrsShape`). `resolve_route` deduced `Shape` as a third template parameter and sliced
it back to `OpShape` on the way into coverage, so **a derived shape must never shadow an `OpShape`
field** — the builder wrote the shadow, the slice copied the base, and every coverage row reported the
default.

The convention for build capabilities was uniform and load-bearing: the builder asked the kernel TU
what existed and stored the answer in a shape field, and the *absent* value made the native route
**unsupported** rather than selectable-but-unimplemented (`TrsmShape::cta_max_n == 0` meant the CTA
kernel was not in the build). A literal constant in the table header instead would have laundered a
hypothesis into a compile-time fact. The same convention holds for `can_run`: a family whose kernel is
not linked is not runnable.

**The four level-3 tile ops never had a `RouteTable`.** `symm`, `syrk`, `syr2k` and `trmm` kept
hand-rolled `if`-chains, expressed as neither `supports()` nor `preferred()`, with their gates in the
facade and their terminals instrumented directly. Wiring them to the resolver was a real change, not a
transcription: the live thresholds were gate-only, so transcribing them into `preferred()` rejected the
tile route for shapes it served (negative result 1 on [`docs/perf/dispatch.md`](../perf/dispatch.md)).

`Op::iluk` existed in the enum and was referenced by nothing but `op_name`: ILU(k) is a BatchLAS
algorithm with no vendor alternative and dispatches through `BATCHLAS_DISPATCH_ON_QUEUE`.
`extensions.hh`'s entry points were absent from `Op` for the same reason.

### The entry-point facade

> **Now:** the facade directory is gone; each op's public definition lives in
> `src/ops/<op>/<op>.cc`, compiled unconditionally into `batchlas_dispatch_obj`. Everything below
> about *where definitions live and why* still holds for those files; the `level3.cc` /
> `factorization.cc` line citations are to the deleted facade TUs.

The original obstacle to vendor independence was not routing at all — it was **definition ownership**.
`gemm<Backend::CUDA, float>` was defined *inside* `cublas.cc` and instantiated there, so "build without
cuBLAS" did not mean "lose the cuBLAS gemm path", it meant "lose `batchlas::gemm` entirely". The same
held in `rocblas.cc` (around `:99`) and `netlib_lapack.cc` (around `:288`), and in `cublas.cc` (around
`:1568`; line numbers as of WP0 S5). No amount of enum, CMake or predicate work addresses that. The
declaration-only `backend::*_vendor` pattern the facade adopted had been used all along by
`syev_vendor` (`functions/syev.hh`) and `ormqr_vendor` (`functions/ormqr.hh`).

`src/dispatch/entry_points/` then owned the public definitions, compiled unconditionally and gated on
no vendor library. The split was:

| translation unit | defines | count |
|---|---|---|
| `entry_points/level3.cc` | `gemm` `gemv` `trsm` `symm` `hemm` `herk` `her2k` `syrk` `syr2k` `trmm` | 10 |
| `entry_points/factorization.cc` | `geqrf` `orgqr` `getrf` `getrs` `getri` `potrf`, each with its `*_buffer_size` | 6 + 6 |
| `entry_points/sparse.cc` | `spmm`, `spmm_buffer_size` | 1 + 1 |
| `entry_points/eigen.cc` | nothing — it relocates the **instantiations** of `syev` and `ormqr` | 2 |

and the shape of each op is still:

```
vendor TU (cublas.cc, …)   defines and instantiates  backend::<op>_vendor<B, T>
src/ops/<op>/<op>.cc       defines and instantiates  <op><B, T>, which selects and may call it
```

Five properties of this layer are load-bearing, and all five carried over to `src/ops/`:

* **Instantiation is keyed on the device family, not the library** (`level3.cc:394`). The bodies
  compile to a throw when the library is absent, so the public symbol exists in every build that has
  the device — which is exactly what stopped being true when the definitions lived in the vendor TUs.
* **An instantiation binds as hard as a definition.** `syev` and `ormqr` were already *defined* in
  headers, but their explicit instantiations lived in `cusolver.cc`/`cublas.cc`, which is enough to
  make them vanish from a build without those libraries; moving the instantiation was the whole change
  for those two. `gesvd`'s public template was then `inline` in `functions/gesvd.hh` and forwarded to
  `gesvd_dispatch`.
* **The selection runs before the vendor-available test.** Anything below `if constexpr
  (!<group>_vendor_available<Back>)` is unreachable in the vendor-free build, which is the build the
  campaign exists for. Every op chooses first and throws second.
* **An op moves together with its `*_buffer_size` query.** Splitting them lets the two choose
  differently, which is exactly the `ormqr` 108x sizing defect (`factorization.cc:6-7`).
* **Backend asymmetries are preserved, not normalised.** rocBLAS has no `hemm`/`herk`/`her2k`/`symm`
  wrapper, so the ROCm backend instantiates only the ops it implements (`level3.cc:398`).

The public definition is also the **injection point** for native drivers that need a selected
sub-operation. A native driver is instantiated per scalar type with no `Backend` parameter, so it
cannot name `gemm<B, T>` itself; the op file passes a lambda. `trsm`'s blocked driver takes its
trailing GEMM this way, `potrf`/`getrf`/`getrs`/`getri` take the public `gemm`/`trsm`, and `orgqr`'s
native arm takes the public `ormqr`. The alternative — the driver calling `sycl_gemm::gemm_custom`
directly (deleted in P3.4) — bypasses gemm's selection and pins the native GEMM even on shapes it is
measured to lose; see [`docs/perf/trsm.md`](../perf/trsm.md) and [`docs/perf/gemm.md`](../perf/gemm.md).

`scripts/facade_symbol_check.sh` verifies the move by symbol rather than by diff, because a forwarder
left behind, or an instantiation pointing at the wrong template, still compiles and links.

### Environment overrides

Canonical spelling was `BATCHLAS_<OP>_ROUTE`, synthesised from `op_env_stem` (`route_env.hh:135-164`).
A value was an origin (`vendor`, `native`), an algorithm (`cta`, `expand_gemm`, …), or both joined by
a colon (`native:register_tiled`); parser at `route_env.hh:50-70`. A bare algorithm implied `Native`,
**except** `FusedDevice`, which is vendor code by definition (`:92-97`). `netlib` mapped to `Vendor`,
not to an algorithm, because netlib LAPACK is somebody else's code (`:47-53`). The variable name
survives; its values are now a choice spelling or a class word (@ref selection, "Pins").

Unset meant `{Auto, Auto}` for **every** op (`legacy_unset_default`, `:145-148`). GEMM used to be the
odd one out at `{Vendor, Auto}`; that asymmetry went with WP2 E6, and the reasoning is in
[`docs/perf/gemm.md`](../perf/gemm.md).

Eight ops had a legacy variable, read only when the canonical one was unset (`:109-121`,
`:214-245`). They were kept because they appeared in committed benchmark scripts and in the provenance
of recorded results. Three collisions between the two vocabularies were deliberate (`:150-203`):

| spelling | canonical meaning | legacy meaning | legacy maps to |
|---|---|---|---|
| `BATCHLAS_GEMM_VARIANT=native` | BatchLAS's own kernel | the **raw CUDA vendor path**, consumed purely as an exclusion | `{Vendor, Direct}` (`:178-182`) |
| `custom` in `symm`/`syrk`/`syr2k`/`trmm` | the register-tiled GEMM family (`:63`) | the fused cuBLASDx kernel | `{Vendor, FusedDevice}` (`:185`) |
| `gemm` in `syrk`/`syr2k` | the `gemm` op | the deliberately wrong `DiagFullGemm` measurement route | `{Vendor, DiagFullGemm}` (`:190-198`) |

Because the two parsers differed, the two spellings did not agree even for the same op:
`BATCHLAS_SYMM_VARIANT=custom` reached the fused arm and `BATCHLAS_SYMM_ROUTE=custom` did not.
`Algorithm::DiagFullGemm` computed and stored **both** triangles, which is not what `syrk` or `syr2k`
mean, and existed only so the arithmetic the triangular kernels save could be measured against it.

Two routing variables were not op-keyed: `BATCHLAS_EXPAND_ROUTE=expand|loop`, which pins the
mirrored-expansion decision for `symm`/`hemm`/`herk`/`her2k`/`trmm` and is consulted **before** the
measured window (still so; see [`docs/perf/dispatch.md`](../perf/dispatch.md)), and
`BATCHLAS_COVERAGE_OUT`. Two per-op ad-hoc knobs (`BATCHLAS_ORTHO_GRAM`, `BATCHLAS_ORMQR_IMPL`) were
never folded into the vocabulary.

The vocabulary was pinned by `tests/route_vocabulary_tests.cc`, including every legacy spelling and
every collision above; the GEMM transcription itself was pinned by
`tests/route_gemm_equivalence_tests.cc` until P3.4 deleted it with `route_gemm.hh`. Phase 5 deleted
the legacy spellings, their aliases and `route_vocabulary_tests.cc`; each op's tests assert that the
old spellings now throw.

### Adding an op in the route era

The recipe, for reading old commits: add the `Op` enumerator and its `op_name` case; write
`RouteTable<Op::x, T>` (order array, `supports()` as correctness only, `preferred()` as a measured
window, `native_tier_preferred()` only with more than one native tier, pure); write the shape builder
in `src/backends/<op>_route.hh`; define the public entry point in `src/dispatch/entry_points/` with its
`*_buffer_size` in the same TU, resolving before the vendor test; add the op's row to the coverage
table, flipping its `native` column in the same step as the wiring; capture a `route_diff.sh` baseline
before and compare after; run `facade_symbol_check.sh` and `rocm_syntax_check.sh` as relevant. The
current recipe is [Adding entry points](../extending.md).

## Where the rest of it is

How selection works now: @ref design_flat_selection, @ref selection and @ref selection_tables.
Per-op measured windows, the grids that justify each boundary, the built-and-rejected designs and the
correctness findings live under [`docs/perf/`](../perf/README.md):
[`dispatch`](../perf/dispatch.md), [`gemm`](../perf/gemm.md), [`level3`](../perf/level3.md),
[`trsm`](../perf/trsm.md), [`potrf`](../perf/potrf.md), [`qr`](../perf/qr.md), [`lu`](../perf/lu.md),
[`gemv`](../perf/gemv.md), [`spmm`](../perf/spmm.md). The superseded root design documents
(`WP0_DISPATCH_SPEC.md`, `WP1_LEVEL3_SPEC.md`, `VENDOR_INDEPENDENCE_PLAN.md`,
`VENDOR_FREE_BASELINE.md` and the per-work-package specs) are retained at the git tag
`perf-evidence/vendor-independence` and retrievable with
`git show perf-evidence/vendor-independence:<path>`.

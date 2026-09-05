# Vendor independence: how dispatch works

BatchLAS configures, compiles, links, loads and runs with no vendor math library. `cmake -B
build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` yields
`BATCHLAS_HAS_CUDA_BACKEND 1` with every CUDA math library at `0` — a CUDA device with no cuBLAS,
cuSOLVER or cuSPARSE. That is not a special build mode; it is the ordinary build with one axis
switched off, and every mechanism on this page exists to keep it expressible.

This is the architecture document: the vocabulary, the predicates, the gate, the facade, the
instrument and the tooling. Read it before adding an op or moving a route. It contains no performance
numbers — every measured window, every rejected design and every open perf debt lives under
[`docs/perf/`](../perf/README.md), one page per op, starting at
[`docs/perf/dispatch.md`](../perf/dispatch.md).

## The three axes

Three questions were once answered by one enum (`Provider`, now deleted). They are independent and
are now spelled independently.

| axis | type | question it answers | where |
|---|---|---|---|
| device family | `Backend` | which SYCL runtime family is this call compiled for | `include/batchlas/blas/enums.hh` |
| library present | `BATCHLAS_HAS_<LIB>` | which third-party math library exists in this build | `cmake/BatchLASOptions.cmake:188-199` → `batchlas/backend_config.h` |
| route | `Route{Origin, Algorithm}` | whose code runs for this call, and which strategy | `include/batchlas/blas/dispatch/route.hh:43-105` |

`Origin ∈ {Auto, Native, Vendor}` answers *whose code* (`route.hh:16-58`); `Algorithm` answers *which
strategy* (`route.hh:22-87`). "NVIDIA GPU with no cuBLAS" is therefore `Backend::CUDA` +
`BATCHLAS_HAS_CUBLAS == 0`, not a new device family — the SYCL runtime is still targeting CUDA, and a
build with no vendor library must not change the answer to "what am I running on". There is
deliberately no `Origin::SYCL`: every route in the tree is SYCL, so the value would name nothing and
would collide with the device-family axis (`route.hh:6-30`).

The MathDx device libraries (cuBLASDx, cuSolverDx) are `Origin::Vendor` even though their kernels
compile into our `.so`: the source is NVIDIA's and ships only for NVIDIA, so vendor independence has
to be measurable without them (`route.hh:20-57`, `cmake/BatchLASOptions.cmake:184-191`).

Two predicates sit on `Origin`, and confusing them is a shipped-and-fixed defect. `is_vendor(r)` is
the gate question (`route.hh:56-111`). `is_plain_vendor(r)` — `Vendor` **and** `Algorithm::Auto` — is
"the ordinary library call" (`route.hh:61-124`); the level-3 dispatchers' `request == Vendor` tests
meant `cublasSsyrk` specifically, and rendering them as `is_vendor()` makes a forced cuBLASDx request
answer yes to "did the caller ask for the vendor?".

**Known wrong, deliberately left.** `Route::library` and `Route::library_valid` are declared as
resolver outputs and excluded from `operator==` (`route.hh:48-103`), but nothing in the tree ever
writes them: no resolver, no table, no facade. Every resolved `Route` still carries the default
`BackendLibrary::CBLAS` with `library_valid == false`, so a consumer that believes the header comment
and reads the field gets a wrong answer silently. The library name a coverage `miss` row carries comes
from `throw_no_vendor_route`'s own `library` argument (`no_route.hh:137-147`), not from this field.

## The three routing predicates

This is the part to get right. Each `RouteTable<Op, T>` supplies two required static predicates and
one optional third, and they answer genuinely different questions.

| predicate | question | `false` means | consulted |
|---|---|---|---|
| `supports(r, s)` | can `r` produce the **correct answer** for shape `s` | the kernel would compute a **wrong answer** or index out of bounds | always, including for a forced route |
| `preferred(r, s)` | is `r` the best route available, **vendor included** | merely **slower** | on the automatic walk, in every build |
| `native_tier_preferred(r, s)` | among the **native** routes that serve `s`, is `r` the better one | another native tier is better here | only on the vendor-free walk |

Three rules follow, and each has cost this codebase something:

* **Never put a speed threshold in `supports()`.** A forced route bypasses `preferred()` — that is
  what forcing is for — but never `supports()` (`route_resolve.hh:76`). A speed cutoff there makes a
  pinned route fall through to `automatic()`, so the test that pinned it silently measures something
  else. `route_potrf.hh` and `route_geqrf.hh` both warn against this at their own tables. Conversely,
  moving a measured window into `supports()` leaves a working shape with **no supported route at all**
  the moment the vendor goes away.
* **Never fix a vendor-free tier choice in `preferred()`.** `preferred()` is consulted by the loop
  above the vendor-free walk, which runs regardless of `vendor_available`. A window written to pick
  the right native tier therefore also moves vendor-present traffic onto that tier — including at
  shapes where the vendor beats both natives. That is what `native_tier_preferred` exists for
  (`route_resolve.hh:18-83`).
* **Tables are pure.** Everything a table reads comes from its arguments: no `getenv`, no SYCL query,
  no dereference of operand data. That is what makes an op and its `*_buffer_size` query reach the
  same route *by construction* (`route_resolve.hh:5-21`) — the `ormqr` defect where the sizing query
  returned 2560 bytes and the call then demanded 276480 is structurally unreachable now.

`native_tier_preferred` is optional and defaults to `true` (`native_tier_preferred_or_default`,
`route_resolve.hh:18-83`), which makes the two vendor-free passes identical for a table that does not
declare it — and it defaults to `true` rather than `false` because a table that has not thought about
the question must keep its old answer. Three tables declare it today: `geqrf` (`route_geqrf.hh:79`),
`getrf` (`route_getrf.hh:78`) and `getrs` (`route_getrs.hh:102`). The comment at
`route_resolve.hh:40-118` still says "every op but geqrf" — stale.

## The resolver

`dispatch::resolve_route<Op, T>` (`route_resolve.hh:85-217`) is the instrumented entry point and the
only one ops call; it wraps a pure `resolve_route_uninstrumented` (`:89-176`). As implemented:

1. `forced.origin == Auto` → `automatic()` (`:132-134`).
2. `automatic()` walks `Table::order_begin()..order_end()` and takes the first route that is both
   `supports` and `preferred` (`:110-112`).
3. Only if `vendor_available == false` does it then accept a merely **supported** native route — in
   two passes, the first honouring `native_tier_preferred`, the second the raw order (`:113-128`).
   Taking "first merely supported" unconditionally inverts the default for small shapes, because the
   orders list natives first.
4. Falling all the way through returns `{Vendor, Auto}` (`:129`). That is the honest "this needs a
   vendor and there isn't one" signal; the caller turns it into a diagnostic, not a wrong answer.
5. A forced **vendor** that does not exist falls back to `automatic()` rather than being honoured
   (`:142-144`).
6. A forced **bare origin** (`native`, no algorithm) walks the order restricted to that origin,
   preference first then mere support (`:153-163`). Returning `{Native, Auto}` verbatim would hand the
   caller a route no dispatch tail can map to a kernel.
7. A forced route that `supports()` the shape is returned (`:165`); one that does not falls back to
   the **ordinary automatic choice**, not to the vendor (`:175`).

**The silent trap in rule 7, which this campaign has paid for repeatedly.** Pinning a route the shape
cannot take does not fail and does not warn — it resolves to `automatic()`, which in a vendor-present
build *is* the vendor. `BATCHLAS_SPMM_ROUTE=cta` resolves to `{Native, CTA}`, `supports()` rejects it
because no CTA body exists, and the run silently measures cuSPARSE. A **misspelled** value is worse:
`parse_route_value` fails, the resulting `ParsedRouteEnv::unparsed` flag is discarded at every
adapter's `parsed.found ? parsed.route : legacy_unset_default(...)` (`gemv_route.hh:151`,
`trsm_route.hh:75`, `spmm_route.hh:102`, and eight more identically), and every decision goes to the
vendor with no message. Confirm a pin with the resolved-route column, never with the exit status.

## Route tables and shape structs

Thirteen ops have a `RouteTable<Op, T>` specialisation: `gemm`, `gemv`, `trsm`, `potrf`, `getrf`,
`getrs`, `getri`, `geqrf`, `orgqr`, `ormqr`, `gesvd` and `spmm` get one header each under
`include/batchlas/blas/dispatch/`; `syev`'s lives with the op, in
`include/batchlas/blas/functions/syev.hh`.

Each table is paired with a **shape builder** — `src/backends/<op>_route.hh`, or the op header for
`gemm`/`gesvd`/`syev`/`ormqr` — which is where everything impure happens: the `getenv`, the SYCL
device query, the operand-agreement checks, and the calls into kernel TUs that ask what this build
actually contains. A builder returns `std::optional<Shape>`, and `nullopt` means "these views do not
describe one call of this op", which resolves to the vendor (`spmm_route.hh:24-53`). These headers
must include only public headers plus at most one private kernel header — no `src/queue.hh`, no
`<sycl/sycl.hpp>`. That constraint is what lets the vendor-free facade include them at all.

An op whose routing reads something `OpShape` (`route.hh:148-262`) has no field for **extends** it
rather than growing `OpShape` into a union of every op's arguments: `TrsmShape`, `GesvdShape`,
`SpmmShape`, `GeqrfShape`, `GetrfShape`, `GetrsShape`. `resolve_route` deduces `Shape` as a third
function template parameter (`route_resolve.hh:28-91`) and slices it back to `OpShape` on the way into
coverage (`:212`), so **never shadow an `OpShape` field in a derived shape** — the builder writes the
shadow, the slice copies the base, and every coverage row reports the default.

The convention for build capabilities is uniform and load-bearing: the builder asks the kernel TU what
exists and stores the answer in a shape field, and the *absent* value makes the native route
**unsupported** rather than selectable-but-unimplemented. `TrsmShape::cta_max_n == 0` means the CTA
kernel is not in this build (`route_trsm.hh:18-97`); `SpmmShape`'s two flags do the same per
transposition half. A literal constant in the table header instead would launder a hypothesis into a
compile-time fact, and would give the header a link dependency on a TU that may not exist yet
(`route_trsm.hh:13-82`).

**The four level-3 tile ops have no `RouteTable` and never call `resolve_route`.** `symm`, `syrk`,
`syr2k` and `trmm` keep hand-rolled `if`-chains, expressed as neither `supports()` nor `preferred()`,
with their gates in the facade (`src/dispatch/entry_points/level3.cc:189`, `:380`, `:418`, `:457`) and
their terminals instrumented directly (`src/backends/level3_coverage.hh:18-37`). Wiring them to the
resolver is a real change, not a transcription: the live thresholds are gate-only, so transcribing
them into `preferred()` rejects the tile route for shapes it serves today. See
[`docs/perf/dispatch.md`](../perf/dispatch.md) for the measurement that established this and for the
windows themselves.

`Op::iluk` exists in the enum (`route.hh:69-139`) and is referenced by nothing but `op_name`: ILU(k)
is a BatchLAS algorithm with no vendor alternative and dispatches through `BATCHLAS_DISPATCH_ON_QUEUE`
(`functions/iluk.hh:176-179`). `extensions.hh`'s entry points are absent from `Op` for the same
reason (`route.hh:69-133`).

## The vendor gate

`include/batchlas/blas/dispatch/vendor_available.hh` asks per **library**, not per device family,
because the map is not uniform: on NVIDIA `geqrf`/`getrf`/`ormqr` come from cuBLAS while
`potrf`/`syev` come from cuSOLVER; on AMD all of them come from rocSOLVER.

| predicate | ops | CUDA | ROCM | NETLIB |
|---|---|---|---|---|
| `level3_vendor_available<B>` (`:34-38`) | `gemm` `gemv` `trsm` `trmm` `symm` `hemm` `syrk` `herk` `syr2k` `her2k` | `BATCHLAS_HAS_CUBLAS` | `ROCBLAS` | `kHasNetlib` |
| `factorization_vendor_available<B>` (`:42-45`) | `geqrf` `orgqr` `getrf` `getrs` `getri` `ormqr` | `CUBLAS` | `ROCSOLVER` | `kHasNetlib` |
| `solver_vendor_available<B>` (`:49-52`) | `potrf` `syev` | `CUSOLVER` | `ROCSOLVER` | `kHasNetlib` |
| `sparse_vendor_available<B>` (`:56-59`) | `spmm` | `CUSPARSE` | `ROCSPARSE` | `kHasNetlib` |

`kHasNetlib` is `BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS` (`:31`), tested together because
`netlib_lapack.cc` calls both and is compiled only when both were found. Parallel `k<Group>Library<B>`
constants (`:62-73`) give the library name a diagnostic should quote.

The gate is an `if constexpr` in the facade, so **the vendor call is not compiled at all** when the
library is absent and there is no symbol to satisfy. The alternative design — a stub TU per absent
library defining a throwing `backend::<op>_vendor` — was declined because it restates all 26 vendor
signatures a second time, and signature divergence between restated copies is a defect class this tree
has already shipped (`vendor_available.hh:15-21`). Where a *header* needs the same gate,
`*_vendor_or_throw` shims do it inline (`functions/gesvd.hh:206-226`).

When nothing serves a call, `throw_no_vendor_route<T>` (`no_route.hh:137-147`) records a coverage miss
and throws `NoRouteError`, whose message names the op, the scalar type and the build switch that would
restore it (`:111-128`) — and deliberately not the backend, which `NoRouteError` carries but discards
when formatting (`:126`).

A fifth, separate question is "is the native kernel **linked**": `level3_tile_route_available<B, T>`
(`route_compiled.hh:211-213`), which is `B == Backend::CUDA && (std::is_same_v<T, float> ||
bool(BATCHLAS_HAS_CUBLAS))`. Four sites in `src/extensions/` and one in `coverage.cc` used to spell
this `B == Backend::CUDA`, which is wrong in the vendor-free build — the backend is still
`Backend::CUDA` and the tile TUs are not compiled. It takes a **scalar** parameter because the answer
varies per `(backend, scalar)`: the float tile routes are reachable everywhere, while `syrk`'s
non-float gram branch and `trmm`'s non-float tile branch live in `cublas.cc`, and `syr2k` has no
non-float tile route at all.

## The entry-point facade

The original obstacle to vendor independence was not routing at all — it was **definition ownership**.
`gemm<Backend::CUDA, float>` was defined *inside* `cublas.cc` and instantiated there, so "build without
cuBLAS" did not mean "lose the cuBLAS gemm path", it meant "lose `batchlas::gemm` entirely". The same
held in `rocblas.cc` and `netlib_lapack.cc`. No amount of enum, CMake or predicate work addresses
that.

`src/dispatch/entry_points/` now owns the public definitions, compiled unconditionally and gated on no
vendor library (`src/dispatch/CMakeLists.txt`). The split is:

| translation unit | defines | count |
|---|---|---|
| `entry_points/level3.cc` | `gemm` `gemv` `trsm` `symm` `hemm` `herk` `her2k` `syrk` `syr2k` `trmm` | 10 |
| `entry_points/factorization.cc` | `geqrf` `orgqr` `getrf` `getrs` `getri` `potrf`, each with its `*_buffer_size` | 6 + 6 |
| `entry_points/sparse.cc` | `spmm`, `spmm_buffer_size` | 1 + 1 |
| `entry_points/eigen.cc` | nothing — it relocates the **instantiations** of `syev` and `ormqr` | 2 |

and the shape of each op is:

```
vendor TU (cublas.cc, …)   defines and instantiates  backend::<op>_vendor<B, T>
entry_points/*.cc          defines and instantiates  <op><B, T>, which routes and may call it
```

Five properties of this layer are load-bearing:

* **Instantiation is keyed on the device family, not the library** (`level3.cc:394-521`). The bodies
  compile to a throw when the library is absent, so the public symbol exists in every build that has
  the device — which is exactly what stopped being true when the definitions lived in the vendor TUs.
* **An instantiation binds as hard as a definition.** `syev` and `ormqr` were already *defined* in
  headers, but their explicit instantiations lived in `cusolver.cc`/`cublas.cc`, which is enough to
  make them vanish from a build without those libraries; moving the instantiation is the whole change
  for those two (`eigen.cc:1-15`). `gesvd` needs no facade TU at all: its public template is `inline`
  in `functions/gesvd.hh:420-467` and forwards to `gesvd_dispatch`.
* **The route gate runs before the vendor-available test.** Anything below `if constexpr
  (!<group>_vendor_available<Back>)` is unreachable in the vendor-free build, which is the build the
  campaign exists for. Every op in `level3.cc` resolves first and throws second.
* **An op moves together with its `*_buffer_size` query.** Splitting them lets the two resolve
  differently, which is exactly the `ormqr` 108x sizing defect (`factorization.cc:8-10`).
* **Backend asymmetries are preserved, not normalised.** rocBLAS has no `hemm`/`herk`/`her2k`/`symm`
  wrapper, so the ROCm backend instantiates only the ops it implements (`level3.cc:398-536`).

The facade is also the **injection point** for native drivers that need a routed sub-operation. A
native driver is instantiated per scalar type with no `Backend` parameter, so it cannot name
`gemm<B, T>` itself; the facade passes a lambda. `trsm`'s blocked driver takes its trailing GEMM this
way (`level3.cc:155-265`), `potrf`/`getrf`/`getrs`/`getri` take routed `gemm`/`trsm`, and `orgqr`'s
native arm takes a routed `ormqr` (`factorization.cc:17-26`, `:60-65`). The alternative — the driver
calling `sycl_gemm::gemm_custom` directly — bypasses `RouteTable<Op::gemm>` and pins the native GEMM
even on shapes it is measured to lose; see [`docs/perf/trsm.md`](../perf/trsm.md) and
[`docs/perf/gemm.md`](../perf/gemm.md).

`scripts/facade_symbol_check.sh` verifies the move by symbol rather than by diff, because a forwarder
left behind, or an instantiation pointing at the wrong template, still compiles and links.

## Environment overrides

Canonical spelling is `BATCHLAS_<OP>_ROUTE`, synthesised from `op_env_stem` — **no `route_env.hh` edit
creates one** (`route_env.hh:135-217`). A value is an origin (`vendor`, `native`), an algorithm (`cta`,
`expand_gemm`, …), or both joined by a colon (`native:register_tiled`); parser at `route_env.hh:50-99`.
A bare algorithm implies `Native`, **except** `FusedDevice`, which is vendor code by definition
(`:92-97`). `netlib` maps to `Vendor`, not to an algorithm, because netlib LAPACK is somebody else's
code (`:47-53`).

Unset means `{Auto, Auto}` for **every** op (`legacy_unset_default`, `:145-148`). GEMM used to be the
odd one out at `{Vendor, Auto}`; that asymmetry is gone, and the reasoning — including what the flip
does and does not turn on — is at `:123-144` and in [`docs/perf/gemm.md`](../perf/gemm.md).

Eight ops have a legacy variable, read only when the canonical one is unset (`:109-121`, `:214-245`).
They keep working because they appear in committed benchmark scripts and in the provenance of recorded
results; silently changing what they mean would invalidate measurements still being compared against.
Three collisions between the two vocabularies are deliberate and must not be "simplified" away
(`:150-203`):

| spelling | canonical meaning | legacy meaning | legacy maps to |
|---|---|---|---|
| `BATCHLAS_GEMM_VARIANT=native` | BatchLAS's own kernel | the **raw CUDA vendor path**, consumed purely as an exclusion | `{Vendor, Direct}` (`:178-182`) |
| `custom` in `symm`/`syrk`/`syr2k`/`trmm` | the register-tiled GEMM family (`:63`) | the fused cuBLASDx kernel | `{Vendor, FusedDevice}` (`:185`) |
| `gemm` in `syrk`/`syr2k` | the `gemm` op | the deliberately wrong `DiagFullGemm` measurement route | `{Vendor, DiagFullGemm}` (`:190-198`) |

Because the two parsers differ, the two spellings do not agree even for the same op:
`BATCHLAS_SYMM_VARIANT=custom` reaches the fused arm and `BATCHLAS_SYMM_ROUTE=custom` does not.
`tiles` and `narrow` exist only in the level-3 legacy parser (`:186-189`).
`Algorithm::DiagFullGemm` is retained on purpose: it computes and stores **both** triangles, which is
not what `syrk` or `syr2k` mean, and it exists only so the arithmetic the triangular kernels save can
be measured against it (`route.hh:38-86`). `Auto` cannot reach it; naming it can.

Two routing variables are not op-keyed. `BATCHLAS_EXPAND_ROUTE=expand|loop` pins the
mirrored-expansion decision for `symm`/`hemm`/`herk`/`her2k`/`trmm` and is consulted **before** the
measured window (`src/backends/triangular_expand.hh:50-60`), so a pin overrides the measurement — a
call-site guard that replicates only half of such a predicate is a shipped-and-fixed defect.
`BATCHLAS_COVERAGE_OUT` turns the dynamic instrument on. Two per-op ad-hoc knobs
(`BATCHLAS_ORTHO_GRAM`, `BATCHLAS_ORMQR_IMPL`) have not been folded into this vocabulary
(`route_env.hh:7-15`).

The vocabulary is pinned by `tests/route_vocabulary_tests.cc`, including every legacy spelling and
every collision above; the GEMM transcription itself is pinned by
`tests/route_gemm_equivalence_tests.cc`.

## The coverage instrument

Two tables answer different questions, and reading either as the other is how a working vendor-free
`gemm` came to be claimed at a point when every such call threw (`coverage.hh:11-25`).

| row kind | question | how produced | cost |
|---|---|---|---|
| `linked` | is the kernel **in this build** — the planning question | iterates `(Op × Backend × ScalarKind)` over the route predicates, no kernel run | exact, instant, no GPU needed |
| `reached` | did a call **get there** — the burn-down question | one row per `(op, scalar, backend, shape_class, variant)`, recorded from `resolve_route` | one predicted branch per op invocation |
| `miss` | nothing served this call at all | recorded by `throw_no_vendor_route` | rare by construction |

**`linked` is not `reached`, and a symbol being present is never evidence it runs.** Several ops report
`native = 1` in the `linked` table while `preferred()` is all-false for them, so a vendor-present build
sends them nothing and only a vendor-free build reaches them, through the fallback at
`route_resolve.hh:38-128`. `coverage.cc:164-219` states this at the point where it is easiest to get
wrong. The `linked` column is reported **for `float`**, because the level-3 tile routes are float-only
outside a cuBLAS build and a type-blind column would restate exactly the overclaim it exists to
prevent (`coverage.cc:190-315`).

`native_route_supported` on a `reached` row is a **tri-state**: `1` yes, `0` no, `-1` the call site
could not tell (`coverage.hh:62-66`, `level3_coverage.hh:47-62`). The third value is load-bearing: a
declining gate never enters `*_cuda_custom`, so it cannot distinguish "nothing native serves this
shape" from "something does but the heuristic preferred the vendor", and recording either as definite
would be a claim the call site cannot support. The gate-declined half is recorded explicitly, beside
each `return` and never in place of one — a shape moving *off* a native kernel is otherwise invisible.

Four properties of the row key:

* `uplo`/`side`/`diag`/`transA`/`transB` are part of the **key**, not decoration (`coverage.cc:35-58`).
  They select which triangle or which operand an op touches, so two calls differing only in `uplo` must
  not collapse into one first-writer-wins row.
* `shape_class` buckets `max(m,n,k)` and `batch` by power of two (`route.hh:179-261`), so a
  10,000-iteration test collapses to a handful of rows rather than 10,000.
* Rows are **first-writer-wins**, so the `m`/`n`/`k`/`batch` columns can report a *different* call's
  exact shape. A coverage row cannot confirm that a particular shape ran; prove that with a deliberate
  break that is red only for it.
* `reached` rows read `backend = AUTO`, expected and not a defect: the backend is a template parameter
  at the call site, so the shape builder never learns it (`coverage.cc:90-118`).

The instrument is gated at **runtime** on `$BATCHLAS_COVERAGE_OUT` — the same variable `emit()` reads,
so recording and emission cannot disagree about whether coverage is on. It was a compile-time macro
first, and that could not work: `resolve_route` is an inline function template, every TU instantiates
its own weak copy, and ELF resolves the executable's weak symbols ahead of a shared library's — so a
test compiled without the macro interposed its uninstrumented copy and the library recorded nothing,
producing a file with a correct header and **zero `reached` rows** (`coverage.hh:27-49`).
`cmake/BatchLASOptions.cmake:109-117` records that the option was deliberately never added. Two further
failure modes are fixed in place and worth not reintroducing: the tables are deliberately **leaked** so
an `atexit` handler cannot walk a destroyed container (`coverage.cc:64-88`), and `emit()` writes **one
file per pid** because a `ctest` run is dozens of binaries and a shared `"w"` handle meant each
truncated the last (`:129-146`).

`route_resolve.hh:85-184` still claims a `-DBATCHLAS_ENABLE_COVERAGE=ON` build option — stale, as is
the same claim in `tests/route_vocabulary_tests.cc:742`. The static table's `trsm` row is hardcoded
`false` (`coverage.cc:168`) although a native `trsm` ships; the `linked` half answers "does this build
have a native route *registered*", and it is stale in both directions. Read the `reached` rows and the
resolved route.

## Verification tooling

| script | what it is for | the property that makes it trustworthy |
|---|---|---|
| `scripts/route_diff.sh capture\|compare` | prove a change moved **no dispatch decision** | treats a capture with **zero `reached` rows as a hard error**, not as "nothing changed" — the instrument has produced a correct header with no rows twice, for unrelated reasons, and both times it looked clean |
| `scripts/coverage_merge.sh` | collapse the per-pid shards a `ctest` run produces | sums `calls`, de-duplicates the identical `linked` block every process emits |
| `scripts/facade_symbol_check.sh` | prove the public entry points actually **left** the vendor component | matches **Itanium mangling directly**: `nm -C` silently fails to demangle concept-constrained templates and would report `symm`/`herk` as missing when present |
| `scripts/rocm_syntax_check.sh` | `-fsyntax-only` the three ROCm vendor TUs this machine never compiles | the gate is "**exactly one** expected error" (a `get_native<ext_oneapi_hip>` overload this CUDA-only DPC++ lacks); anything else is a real defect. It forces the CUDA macros off, exercising the per-library `#if` structure nothing else here builds. The ROCm headers live under `/opt/rocm/include/roc*/`, a subdirectory, which is why a naive probe reads them as absent |
| `scripts/register_probe.sh` | register/spill residency of the SYCL device link | replays a target's `link.txt` verbatim, so the flags stay exactly the real build's; **fails loudly** when the named target has no `link.txt` rather than silently probing the default library |

`route_diff.sh` is the only tool that can see a **vendor-to-vendor** route change: the kernel trace
cannot (its `Record` holds a `sycl::event`) and timing cannot (an unsaturated ratio is overhead, and
routing a shape to the vendor may well be faster, so a perf gate cannot flag a wrong route). Its
comparison key is `(kind, op, scalar, backend, shape_class, origin, algo, native flags, uplo, side,
diag, transA, transB)` — `m`/`n`/`k`/`batch` and the call count are dropped on purpose, since counts
vary with test scheduling and are not part of the decision. It applies **no `backend != AUTO` filter**,
so pure-layer test shapes recorded with `backend = AUTO` can make a small real move look like hundreds
of lines of churn.

`register_probe.sh`'s gate is **not** "stack frame == 0": on this tree 220 of 376 entry functions carry
a non-zero stack frame with zero spills, so gating on it rejects healthy kernels. Use `0 bytes spill
stores, 0 bytes spill loads` on the kernel's lines, and `Used N registers × work-group size <= 65536`,
the per-block limit whose failure mode is a launch abort rather than a slowdown. Each kernel appears
twice, as `<name>` and `<name>_with_offset`; take the max.

## Adding an op, or moving a route

1. Add the `Op` enumerator and its `op_name` case (`route.hh:69-139`, `:188-203`). The environment
   variable name is synthesised from it; do **not** add a `legacy_variable_for` case for an op that
   never had a legacy spelling — that invents a legacy variable that never shipped.
2. Write `RouteTable<Op::x, T>` under `include/batchlas/blas/dispatch/`: an order array, `supports()`
   as correctness only, `preferred()` as a measured window, `native_tier_preferred()` only if there is
   more than one native tier. Keep it pure. If it needs a field `OpShape` lacks, derive an `XShape` —
   and shadow nothing.
3. Write the shape builder in `src/backends/<op>_route.hh`: public headers plus at most one private
   kernel header, `std::optional<XShape>` return, capability flags read from the kernel TU.
4. Define the public entry point in `src/dispatch/entry_points/`, with its `*_buffer_size` query in the
   same TU, resolving **before** the `if constexpr (!<group>_vendor_available<Back>)` test and calling
   `throw_no_vendor_route<T>` in the vendor-free arm. Instantiate on the device family.
5. Add the op's row to `coverage.cc`'s `entries[]`, and flip its `native` column **in the same step as
   the entry-point wiring** — never before, or the table claims traffic that cannot reach the kernel.
6. Capture a `route_diff.sh` baseline before the change and compare after. On a vendor-present box a
   pure relocation must be byte-identical; a real route change has to be argued for and measured, on
   the op's `docs/perf/` page, not discovered later.
7. Run `facade_symbol_check.sh` if you touched a definition or an instantiation, and
   `rocm_syntax_check.sh` if you touched a vendor TU's declarations.

## What is still open, architecturally

Per-op performance debts live on the `docs/perf/` pages. These belong to the dispatch layer itself:

1. **`Route::library` is never written** (`route.hh:48-103`). The header documents an output the
   resolver does not produce.
2. **The four level-3 tile ops have no `RouteTable`** and never call `resolve_route`. Adding the tables
   as unwired additions alongside an equivalence test is cheap; *wiring* them is the change that moved
   `n = 256` onto a route that writes both triangles.
3. **Two forced level-3 routes are known wrong and preserved deliberately.**
   `BATCHLAS_SYRK_ROUTE=native` falls through to a route that clobbers the triangle the caller did not
   name, and `BATCHLAS_SYR2K_ROUTE=native` throws a cuBLASDx message it did not ask for. No test in the
   tree sets either variable. Both are pre-existing rather than introduced; see
   [`docs/perf/dispatch.md`](../perf/dispatch.md).
4. **The static coverage table is stale in both directions** — `trsm` hardcoded `false`
   (`coverage.cc:168`) — and two comments claim a `BATCHLAS_ENABLE_COVERAGE` option that does not
   exist (`route_resolve.hh:85-184`, `tests/route_vocabulary_tests.cc:742`).
5. **`route_diff.sh compare` applies no backend filter**, so `AUTO` rows from pure-layer tests inflate
   every diff.
6. **`Backend::INTEL` is hard-wired false and oneMKL cannot be tested here**; only the dead branch that
   produced undefined references was removed. ROCm is reachable only through `rocm_syntax_check.sh`,
   and MathDx-present boxes (`BATCHLAS_HAS_CUBLASDX 0` here) are untestable in this tree — statements
   about them are stated, not verified.
7. **The runtime `BATCHLAS_NO_VENDOR=1` enforcement knob was never built.** `op_external`
   (`include/batchlas/blas/dispatch/op.hh:10-13`) is still the no-op tag the plan proposed
   instrumenting, so five `src/extensions/` sites that call `backend::*_vendor` directly — bypassing
   the public entry point and therefore the resolver — remain invisible to the instrument. The
   build-time switch `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` is what enforces independence today.
8. **The vendor-free suite is not green, and no dispatch mechanism can make it so** — the remaining gap
   is missing kernels, not routing. The failing **set** is the reviewable artefact, not its count; it
   is tracked per package under [`docs/perf/`](../perf/README.md).

## Where the rest of it is

Per-op measured windows, the grids that justify each boundary, the built-and-rejected designs and the
correctness findings live under [`docs/perf/`](../perf/README.md):
[`dispatch`](../perf/dispatch.md), [`gemm`](../perf/gemm.md), [`level3`](../perf/level3.md),
[`trsm`](../perf/trsm.md), [`potrf`](../perf/potrf.md), [`qr`](../perf/qr.md), [`lu`](../perf/lu.md),
[`gemv`](../perf/gemv.md), [`spmm`](../perf/spmm.md). The superseded root design documents
(`WP0_DISPATCH_SPEC.md`, `WP1_LEVEL3_SPEC.md`, `VENDOR_INDEPENDENCE_PLAN.md`,
`VENDOR_FREE_BASELINE.md` and the per-work-package specs) are retained at the git tag
`perf-evidence/vendor-independence` and retrievable with
`git show perf-evidence/vendor-independence:<path>`.

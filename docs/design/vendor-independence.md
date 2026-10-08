# Vendor independence

> **Status:** current. The [History](#history-the-routetable-layer) section describes deleted code.

BatchLAS configures, builds, loads and runs with no vendor math library. This page describes the
mechanisms that keep that property: the two axes that decide vendor use, the per-library gate, the
contracts every public entry point keeps, the coverage instrument and the verification scripts. It
has no performance numbers. Measured windows and per-op debts are in [`docs/perf/`](../perf/README.md),
starting at [`docs/perf/dispatch.md`](../perf/dispatch.md). Current vendor-free status is in
[`vendor-free-status.md`](vendor-free-status.md).

A vendor-free CUDA build is the ordinary build with one library axis off:
`cmake -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` yields
`BATCHLAS_HAS_CUDA_BACKEND 1` with every CUDA math library at `0`.

## Vendor independence today

Two independent questions decide whether a call can reach a vendor library:

| axis | type | question it answers | where |
|---|---|---|---|
| device family | `Backend` | which SYCL runtime family is this call compiled for | `include/batchlas/blas/enums.hh` |
| library present | `BATCHLAS_HAS_<LIB>` | which third-party math library exists in this build | `cmake/BatchLASOptions.cmake` → `batchlas/backend_config.h` |

"NVIDIA GPU with no cuBLAS" is `Backend::CUDA` with `BATCHLAS_HAS_CUBLAS == 0`, not a new device
family. The third question, whose code runs for a call, is the op's **choice**: a value of the op's
family variant. The vendor library is the family `vendor`.

* **Public definitions** live in `src/ops/<op>/<op>.cc`, compiled into `batchlas_dispatch_obj` and
  instantiated on the device family. The vendor TUs (`cublas.cc`, `cusolver.cc`, `rocblas.cc`,
  `netlib_lapack.cc`, ...) define and instantiate only `backend::<op>_vendor<B, T>`. The public symbol
  therefore exists in every build that has the device.
* **The `vendor` family's `can_run`** is `d.has_vendor`, filled by `select::device_of` from
  `select::has_library<B>(spec.vendor)`. Without the library, every vendor entry is unrunnable and the
  first runnable native entry of the row runs. `can_run` is correctness only; table order is speed.
* **Nothing runnable** is reported by `select::pick` as `NoRouteError`.
* **Ops without `choice.hh`** select by hand: `hemm`, `herk` and `her2k` (entry points in
  `src/ops/level3/level3.cc`; the expand-or-vendor choice is in `src/backends/cublas.cc`, see
  [`docs/perf/level3.md`](../perf/level3.md)). `BATCHLAS_EXPAND_ROUTE=expand|loop` pins that decision. @ref selection_tables lists the ops that select from
  tables.

The MathDx device libraries (cuBLASDx, cuSolverDx) counted as vendor, because their source is NVIDIA's
and ships only for NVIDIA. Flat selection (#147) deleted every cuBLASDx path. The rule still applies to
any header-only vendor library added later.

## The vendor gate

`src/select/vendor.hh` asks per **library**, not per device family, because the map is not uniform.
On NVIDIA, `getrf` and `getri` come from cuBLAS, while `geqrf`, `orgqr`, `ormqr`, `getrs`, `potrf`
and `syev` come from cuSOLVER. An op names its group once in `OpSpec::vendor` (`select::Lib`).

| predicate | `select::Lib` | ops | CUDA | ROCM | NETLIB |
|---|---|---|---|---|---|
| `level3_vendor_available<B>` | `level3` | `gemm` `gemv` `trsm` `trmm` `symm` `hemm` `syrk` `herk` `syr2k` `her2k` | `BATCHLAS_HAS_CUBLAS` | `ROCBLAS` | `kHasNetlib` |
| `factorization_vendor_available<B>` | `factorization` | `geqrf` `orgqr` `getrf` `getrs` `getri` `ormqr` | `CUBLAS` **and** `CUSOLVER` | `ROCSOLVER` | `kHasNetlib` |
| `solver_vendor_available<B>` | `solver` | `potrf` `syev` `gesvd` | `CUSOLVER` | `ROCSOLVER` | `kHasNetlib` |
| `sparse_vendor_available<B>` | `sparse` | `spmm` | `CUSPARSE` | `ROCSPARSE` | `kHasNetlib` |
| — | `none` | `gesv`, `posv` (no vendor family) | — | — | — |

`kHasNetlib` is `BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS`, because `netlib_lapack.cc` needs both.
`select::library_name<B>(lib)` gives the name a diagnostic should quote.

The vendor arm is an `if constexpr` on the same predicate. When the library is absent the vendor call is
not compiled, so there is no symbol to satisfy. Its `else` is `select::no_vendor<B, T>(spec)`, which
records a coverage `miss` and throws `NoRouteError`. The message names the op, the scalar type and the
switch that restores it.

Rejected: a stub TU per absent library that defines a throwing `backend::<op>_vendor`. It restates all
26 vendor signatures, and signature divergence between restated copies is a defect class this tree has
already shipped. Header and non-op callers use `*_vendor_or_throw` shims instead, for example
`syev_vendor_or_throw` in `src/ops/syev/vendor.hh`.

Whether the native kernel is **linked** is a separate question, answered by
`select::level3_tile_route_available<B, T>`: `B == Backend::CUDA && (T is float || BATCHLAS_HAS_CUBLAS)`.
Do not test `B == Backend::CUDA` alone. The backend is still `CUDA` in a vendor-free build, and the
tile TUs are not compiled there.

### The vendor gate: why the tile route predicate is per backend and scalar

The predicate takes the scalar because a bare `true` is wrong in two directions:

* **Too wide in type.** Double and complex tile routes are not linked everywhere: `syr2k` has no
  non-float tile route, and `ortho.cc`'s `gram_via_syrk` admits double. A vendor-free call would throw.
* **Too wide in backend.** The entry-point gate is guarded on `Backend::CUDA`, so claiming ROCM or
  NETLIB would repeat the defect of using the `Backend` enum to answer a library question.

With `BATCHLAS_HAS_CUBLAS` on, the answer is `true` for every type on CUDA, so vendor-present behaviour
is unchanged. The internal callers that check before calling a tile kernel are `ortho.cc`,
`ormqr_blocked.cc`, `sytrd_blocked.cc` and `src/select/coverage.cc`. The level-3 four launch from
`src/ops/{symm,syrk,syr2k,trmm}/<op>.cc`; each native `can_run` has a `kWired = B == Backend::CUDA`
term (for example `src/ops/trmm/trmm.cc:54`). Code site: `src/select/vendor.hh`.

### The vendor gate: history of the per-library predicates

Until WP0 the public entry points were defined inside the vendor TUs, so "no vendor" and "no entry point"
were the same condition, and the answer was always yes.

`factorization_vendor_available` requires both cuBLAS and cuSOLVER. On NVIDIA the group spans both, so
keying it on cuBLAS alone claimed a `geqrf` vendor route whenever cuBLAS was on. A vendor-free user
could not recover it by re-enabling cuBLAS. Before flat-selection phase 5 the predicates lived in
`include/batchlas/blas/dispatch/vendor_available.hh`, behind two `select::Device` flags. `OpSpec::vendor`
and the single `Device::has_vendor` replaced them. A per-op split of the six-op factorization group is
open debt (see [What is still open](#what-is-still-open-architecturally)).

## Vendor independence: headers keyed on the library axis

The vendor includes in the private `src/linalg-impl.hh` (`<cublas_v2.h>`, `<cusolverDn.h>`,
`<cusparse.h>` and the ROCm equivalents) are guarded by `BATCHLAS_HAS_<LIB>`, not by the family flag.
A CUDA device with no math libraries is a valid configuration, and the family guard still tried to
include `<cublas_v2.h>` in it.

* `cuda_runtime.h` stays on the family flag. It is the CUDA runtime, needed for streams and device queries.
* `<cuComplex.h>` is included explicitly on the family flag. The pointer-cast helpers need it, and it
  used to arrive only through `cublas_v2.h`.
* The CUDA handle types (`cublasComputeType_t`, the cuBLAS, cuSPARSE and cuSOLVER handle triple) sit under
  the library flags. Naming them under `BATCHLAS_HAS_CUDA_BACKEND` alone does not compile when a library is absent.
* A CUDA device with some libraries absent still gets a `LinalgHandle<Backend::CUDA>` specialisation with
  no vendor handles. The type must be complete, because native TUs declare one (`src/extensions/ortho.cc`).

Symbols follow the same split. Public instantiations are keyed on the device family, and the vendor TUs
instantiate only their `backend::*_vendor` symbols. See
[the runtime-internals note](runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols).

## Positional validators: reject only what no route can serve

Each positional (workspace-taking) entry point of `geqrf`, `orgqr`, `getrf`, `getrs`, `getri` and `potrf`
has an `<op>_validate_params` in its public header. It runs before the table key is built, because the key
reads `A.rows()` and `A.cols()`. `trsm_validate_params` follows the same order.

**Rule: validate only what no kernel could serve.** A native `can_run` that rejects a shape steers the
choice to another family, usually `vendor`. It does not make the call invalid. A validator that threw on
such a shape would change behaviour for a working call, which belongs in its own commit with its own test.
The stricter checks (`require_square`, `require_span_at_least`, `require_info_span`) live in the option
and arena overloads in `include/batchlas/blas/options.hh`.

What the positional path deliberately does not check:

| op | checked | not checked, and why |
|---|---|---|
| `potrf` | negative extents, squareness, `uplo` | the length of a non-empty `info` span. A short span silently becomes pool scratch (`detail::info_target`); throwing would change behaviour. |
| `geqrf` | negative extents | squareness (rectangular A is the point, and callers pass tall panels); `m >= n` (native `can_run` refuses a wide view and the vendor serves it); `tau` length (arena spellings check it) |
| `orgqr` | negative extents | `n <= m` (every backend accepts such a view and hands it to a vendor); `tau` length |
| `getrf` | negative extents | squareness; pivot length (arena spellings check it) |
| `getrs` | negative extents | squareness of A, `A.rows() == B.rows()`, batch, pivot length (arena spellings check them; a non-conforming pair goes to the vendor through `can_run`) |
| `getri` | negative extents of A | squareness of A and C, order and batch agreement, pivot length (arena spellings check them; a non-square A goes to the vendor through `can_run`) |

`getri` has two arities, forced by the signatures. `getri_buffer_size` takes A alone, and
`ops::getri::key_of` is a function of A alone. The query must validate the same view its choice is
computed from. The second arity also checks C's extents.

`gesv` and `posv` reject more, because they have no vendor family (`Lib::none`). A bad pair would reach
"nothing runnable" and report the wrong cause. They check squareness, `B.rows() == A.rows()` and equal
batch, and `posv` also checks `uplo`.

## Per-item info spans for potrf, getrf and getri

`info` on `potrf`, `getrf` and `getri` (and through them `posv` and `gesv`) is the LAPACK per-item status:
one `int32` per batch item, 0 on success. A value above 0 names the leading minor that is not positive
definite (`potrf`), the column where U is exactly singular (`getrf`), or the zero diagonal of U that
leaves the item without an inverse (`getri`).

* An **empty** span means "not requested". The backend then uses its own scratch.
* The **workspace size is the same either way**, so `*_buffer_size` is correct whether or not the caller asks.
* A **short non-empty** span is treated as "not requested" on the positional path. The option overloads
  reject it with `require_info_span`. See the potrf row of the table above.
* `info` cannot be a defaulted trailing parameter. The `sig::` aliases used by `BATCHLAS_INSTANTIATE`
  (`src/util/template-instantiations.hh`) are function types, and they cannot carry defaults. A separate
  inline overload keeps the old arity compiling. `gesv` and `posv` do the same.

## Info spans on syev, gesvd and steqr: forwarder or default

Five routines (`syev`, `syevx`, `gesvd`, `steqr`, `stedc`) and every tier below them take a trailing
`Span<int32_t> info`. Two declaration shapes carry it. The user-facing contract is in
[the C++ API page](../cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc).

* **Public `syev` and `gesvd`**: an old-arity inline forwarder. The `sig::` alias used for explicit
  instantiation is a function type and cannot carry the default. The declaration and the alias keep the
  same parameters. For `gesvd`, arity plus the `Uplo`/`Span` type at parameter 8 keeps the four overloads
  unambiguous.
* **`backend::syev_vendor` and `backend::gesvd_vendor`**: a defaulted `info_out`. A default is a property
  of the declaration, so the `sig::` aliases and the vendor TUs still match. The default keeps the internal
  six-argument callers compiling (`src/extra/norm.cc`, `src/extra/cond.cc`, `src/extensions/syevx_lobpcg.cc`,
  through `syev_vendor_or_throw`).
* **`steqr`, `steqr_cta`, `stedc`, `syevx*`, the `syev_*` tiers and the SVD tiers** (`extensions.hh`): a
  defaulted trailing parameter. They are instantiated from macros, not `sig::` aliases, and they already
  default `jobz`, `params` and `eigvects`. A forwarder would be ambiguous with the primary.

`gesvd` details are in [the gesvd design page](gesvd.md#gesvd-design-vendor-binding-and-dispatch).

## Vendor independence: the coverage instrument

`src/select/coverage.hh` writes three kinds of row. Do not read one as another: a symbol that is present
is not a kernel that runs.

| row kind | question | how produced | cost |
|---|---|---|---|
| `linked` | is the kernel in this build (planning) | `coverage::static_table()`: per `(op, backend)`, the vendor gate and a native-linked flag | exact, no GPU needed |
| `reached` | did a call get there (burn-down) | one row per `(op, scalar, backend, shape_class, variant, choice)`, from `select::TraceScope` and the hand-selected ops | one predicted branch per op invocation |
| `miss` | nothing served the call | `throw_no_vendor_route` | rare by construction |

* The `linked` column is reported for `float`, because the level-3 tile routes are float-only outside a
  cuBLAS build.
* `native_route_supported` on a `reached` row is tri-state: `1` yes, `0` no, `-1` the call site cannot tell.
  The `-1` value matters for the hand-selected level-3 ops. A declining gate never reaches the tile
  dispatcher, so it cannot separate "nothing native serves this shape" from "the heuristic chose the vendor".
* `uplo`, `side`, `diag`, `transA` and `transB` are part of the row key (`variant_key` in
  `src/select/coverage.cc`). They change which triangle or operand is touched.
* `shape_class` buckets `max(m,n,k)` and `batch` by power of two.
* Rows are **first-writer-wins**. The `m`, `n`, `k` and `batch` columns can show another call's shape, so
  a row cannot prove a particular shape ran. Prove that with a deliberate break that fails only for it.
* `select::run` fills `shape.backend` from its template parameter.
* The instrument is gated at runtime on `BATCHLAS_COVERAGE_OUT`. The tables are deliberately leaked so
  an `atexit` handler does not walk destroyed containers, and `emit()` writes one file per pid.

A compile-time gate (`-DBATCHLAS_ENABLE_COVERAGE`) was rejected. An inline header function is sound only
if every TU in the process agrees, and a library cannot enforce that on its consumers. A test binary that
carried its own weak copy of the resolver produced a valid header and zero `reached` rows.
`cmake/BatchLASOptions.cmake` records that the option was never added.

## Verification tooling

| script | what it is for | why it can be trusted |
|---|---|---|
| `scripts/route_diff.sh capture\|compare` | prove a change moved no selection decision | a capture with zero `reached` rows is a hard error, not "no change". Its key is `(kind, op, scalar, backend, shape_class, origin, algo, native flags, uplo, side, diag, transA, transB)`. It drops `m`, `n`, `k`, `batch` and the call count, and applies no `backend != AUTO` filter. |
| `scripts/coverage_merge.sh` | merge the per-pid shards of a ctest run | sums `calls`, de-duplicates the identical `linked` block |
| `scripts/facade_symbol_check.sh` | prove public entry points are not defined in a vendor component | matches Itanium mangling directly; `nm -C` fails to demangle concept-constrained templates |
| `scripts/rocm_syntax_check.sh` | `-fsyntax-only` the three ROCm vendor TUs | exactly one expected error (a `get_native<ext_oneapi_hip>` overload this DPC++ lacks); anything else is real. The ROCm headers are under `/opt/rocm/include/roc*/`. |
| `scripts/register_probe.sh` | register and spill residency of the device link | replays the target's `link.txt`, and fails if there is none. Gate on `0 bytes spill stores/loads` and `Used N registers × work-group size <= 65536`, not on stack frame == 0 (220 of 376 healthy entry functions have a frame). Take the max over `<name>` and `<name>_with_offset`. |
| `BATCHLAS_SELECT_TRACE=1` | print each decision, its table and the runner-up | one line per call, indented under its parent op |

`route_diff.sh` is the only tool that sees a vendor-to-vendor change. The kernel trace cannot, and timing
cannot, because an unsaturated ratio measures overhead and a perf gate cannot flag a wrong choice.

Adding an op is described in [Adding entry points](../extending.md) and the @ref selection group. The
vendor-specific part is one `OpSpec::vendor` value and the `if constexpr` arm above.

## What is still open, architecturally

Per-op performance debts are on the `docs/perf/` pages. These belong to the vendor seam:

1. `route_diff.sh compare` applies no backend filter, so `AUTO` rows from pure-layer tests inflate diffs.
2. `Backend::INTEL` is hard-wired false, and oneMKL cannot be tested here. ROCm is reachable only through
   `rocm_syntax_check.sh`, so statements about it are unverified.
3. The runtime `BATCHLAS_NO_VENDOR=1` enforcement knob was never built. Call sites that reach
   `backend::*_vendor` or a `*_vendor_or_throw` shim directly (`src/extra/cond.cc`, `src/extra/norm.cc`,
   `src/extensions/syevx_lobpcg.cc`) bypass public selection and throw vendor-free by construction. The
   build switch `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` is the enforcement today.
4. `factorization_vendor_available` covers six ops over two NVIDIA libraries. A per-op split would move every call site.
5. Ops without `choice.hh` still select by hand and keep their windows in code. See
   [`docs/perf/dispatch.md`](../perf/dispatch.md) and @ref selection_tables.
6. The vendor-free suite is not green, and no selection mechanism can make it so. The gap is missing
   kernels, not routing. The failing set is tracked in [`vendor-free-status.md`](vendor-free-status.md).

## History: the RouteTable layer

This section describes code that no longer exists: `include/batchlas/blas/dispatch/`, the
`src/backends/<op>_route.hh` shape builders and `src/dispatch/entry_points/`. Flat kernel selection
phase 5 deleted them (@ref design_flat_selection, section 12). The reasoning carried over, so it is kept here.

The old design answered three questions with `Route{Origin, Algorithm}`: whose code runs, and which
strategy. Each `RouteTable<Op, T>` supplied:

| predicate | question | `false` means | today |
|---|---|---|---|
| `supports(r, s)` | can `r` give the correct answer | a wrong answer or out-of-bounds index | `can_run` |
| `preferred(r, s)` | is `r` the best route, vendor included | merely slower | the table ranking |
| `native_tier_preferred(r, s)` | among native routes, is `r` the better one | another native tier is better | the vendor-free order (the table with the vendor entry skipped) |

Rules that held, and what each cost:

* **No speed threshold in `supports()`.** A pinned route that failed a speed cutoff fell back to the
  automatic choice silently, so the test measured something else.
* **No vendor-free tier choice in `preferred()`.** Such a window also moved vendor-present traffic onto
  that tier. That is why `native_tier_preferred` existed.
* **Tables are pure.** No `getenv`, SYCL query or operand data. Sizing and running then reach the same
  choice, which is what prevents the `ormqr` sizing mismatch (a query returned 2560 bytes and the call
  needed 276480).

An unparsable, uncompiled or non-runnable pin fell back to the automatic choice, which in a vendor build
is the vendor. Flat selection throws instead (rule R6). Legacy spellings and aliases were removed, and
the old spellings now throw.

### The entry-point facade

The original obstacle was definition ownership, not routing. `gemm<Backend::CUDA, float>` was defined in
`cublas.cc`, so building without cuBLAS removed `batchlas::gemm` entirely. The fix moved each public
definition out of the vendor TUs:

```
vendor TU (cublas.cc, ...)  defines and instantiates  backend::<op>_vendor<B, T>
src/ops/<op>/<op>.cc        defines and instantiates  <op><B, T>, which selects and may call it
```

Until flat selection the public definitions were in `src/dispatch/entry_points/`. They now live in `src/ops/<op>/<op>.cc`.

* **Instantiation is keyed on the device family.** The bodies compile to a throw without the library, so
  the public symbol exists in every build that has the device.
* **An instantiation binds as hard as a definition.** `syev` and `ormqr` were defined in headers, but their
  instantiations lived in vendor TUs, so they vanished without those libraries.
* **Selection runs before the vendor-available test.** Each op chooses first and throws second.
* **An op moves together with its `*_buffer_size` query.** Splitting them lets the two choose differently
  (the `ormqr` sizing defect).
* **Backend asymmetries are preserved.** rocBLAS has no `hemm`, `herk`, `her2k` or `symm` wrapper, so ROCm
  instantiates only what it implements.

A native driver has no `Backend` parameter, so it cannot name `gemm<B, T>`. The op file passes the public
`gemm`, `trsm` or `ormqr` in as a lambda. Calling `sycl_gemm::gemm_custom` directly bypassed gemm's
selection and pinned the native GEMM on shapes it loses; see [`docs/perf/trsm.md`](../perf/trsm.md) and
[`docs/perf/gemm.md`](../perf/gemm.md).

`scripts/facade_symbol_check.sh` verifies the move by symbol. A forwarder left behind still compiles and links.

### Adding an op in the route era

Historical recipe, for reading old commits. The current recipe is [Adding entry points](../extending.md).

**See also:** @ref design_flat_selection, @ref selection and @ref selection_tables. Per-op windows and
rejected designs are in [`docs/perf/`](../perf/README.md), for example [`dispatch`](../perf/dispatch.md),
[`gemm`](../perf/gemm.md), [`level3`](../perf/level3.md), [`trsm`](../perf/trsm.md),
[`potrf`](../perf/potrf.md), [`qr`](../perf/qr.md), [`lu`](../perf/lu.md), [`gemv`](../perf/gemv.md) and
[`spmm`](../perf/spmm.md). Superseded root design documents are at the git tag
`perf-evidence/vendor-independence`, readable with `git show <tag>:<path>`.

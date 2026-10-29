# Vendor independence {#design_vendor_independence}

> **Status:** current · checked 2026-10-29 against `src/select/vendor.hh`, the CMake files and `ci.yml`.

BatchLAS configures, builds, loads and runs with no vendor math library. This page covers the axes
that decide vendor use, the per-library gate, the contracts every public entry point keeps, the
coverage instrument, the verification scripts and what the vendor-free build still lacks. Measured
windows are in [docs/perf/](../perf/README.md); open bugs in [known-defects.md](known-defects.md).

## Vendor independence today

Two independent questions decide whether a call can reach a vendor library:

| axis | type | question it answers | where |
|---|---|---|---|
| device family | `Backend` | which SYCL runtime family the call is compiled for | `include/batchlas/blas/enums.hh` |
| library present | `BATCHLAS_HAS_<LIB>` | which third-party math library exists in this build | `cmake/BatchLASDependencies.cmake` → `batchlas/backend_config.h` |

- "NVIDIA GPU with no cuBLAS" is `Backend::CUDA` with `BATCHLAS_HAS_CUBLAS == 0`, not a new family.
  Never test `B == Backend::CUDA` to mean "cuBLAS is here".
- `Backend::SYCL` (AdaptiveCpp Metal) has no vendor library: every vendor predicate is false there,
  so the Apple GPU build is vendor-free by construction.
- `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` (`cmake/BatchLASOptions.cmake`) is the one switch over every
  vendor library (cuBLAS, cuSOLVER, cuSPARSE, rocBLAS, rocSOLVER, rocSPARSE, LAPACKE, CBLAS). There
  are no per-library options. The three CUDA libraries are all-or-nothing: a partial set fails
  configure (`cmake/BatchLASDependencies.cmake`).
- The MathDx device libraries (cuBLASDx, cuSolverDx) count as vendor; every cuBLASDx path is deleted.

Whose code runs is the op's **choice** (@ref design_flat_selection); the vendor library is the
family `vendor`. Its `can_run` is `d.has_vendor`, filled from `select::has_library<B>(spec.vendor)`.
Without the library every vendor entry is unrunnable and the first runnable native entry of the
table row runs. Nothing runnable is a `NoRouteError` (`include/batchlas/no_route.hh`, a
`batchlas::unsupported`) whose message names the op, the scalar and the switch that restores it.

## The vendor gate

`src/select/vendor.hh` asks per **library group**, not per device family. An op names its group once
in `OpSpec::vendor` (`select::Lib`).

| predicate | `select::Lib` | ops | CUDA | ROCM | SYCL | NETLIB |
|---|---|---|---|---|---|---|
| `level3_vendor_available<B>` | `level3` | `gemm` `gemv` `trsm` `trmm` `symm` `hemm` `syrk` `herk` `syr2k` `her2k` | `kHasCudaVendor` | `ROCBLAS` | false | `kHasNetlib` |
| `factorization_vendor_available<B>` | `factorization` | `geqrf` `orgqr` `getrf` `getrs` `getri` `ormqr` | `kHasCudaVendor` | `ROCSOLVER` | false | `kHasNetlib` |
| `solver_vendor_available<B>` | `solver` | `potrf` `syev` `gesvd` | `kHasCudaVendor` | `ROCSOLVER` | false | `kHasNetlib` |
| `sparse_vendor_available<B>` | `sparse` | `spmm` | `kHasCudaVendor` | `ROCSPARSE` | false | `kHasNetlib` |
| — | `none` | `gesv`, `posv` (no vendor family) | — | — | — | — |

- `kHasCudaVendor` is `BATCHLAS_HAS_CUBLAS && BATCHLAS_HAS_CUSOLVER && BATCHLAS_HAS_CUSPARSE`: each
  CUDA vendor TU calls all three. `kHasNetlib` is `BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS`.
- rocBLAS has no `hemm`, `herk`, `her2k` or batched `symm`: on ROCm the first three refuse `vendor`
  and run their native families, and symm's `vendor` arm is symmetrize plus the public gemm.

The vendor arm is an `if constexpr` on the same predicate, so with the library absent the vendor call
is not compiled. Its `else` is `select::no_vendor<B, T>(spec)`: a coverage `miss`, then
`NoRouteError`. Header and non-op callers use `*_vendor_or_throw` shims (`src/ops/syev/vendor.hh`,
`src/ops/ormqr/vendor.hh`).

| Alternative | Result | Verdict |
|---|---|---|
| A stub TU per absent library defining a throwing `backend::<op>_vendor` | Restates every vendor signature (20 ops plus 10 `*_buffer_size` queries); restated copies diverge | Rejected |

### The vendor gate: why the tile route predicate is per backend and scalar

Whether `sytrd_blocked`, `ortho` and `ormqr_blocked` swap a gemm for a level-3 tile kernel is
`select::level3_tile_substitution_enabled<B, T>`: `native_gpu_backend<B> && (T is float || (B ==
Backend::CUDA && BATCHLAS_HAS_CUBLAS))`, with `native_gpu_backend` = CUDA or SYCL. A bare `true` is
too wide in type (non-float tile routes are not linked everywhere: `syr2k` has none, and
`gram_via_syrk` admits double) and in backend (the native level-3 kernels are wired only for CUDA
and SYCL). With cuBLAS present the answer is `true` for every type on CUDA. The gate is narrower
than where those kernels run vendor-free; widening it changes ortho's and ormqr's choices, which are
unmeasured.

## Vendor independence: headers keyed on the library axis

The math-library includes in `src/linalg-impl.hh` (`<cublas_v2.h>`, `<cusparse.h>`,
`<cusolverDn.h>`, `<lapacke.h>`, `<cblas.h>`) and the CUDA handle types are guarded by
`BATCHLAS_HAS_<LIB>`, not by the family flag, because a CUDA device with no math libraries is valid.
`cuda_runtime.h` and `<cuComplex.h>` stay on the family flag (streams, device queries, the
pointer-cast helpers). A CUDA device without the libraries still gets a complete
`LinalgHandle<Backend::CUDA>` with no vendor handles, because native TUs declare one
(`src/ops/ortho/ortho.cc`). The ROCm headers are still on `BATCHLAS_HAS_ROCM_BACKEND` (unverified:
no AMD GPU here). Symbols follow the same split; see
[runtime internals](runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols).

## The entry-point facade

Public definitions live in `src/ops/<op>/<op>.cc`; the vendor TUs (`cublas.cc`, `cusolver.cc`,
`cusparse.cc`, `rocblas.cc`, `netlib_lapack.cc`, ...) define and instantiate only
`backend::<op>_vendor<B, T>`. A public definition in a vendor TU disappears with its library (the
original obstacle: `gemm<Backend::CUDA, float>` lived in `cublas.cc`).

- **Instantiation is keyed on the device family**, so the public symbol exists in every build that
  has the device. An instantiation binds as hard as a definition: a header-defined op instantiated
  in a vendor TU vanishes too.
- **Selection runs before the vendor-available test**: each op chooses first and throws second.
- **An op moves together with its `*_buffer_size` query**, or the two choose differently (the
  `ormqr` sizing defect: a query returned 2560 bytes, the call needed 276480).
- **A native driver calls the public op**, passed in as a lambda, never a native kernel directly:
  calling `sycl_gemm::gemm_custom` bypassed gemm's selection and pinned it on shapes it loses.

`scripts/facade_symbol_check.sh` verifies this by symbol: a forwarder left behind still compiles.

## Positional validators: reject only what no route can serve

Each positional (workspace-taking) entry point of `geqrf`, `orgqr`, `getrf`, `getrs`, `getri` and
`potrf` has an `<op>_validate_params` in its public header, run before the table key is built (the
key reads `A.rows()` and `A.cols()`); `trsm_validate_params` follows the same order. They throw
`batchlas::invalid_argument`.

**Validate only what no kernel could serve.** A native `can_run` that rejects a shape steers the
choice to another family, usually `vendor`; it does not make the call invalid. The stricter checks
(`require_square`, `require_span_at_least`, `require_info_span`) live in the option and arena
overloads (`include/batchlas/blas/options.hh`).

| op | checked | not checked, and why |
|---|---|---|
| `potrf` | negative extents, squareness, `uplo` | a short non-empty `info` span silently becomes pool scratch (`detail::info_target`) |
| `geqrf` | negative extents | squareness and `m >= n` (tall panels are the point; the vendor serves wide views); `tau` length |
| `orgqr` | negative extents | `n <= m` (handed to a vendor); `tau` length |
| `getrf` | negative extents | squareness; pivot length |
| `getrs` | negative extents | A square, `A.rows() == B.rows()`, batch, pivot length (a non-conforming pair goes to the vendor) |
| `getri` | negative extents of A | squareness of A and C, order and batch agreement, pivot length |

The arena spellings check the lengths. `getri` has two arities because `getri_buffer_size` and
`ops::getri::key_of` take A alone, and the query must validate the view its choice is computed from.
`gesv` and `posv` (no vendor family) also check squareness, `B.rows() == A.rows()`, equal batch and
(`posv`) `uplo`, since a bad pair would otherwise report "nothing runnable".

## Per-item info spans for potrf, getrf and getri

`info` on `potrf`, `getrf` and `getri` (and through them `posv` and `gesv`) is one `int32` per batch
item, 0 on success; above 0 it names the failing leading minor (`potrf`), the column where U is
exactly singular (`getrf`) or the zero diagonal of U (`getri`). An empty span means "not requested";
a short non-empty span is treated the same on the positional path and rejected by
`require_info_span` on the option overloads. The workspace size is the same either way. `info` is a
separate inline overload, not a defaulted parameter, because the `sig::` aliases behind
`BATCHLAS_INSTANTIATE` are function types and cannot carry defaults.

## Info spans on syev, gesvd and steqr: forwarder or default

`syev`, `syevx`, `gesvd`, `steqr`, `stedc` and every tier below them take a trailing
`Span<int32_t> info` ([contract](../cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc)).

- **Public `syev` and `gesvd`**: an old-arity inline forwarder, for the same `sig::` reason. For
  `gesvd`, arity plus the `Uplo`/`Span` type at parameter 8 keeps the four overloads unambiguous.
- **`backend::syev_vendor` and `backend::gesvd_vendor`**: a defaulted `info_out` (a default belongs
  to the declaration, so the aliases still match), keeping the internal six-argument callers compiling.
- **`steqr`, `steqr_cta`, `stedc`, `syevx*`, the `syev_*` and SVD tiers** (`extensions.hh`): a
  defaulted trailing parameter. They are instantiated from macros and already default `jobz`,
  `params` and `eigvects`; a forwarder would be ambiguous with the primary.

`gesvd` details: [the gesvd design page](gesvd.md#gesvd-design-vendor-binding-and-dispatch).

## Vendor independence: the coverage instrument

`src/select/coverage.hh` writes three kinds of row to `$BATCHLAS_COVERAGE_OUT.<pid>`. A symbol that
is present is not a kernel that runs.

| row kind | question | how produced |
|---|---|---|
| `linked` | is the kernel in this build | `coverage::static_table()`: per `(op, backend)`, the vendor gate and a native-linked flag, reported for `float` |
| `reached` | did a call get there | `coverage::record_choice` from `select::run`, one row per `(op, scalar, backend, shape_class, variant, origin, choice)` |
| `miss` | nothing served the call | `select::throw_no_vendor_route` (also behind `select::no_vendor`) |

- `native_route_supported` is `1`, `0`, or `-1` (only for a `TraceScope` built without facts, in tests).
- `uplo`, `side`, `diag`, `transA`, `transB` are in the row key; `shape_class` buckets `max(m,n,k)`
  and `batch` by power of two.
- Rows are **first-writer-wins**: the `m`, `n`, `k`, `batch` columns can show another call's shape.
  Prove a particular shape ran with a deliberate break that fails only for it.
- Gated at runtime on `BATCHLAS_COVERAGE_OUT`; the tables are leaked so the `atexit` writer does
  not walk destroyed containers.

| Alternative | Result | Verdict |
|---|---|---|
| A compile-time `-DBATCHLAS_ENABLE_COVERAGE` | Sound only if every TU in the process agrees; a test binary with its own weak resolver copy wrote zero `reached` rows | Rejected (`cmake/BatchLASOptions.cmake` records it) |

## Verification tooling

| script | what it is for | why it can be trusted |
|---|---|---|
| `scripts/route_diff.sh capture\|compare` | prove a change moved no selection decision | zero `reached` rows is a hard error. Keys on `(kind, op, scalar, backend, shape_class, origin, algo, native flags, uplo, side, diag, transA, transB)`; drops `m`, `n`, `k`, `batch`, call count. The only tool that sees a vendor-to-vendor change. |
| `scripts/coverage_merge.sh` | merge the per-pid shards of a ctest run | sums `calls`, de-duplicates `linked` |
| `scripts/facade_symbol_check.sh` | public entry points are not defined in a vendor component | matches Itanium mangling; `nm -C` fails on concept-constrained templates |
| `scripts/rocm_syntax_check.sh` | `-fsyntax-only` the three ROCm vendor TUs | only `sycl::get_native<ext_oneapi_hip>` errors are expected on a CUDA-only DPC++; anything else is real |
| `scripts/register_probe.sh` | register and spill residency of the device link | gate on `0 bytes spill` and `resident::sm89_fits(regs, wg)` (\f$\lceil \mathrm{warps}/4\rceil \cdot 32 \cdot \mathrm{roundup}_8(\mathrm{regs}) \le 16384\f$), max over `<name>` and `<name>_with_offset`; a non-zero stack frame is normal |
| `BATCHLAS_SELECT_TRACE=1` | print each decision, its table and the runner-up | one line per call, indented under its parent op |

## The vendor-free build {#vendor-free-status}

| | vendor-free CUDA | Metal (AdaptiveCpp, Apple GPU) |
|---|---|---|
| configure | `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON` | preset `acpp-metal-tests` |
| result | `BATCHLAS_HAS_CUDA_BACKEND 1`, every vendor library 0 | `BATCHLAS_HAS_SYCL_BACKEND 1`; double on the host (NETLIB, OpenBLAS) |
| CI | nightly "Vendor-free configure, build and ctest", `continue-on-error`: reports, never gates | not in CI |
| ledger | none: no per-case list has been recorded on the current tree | `tests/known-failures-acpp-metal.txt` |

Both build, link, load and run; the vendor-free CUDA suite is not green. Do not measure it by pass
count: level-3 and factorization suites also run host-backend rows, which a vendor-free build
cannot serve. Count the `NoRouteError`s by op and diff them with the failing test names:

```
ctest --test-dir build-novendor -LE slow --output-on-failure \
  | grep -o 'no route for [a-z0-9_]*' | sort | uniq -c | sort -rn
```

For routes, read the `reached` rows (`ctest -R`-scoped, or the `batchlas_coverage` target), never
the `linked` rows.

## What is still open, architecturally

Per-op performance debts (cells where the vendor still wins) are on the `docs/perf/` pages.

1. **The host path.** Native `can_run` requires `d.is_gpu` for almost every family; the exceptions
   are `gemv` and `spmm` `direct`, and `gemm`, whose natives run when `d.is_gpu || !d.has_vendor`.
   Host (`Backend::NETLIB`) rows therefore fail vendor-free. Whether a CPU SYCL device must be fast
   or only correct is undecided (standing answer: correct only).
2. **Vendor-only cells on a GPU:** `syr2k` complex<double>, `syrk` complex<double> beyond the `gram`
   range (`triangular` excludes it), and shapes a native `can_run` refuses.
3. **Unrecorded status.** The vendor-free CUDA failure list was last counted 2026-09-06, before flat
   selection; the Metal ledger still lists `hemm`, `herk`, `her2k` and `posv` Upper cases that
   2026-10-22 native work should have closed. Both need a re-run.
4. `route_diff.sh compare` applies no backend filter, so `AUTO` rows from pure-layer tests inflate diffs.
5. There is no Intel or oneMKL backend (`Backend::MKL`, `Backend::MAGMA` are reserved). ROCm is
   reachable only through `rocm_syntax_check.sh`, so statements about it are unverified.
6. No runtime `BATCHLAS_NO_VENDOR=1` knob exists; the build switch is the enforcement. LOBPCG's
   projected solves (`src/ops/syevx/kernels/syevx_lobpcg.cc`) call `syev_vendor_or_throw` directly
   under the opt-in `BATCHLAS_SYEVX_PROJECTED_VENDOR`, bypassing selection.
7. `factorization_vendor_available` covers six ops over two NVIDIA libraries; a per-op split would
   move every call site. Moot on CUDA while the libraries are all-or-nothing.

## History: the RouteTable layer

Flat selection replaced `include/batchlas/blas/dispatch/`, the `src/backends/<op>_route.hh` shape
builders and `src/dispatch/entry_points/`. Their rules carried over: no speed threshold in
correctness predicates (now `can_run`), and pure tables (no `getenv`, SYCL query or operand data),
so sizing and running reach the same choice. A bad pin used to fall back silently; it now throws.

### Adding an op in the route era

Superseded; the current recipe is [Adding entry points](../extending.md). Old design documents are
at the git tag `perf-evidence/vendor-independence` (`git show <tag>:<path>`).

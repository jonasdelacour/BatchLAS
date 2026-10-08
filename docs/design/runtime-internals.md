# Runtime internals {#design_runtime_internals}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · checked 2026-10-06

This page covers the private runtime under `src/` that every entry point runs through: the `Queue`
implementation, the workspace arena, the settings loader, the per-item `info` span, matrix layout
checks, explicit-instantiation macros, symbol visibility, the build-time object-library cut and the
embedded tuned tables. Caller-facing contracts are in `<batchlas/settings.hh>`,
`<batchlas/util/workspace.hh>` and `<batchlas/util/sycl-device-queue.hh>`.

## Symbol visibility for private headers {#runtime-internals-symbol-visibility-for-private-headers}

Under `BATCHLAS_MONOLITHIC_LIBRARY` the object libraries build with `-fvisibility=hidden`.
`BATCHLAS_API` (from the generated `<batchlas/export.hh>`) restores default visibility on public
declarations (@ref design_symbol_visibility). Private headers follow the additional rules below.

- **`BATCHLAS_INTERNAL_API`** (`src/util/internal-api.hh`, never installed). 199 of the 1,083 symbols
  that the 63 test binaries resolve from `libbatchlas*.so` belong to 55 entities declared in no
  public header. The tests reach them by including private headers by relative path. The macro
  gives them default visibility on ELF, so the links work, without promising them to consumers: a
  consumer cannot name a symbol it cannot reach. When the tests stop including private headers,
  delete the macro, or link the tests against the object libraries instead.
- **Process-wide state in private headers.** Inline variables and statics such as `g_enabled` in
  `src/util/kernel-trace.hh` and `QueueImpl::device_arrays` in `src/queue.hh` must fold to one
  instance per process. Hidden library copies do not fold with a test TU's default-visibility copy,
  so the process silently gets two copies (traces invisible to the test's flush, two SYCL device
  caches). Mark such declarations `BATCHLAS_INTERNAL_API`.
- **`BATCHLAS_QUEUE_EXPORTED_INLINE`** (`src/queue.hh`) expands to `[[gnu::used]]`. The out-of-line
  `Queue`/`Event` members must be emitted into `libbatchlas` for consumers. Plain `inline` drops them
  when nothing in the tree calls them. It is not applied in the SYCL device pass.
- **`template struct BATCHLAS_API UnifiedVector<std::byte>;`** (`src/util/sycl-util-impl.cc`). An
  instantiation takes the minimum visibility of the template and its arguments. libstdc++ gives
  `std::byte` no visibility attribute, so this specialisation would be hidden. Do not remove the
  attribute: `resize`, both constructors and the destructor silently stop being exported, and the
  only symptom is a consumer link error.
- **Include spelling.** `src/util/` is private, never installed and on no `-I` line. Its headers are
  included only by quoted relative paths. Public util headers live in `include/batchlas/util/` and
  are spelled `<batchlas/util/...>`. Do not add `-I${PROJECT_SOURCE_DIR}/src` and do not convert
  quoted includes to `<>`.

## Namespace placement of out-of-line definitions {#runtime-internals-namespace-placement-of-out-of-line-definitions}

`src/util/queue-impl.cc`, `src/util/sycl-util-impl.cc` and `src/queue.hh` place their definitions in
`namespace batchlas` until the end of the file. `Queue`, `Event`, `Device`, `QueueImpl`, `EventImpl`,
`UnifiedVector` and `Span` are declared there. The global-namespace shim in
`<batchlas/util/sycl-device-queue.hh>` only introduces the names.

- Out-of-line members must be defined in their class's namespace. At global scope, `QueueImpl`
  would define an unrelated type and leave `Queue::impl_` pointing at an incomplete one. This is the
  only one of the three rules that fails loudly.
- Explicit instantiations that name a template by unqualified-id must be in the same namespace.
- The `operator<<` for `ReferenceWrapper`, `UnifiedVector` and `Span` exists only as in-class friend
  templates, so it belongs to the enclosing namespace. At global scope the explicit instantiations
  would instantiate the wrong template. Consumers then get an undefined reference, and the library
  build does not fail.

The `std::array` `operator<<` in `sycl-util-impl.cc` is found only by ordinary lookup inside that
file. It is not reachable by ADL, so do not rely on it from another TU.

Two definitions stay at global scope on purpose. The anonymous namespace in `queue-impl.cc` holds
SYCL kernel-name tags, and moving it would rename every kernel mangled from them.
`batchlas_throw_queue_wrong_thread` in `src/queue.hh` already carries its prefix.

## Explicit instantiation macros {#runtime-internals-explicit-instantiation-macros}

`src/util/template-instantiations.hh` defines the macros that backend and entry-point TUs use to
emit their symbols.

- **`BATCHLAS_INSTANTIATE(SIG, FN, ...)`** expands to `template SIG FN<...>;`. `SIG` is a function
  type from the `sig` namespace beside each declaration in `include/batchlas/blas/functions/`. Each
  public entry point is an overload set (a `MatrixView` primary plus an inline `Matrix` forwarder),
  so `template decltype(FN<Args...>) FN<Args...>;` is ill-formed. A signature change is one header
  edit. Function types cannot carry default arguments, so spell out every parameter.
  `BATCHLAS_COMMA` passes a second template argument through macro splitting, as in
  `sig::spmm<fp BATCHLAS_COMMA F>`.
- **`BATCHLAS_INSTANTIATE_OP(Backend::CUDA, (float), potrf)`** gives
  `template sig::potrf<float> potrf<Backend::CUDA, float>;`. The type is parenthesised so it composes
  with the `BATCHLAS_FOR_EACH_*_TYPE_1` drivers.
  - `_BACKEND_OP` is for the `backend::`-qualified vendor entry points (`syev_vendor`,
    `gesvd_vendor`, `ormqr_vendor`), whose aliases live in `sig`.
  - `_FORMAT_OP` carries the second template argument of the sparse ops. There is no
    `_BACKEND_FORMAT_OP`; the sparse vendor TUs use a local shim or a raw `BATCHLAS_INSTANTIATE`.
- Scalar domains differ per op: `symm`, `syrk` and `syr2k` are real-only; `hemm`, `herk` and
  `her2k` are complex-only. Each domain has its own driver.
- **`BATCHLAS_FOR_EACH_ENABLED_BACKEND`** instantiates an entry point once per backend the build
  compiled. Its arms are pre-guarded (`BATCHLAS_IF_CUDA`, `BATCHLAS_IF_ROCM`, `BATCHLAS_IF_HOST`), so
  a backend that was not compiled expands to nothing. `BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS` and
  `_SCALAR_ALL_BACKENDS` combine the backend and type loops.
- MKL is deliberately absent from these drivers. `ritz_values.cc`, `symm.cc`, `syrk.cc`, `syr2k.cc`,
  `trmm.cc` and `steqr_legacy.cc` instantiate different sets. Folding them in would silently add or
  drop exported symbols, so they keep hand-written blocks.

**No `BATCHLAS_API` inside the macro.** The emitted definitions need default visibility, but the
annotation belongs on the primary declaration in the public header
(`template <Backend Back, typename T> BATCHLAS_API Event gemm(...)`):

1. An attribute on an explicit instantiation is not portable. g++ 13 with `-std=c++20` rejects
   `template class __attribute__((visibility("default"))) F<double,1>;`, and the
   decl-specifier-seq is a type alias, not a return type.
2. It is unnecessary. An instantiation takes the minimum of the template's and its arguments'
   visibility, so `Backend` and `MatrixFormat` carry `BATCHLAS_API` themselves. With that, the
   declaration annotation covers the definition.
3. Only the declaration reaches consumers, and on ELF `BATCHLAS_API` is default visibility on both
   the build and the consume side. Annotating the instantiation would export the definition and
   leave the consumer's reference hidden.

The `sig::` aliases are types, so they need no annotation.

## Vendor TUs instantiate only vendor symbols {#runtime-internals-vendor-tus-instantiate-only-vendor-symbols}

The explicit-instantiation tables in vendor TUs (`cusolver.cc`, `rocblas.cc`, `rocsolver.cc`,
`rocsparse.cc`, `netlib_lapack.cc`, and the cuBLAS and cuSPARSE TUs) name only `backend::`-qualified
`*_vendor` symbols. Public entry points and their `*_buffer_size()` queries are instantiated in
`src/ops/<op>/<op>.cc` (the level-3 family shares one TU under `src/ops/`). Those files compile into
`batchlas_dispatch_obj` in every build, keyed on device family. The vendor call inside is
`if constexpr`-gated, so the public symbol exists even without the library. A public-op row in a
vendor TU would be a duplicate symbol, and a vendor-free build would lose the op. The rule is per op:
a new op under `src/ops/` follows it with no change here. Design record:
[flat kernel selection](flat-kernel-selection.md).

- `sig::trsm_vendor` is not an alias of `sig::trsm`. The vendor order puts `alpha` last, as cuBLAS
  defines it; the public `trsm` takes it third. The vendor forms must agree with each other.
- `gesvd_vendor` is declared in `functions/gesvd.hh` and defined only by backends that have one.
  `rocsolver.cc` defines a throwing stub, so a ROCM build fails at the call, not at link time.
- The netlib `gemm` lives in `backend` under its vendor name, because its public `gemm` is the CBLAS
  call. Its `gesvd_vendor` keeps the synchronous `ctx.wait()`: LAPACKE `?gesvd` needs `A` on the host.
- `rocblas.cc` has no `symm`, `hemm`, `herk` or `her2k` wrapper. The ROCM arm of the level-3 TU
  under `src/ops/` mirrors that and instantiates only the ops rocBLAS implements.
- ROCm 6.3 changed `rocblas_[sdcz]trmm` to the 14-argument out-of-place form (`..., A, lda, B, ldb,
  C, ldc`, with `B` input and `C` output). All four complex and real arms are one call.

**Unbatched vendor level-3 loop (`src/backends/batch_launch.hh`).** cuBLAS and rocBLAS `symm`,
`hemm`, `syrk`, `herk`, `syr2k`, `her2k` and `trmm` handle one matrix per call, so
`for_each_batch_item` issues one launch per batch member. A batch of one launches against the
caller's view, not `view[0]`. The loop tests `<= 1`, not `== 1`: an empty view still issues one
launch, and changing that would change what an empty batch does. The callable comes first because
the trailing pack varies (two views for `syrk`/`herk`, three otherwise). The count comes from the
first view.

## Per-queue workspace arena {#runtime-internals-the-per-queue-workspace-arena}

`WorkspaceArena` (`src/queue.hh`) is the scratch memory behind `Queue::workspace()` and
`WorkspaceLease`. Caller rules are in `<batchlas/util/workspace.hh>`. The lease design (nesting,
reassignment, out-of-order queues, sizing) is @ref design_workspace.

- **Append-only blocks.** Blocks are never moved, because a live lease keeps its pointer. A borrow
  that does not fit in the current block opens a new one. Released bytes are rewound, not freed, so
  the steady state is one allocation per high-water mark.
- **Release order is enforced.** Only the innermost outstanding loan can move the cursor. An
  out-of-order return is recorded, and its bytes are reclaimed when the loans stacked above it come
  back. The cost is held memory, not correctness.
- **Live-loan stack.** One entry per outstanding loan, innermost last, in a `std::vector` that keeps
  its capacity. A returned entry stays only while something above it is live, so a non-empty stack
  means at least one lease is outstanding. A sequence number cannot answer that once loans return
  out of order.
- **`diagnose_out_of_order`** is false for exactly one caller: `WorkspaceLease` move-assignment,
  which acquires the right-hand lease before releasing the left (`ws = q.workspace(n);`). The assert
  holds everywhere else.
- **Drain on out-of-order queues** (`WorkspaceLease::release` in `queue-impl.cc`). Reclaimed bytes can
  go to a borrow whose kernels would run beside kernels still reading the old bytes. The release
  waits only on an out-of-order queue and only when it reclaims (`release_reclaims`). A return under
  a live lease only sets a flag; the release that pops it drains. `release()` cannot throw, because
  `~WorkspaceLease` calls it and move-assignment is `noexcept`. A runtime error surfaces at the
  caller's next `wait`.
- **Not covered by the drain:** work on a derived in-order queue, as `gesvd` and `iluk` build from
  `ctx`. Waiting on `ctx` does not wait for it; the derived queue's destructor does. See `release()`
  in `<batchlas/util/workspace.hh>`.
- **Freeing.** `trim()` refuses while any lease is outstanding and reports it. `trim()` and
  `~QueueImpl` drain the queue before freeing, because a released lease means only that the caller
  is done with the bytes, not that the kernels over them have finished. `~QueueImpl` asserts that no
  lease is live. A `WorkspaceLease` holds a `Queue*`, so releasing it after the arena is gone would
  hand a stale block to a later arena. Moving a `Queue` out from under a live lease is a caller bug,
  and draining would hide it.
- **`Queue::operator=(Queue&&)` is written out.** Move-assignment destroys the destination's
  `QueueImpl` without running `~Queue`, so per-queue teardown must run on that path. Per-queue state
  belongs in `QueueImpl`. A side table keyed on `impl_` would inherit stale entries when the address
  is reused.

## Queue thread ownership and the last-event holder {#runtime-internals-queue-thread-ownership-and-the-last-event-holder}

A `Queue` is single-threaded by contract (see `struct Queue` in `<batchlas/util/sycl-device-queue.hh>`).
`QueueThreadOwner` records the building thread and compares it on paths that mutate shared state: a
TLS load and a word compare. It is deliberately not a mutex. The arena is a bump allocator whose
cursor rewinds, so serialised calls would still interleave leases within a block. A lock would hide
the constraint.

The check sits where bytes are handed out and blocks are freed (`acquire()`, `trim()`), not in
`release()`. Every release comes from an acquire that was checked, and `~WorkspaceLease` is
`noexcept`, so a throw there would terminate.

**`LastEvent`** holds the last event submitted through the queue's wrappers, which gives a cheap
`get_event()` on an in-order queue. It is a guarded holder rather than a bare
`std::optional<sycl::event>`. Two threads submitting on one `Queue` tear an unsynchronised optional
and abort with `UR_RESULT_ERROR_INVALID_EVENT`, even when each supplies its own workspace. The check
on the member covers every reader and writer. Its interface is the subset of `std::optional` in use.

**Profiling** is opt-in. Kernel trace implies profiling. `settings().diagnostics.profiling` is the OR
of `BATCHLAS_QUEUE_PROFILING` and `BATCHLAS_BENCH_PROFILING`, folded in `settings.cc`. Reading the
field, not the environment, keeps both knobs under `configure()` and `ScopedEnvVar`.

**`BATCHLAS_LAUNCH_BOUNDS`** expands to SYCL kernel attributes on NVPTX only. On SPIR-V the
attributes emit `KernelAttributesINTEL`, which icpx 2026.0's CPU AOT compiler rejects ("unsupported
capability 5892"). That fails any `spir64_x86_64` link.

## Settings loader {#runtime-internals-the-settings-loader}

`src/util/settings.cc` is the one place BatchLAS calls `std::getenv` for its own knobs. The reason the
`Settings` snapshot exists is documented on `<batchlas/settings.hh>`.

- **Same parser as the call site.** Each field uses the parser its call site used before the
  snapshot existed, so the loader changed where a string comes from, not what it means. Sites with
  a bespoke parser keep it: the field is an `EnvValue`, and the loader only captures the string.
- **Preserved quirks.**
  - `env_positive_int_or` and the bare `atoi` form differ only for an integer literal too large for
    `int`.
  - The syevx and LOBPCG knobs keep `atoi`, because they accept `0` as a meaningful value.
  - `BATCHLAS_KERNEL_TRACE_PATH`, then `BATCHLAS_TRACE_PATH`: the first **non-empty** value wins.
  - `BATCHLAS_DUMP_BANDR1_DIR=` (set but empty) yields an empty root, not the default.
- **Synthesised route variables.** The names are built as `"BATCHLAS_" + upper(op) + "_ROUTE"` for
  each op in `RoutingSettings::ops`, so grepping for string literals misses them. The selection layer
  reads them as pins (`pin_text` in `src/select/select.cc`). An op that reads a pin must be in that
  list, or it silently has no environment pin. The meaning of each value is in
  [flat kernel selection](flat-kernel-selection.md) and on `RoutingSettings`.
- **Initialisation.** The snapshot and the last `configure()` value are function-local statics. Selection
  coverage reads its variable from a namespace-scope initialiser (`src/select/coverage.cc`), which runs
  before `main`. The only flag that can race is "a Queue has been constructed"; it is atomic because
  a Queue may be built on another thread.
- **`configure()` closes at the first Queue.** It is accepted only until the first `Queue` exists.
  The three root `Queue` constructors enforce this; the delegating and defaulted move constructors are
  covered by them. `*_buffer_size()` queries read routing and geometry, so a later change would let
  two calls in one process disagree on scratch size. `note_queue_constructed()` is idempotent and
  `noexcept`.
- **Unsafe gate.** It warns once per variable per process, not per load, because `reload_settings()`
  runs on every `ScopedEnvVar` construction and destruction. Each field is read as its call site read
  it, then gated, so a warning names a variable that was set. The gate refuses the unsafe direction,
  not every non-default: `BlasHealth::Error` is stricter and passes. The pointer-check skip treats any
  non-empty value not starting with `0` as skip, including `false`, `off` and `no`; it is not
  `env_truthy`. The gate is compiled out when `BATCHLAS_ALLOW_UNSAFE_ENV` is ON. That option is set in
  `cmake/backend_config.h.in`, and `settings.cc` defaults it to the safe reading.
- **Reload.** A reload reads the environment on top of what `configure()` last set, not on top of the
  defaults. Starting from the defaults would discard `configure()` on any unrelated `ScopedEnvVar`.
  Ignoring the environment after `configure()` would make every `ScopedEnvVar` a no-op, so an A/B test
  would compare an arm with itself (`tests/settings_tests.cc`, case (c), guards this). An explicitly
  set variable still wins over `configure()`. To pin a knob against the environment, set it in the
  `Settings` and do not export it.

## Per-item info span contract {#runtime-internals-the-per-item-info-span-contract}

`src/extensions/info_span.hh` implements the per-item status span. The public declarations in
`blas/extensions.hh` point to it. The caller-facing convention is in
[the API conventions](api-conventions.md#api-conventions-per-item-info-spans).

- `info` is one `int32` per batch item, in the caller's USM memory, written in place. It needs no
  copy back, no workspace, and does not change any `*_buffer_size()` result. An empty span costs nothing.
- `0` means converged. A value `> 0` is LAPACK-like: the number of off-diagonal elements that failed
  to converge, or `1` where a tier tracks only "did not converge".
- An **empty** span means "not requested": `info_ptr()` yields `nullptr` and every helper is a no-op.
  A too-short span is ignored. The deducing overloads reject short spans up front
  (`detail::require_info_span` in `blas/options.hh`).
- **It is an accumulator.** Producers only raise a value (`info_report` uses `fetch_max`). The entry
  point the caller invoked zeroes the span once, through `info_clear`.

The last rule is load-bearing. Nested clears are harmless (`syev -> syev_blocked -> stedc` clears the
same span three times) because no producer runs between them. Two clears with a producer between them
is the bug the rule prevents. That is why `stedc`'s merge kernels accumulate with `fetch_max`, and why
`stedc` does not pass `info` to its leaf `steqr` calls: both drivers would clear it again. The
level-synchronous driver solves `leaves * batch` problems in one `steqr` call. `info_item()` maps the
finer index back to the caller's item, which keeps `info` a plain span of length `batch`.

- **Single-writer stores.** `info_store` and `info_from_flags` store rather than raise. They are for
  tiers that are the only writer of each item (`bdsqr`, `syevx_lobpcg`, `syevx_filtered`). A store sets
  a converged item to `0` without a separate clear. A clear plus a raising kernel is two submissions
  that can race. Never use these where several producers share the span.
- **`one_means_converged`** in `info_from_flags` is needed because `syevx_lobpcg` and `syevx_filtered`
  write `1` for converged, the opposite of LAPACK. The caller keeps `flags` alive until the kernel has run.

**Vendor status arrays.** `detail::info_target` (`src/linalg-impl.hh`) routes the caller's span in when
one is supplied, and otherwise falls back to pool scratch. Supplying a span only removes a pool draw.
Every `*_vendor_buffer_size` therefore keeps its unconditional `int` term and stays correct in both
modes. The netlib `potrf`, `getrf`, `getri` and `syev` report their LAPACKE status through the span.
`call_backend_nh_r` sits beside `call_backend_nh`. The `_nh` form discards the return value and also
serves `void` CBLAS calls. An `auto` deduced from branches where one is `void` is ill-formed once the
result is used, so the two stay separate.

## Matrix layout checks {#runtime-internals-matrix-layout-checks}

`src/matrix.cc` validates layout metadata in constructors and conversions.

- **`Matrix` allocating constructor: `ld >= rows`.** `ld` is the distance between columns, and batch
  items must not overlap. Before the check, an `ld < rows` matrix read into the next column.
- **`MatrixView(data, rows, cols, ld, stride, batch_size)`** checks `ld >= rows`. Its argument order
  differs from `Matrix(rows, cols, batch_size, ld, stride)`. Writing `MatrixView<float> V(p, n, n, batch)`
  yields `ld = batch`, `stride = batch*n` and `batch_size = 1` with no error. The check fires when
  `batch < n` and names the intended spelling. Deliberately not checked:
  - null `data`, or zero `rows` or `cols`. About 37 in-repo sites build shape-only stand-ins for
    workspace queries, such as `(nullptr, 0, 0, 1, 1, batch)`.
  - `stride >= ld*cols`. Two live sites violate it: the transposed CGS view in `src/extensions/ortho.cc`
    ([known defect 1](known-defects.md#defect-1-orthos-transposed-arm-builds-a-view-that-does-not-describe-the-memory))
    and the `syevx_lobpcg` sizing dummy. A throw would break `ortho` and `syevx_lobpcg_buffer_size`.
- **Row-major pitch is a parameter.** `to_column_major` takes the row pitch explicitly. It is not
  inferred from `ld`, which is the column distance. For `rows > cols`, a legal pitch in `(cols, rows]`
  looks packed. The default (pitch `cols`) is accepted only when packed is the only layout the matrix
  can hold: `ld == rows` and `stride == rows*cols`. Otherwise the default refuses. Guessing read a
  padded buffer (`rows=8, cols=6, ld=8, stride=64`) at the wrong pitch and returned 42 of 48 elements
  wrong, all in bounds. `to_row_major` produces packed data, and its pitch is not recoverable from
  column-major metadata. The helper also checks that reads stay inside the buffer and do not straddle
  the next item.
- **`triangularize`: `uplo` names the triangle to keep.** `Uplo::Upper` zeroes the strict lower
  triangle. Storage is column-major (`data_[b*stride_ + j*ld_ + i]`). The element count is
  `batch*rows*cols`, not the span extent, which is wrong for sliced or strided views. `n` is taken from
  both dimensions. Extracting `R` from a `geqrf` result therefore needs `Upper`.
- **CSR from a dense matrix.** The offset scan writes offsets `1..rows` of each item and never element
  0. The whole offsets array is zeroed, so `row_offsets()[b*(rows+1)]` is defined. The value and index
  padding above each item's count is deterministic. `max_nnz` is the batch maximum.

## Host BLAS dgemm health check {#runtime-internals-the-host-blas-dgemm-health-check}

Some OpenBLAS CPU-dispatch kernels compute `dgemm` wrongly on the machine that auto-detection picks.
The known case is OpenBLAS 0.3.20's Cooperlake kernel on recent Intel parts. It is off by O(1) to
O(100) at some sizes, while `sgemm` is correct. Known Pitfalls in `docs/developer/agent-guide.md`
cover the symptom.

- `cmake/BatchLASBlasHealthCheck.cmake` detects the defect at configure time and records the
  `OPENBLAS_CORETYPE` that repairs it. An install can be consumed on another machine, so the value is
  compiled in as `BATCHLAS_REQUIRED_OPENBLAS_CORETYPE`. `src/backends/netlib_lapack.cc` reruns a cheap
  version of the probe once, on first double-precision use.
- The probe evaluates the naive reference on a strided sample of `C`, at most 32x32 entries per size,
  which costs milliseconds. The sizes are chosen to expose the defect: `n = 64` and `n = 256` happen to
  be correct, so one small size proves nothing.
- Single precision is skipped. Paths that call LAPACKE directly, such as netlib `gesvd`, run the guard
  themselves.
- The probe does not call `setenv()`. OpenBLAS reads `OPENBLAS_CORETYPE` in a library constructor that
  runs before any BatchLAS code. Only the environment at process start can fix it. The diagnostic reads
  `OPENBLAS_CORETYPE` directly and does not capture it into `Settings`.
- `BATCHLAS_BLAS_HEALTH` (`off | warn | error`) is a parsed enum on `settings().unsafe`. `off` disables
  the only detection of a wrong host `dgemm`, so the unsafe gate refuses it. `error` is stricter and
  passes.

## Build-time structure of the library {#runtime-internals-build-time-structure}

`src/CMakeLists.txt` cuts the library into OBJECT libraries, one per area, so an edit recompiles only
that area's objects.

| Object library | Holds |
| --- | --- |
| `batchlas_core_obj` | `matrix.cc`, `csr_generators.cc`; the target `generate_export_header()` is keyed on (@ref design_symbol_visibility) |
| `batchlas_dispatch_obj` | flat kernel selection: `src/select/*.cc`, every `src/ops/<op>/<op>.cc`, and the generated tuned-tables TU |
| `batchlas_backends_obj`, `batchlas_backends_cuda_obj`, `batchlas_backends_rocm_obj` | vendor wrapper TUs; the CUDA and ROCm libraries exist only when a vendor library of that family is enabled |
| `batchlas_extensions_*_obj` (eigen, factorization, symmetric, tridiag, sytrd, latrd, stedc, cta) | native drivers under `src/extensions/` |
| `batchlas_sycl_obj` | native SYCL kernels under `src/sycl/` |
| `batchlas_util_obj`, `batchlas_extra_obj` | `src/util/` (queue, settings, SYCL utilities) and `src/extra/` (norms, cond, transpose, random matrices) |

- **Selection is compiled unconditionally.** `batchlas_dispatch_obj` has no vendor condition. Its
  vendor calls are `if constexpr`-gated on the `BATCHLAS_HAS_*` macros. This makes
  `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` a complete library rather than a link error.
- **An empty vendor object library is a CMake error.** "A CUDA device is present" is not enough to
  create the CUDA backend library. It requires at least one enabled CUDA math library.
- **`batchlas_extensions_cta_obj` is the only object library built with `NO_CPU_TARGETS`.** Its kernels
  are GPU-only (32-lane CTA and tiny tiers). Nothing warns if a source is moved out of that list; it
  silently gains a CPU image. `batchlas_sycl_obj` keeps the CPU target, because the native-CPU image is
  what makes a vendor-free CPU build complete.
- **A source sits with the sources whose device symbols it calls.** The shared library is the
  device-link unit. `potrf_blocked.cc` and `potrf_cta.cc` share `potrf_cta_body`, so they must be in
  the same object library. Splitting them links today only because the helpers are inline templates.
  Once one is marked `SYCL_EXTERNAL`, the split fails with `ptxas fatal: Unresolved extern function`.
  The rationale for each placement is in the comments of `src/extensions/CMakeLists.txt`.
- **Tiny-kernel TUs** (`getrf_tiny.cc`, `gesv_tiny.cc`, `posv_tiny.cc`, `trsm_sg_left.cc`) build with
  `-mllvm -pragma-unroll-threshold=262144`. Without it LLVM silently declines `#pragma unroll`, and
  the register arrays go to the stack.
- **Split by default, monolithic for release.** The default build makes each object library its own
  shared library. `BATCHLAS_MONOLITHIC_LIBRARY` links them into one `libbatchlas.so`. The device link
  runs per shared library, so the split keeps a one-file edit's relink small. The merged link took
  575 s for a 273 MiB `.so` on the RTX 4090 box, out of 11 m 38 s for a `-j16` build. Hidden visibility
  applies only to the monolithic build. In split mode, process-wide inline state would split into one
  copy per `.so` (see [symbol visibility](#runtime-internals-symbol-visibility-for-private-headers)).

## Embedded tuned tables {#embedded-tuned-tables}

The tuned selection tables (`tuned/<op>.<dtype>.<device>.txt`; format in `tuned/README.md` and
[flat kernel selection](flat-kernel-selection.md)) are compiled into the library. An installed
BatchLAS therefore needs no data files at run time.

- **Generation.** `src/CMakeLists.txt` globs `tuned/*.txt` with `CONFIGURE_DEPENDS`, so adding or
  removing a table reconfigures. It runs `cmake/BatchLASEmbedTables.cmake` as a custom command that
  depends on the tables. The script writes `<build>/generated/select/tuned_tables.cc`, which defines
  `select::embedded_tables()` (declared in `src/select/select.hh`) as a sorted array of
  `{file name, text}` pairs. The TU is compiled into `batchlas_dispatch_obj`.
- A file not named `<op>.<dtype>.<device>.txt` is a configure error. The loader would otherwise skip
  it silently.
- Each table is embedded as `R"batchlas_tbl(...)batchlas_tbl"`. A table that contains the delimiter is
  a configure error.
- Each table is one `char` array whose length comes from `sizeof`. A `string_view` built from a bare
  literal runs a constexpr `strlen`, and about 0.5 MB of tables exceeds clang's constexpr step limit.
  C++23 `#embed` is not available in DPC++.
- Zero tables is valid. The array keeps a sentinel entry, and the span is empty.
- An unchanged table set leaves the generated file untouched, so its timestamp holds and the TU is not
  recompiled.

**Staleness.** `cmake/BatchLASTunedStaleness.cmake` recomputes at configure time the kernel-source hash
that each table header records (`kernels=<hash>`, over the sources listed in `tools/tune/<op>_spec.cc`)
and warns on a mismatch. It is a warning only: a stale table stays in use, since a slightly stale
ranking beats none. `.github/ci/check_tuned_tables.py` and the tuner compute the same hash.

**Run time.** Tables are parsed lazily per `(op, dtype)`, on the first selection that needs them, and
cached together with the `BATCHLAS_TUNED_DIR` value and a generation counter. A file in
`BATCHLAS_TUNED_DIR` with the same name replaces the embedded one, and the trace tag then reads
`override`. A non-table file there is skipped with a one-time warning. The cache, its mutex and the
parsed tables are deliberately leaked. `choose()` can run from static destructors, and the coverage
writer runs from `atexit`, so neither may find the tables already destroyed. Tests swap the embedded
set with `select::testing::set_builtin_tables` and restore it with `use_embedded_tables`. Each call
bumps the generation, so no cached parse survives.

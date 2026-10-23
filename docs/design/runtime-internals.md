# Runtime internals {#design_runtime_internals}

> **Status:** current · checked 2026-10-21

The private runtime under `src/` that every entry point runs through: the `Queue` implementation,
the workspace arena, the settings loader, the per-item `info` span, matrix layout checks,
explicit-instantiation macros, symbol visibility, the object-library cut and the embedded tuned
tables. Caller-facing contracts are in `<batchlas/settings.hh>`, `<batchlas/util/workspace.hh>` and
`<batchlas/util/sycl-device-queue.hh>`.

## Symbol visibility for private headers {#runtime-internals-symbol-visibility-for-private-headers}

With `BATCHLAS_MONOLITHIC_LIBRARY` and `BATCHLAS_HIDDEN_VISIBILITY` (default ON, OFF when the tree
builds tests) the object libraries build with `-fvisibility=hidden`. `BATCHLAS_API` (from the
generated `<batchlas/export.hh>`) restores default visibility on public declarations
(@ref design_symbol_visibility). Private headers follow these extra rules.

- **`BATCHLAS_INTERNAL_API`** (`src/util/internal-api.hh`, never installed) gives default ELF
  visibility to entities declared in no public header that the tests reach through private headers
  (`<src/...>`), without promising them to consumers. At introduction: 199 of the 1,083 symbols the
  63 test binaries resolved from `libbatchlas*.so`, 55 entities. When the tests stop including
  private headers, delete the macro or link the tests against the object libraries.
- **Process-wide state in private headers.** Inline variables and statics (`g_enabled` in
  `src/util/kernel-trace.hh`, `QueueImpl::device_arrays` in `src/queue.hh`) must fold to one instance
  per process. A hidden library copy does not fold with a test TU's default-visibility copy, so the
  process silently gets two (traces invisible to the test's flush, two SYCL device caches). Mark
  them `BATCHLAS_INTERNAL_API`.
- **`BATCHLAS_QUEUE_EXPORTED_INLINE`** (`src/queue.hh`) is `[[gnu::used]] inline`, so the out-of-line
  `Queue`/`Event` members are emitted into `libbatchlas` for consumers; plain `inline` drops them
  when nothing in the tree calls them. Plain `inline` in the SYCL device pass.
- **`template struct BATCHLAS_API UnifiedVector<std::byte>;`** (`src/util/sycl-util-impl.cc`). An
  instantiation takes the minimum visibility of the template and its arguments, and libstdc++ gives
  `std::byte` none. Without the attribute `resize`, both constructors and the destructor silently
  stop being exported; the only symptom is a consumer link error.
- **Include spelling.** `src/` is private and never installed. In-tree code includes it root-based,
  `<src/util/...>`; `batchlas_add_source_root` (`cmake/BatchLASTargetHelpers.cmake`) puts the
  repository root on in-tree targets' PRIVATE `-I` line only. Public util headers are
  `<batchlas/util/...>` (`include/batchlas/util/`). Never add `-I${PROJECT_SOURCE_DIR}/src`: `<util/...>`
  would then name a private header with the spelling `check_no_unprefixed_includes.py` reserves. A
  public header that spells `<src/...>` fails `check_public_headers.py`.

## Namespace placement of out-of-line definitions {#runtime-internals-namespace-placement-of-out-of-line-definitions}

`src/util/queue-impl.cc`, `src/util/sycl-util-impl.cc` and `src/queue.hh` stay in `namespace
batchlas` to the end of the file, where `Queue`, `Event`, `Device`, `QueueImpl`, `EventImpl`,
`UnifiedVector` and `Span` are declared. The global-namespace shim in
`<batchlas/util/sycl-device-queue.hh>` only introduces the names.

| Rule | Failure at global scope |
| --- | --- |
| Out-of-line members are defined in their class's namespace | `QueueImpl` defines an unrelated type and `Queue::impl_` points at an incomplete one; the only loud failure |
| Explicit instantiations naming a template by unqualified-id sit in its namespace | the wrong template is instantiated |
| The `UnifiedVector`/`Span` `operator<<` exists only as in-class friend templates, so it belongs to the enclosing namespace | the instantiations hit the wrong template; consumers get an undefined reference, the library build passes |

The `std::array` `operator<<` in `sycl-util-impl.cc` is found only by ordinary lookup inside that
file, not by ADL; do not rely on it from another TU.

Two definitions stay global on purpose: the anonymous namespace at the top of `queue-impl.cc`
(SYCL kernel-name tags; moving it renames every kernel mangled from them) and
`batchlas_throw_queue_wrong_thread` in `src/queue.hh`, which carries its own prefix.

## Explicit instantiation macros {#runtime-internals-explicit-instantiation-macros}

`src/util/template-instantiations.hh` defines the macros backend and entry-point TUs use to emit
their symbols.

| Macro | Expands to / does |
| --- | --- |
| `BATCHLAS_INSTANTIATE(SIG, FN, ...)` | `template SIG FN<...>;`. `SIG` is a function type from the `sig` namespace beside each declaration in `include/batchlas/blas/functions/` |
| `BATCHLAS_COMMA` | a comma that survives macro splitting: `sig::spmm<fp BATCHLAS_COMMA F>` |
| `BATCHLAS_INSTANTIATE_OP(Backend::CUDA, (float), potrf)` | `template sig::potrf<float> potrf<Backend::CUDA, float>;`; the type is parenthesised to compose with the `_TYPE_1` drivers |
| `BATCHLAS_INSTANTIATE_BACKEND_OP` | the `backend::`-qualified vendor entry points (`syev_vendor`, `gesvd_vendor`, `ormqr_vendor`), aliases in `sig` |
| `BATCHLAS_INSTANTIATE_FORMAT_OP` | the sparse ops' second template argument; there is no `_BACKEND_FORMAT_OP` (sparse vendor TUs use a local shim or raw `BATCHLAS_INSTANTIATE`) |
| `BATCHLAS_FOR_EACH_{REAL,COMPLEX,SCALAR}_TYPE_1` | the type loops; `symm`, `syrk`, `syr2k` are real-only, `hemm`, `herk`, `her2k` complex-only |
| `BATCHLAS_FOR_EACH_DEVICE_*_TYPE_1` | the `Backend::SYCL` type loops: float and `complex<float>`, the double pair only under `BATCHLAS_IF_DEVICE_FP64` |
| `BATCHLAS_FOR_EACH_ENABLED_BACKEND_1` | once per compiled backend; arms pre-guarded by `BATCHLAS_IF_CUDA`, `_ROCM`, `_SYCL`, `_HOST`, so an absent backend expands to nothing |
| `BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS`, `_SCALAR_ALL_BACKENDS` | backend loop times type loop (device type loop on the SYCL arm) |

- Each public entry point is an overload set, so `template decltype(FN<Args...>) FN<Args...>;` is
  ill-formed; the `sig` alias makes a signature change one header edit. Function types carry no
  default arguments: spell out every parameter.
- No MKL arm: `Backend::MKL` is reserved and never available (`include/batchlas/blas/enums.hh`).

**No `BATCHLAS_API` inside the macro.** The annotation belongs on the primary declaration in the
public header (`template <Backend Back, typename T> BATCHLAS_API Event gemm(...)`):

1. An attribute on an explicit instantiation is not portable. g++ 13 with `-std=c++20` rejects
   `template class __attribute__((visibility("default"))) F<double,1>;`, and the
   decl-specifier-seq is a type alias, not a return type.
2. It is unnecessary. An instantiation takes the minimum of the template's and its arguments'
   visibility; `Backend` and `MatrixFormat` carry `BATCHLAS_API`, so the declaration's annotation
   covers the definition.
3. Only the declaration reaches consumers. On ELF `BATCHLAS_API` is default visibility on both sides;
   annotating the instantiation would export the definition and leave the consumer's reference
   hidden.

The `sig::` aliases are types and need no annotation.

## Vendor TUs instantiate only vendor symbols {#runtime-internals-vendor-tus-instantiate-only-vendor-symbols}

The instantiation tables in vendor TUs (`src/backends/`: `cublas.cc`, `cusolver.cc`, `cusparse.cc`,
`rocblas.cc`, `rocsolver.cc`, `rocsparse.cc`, `netlib_lapack.cc`) name only `backend::`-qualified
`*_vendor` symbols. Public entry points and their `*_buffer_size()` queries are instantiated in
`src/ops/<op>/<op>.cc` (`src/ops/level3/level3.cc` for hemm/herk/her2k), compiled into
`batchlas_selection_obj` in every build. The vendor call inside is `if constexpr`-gated, so the public
symbol exists without the library. A public-op row in a vendor TU would be a duplicate symbol, and a
vendor-free build would lose the op. Design record: @ref design_flat_selection.

- `sig::trsm_vendor` is not an alias of `sig::trsm`: the vendor order puts `alpha` last, as cuBLAS
  does; the public `trsm` takes it third. The vendor forms agree with each other.
- `gesvd_vendor` is declared in `functions/gesvd.hh` and defined by backends that have one;
  `rocsolver.cc` defines a stub that throws `batchlas::unsupported`, so a ROCm build fails at the
  call, not at link time.
- The netlib `gemm_vendor` is the CBLAS call. Netlib `gesvd_vendor` keeps a synchronous
  `ctx.wait()`: LAPACKE `?gesvd` needs `A` on the host.
- `rocblas.cc` has no `symm`, `hemm`, `herk` or `her2k` wrapper; the ROCm arm of the level-3 TU
  instantiates only what rocBLAS implements.
- ROCm 6.3 changed `rocblas_[sdcz]trmm` to the 14-argument out-of-place form (`..., A, lda, B, ldb,
  C, ldc`, `B` input, `C` output); all four type arms are one call.

**Unbatched vendor level-3 loop (`src/backends/batch_launch.hh`).** cuBLAS and rocBLAS `symm`,
`hemm`, `syrk`, `herk`, `syr2k`, `her2k` and `trmm` take one matrix per call, so
`for_each_batch_item` issues one launch per item. A batch of one launches against the caller's view,
not `view[0]`. The test is `<= 1`, not `== 1`: an empty view still issues one launch. The callable
comes first because the trailing pack varies (two views for `syrk`/`herk`, three otherwise); the
count comes from the first view.

## Per-queue workspace arena {#runtime-internals-the-per-queue-workspace-arena}

`WorkspaceArena` (`src/queue.hh`) is the scratch behind `Queue::workspace()` and `WorkspaceLease`.
Caller rules: `<batchlas/util/workspace.hh>`; lease design: @ref design_workspace.

- **Append-only blocks.** A live lease keeps its pointer, so blocks never move. A borrow that does not
  fit opens a new USM-shared block of `max(request, 2 x last block, 64 KiB)`. Released bytes are
  rewound, not freed: the steady state is one allocation per high-water mark.
- **Release order is enforced.** Only the innermost outstanding loan moves the cursor. An
  out-of-order return is marked, and its bytes are reclaimed when the loans above it come back. The
  cost is held memory, not correctness.
- **Live-loan stack.** One entry per outstanding loan, innermost last, in a `std::vector` that keeps
  its capacity. A returned entry stays only while something above it is live, so a non-empty stack
  means a lease is outstanding; a sequence number alone cannot answer that once loans return out of
  order. Sequence numbers are never reused, even after `free_all`.
- **`diagnose_out_of_order`** is false for exactly one caller: `WorkspaceLease` move-assignment, which
  acquires the right-hand lease before releasing the left (`ws = q.workspace(n);`). The debug assert
  holds everywhere else.
- **Drain on out-of-order queues** (`WorkspaceLease::release_` in `queue-impl.cc`). Reclaimed bytes can
  go to a borrow whose kernels run beside kernels still reading them, so a release waits, only on an
  out-of-order queue and only when it reclaims (`release_reclaims`). A return under a live lease only
  marks it; the release that pops it drains. `release()` cannot throw (`~WorkspaceLease` calls it,
  move-assignment is `noexcept`); a runtime error surfaces at the caller's next `wait`.
- **Not covered by the drain:** work on a derived in-order queue (`Queue(base, true)`). The drivers
  that build one (`select::on_in_order_queue`: syev, gesvd, ormqr, posv, the `linalg` wrappers) make
  `base` wait on its last event when they return; `iluk` waits on its derived queue. A lease on
  `base` released while the derived queue still runs is not covered. See `release()` in
  `<batchlas/util/workspace.hh>`.
- **Freeing.** `trim()` refuses while any lease is outstanding. `trim()` and `~QueueImpl` drain the
  queue before freeing (a released lease means the caller is done, not the kernels). `~QueueImpl`
  asserts that no lease is live and drains only when it has blocks, except on Metal, where it always
  drains: coarse events leave the command buffer open, so a derived queue's work outlived the
  caller's wait. A `WorkspaceLease` holds a `Queue*`; releasing it after the arena is gone would hand
  a stale block to a later arena. Moving a `Queue` from under a live lease is a caller bug, and
  draining would hide it.
- **`Queue::operator=(Queue&&)` is written out.** It destroys the destination's `QueueImpl` without
  running `~Queue`, so per-queue state belongs in `QueueImpl`; a side table keyed on `impl_` inherits
  stale entries when the address is reused.

## Queue thread ownership and the last-event holder {#runtime-internals-queue-thread-ownership-and-the-last-event-holder}

A `Queue` is single-threaded by contract (`struct Queue` in `<batchlas/util/sycl-device-queue.hh>`).
`QueueThreadOwner` (`src/queue.hh`) records the building thread and compares it on the paths that
mutate shared state, a TLS load and a word compare; a mismatch throws `batchlas::api_misuse`. It is
deliberately not a mutex: the arena's cursor rewinds, so serialised calls from two threads would still
interleave their leases.

- The check sits where bytes are handed out and blocks freed (`acquire()`, `trim()`), not in
  `release()`: every release comes from a checked acquire, and `~WorkspaceLease` is `noexcept`.
- `Queue::attach_to_current_thread()` moves ownership (arena and last event) to the calling thread.
  It throws `batchlas::api_misuse` while a lease is outstanding.
- **`LastEvent`** holds the last event submitted through the queue's wrappers, for a cheap
  `get_event()` on an in-order queue. It is a guarded holder, not a bare `std::optional<sycl::event>`:
  two threads submitting on one `Queue` tear an unsynchronised optional and abort with
  `UR_RESULT_ERROR_INVALID_EVENT`, even when each supplies its own workspace. Its interface is the
  subset of `std::optional` in use.
- **Profiling** is opt-in; kernel trace implies it. `settings().diagnostics.profiling` is the OR of
  `BATCHLAS_QUEUE_PROFILING` and `BATCHLAS_BENCH_PROFILING`, folded in `settings.cc`; reading the
  field keeps both under `configure()` and `ScopedEnvVar`.
- **`BATCHLAS_LAUNCH_BOUNDS`** (`src/sycl/kernel_attrs.hh`) expands to kernel attributes on DPC++ NVPTX
  only, and to nothing under AdaptiveCpp. On SPIR-V the attributes emit `KernelAttributesINTEL`, which
  icpx 2026.0's CPU AOT compiler rejects ("unsupported capability 5892"), failing any `spir64_x86_64`
  link.

## Settings loader {#runtime-internals-the-settings-loader}

`src/util/settings.cc` is the one place BatchLAS calls `std::getenv` for its own knobs (exceptions:
@ref design_environment). Why the `Settings` snapshot exists is documented on `<batchlas/settings.hh>`.

- **Same parser as the call site.** Each field uses the parser its call site used, so the loader
  changes where a string comes from, not what it means. A site with a bespoke parser keeps it: the
  field is an `EnvValue` and the loader only captures the string.
- **Preserved quirks.**
  - `env_positive_int_or` and the bare `atoi` form differ only for an integer literal too large for
    `int`.
  - The syevx and LOBPCG knobs keep `atoi`: they accept `0` as a meaningful value.
  - `BATCHLAS_KERNEL_TRACE_PATH`, then `BATCHLAS_TRACE_PATH`: the first **non-empty** value wins.
  - `BATCHLAS_DUMP_BANDR1_DIR=` (set but empty) yields an empty root, not the default.
- **Synthesised route variables.** `"BATCHLAS_" + upper(op) + "_ROUTE"` for each op in
  `RoutingSettings::ops`, so a grep for string literals misses them. Selection reads them as pins
  (`pin_text` in `src/select/select.cc`). An op that reads a pin must be in that list, or it silently
  has no environment pin. Value meanings: @ref design_flat_selection and `RoutingSettings`.
- **Initialisation.** The snapshot and the last `configure()` value are function-local statics.
  Selection coverage reads its variable from a namespace-scope initialiser (`src/select/coverage.cc`)
  that runs before `main`. The only flag that can race is "a Queue has been constructed"; it is atomic
  because a Queue may be built on another thread.
- **`configure()` closes at the first Queue.** The three root `Queue` constructors call
  `note_queue_constructed()` (idempotent, `noexcept`); the delegating and move constructors are
  covered by them. `*_buffer_size()` queries read routing and geometry, so a later change would let two
  calls in one process disagree on scratch size.
- **Unsafe gate.** Warns once per variable per process, not per load, because `reload_settings()` runs
  on every `ScopedEnvVar` construction and destruction. Each field is read as its call site read it,
  then gated, so a warning names a variable that was set. It refuses the unsafe direction only:
  `BlasHealth::Error` is stricter and passes. The pointer-check skip treats any non-empty value not
  starting with `0` as skip (`false`, `off`, `no` included); it is not `env_truthy`. The gate is
  compiled out with `-DBATCHLAS_ALLOW_UNSAFE_ENV=ON` (`cmake/BatchLASOptions.cmake`, written to
  `cmake/backend_config.h.in`); `settings.cc` defaults it to the safe reading.
- **Warnings.** Every user-facing warning is one stderr line `batchlas: <text>` from
  `batchlas::detail::warn` (`src/util/internal-api.hh`). `warn_once(tag, text)` prints a tag once per
  process (`unsafe:<variable>`, `borrow:<op>`, `native:<op>`/`vendor:<op>`, `tuned_dir:`, `tuned_name:`,
  `syevx_degrade:`); `reset_warnings()` clears the tags for tests.
- **Reload.** A reload reads the environment on top of what `configure()` last set (the defaults if it
  was never called). Each loader assigns a field only when its variable is exported
  (`set_if_present`); the unsafe gate likewise refuses only a move the environment made. Starting from
  the defaults would discard `configure()` on any unrelated `ScopedEnvVar`
  (`SettingsConfigure.ConfigureSurvivesUnrelatedReload`). Ignoring the environment after
  `configure()` would make every `ScopedEnvVar` a no-op and an A/B test compare an arm with itself
  (`tests/settings_tests.cc`, case (c)). An exported variable wins over `configure()`; to pin a knob,
  set it in `Settings` and do not export it.

## Per-item info span contract {#runtime-internals-the-per-item-info-span-contract}

`src/kernels/common/info_span.hh` implements the per-item status span. Caller convention:
[API conventions](api-conventions.md#api-conventions-per-item-info-spans).

- `info` is one `int32` per batch item in the caller's USM, written in place: no copy back, no
  workspace, no change to any `*_buffer_size()`. An empty span costs nothing.
- `0` means converged. `> 0` is LAPACK-like: the number of off-diagonal elements that failed to
  converge, or `1` where a tier tracks only "did not converge".
- **Empty** means "not requested": `info_ptr()` yields `nullptr` and every helper is a no-op. A
  too-short span is ignored here; the deducing overloads reject it first (`detail::require_info_span`,
  `blas/options.hh`).
- **It is an accumulator.** Producers only raise (`info_report`, `fetch_max`). The entry point the
  caller invoked zeroes the span once (`info_clear`).

The last rule is load-bearing: two clears with a producer between them lose a failure. Nested clears
with no producer between them are harmless. `stedc` therefore calls its leaves through
`steqr_dispatch`, which raises without clearing (`steqr` would clear). The level-synchronous driver
solves `leaves * batch` problems in one call, so the leaves get their own per-leaf status array, folded
into the caller's span with `info_item()` (leaf \f$j\f$ belongs to item \f$j / 2^L\f$) and
`info_report`; `info` stays a plain span of length `batch`.

- **Single-writer stores.** `info_store` and `info_from_flags` store rather than raise, for tiers that
  are the only writer of each item: `bdsqr`, `syevx_lobpcg`, `syevx_filtered`, `syev_cta_fused`,
  `syev_jacobi_cta`, `gesvdj_cta`. A store sets a converged item to `0` without a separate clear; a
  clear plus a raising kernel is two submissions that can race. Never use them where several
  producers share the span.
- **`one_means_converged`** in `info_from_flags`: `syevx_lobpcg` and `syevx_filtered` flag `1` for
  converged, the opposite of LAPACK. The caller keeps `flags` alive until the kernel has run.

**Vendor status arrays.** `detail::info_target` (`src/linalg-impl.hh`) routes the caller's span in when
one is supplied and otherwise falls back to pool scratch, so a span only removes a pool draw and every
`*_vendor_buffer_size` keeps its unconditional `int` term. The netlib `potrf`, `getrf`, `getri` and
`syev` report their LAPACKE status through it via `call_backend_nh_r`, beside `call_backend_nh`. The
`_nh` form discards the return value and also serves `void` CBLAS calls; an `auto` deduced from
branches where one is `void` is ill-formed once the result is used, so the two stay separate.

## Matrix layout checks {#runtime-internals-matrix-layout-checks}

`src/matrix.cc` validates layout metadata in constructors and conversions; every failure is
`batchlas::invalid_argument`. Caller view: @ref design_matrix_model.

- **`Matrix` allocating constructor: `ld >= rows`, `stride >= ld*cols`.** `ld` is the column
  distance; items must not overlap. An `ld < rows` matrix reads into the next column.
- **`MatrixView(data, rows, cols, ld, stride, batch_size)`** checks `ld >= rows`. Its order differs from
  `Matrix(rows, cols, batch_size, ld, stride)`: `MatrixView<float> V(p, n, n, batch)` means
  `ld = batch`, `stride = batch*n`, `batch_size = 1`. The check fires when `batch < n` and the message
  names the intended spelling. Deliberately not checked:
  - null `data`, or zero `rows` or `cols`: about 37 in-repo sites build shape-only stand-ins for
    workspace queries, such as `(nullptr, 0, 0, 1, 1, batch)`.
  - `stride >= ld*cols`: the recorded violators were the transposed CGS view in `ortho` (fixed,
    [known defect 1](known-defects.md#defect-1-orthos-transposed-arm-builds-a-view-that-does-not-describe-the-memory))
    and a `syevx_lobpcg` sizing dummy. Whether any live site still violates it is not re-checked
    (open); the comment in `src/matrix.cc` cites the `syevx_lobpcg` one.
- **Row-major pitch is a parameter.** `to_column_major` takes the row pitch explicitly; it is not
  inferred from `ld` (the column distance). For `rows > cols` a legal pitch in `(cols, rows]` looks
  packed, so the default (pitch `cols`) is accepted only when packed is the only layout the matrix can
  hold (`ld == rows`, `stride == rows*cols`, or one row). Guessing read a padded buffer
  (`rows=8, cols=6, ld=8, stride=64`) at the wrong pitch: 42 of 48 elements wrong, all in bounds.
  `to_row_major` writes packed data, whose pitch is not recoverable from column-major metadata. Both
  check that reads stay inside the buffer and do not straddle the next item.
- **`triangularize`: `uplo` names the triangle to keep.** `Uplo::Upper` zeroes the strict lower
  triangle (column-major, `data_[b*stride_ + j*ld_ + i]`). The element count is `batch*rows*cols`, not
  the span extent (wrong for sliced or strided views). Extracting `R` from a `geqrf` result needs
  `Upper`.
- **CSR from a dense matrix.** The offset scan writes offsets `1..rows` of each item, never element 0;
  the whole offsets array is zeroed first, so `row_offsets()[b*(rows+1)]` is defined. Value and index
  padding above each item's count is deterministic. `max_nnz` is the batch maximum.

## Host BLAS dgemm health check {#runtime-internals-the-host-blas-dgemm-health-check}

Some OpenBLAS CPU-dispatch kernels compute `dgemm` wrongly on the machine auto-detection picks: the
known case is OpenBLAS 0.3.20's Cooperlake kernel on recent Intel parts, off by O(1) to O(100) at some
sizes while `sgemm` is correct (Known Pitfalls in @ref dev_agent_guide).

- `cmake/BatchLASBlasHealthCheck.cmake` detects it at configure time and records the
  `OPENBLAS_CORETYPE` that repairs it, compiled in as `BATCHLAS_REQUIRED_OPENBLAS_CORETYPE` because an
  install can run on another machine.
- `src/backends/host_blas_health.hh` holds the run-time port, shared by the NETLIB backend
  (`netlib_lapack.cc`, once, on first double use) and the `batchlas::verify` references. It checks a
  strided sample of `C` (step \f$\max(1, n/32)\f$, at most about 32 x 32 entries) at
  \f$n \in \{128, 200, 512\}\f$, in milliseconds. `n = 64` and `n = 256` happen to be correct, so one
  small size proves nothing.
- Single precision is skipped. Paths that call LAPACKE directly, such as netlib `gesvd`, run the guard
  themselves.
- The probe never calls `setenv()`: OpenBLAS reads `OPENBLAS_CORETYPE` in a library constructor before
  any BatchLAS code, so only the environment at process start fixes it. The diagnostic reads
  `OPENBLAS_CORETYPE` directly, not through `Settings`.
- `BATCHLAS_BLAS_HEALTH` (`off | warn | error`, default `warn`) is a parsed enum on
  `settings().unsafe`. `off` disables the only detection of a wrong host `dgemm`, so the unsafe gate
  refuses it; `error` (throws `batchlas::device_error`) is stricter and passes.

## Build-time structure of the library {#runtime-internals-build-time-structure}

`src/CMakeLists.txt` cuts the library into OBJECT libraries, so an edit recompiles only its area.

| Object library | Holds |
| --- | --- |
| `batchlas_core_obj` | `matrix.cc`, `csr_generators.cc`; the target `generate_export_header()` is keyed on (@ref design_symbol_visibility) |
| `batchlas_selection_obj` | flat selection: `src/select/*.cc`, every `src/ops/<op>/<op>.cc`, the generated tuned-tables TU |
| `batchlas_backends_obj`, `_cuda_obj`, `_rocm_obj` | vendor wrapper TUs (`src/backends/`) |
| `batchlas_extensions_{eigen,factorization,tridiag,sytrd,latrd,stedc,cta}_obj` | native drivers and kernels under `src/ops/` and `src/kernels/common/`; lists in `src/ops/CMakeLists.txt` group by device-link cluster, not folder |
| `batchlas_sycl_obj` | native BLAS kernels: `src/ops/{gemm,trsm,gemv,spmm}/kernels/` |
| `batchlas_util_obj`, `batchlas_extra_obj` | `src/util/`: `UTIL_SOURCES` (queue, settings, SYCL utilities); `EXTRA_SOURCES` (norm, cond, random cond, transpose, elementwise, the verification checks `matrix_checks.cc`) |

- **Selection is compiled unconditionally.** Its vendor calls are `if constexpr`-gated on
  `BATCHLAS_HAS_*`, so `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` is a complete library, not a link error.
- **An empty OBJECT library is a CMake error.** The CUDA vendor TUs each call all three libraries, so
  the CUDA set is all-or-nothing (`BATCHLAS_HAS_CUDA_VENDOR_TUS`, `cmake/BatchLASDependencies.cmake`):
  a partial cuBLAS/cuSOLVER/cuSPARSE set is a configure error. The ROCm library exists when at least
  one ROCm math library is enabled.
- **`batchlas_extensions_cta_obj` is the only `NO_CPU_TARGETS` library.** Its kernels are GPU-only
  (32-lane CTA and tiny tiers). Nothing warns if a source moves out of it; it silently gains a CPU
  image. `batchlas_sycl_obj` keeps the CPU target: its native-CPU image completes a vendor-free CPU
  build.
- **A source sits with the sources whose device symbols it calls.** The shared library is the
  device-link unit. `potrf_blocked.cc` and `potrf_cta.cc` share `potrf_cta_body` and must stay in one
  object library; split, they link only while the helpers are inline templates, and fail with
  `ptxas fatal: Unresolved extern function` once one is `SYCL_EXTERNAL`. Placement rationale: comments
  in `src/ops/CMakeLists.txt`.
- **Tiny-kernel TUs** (`getrf_tiny.cc`, `gesv_tiny.cc`, `posv_tiny.cc`, `trsm_sg_left.cc`) build with
  `-mllvm -pragma-unroll-threshold=262144` under DPC++; without it LLVM silently declines
  `#pragma unroll` and the register arrays go to the stack. AdaptiveCpp's JIT never sees stage-1
  `-mllvm` options; `BATCHLAS_UNROLL_FULL` covers it there.
- **Split by default, monolithic for release.** The default build makes each object library its own
  shared library (14 components); `BATCHLAS_MONOLITHIC_LIBRARY` (default ON on macOS, where split
  dylibs are rejected) links one `libbatchlas`. The device link runs per shared library, so the split
  keeps a one-file relink small: the merged link took 575 s for a 273 MiB `.so` on the RTX 4090 box,
  out of 11 m 38 s for a `-j16` build. Hidden visibility applies only to the monolithic build; in split
  mode process-wide inline state would split into one copy per `.so`
  ([symbol visibility](#runtime-internals-symbol-visibility-for-private-headers)).

## Embedded tuned tables {#embedded-tuned-tables}

The selection tables (`tuned/<op>.<device>.txt`; format in `tuned/README.md` and
@ref design_tuning_data) are compiled into the library, so an install needs no data files at run time.

- **Generation.** `src/CMakeLists.txt` globs `tuned/*.txt` with `CONFIGURE_DEPENDS` (adding or removing
  a table reconfigures) and runs `cmake/BatchLASEmbedTables.cmake` as a custom command on them. It
  writes `<build>/generated/select/tuned_tables.cc`, defining `select::embedded_tables()` (declared in
  `src/select/select.hh`) as a sorted array of `{file name, text}`, compiled into
  `batchlas_selection_obj`.
- A file not named `<op>.<device>.txt` is a configure error (the loader would skip it silently). At
  run time `select::parse_tables` splits each file into one table per dtype (first key
  `dtype:exact`).
- Each table is embedded as `R"batchlas_tbl(...)batchlas_tbl"`; a table containing the delimiter is a
  configure error.
- Each table is one `char` array sized by `sizeof`. A `string_view` built from a bare literal runs a
  constexpr `strlen`, and the tables (about 50 MB over 39 files on 2026-10-21) exceed clang's
  constexpr step limit. C++23 `#embed` is not available in DPC++.
- Zero tables is valid: the array keeps a sentinel and the span is empty.
- An unchanged table set leaves the generated file untouched, so the TU is not recompiled.

**Staleness:** @ref design_tuning_data. A stale table stays in use.

**Run time.** Tables are parsed lazily per `(op, dtype)` on the first selection that needs them, and
cached with the `BATCHLAS_TUNED_DIR` value and a generation counter. A same-named file in
`BATCHLAS_TUNED_DIR` replaces the embedded one dtype by dtype (a dtype the file lacks keeps the
embedded rows); the trace tag then reads `override`. A non-table file there is skipped with a one-time
warning. The cache, its mutex and the parsed tables are deliberately leaked: `choose()` can run from
static destructors and the coverage writer from `atexit`. Tests swap the embedded set with
`select::testing::set_builtin_tables` and restore it with `use_embedded_tables`; each call bumps the
generation, so no cached parse survives.

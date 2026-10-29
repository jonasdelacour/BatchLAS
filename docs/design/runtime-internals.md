# Runtime internals {#design_runtime_internals}

> **Status:** current · checked 2026-10-29

The private runtime under `src/` that every entry point runs through: the `Queue` implementation,
the workspace arena, the settings loader, the per-item `info` span, matrix layout checks,
instantiation macros, symbol visibility, the object-library cut and the embedded tuned tables.
Caller-facing contracts are in `<batchlas/settings.hh>`, `<batchlas/util/workspace.hh>` and
`<batchlas/util/sycl-device-queue.hh>`.

## Symbol visibility for private headers {#runtime-internals-symbol-visibility-for-private-headers}

`BATCHLAS_HIDDEN_VISIBILITY` (monolithic builds only; default ON, OFF when the tree builds tests)
builds the object libraries with `-fvisibility=hidden`; `BATCHLAS_API` restores default visibility
on public declarations (@ref design_symbol_visibility).

- **`BATCHLAS_INTERNAL_API`** (`src/util/internal-api.hh`, never installed) exports entities the
  tests reach through private `<src/...>` headers without promising them to consumers.
- **Process-wide state in private headers** (`g_enabled` in `src/util/kernel-trace.hh`,
  `QueueImpl::device_arrays`) must be `BATCHLAS_INTERNAL_API`: a hidden library copy does not fold
  with a test TU's copy, so the process silently gets two.
- **`BATCHLAS_QUEUE_EXPORTED_INLINE`** (`src/queue.hh`) is `[[gnu::used]] inline` (plain `inline` in
  the device pass), so out-of-line `Queue`/`Event` members are emitted even when nothing in the tree
  calls them.
- **`template struct BATCHLAS_API UnifiedVector<std::byte>;`** (`src/util/sycl-util-impl.cc`): an
  instantiation takes the minimum visibility of its arguments, and libstdc++ gives `std::byte` none.
  Without the attribute the members stop being exported; the only symptom is a consumer link error.
- **Include spelling.** `batchlas_add_source_root` puts the repository root on in-tree targets'
  PRIVATE `-I` line, so private headers are `<src/...>`. Never add `-I${PROJECT_SOURCE_DIR}/src`:
  `<util/...>` would then alias a private header (`check_no_unprefixed_includes.py`); a public header
  spelling `<src/...>` fails `check_public_headers.py`.

## Namespace placement of out-of-line definitions {#runtime-internals-namespace-placement-of-out-of-line-definitions}

`src/util/queue-impl.cc`, `src/util/sycl-util-impl.cc` and `src/queue.hh` stay in `namespace
batchlas` to the end of the file; the global shim in `<batchlas/util/sycl-device-queue.hh>` only
introduces the names. At global scope an out-of-line `QueueImpl` defines an unrelated type, an
explicit instantiation by unqualified-id hits the wrong template, and the `UnifiedVector`/`Span`
`operator<<` (in-class friend templates) leaves consumers with an undefined reference while the
library build passes. The `std::array` `operator<<` in `sycl-util-impl.cc` is found by ordinary
lookup only, not ADL. Two things stay global on purpose: the anonymous namespace of kernel-name tags
at the top of `queue-impl.cc` (moving it renames every kernel) and `batchlas_throw_queue_wrong_thread`.

## Explicit instantiation macros {#runtime-internals-explicit-instantiation-macros}

`src/util/template-instantiations.hh`: `BATCHLAS_INSTANTIATE(SIG, FN, ...)` expands to
`template SIG FN<...>;`, where `SIG` is a function type from the `sig` namespace beside each
declaration in `include/batchlas/blas/functions/`. Each entry point is an overload set, so
`decltype(FN<...>)` is ill-formed; function types carry no default arguments, so spell out every
parameter. `BATCHLAS_INSTANTIATE_OP`, `_BACKEND_OP` (the `backend::*_vendor` symbols) and
`_FORMAT_OP` (sparse) wrap it; `BATCHLAS_FOR_EACH_{REAL,COMPLEX,SCALAR}_TYPE_1`,
`_DEVICE_*_TYPE_1` (doubles only under `BATCHLAS_IF_DEVICE_FP64`) and
`BATCHLAS_FOR_EACH_ENABLED_BACKEND_1` (absent backends expand to nothing) are the loops.
`Backend::MKL` is reserved and has no arm.

**No `BATCHLAS_API` inside the macro.** g++ 13 rejects an attribute on an explicit instantiation,
and it is unnecessary: the instantiation inherits the visibility of the annotated primary
declaration (`Backend` and `MatrixFormat` carry `BATCHLAS_API`), and only the declaration reaches
consumers.

## Vendor TUs instantiate only vendor symbols {#runtime-internals-vendor-tus-instantiate-only-vendor-symbols}

The vendor TUs in `src/backends/` instantiate only `backend::*_vendor` symbols. Public entry points
and their `*_buffer_size()` are instantiated in `src/ops/<op>/<op>.cc`, compiled in every build,
with the vendor call `if constexpr`-gated (@ref design_flat_selection). A public-op row in a vendor
TU would be a duplicate symbol, and a vendor-free build would lose the op.

- `sig::trsm_vendor` puts `alpha` last, as cuBLAS does; public `trsm` takes it third.
- `rocsolver.cc` defines a `gesvd_vendor` stub that throws `batchlas::unsupported`, so ROCm fails at
  the call, not at link. Netlib `gesvd_vendor` keeps a synchronous `ctx.wait()`: LAPACKE needs `A` on
  the host.
- `rocblas.cc` has no `symm`, `hemm`, `herk` or `her2k`. ROCm 6.3's `rocblas_[sdcz]trmm` is the
  14-argument out-of-place form (`B` input, `C` output).
- **`for_each_batch_item`** (`src/backends/batch_launch.hh`) loops the unbatched cuBLAS/rocBLAS
  level-3 calls. A batch `<= 1` (including empty) launches once against the caller's view, not
  `view[0]`; the count comes from the first view.

## Per-queue workspace arena {#runtime-internals-the-per-queue-workspace-arena}

`WorkspaceArena` (`src/queue.hh`) backs `Queue::workspace()` and `WorkspaceLease`; lease design:
@ref design_workspace.

- **Append-only blocks.** Blocks never move under a live lease. A borrow that does not fit opens a
  USM-shared block of `max(request, 2 x last block, 64 KiB)`; released bytes are rewound, not freed.
- **Release order.** Only the innermost outstanding loan moves the cursor; an out-of-order return is
  marked and reclaimed when the loans above it return. A non-empty live-loan stack means a lease is
  outstanding. Sequence numbers are never reused. `diagnose_out_of_order` is false only for
  `WorkspaceLease` move-assignment, which acquires before releasing.
- **Drain on out-of-order queues.** A release that reclaims (`release_reclaims`) waits on the
  queue, since reclaimed bytes could go to kernels running beside readers. `release()` cannot throw;
  errors surface at the next `wait`. Work on a derived in-order queue (`Queue(base, true)`) is not
  covered: `select::on_in_order_queue` drivers make `base` wait on its last event on return.
- **Freeing.** `trim()` refuses while a lease is outstanding; `trim()` and `~QueueImpl` drain before
  freeing (`~QueueImpl` only when it has blocks, except on Metal, where coarse events leave the
  command buffer open). Moving a `Queue` from under a live lease is a caller bug.
- **`Queue::operator=(Queue&&)`** destroys the destination's `QueueImpl` without `~Queue`, so
  per-queue state belongs in `QueueImpl`, never a side table keyed on `impl_`.

## Queue thread ownership and the last-event holder {#runtime-internals-queue-thread-ownership-and-the-last-event-holder}

A `Queue` is single-threaded by contract. `QueueThreadOwner` (`src/queue.hh`) records the building
thread and checks it in `acquire()` and `trim()` (not `release()`, which is `noexcept`), throwing
`batchlas::api_misuse`. It is not a mutex: serialised calls from two threads would still interleave
leases. `Queue::attach_to_current_thread()` moves ownership and throws while a lease is outstanding.

- **`LastEvent`** is a guarded holder for the last submitted event: two threads on one unsynchronised
  `std::optional<sycl::event>` abort with `UR_RESULT_ERROR_INVALID_EVENT`.
- **Profiling** is opt-in: `settings().diagnostics.profiling` is the OR of `BATCHLAS_QUEUE_PROFILING`
  and `BATCHLAS_BENCH_PROFILING`.
- **`BATCHLAS_LAUNCH_BOUNDS`** (`src/sycl/kernel_attrs.hh`) is empty except on DPC++ NVPTX: on SPIR-V
  it emits `KernelAttributesINTEL`, which icpx 2026.0's CPU AOT rejects ("unsupported capability
  5892").

## Settings loader {#runtime-internals-the-settings-loader}

`src/util/settings.cc` is the one place BatchLAS calls `std::getenv` for its own knobs (exceptions:
@ref design_environment).

- **Same parser as the call site.** A bespoke parser stays at its site; the field is an `EnvValue`.
  The syevx and LOBPCG knobs keep `atoi` (they accept `0`); `BATCHLAS_KERNEL_TRACE_PATH`, then
  `BATCHLAS_TRACE_PATH`, first non-empty wins; `BATCHLAS_DUMP_BANDR1_DIR=` yields an empty root.
- **Synthesised route variables** `"BATCHLAS_" + upper(op) + "_ROUTE"` for each op in
  `RoutingSettings::ops`, so a string-literal grep misses them; an op absent from that list silently
  has no environment pin (`pin_text` in `src/select/select.cc`).
- **`configure()` closes at the first Queue** (`note_queue_constructed()` in the three root
  constructors; the flag is atomic): `*_buffer_size()` reads routing, so a later change would let two
  calls disagree on scratch size.
- **Unsafe gate.** Warns once per variable per process (`reload_settings()` runs on every
  `ScopedEnvVar`) and refuses only the unsafe direction. The pointer-check skip treats any non-empty
  value not starting with `0` as skip; it is not `env_truthy`. `-DBATCHLAS_ALLOW_UNSAFE_ENV=ON`
  compiles the gate out.
- **Warnings** are one stderr line `batchlas: <text>` from `detail::warn`; `warn_once(tag, text)`
  dedups per process and `reset_warnings()` clears the tags for tests.
- **Reload** applies only exported variables (`set_if_present`) on top of the last `configure()`.
  Starting from defaults would discard `configure()` on any `ScopedEnvVar`; ignoring the environment
  would make A/B tests compare an arm with itself (`tests/settings_tests.cc`). An exported variable
  beats `configure()`.

## Per-item info span contract {#runtime-internals-the-per-item-info-span-contract}

`src/kernels/common/info_span.hh` implements the per-item status span. Caller conventions are in
[API conventions](api-conventions.md#api-conventions-per-item-info-spans) and, for the vendor-backed
ops, @ref design_vendor_independence.

- One `int32` per item in the caller's USM, written in place, no workspace. Empty means "not
  requested" (`info_ptr()` is `nullptr`, helpers are no-ops). `0` is converged; `> 0` counts failed
  off-diagonals, or is `1` where a tier tracks only "did not converge".
- **It is an accumulator.** Producers only raise (`info_report`, `fetch_max`); the entry point the
  caller invoked clears once (`info_clear`). Two clears with a producer between them lose a failure,
  so `stedc` calls its leaves through `steqr_dispatch` (raises, does not clear) and folds per-leaf
  status in with `info_item()` (leaf \f$j\f$ belongs to item \f$j / 2^L\f$).
- **Single-writer stores** (`info_store`, `info_from_flags`) are for tiers that alone write each
  item: `bdsqr`, `syevx_lobpcg`, `syevx_filtered`, `syev_cta_fused`, `syev_jacobi_cta`,
  `gesvdj_cta`. `one_means_converged` covers `syevx_lobpcg`/`syevx_filtered`, whose flags are
  inverted.
- **Vendor status arrays.** `detail::info_target` (`src/linalg-impl.hh`) uses the caller's span or
  pool scratch, so a span only removes a pool draw. Netlib LAPACKE status goes through
  `call_backend_nh_r`; `call_backend_nh` stays separate because its `void` CBLAS users break an
  `auto` return.

## Matrix layout checks {#runtime-internals-matrix-layout-checks}

`src/matrix.cc` throws `batchlas::invalid_argument` on bad layout metadata (@ref design_matrix_model).

- **`Matrix`**: `ld >= rows`, `stride >= ld*cols`. **`MatrixView(data, rows, cols, ld, stride,
  batch)`** checks `ld >= rows` only; its argument order differs from `Matrix(rows, cols, batch, ld,
  stride)`, and the message names the intended spelling. Null data, zero extents and `stride` are not
  checked: shape-only stand-ins like `(nullptr, 0, 0, 1, 1, batch)` size workspaces. Whether a live
  site still violates `stride >= ld*cols` is open.
- **Row-major pitch is a parameter**, never inferred from `ld`. The default (pitch `cols`) is accepted
  only when packed is the only layout the item can hold; guessing read a padded `8 x 6` buffer at the
  wrong pitch, 42 of 48 elements wrong and all in bounds.
- **`triangularize(Uplo::Upper)` keeps the upper triangle** (extracting `R` from `geqrf`).
- **Dense to CSR** zeroes the whole offsets array, so `row_offsets()[b*(rows+1)]` is defined; `max_nnz`
  is the batch maximum.

## Host BLAS dgemm health check {#runtime-internals-the-host-blas-dgemm-health-check}

OpenBLAS 0.3.20's Cooperlake `dgemm` is wrong by O(1) to O(100) on recent Intel parts (Known
Pitfalls in @ref dev_agent_guide). `cmake/BatchLASBlasHealthCheck.cmake` records the repairing
`OPENBLAS_CORETYPE` as `BATCHLAS_REQUIRED_OPENBLAS_CORETYPE`; `src/backends/host_blas_health.hh`
probes a strided sample at \f$n \in \{128, 200, 512\}\f$ (64 and 256 happen to be correct) on first
double use. It never calls `setenv()`: OpenBLAS reads the variable in a library constructor.
`BATCHLAS_BLAS_HEALTH=off|warn|error` (default `warn`); `off` is refused by the unsafe gate.

## Build-time structure of the library {#runtime-internals-build-time-structure}

`src/CMakeLists.txt` cuts the library into OBJECT libraries: `batchlas_core_obj`,
`batchlas_selection_obj` (selection, every `src/ops/<op>/<op>.cc` but the level-3 seven, the tuned
tables), `batchlas_extensions_*_obj` (native drivers, grouped by device-link cluster in
`src/ops/CMakeLists.txt`), `batchlas_sycl_obj` (native BLAS kernels), the vendor
`batchlas_backends*_obj`, `batchlas_util_obj` and `batchlas_extra_obj`.

- **Selection is compiled unconditionally**; vendor calls are `if constexpr`-gated, so
  `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` is a complete library. The CUDA vendor TUs are all-or-nothing
  (`BATCHLAS_HAS_CUDA_VENDOR_TUS`) because an empty OBJECT library is a CMake error.
- **`BATCHLAS_NO_CPU_OBJECT_LIBS`** (cta, posv, gesv, level3) build GPU-only; nothing warns if a
  source moves out of them and silently gains a CPU image.
- **A source sits with the sources whose device symbols it calls** (the `.so` is the device-link
  unit): split `potrf_blocked.cc`/`potrf_cta.cc` fail with `ptxas fatal: Unresolved extern function`
  once a shared helper is `SYCL_EXTERNAL`.
- **Tiny-kernel TUs** build with `-mllvm -pragma-unroll-threshold=262144` under DPC++, else
  `#pragma unroll` is silently declined; AdaptiveCpp uses `BATCHLAS_UNROLL_FULL`.
- **Split by default, monolithic for release** (`BATCHLAS_MONOLITHIC_LIBRARY`, required on macOS). The
  merged device link took 575 s of an 11 m 38 s `-j16` build on the RTX 4090 box. Hidden visibility
  is monolithic-only: split, inline process-wide state would get one copy per `.so`.

## Embedded tuned tables {#embedded-tuned-tables}

`tuned/*.txt` (@ref design_tuning_data) is globbed with `CONFIGURE_DEPENDS` and written by
`cmake/BatchLASEmbedTables.cmake` into `<build>/generated/select/tuned_tables.cc`
(`select::embedded_tables()`), rewritten only when a table changes.

- Each table is one `char` array sized by `sizeof`: a `string_view` from a literal runs a constexpr
  `strlen`, and the tables (about 50 MB) exceed clang's constexpr step limit. DPC++ has no `#embed`.
  A table containing the raw-string delimiter `batchlas_tbl` is a configure error.
- Tables are parsed lazily per `(op, dtype)` and cached with the `BATCHLAS_TUNED_DIR` value and a
  generation counter. The cache is deliberately leaked: `choose()` can run from static destructors
  and the coverage writer from `atexit`. Tests swap tables with `select::testing::set_builtin_tables`
  / `use_embedded_tables`, each bumping the generation.

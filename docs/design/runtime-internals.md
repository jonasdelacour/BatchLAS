# Runtime internals: Queue, workspace arena, settings, matrices and instantiation {#design_runtime_internals}

> **Covers:** the private runtime under `src/`: the `Queue` implementation (`src/queue.hh`,
> `src/util/queue-impl.cc`), the per-queue workspace arena, the settings loader
> (`src/util/settings.cc`), the per-item `info` span, the layout checks in `src/matrix.cc`,
> the explicit-instantiation macros (`src/util/template-instantiations.hh`), the
> symbol-visibility rules that private headers must follow, how the library is cut into object
> libraries at build time, and how the tuned selection tables get into the binary
> (`cmake/BatchLASEmbedTables.cmake`).
> **Status:** current. Assembled 2026-09-30 from the source comments it replaces; revised
> 2026-10-06 for flat kernel selection (`src/select/`, `src/ops/`), which replaced the
> `src/dispatch/` layer. Cited sections are pointed at from the code by `evidence:` pointers.

This page is the design record for code that callers never see but every entry point runs
through. The caller-facing contracts are in the public headers (`<batchlas/settings.hh>`,
`<batchlas/util/workspace.hh>`, `<batchlas/util/sycl-device-queue.hh>`); what is here is why the
implementation is shaped the way it is, and the traps an editor of these files has to know.

## Runtime internals: symbol visibility for private headers

Under `BATCHLAS_MONOLITHIC_LIBRARY` the object libraries are compiled `-fvisibility=hidden`, and
`BATCHLAS_API` (from the generated `<batchlas/export.hh>`) is what pulls a declaration back to
default visibility. How the macro is generated and where the public headers must place it is
@ref design_symbol_visibility; private headers need a second rule set on top of that.

**`BATCHLAS_INTERNAL_API` (`src/util/internal-api.hh`).** A measurable slice of what the library
exports is needed only by the in-tree tests, not by any consumer: **199 of the 1,083 symbols the
63 test binaries resolve out of `libbatchlas*.so` belong to 55 entities that are declared in no
public header at all.** They are reachable because eight-plus test TUs include a private header by
relative path (`tests/gemm_tests.cc` includes `../src/sycl/gemm_kernels.hh`,
`tests/getrf_tests.cc` includes `../src/extensions/getrf_native.hh`, and so on) and then link the
shared library. Hidden visibility breaks those links.

Spelling them `BATCHLAS_API` would fix the link and re-widen the ABI the hidden-visibility work
exists to bound: every one of those 55 would become a name the library promises.
`BATCHLAS_INTERNAL_API` expands to the same thing on ELF and carries the opposite promise. It lives
in a header that is never installed precisely so that the distinction survives: a symbol marked
with it cannot appear in a consumer's translation unit, because the consumer cannot reach the
declaration that names it.

*Exit condition.* If the test suite stops including private headers, delete the macro. Relinking
those test targets against the object libraries instead of the `.so` is the better fix: it removes
the symbols from the dynamic table entirely rather than exporting them under a different name.

**Vague-linkage process-wide state.** Inline variables and inline statics in a private header
(`src/util/kernel-trace.hh`'s `g_enabled` and record vectors, `QueueImpl::device_arrays` in
`src/queue.hh`) are emitted by every TU that includes the header, and the linker is expected to
fold them into **one instance per process**. Under `-fvisibility=hidden` the library's copies
become hidden and stop participating in the fold, while a test TU, which is not compiled with
hidden visibility and includes the header directly (`tests/stedc_tests.cc`,
`tests/device_blas_tests.cc`), keeps a default-visibility copy of its own. The result is two
copies of the state, silently: traces written inside the library are invisible to the test's
flush, and a hidden `device_arrays` gives the process two SYCL device caches, a duplicated-state
bug no undefined reference ever points at. `BATCHLAS_INTERNAL_API` on those declarations keeps the
fold. The same applies to any process-wide state that lives in a private header.

**`BATCHLAS_QUEUE_EXPORTED_INLINE` (`src/queue.hh`).** The out-of-line `Queue`/`Event` members
defined in the private header must reach consumers that only see the declaration in the installed
public header, so they must be emitted into `libbatchlas` rather than dropped as unreferenced.
`[[gnu::used]]` does that; plain `inline` does not (verified: nothing in-tree calls them, so
without it the symbol is absent from every object file). It is not applied in the SYCL device
pass, which has no business emitting host-only queue plumbing.

**`template struct BATCHLAS_API UnifiedVector<std::byte>;` (`src/util/sycl-util-impl.cc`).** An
instantiation takes the **minimum** of the template's visibility and its arguments'. `std::byte`
is `enum class byte : unsigned char`, declared by libstdc++ with no visibility attribute, so under
`-fvisibility=hidden` it is hidden, and the class-level `BATCHLAS_API` on `UnifiedVector` is not
enough for that one specialisation. Unlike `Backend`, `MatrixFormat` and `BinaryOp`, the argument
cannot be annotated because it is not ours. The sibling instantiations need nothing (their
arguments are builtins or our own class types). Removing the attribute silently un-exports
`resize`, both constructors and the destructor; the only symptom is a consumer link error.

**Include spelling.** `src/util/` is a private directory that is never installed and is on no
`-I` line; it holds the only headers in the tree spelled `util/...`, reached exclusively by quoted
relative includes. The public tree moved to `include/batchlas/util/` (spelled
`<batchlas/util/...>`) so that no angle-form `<util/...>` exists anywhere. Do not add
`-I${PROJECT_SOURCE_DIR}/src` to a target and do not convert those includes to `<>`.

## Runtime internals: namespace placement of out-of-line definitions

`src/util/queue-impl.cc`, `src/util/sycl-util-impl.cc` and the definitions in `src/queue.hh` sit
in `namespace batchlas` to the end of the file, and have to. `Queue`, `Event`, `Device`,
`QueueImpl`, `EventImpl`, `UnifiedVector` and `Span` are declared inside `batchlas`
(`<batchlas/util/sycl-device-queue.hh>`); the compatibility shim at the bottom of that header only
introduces the *names* into the global namespace, which is enough to spell a type but not to
define its members. Three separate rules force the placement, and only the first fails loudly:

- out-of-line members must be defined in the namespace their class was declared in (defining
  `QueueImpl` at global scope would define an unrelated type and leave `Queue::impl_` pointing at
  an incomplete one);
- explicit instantiations that name their template by an unqualified-id must be in that
  namespace too;
- the `operator<<` for `ReferenceWrapper`/`UnifiedVector`/`Span` are declared **only** as in-class
  friend templates, so each is a member of the namespace enclosing the befriending class. Left at
  global scope they would silently define an unrelated `::operator<<` that nothing can call, and
  the explicit instantiations would instantiate the wrong template: an undefined reference in a
  consumer, with nothing failing in the library build.

The `std::array` `operator<<` in `sycl-util-impl.cc` came along for the ride. It is used only
from inside that file, where ordinary lookup finds it; it is no longer reachable by ADL on
`std::array`, so it must not be relied on from another TU.

Two deliberate exceptions stay at global scope. The anonymous namespace in `queue-impl.cc` holds
SYCL kernel-name tags, and moving it would rename every kernel mangled from them for no benefit
(ordinary lookup still finds them). `batchlas_throw_queue_wrong_thread` in `src/queue.hh` already
carries the prefix, has no caller outside that header, and is found from
`QueueThreadOwner::check` by ordinary lookup. The header is private, so the public header's
compatibility shim does not reach it at all.

## Runtime internals: explicit instantiation macros

`src/util/template-instantiations.hh` defines the machinery every backend and entry-point TU uses
to emit its symbols.

**`BATCHLAS_INSTANTIATE(SIG, FN, ...)`** expands to `template SIG FN<...>;`, for example
`BATCHLAS_INSTANTIATE(sig::gemm<float>, gemm, Backend::NETLIB, float)` gives
`template Event gemm<Backend::NETLIB, float>(Queue&, ...);`. `SIG` must be a *function type*: the
`sig` namespaces next to each public declaration in `include/batchlas/blas/functions/` hold them.
Naming the type is what makes it work: every public entry point is an overload set (the
`MatrixView` primary plus an inline `Matrix`-taking forwarder with an identical template parameter
list), so the tempting `template decltype(FN<Args...>) FN<Args...>;` is ill-formed ("reference to
overloaded function could not be resolved"). Because the alias lives beside the declaration, a
signature change is one header edit instead of one per backend: `gemm`'s signature alone used to be
restated in `netlib_lapack.cc`, `cublas.cc`, `rocblas.cc` and `mkl.cc`. Function types cannot
carry default arguments, so write every alias with every parameter spelled out and no `= default`.
`BATCHLAS_COMMA` smuggles a second template argument through macro splitting
(`sig::spmm<fp BATCHLAS_COMMA F>`).

**No `BATCHLAS_API` in the macro, deliberately.** Each of the ~1,300 definitions the macro emits
needs default visibility to reach a consumer, and the annotation lives on the primary
*declaration* in the public header instead (`template <Backend Back, typename T> BATCHLAS_API
Event gemm(...)` and its ~178 siblings). The macro is the wrong place on three counts:

1. *It does not portably compile.* An attribute on an explicit instantiation is not honoured
   consistently: measured, g++ 13 `-std=c++20` rejects
   `template class __attribute__((visibility("default"))) F<double,1>;` with "'F' is not a class
   template", where clang accepts it. There is also no natural slot in this expansion: the
   decl-specifier-seq is a type alias naming a function type, not a return type followed by a
   declarator.
2. *It is unnecessary, but not for the reason first written in the source.* The earlier comment
   said both compilers propagate a visibility attribute from the primary template to every
   specialisation. That is half the rule: an instantiation gets the **minimum** of the template's
   visibility and its template arguments'. These are all instantiated on `Backend`, so an
   unannotated enum would hold every one of them at hidden whatever the declaration says.
   `Backend` and `MatrixFormat` therefore carry `BATCHLAS_API` themselves (the note at the top of
   `include/batchlas/blas/enums.hh` records the measurement), and given that, annotating the
   declaration covers the definition the macro emits.
3. *Only the declaration reaches the consumer.* A consumer's TU sees the header, never this file,
   and `BATCHLAS_API` expands to `__attribute__((visibility("default")))` on both the build and
   the consume side on ELF, so one token on one line does both jobs. Annotating the instantiation
   would export the definition and leave the consumer's reference hidden.

The `sig::` aliases need nothing: they are types, not entities.

**Op-token drivers.** The signature alias and its function share a spelling, so one op token
drives both halves: `BATCHLAS_INSTANTIATE_OP(Backend::CUDA, (float), potrf)` gives
`template sig::potrf<float> potrf<Backend::CUDA, float>;`. The type arrives parenthesised so the
macros compose with the `BATCHLAS_FOR_EACH_*_TYPE_1` drivers, which hand out `(float)`,
`(double)`, `(std::complex<float>)`, `(std::complex<double>)`; `BATCHLAS_UNPAREN` strips the
parentheses on both uses. This replaced the per-TU `#define X_INSTANTIATE` / aggregate-list /
`#undef` triple: a TU writes one line per op and the driver supplies the type.
`_BACKEND_OP` is for the `backend::`-qualified vendor entry points (`syev_vendor`,
`gesvd_vendor`, `ormqr_vendor`, ...), whose aliases still live in `sig`, not `backend::sig`.
`_FORMAT_OP` carries the second template argument of the sparse ops. There is no
`_BACKEND_FORMAT_OP`, so the sparse vendor TUs (`netlib_lapack.cc`, `rocsparse.cc`) use a local
shim or a raw `BATCHLAS_INSTANTIATE` with `BATCHLAS_COMMA` and `BATCHLAS_UNPAREN`.

Scalar domains differ per op (`symm`/`syrk`/`syr2k` are real-only, `hemm`/`herk`/`her2k`
complex-only), so there is a driver per domain rather than one blanket loop.

**The backend member of the family.** Every backend-parameterised entry point is instantiated once
per backend the build compiled. Each `.cc` used to carry three hand-written
`#if BATCHLAS_HAS_*_BACKEND` blocks plus, usually, a one-off `X_INSTANTIATE_FOR_BACKEND` binder:
the same eleven lines in more than forty TUs. `BATCHLAS_FOR_EACH_ENABLED_BACKEND` replaces them.
Its arms are pre-guarded (`BATCHLAS_IF_CUDA`/`_ROCM` from `backend_config.h`, `BATCHLAS_IF_HOST`
supplied here) because the preprocessor cannot emit an `#if` from a macro expansion: a backend
that was not compiled has to expand to nothing at all. `BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS`
and `_SCALAR_ALL_BACKENDS` combine the backend and type loops, so the binder, the three
`#if`/`#endif` pairs and the three invocations become one line.

*MKL is deliberately absent.* The files that carry an MKL arm (`ritz_values.cc`, `symm.cc`,
`syrk.cc`, `syr2k.cc`, `trmm.cc`) do not instantiate the same set as the {CUDA, ROCM, HOST}
triple, and `steqr_legacy.cc` has no ROCM arm; folding any of them into a blanket loop would
silently add or drop exported symbols. Those files keep their hand-written blocks.

## Runtime internals: vendor TUs instantiate only vendor symbols

Every explicit-instantiation table in a vendor TU (`cusolver.cc`, `rocblas.cc`, `rocsolver.cc`,
`rocsparse.cc`, `netlib_lapack.cc`, and the cuBLAS/cuSPARSE ones) names only
`backend::`-qualified `*_vendor` symbols, and that is the invariant rather than an oversight.
The public entry points and their `*_buffer_size` queries are instantiated in the op files of
flat kernel selection, `src/ops/<op>/<op>.cc` (the level-3 family shares one TU under
`src/ops/`), which are compiled into `batchlas_dispatch_obj` in every build and instantiate
keyed on the **device family** rather than on any vendor library: the vendor call inside is
`if constexpr`-gated, so the public symbol exists even when the library does not. A public-op
row in a vendor TU would be a duplicate symbol against those, and a vendor-free build would lose
the op. The convention is per op, not per list: an op added under `src/ops/` follows it with no
change here. Design record: [flat kernel selection](flat-kernel-selection.md).

*History.* WP0b first moved the public entry points out of the vendor TUs into
`src/dispatch/entry_points/`, keyed the same way
([the entry-point facade](vendor-independence.md#the-entry-point-facade)); the phase 5 rip of flat
kernel selection deleted that directory and the op files took over its role.

- `sig::trsm_vendor` is deliberately **not** an alias of `sig::trsm`: the vendor order puts
  `alpha` last (as cuBLAS defines it), the public `trsm` takes it third. The two orders coexisted
  while each TU declared its own public `trsm`; one declaration now serves every backend, so the
  vendor forms have to agree with each other.
- `gesvd_vendor` is declared in `functions/gesvd.hh` and defined only by the backends that have
  one. `rocsolver.cc` defines a throwing `gesvd_vendor` stub because that header is
  declaration-only (it used to define a generic throwing template, which is what blocked a
  cuSOLVER implementation); without the stub a ROCM build fails to link rather than failing at
  the call. (Until phase 5 gesvd had no entry-point TU at all; its public forms now live in
  `src/ops/gesvd/gesvd.cc` like every other op's.)
- The netlib `gemm` moved into `backend` under its vendor name instead of being deleted: unlike
  `cublas.cc` and `rocblas.cc`, that TU had no separate `gemm_vendor` to forward to; its public
  `gemm` *was* the CBLAS call. Its `gesvd_vendor` was moved verbatim from `functions/gesvd.hh`,
  including the synchronous `ctx.wait()` (LAPACKE `?gesvd` needs `A` on the host, and this path is
  the reference implementation, not a fast one).
- `rocblas.cc` carries no `symm`/`hemm`/`herk`/`her2k` wrapper at all, which is the omission the
  ROCM arm of the level-3 op TU under `src/ops/` mirrors exactly: it instantiates only the
  level-3 ops rocBLAS implements.
- ROCm 6.3 changed `rocblas_[sdcz]trmm` to the 14-argument out-of-place form
  (`..., A, lda, B, ldb, C, ldc`, `B` input, `C` output); the old 16-argument variant with a
  duplicate output pair was removed. The `rocblas_float_complex`/`rocblas_double_complex` casts
  the complex arms used to spell by hand are what `ptr_convert` already emits for
  `BackendLibrary::ROCBLAS`, so all four arms are one call.

**The unbatched vendor level-3 host loop (`src/backends/batch_launch.hh`).** cuBLAS and rocBLAS
spell `symm`/`hemm`/`syrk`/`herk`/`syr2k`/`her2k`/`trmm` only for a single matrix (no strided- or
pointer-batched form), so a batch is a host loop issuing one launch per member. Ten call sites
across the two backends wrote that loop by hand, all with the same single-item short circuit: a
batch of one is launched against the caller's own view rather than `view[0]`, so the common case
does not pay for a sub-view. `for_each_batch_item` is that loop. `<= 1` rather than `== 1` is
carried over verbatim: an empty view still issues exactly one launch, whose `m` or `n` the vendor
sees as zero, and tightening it would silently change what an empty batch does, a decision for the
callers. The callable comes first because the trailing pack varies (`syrk`/`herk` pass two views,
everything else three); members are passed in view order and the count is taken from the first
view, as the hand-written loops took it from `A`.

## Runtime internals: the per-queue workspace arena

`WorkspaceArena` (`src/queue.hh`) is the per-queue scratch memory behind `Queue::workspace()` and
`WorkspaceLease`. The caller-facing rules are in `<batchlas/util/workspace.hh>`, and the lease
design (nesting, reassignment, out-of-order queues, sizing) is @ref design_workspace; this section
records the implementation choices inside the arena and their traps.

**Append-only blocks.** Blocks are never reallocated or moved, only appended to, because a lease
that is still live must keep its pointer: an inner borrow that does not fit in the current block
opens a new one. Released bytes are rewound, not freed, so the steady state is one allocation per
distinct high-water mark rather than one per call.

**Release order is enforced, not documented.** The rewind is what makes release order matter.
Documenting the LIFO invariant was not enough, because `WorkspaceLease::release()` exists
precisely so a caller can hand bytes back early, and doing that to anything but the innermost
lease used to re-serve memory a live lease was still pointing at: the next borrow silently aliased
it, a wrong answer rather than a diagnosable failure. Now only the innermost outstanding loan may
move the cursor. An out-of-order return is recorded in place and its bytes are reclaimed later,
when the loans stacked on top of it come back. The arena holds more than it needs for a while,
which is a cost, not a correctness problem.

- The live-loan stack (one entry per outstanding loan, innermost last) exists so `release()` can
  tell "this is the innermost loan" from "this loan has live leases stacked on top of it"; a
  sequence number alone cannot answer the second question once more than one loan has been
  returned out of order. It is a `std::vector` because nesting depth is a property of the call
  graph; it keeps its capacity, so after the first few leases the LIFO path is a
  `push_back`/`pop_back` into a warm buffer and allocates nothing. A returned entry is kept only
  while something above it is live, so a non-empty stack always means at least one lease is
  genuinely outstanding.
- `diagnose_out_of_order` is false for exactly one caller: `WorkspaceLease`'s move-assignment,
  where the right-hand lease is necessarily acquired before the left-hand one is released.
  Asserting there would abort legal code (`ws = q.workspace(n);`) in every non-`NDEBUG` build.
  Everywhere else the assert is the point: an out-of-order return from a scope-bound lease means
  the scopes are not nested the way the caller thinks.

**Draining before reuse on an out-of-order queue** (`WorkspaceLease::release` in
`queue-impl.cc`). Every release funnels through one function, so the out-of-order-queue wait lives
there rather than at the call sites: reclaiming hands the bytes to the next borrow, and on an
out-of-order queue nothing stops the runtime running that borrow's kernels alongside the ones still
reading ours. An in-order queue orders them, so it must not pay. The wait is conditioned on the
release actually reclaiming (`release_reclaims`): a return that lands under a live lease only flips
a flag, the bytes are not re-servable until the loans above come back, and the release that pops
them drains then. Before this was scoped, every convenience overload holding a scope-bound lease
drained the device twice on a nested call. The release path cannot throw
(`~WorkspaceLease` calls it and move-assignment is `noexcept`); a failure the runtime surfaces at
that wait is not this lease's and is reported again at the caller's next `wait`/`wait_and_throw`.

*What the drain does not order against:* work submitted to a **derived** in-order queue (`gesvd`
and `iluk` build one from `ctx` and run kernels there while the lease belongs to `ctx`). Waiting on
`ctx` does not wait on that; the derived queue's destructor does. See `release()` in
`<batchlas/util/workspace.hh>`.

**Freeing.** `trim()` refuses while any lease is outstanding (the blocks are what those leases
point at) and reports that rather than trimming partially. Both `trim()` and `~QueueImpl` drain the
queue before freeing: a released lease only says the *caller* is finished with the bytes, not that
the kernels it enqueued over them have finished, and freeing shared USM under in-flight work is a
use-after-free that usually only shows up under load. `~QueueImpl` asserts that no lease is live
rather than defending: `WorkspaceLease` holds a `Queue*`, so releasing it after the arena is gone
would hand a stale block/offset to whatever arena the `Queue` has next. Scope-bound leases make
that impossible for `~Queue`, but `Queue`'s move-assignment also runs `~QueueImpl` (it replaces
`impl_`), and nothing forces the leases to be gone first; a `Queue` moved out from under a live
lease is a caller bug, and silently draining and freeing would hide it.

**`Queue::operator=(Queue&&)` is written out on purpose.** Move-assignment destroys the
destination's `QueueImpl` without running `~Queue`, so per-queue state has to be torn down on that
path too. It is, because the arena is a member of `QueueImpl`. Keeping the operator written out
records the requirement and gives it somewhere to live if state is ever added outside `QueueImpl`;
storing such state in a side table keyed on `impl_` would be a bug, since a later heap reuse of
the same address would inherit the entry.

## Runtime internals: Queue thread ownership and the last-event holder

A `Queue` is single-threaded by contract; the reasoning is on `struct Queue` in
`<batchlas/util/sycl-device-queue.hh>`. The enforcement (`QueueThreadOwner`) records the thread
that built the queue and compares against it on the paths that mutate state shared across a
queue's calls: a TLS load and a word compare, on paths that are about to talk to a driver. It is
deliberately **not a mutex**: the arena is a bump allocator whose cursor rewinds on release, so
serialising the calls would still interleave two threads' leases within one block and still
corrupt them. A lock would hide the design constraint rather than satisfy it.

The arena's owner is checked where bytes are handed out and where blocks are freed (`acquire()`,
`trim()`), not in `release()`: every release comes from a lease, every lease from an `acquire`
that was checked, and `~WorkspaceLease` is `noexcept`, so a throw there would terminate instead of
diagnosing.

**`LastEvent`.** The queue tracks the last event submitted through its wrappers, which gives a
cheap `get_event()` on an in-order queue. It is a guarded holder rather than a bare
`std::optional<sycl::event>` because the unsynchronised optional is a race of its own,
independent of the arena: two threads submitting on one `Queue` tear it and abort inside the SYCL
runtime with `UR_RESULT_ERROR_INVALID_EVENT`, even when both callers supply their own workspace.
Putting the check on the member covers every reader and writer (the submit wrappers and
`Queue::enqueue`/`get_event`/`create_event_after_external_work`) without each having to remember.
Its interface is the subset of `std::optional` those callers use.

**Profiling.** Queue profiling is opt-in to avoid overhead outside benchmarks. Kernel trace implies
profiling; benchmarks can enable profiling without tracing. `settings().diagnostics.profiling` is
the OR of `BATCHLAS_QUEUE_PROFILING` and `BATCHLAS_BENCH_PROFILING` (two names, one field, folded
in `settings.cc`); reading the field rather than the environment is what puts both knobs under
`configure()` and under `ScopedEnvVar`'s reload.

**`BATCHLAS_LAUNCH_BOUNDS`** expands to SYCL kernel attributes on NVPTX only and to nothing
elsewhere (an empty attribute-list entry is legal): the values were tuned on NVIDIA, and on SPIR-V
the attributes emit `KernelAttributesINTEL`, which icpx 2026.0's CPU AOT compiler rejects
("unsupported capability 5892"), failing the link of any `spir64_x86_64` build.

## Runtime internals: the settings loader

`src/util/settings.cc` is **the one place BatchLAS calls `std::getenv`** for its own knobs. Why the
`Settings` snapshot exists is on `<batchlas/settings.hh>`; this section is about the load.

**Same parser as the call site.** Each field is produced by the same parser its call site used
before the migration, so that migrating a site changed where the string comes from and never what
it means. Where a site had a bespoke parser the field is an `EnvValue` and the parser stays at the
site; the loader only captures the string. Assignments follow the field order of the header so the
two read side by side. Consequences that look like inconsistencies but are preserved on purpose:
`env_positive_int_or` and the bare-`atoi` form agree except for an integer literal too large for
`int` (undefined for `atoi`, the fallback for stoi-in-a-try: a change in the safe direction); the
syevx/LOBPCG knobs keep their `atoi` form because two accept `0` as a meaning and one accepts a
value below its own default; `BATCHLAS_KERNEL_TRACE_PATH` then `BATCHLAS_TRACE_PATH`, first
**non-empty** wins, while `BATCHLAS_DUMP_BANDR1_DIR=` (set, empty) yields an empty root, not the
default.

**Synthesised route variables.** The routing variable names are built as
`"BATCHLAS_" + upper(op) + "_ROUTE"` for every op in `RoutingSettings::ops`
(`<batchlas/settings.hh>`), which is why a grep for `BATCHLAS_*` string literals misses every one
of them: no literal exists anywhere in the tree. The loader only captures the strings; the
selection layer reads them as pins (`pin_text` in `src/select/select.cc`, after any `ScopedPin`),
spelling the variable name the same way for its diagnostics. An op that reads a pin has to be in
that list; an op missing from it silently has no environment pin. What a value means (an
unknown value throws, `native` and `vendor` fall back to `auto` with a warning when nothing in
that class can run) is documented on `RoutingSettings` and in
[flat kernel selection](flat-kernel-selection.md). The old `parse_route_env`, deleted with the
route tables, built the same names.

**Initialisation.** The mutable snapshot and the last `configure()` value are function-local
statics, initialised on first use rather than in static-initialisation order, because selection
coverage reads its variable from a namespace-scope dynamic initialiser (`src/select/coverage.cc`
runs at static init, before `main` and so before any `configure()`; `settings()` is documented
safe to call there, and that is the site that requires it; before phase 5 the same site was
`src/dispatch/coverage.cc`). The only flag that can race is
"a Queue has been constructed", which is atomic because a Queue may be built on another thread.

**`configure()` closes at the first Queue.** `batchlas::configure()` is permitted only until the
first `Queue` exists. Routing and geometry settings are read by `*_buffer_size()` queries as well
as by the matching solve, so a change taken after work has started lets two calls in one process
disagree about how much scratch a solve needs. The close is in the three root `Queue`
constructors; the delegating constructor and the defaulted move constructor are covered by them.
`note_queue_constructed()` is idempotent and `noexcept`.

**The unsafe gate.** One warning per **variable** per process, not per load: `reload_settings()`
runs once per `ScopedEnvVar` construction and destruction, so a per-load warning would print
thousands of lines in the test suite, and the flags are never reset. Each field is first read
exactly as its call site read it and only then gated, so a warning names a variable that was
actually set. The gate refuses the unsafe **direction**, not every non-default value
(`BlasHealth::Error` is stricter than the default and passes). The pointer-check skip treats any
non-empty value not starting with `0` as "skip", including `false`, `off` and `no`; it is not
`env_truthy`. The gate is compiled out when `BATCHLAS_ALLOW_UNSAFE_ENV` is ON.
`BATCHLAS_ALLOW_UNSAFE_ENV` is set through `cmake/backend_config.h.in` and defaulted to the safe
reading in `settings.cc` so the file still compiles against an older generated header.

**Reload semantics.** A reload re-reads the environment **on top of** what `configure()` last set,
not on top of the defaults. Each half is a defect the other way round:

- Starting from the defaults discarded `configure()` entirely, and a `ScopedEnvVar` on a completely
  unrelated variable, built anywhere in an embedding application's own harness, was enough to do
  it, silently. That is the ambient-state defect the settings work removed.
- Ignoring the environment once `configure()` had been called made every `ScopedEnvVar` in the
  process a no-op, which turns an A/B test into two runs of the same arm that agree by
  construction. Twelve guards in the tree had already failed that way; `tests/settings_tests.cc`'s
  regression case (c) is exactly it, and it calls `configure()` in case (a) first, so the whole
  binary would have gone quietly green.

So `configure()` is the base and an explicitly set variable still wins over it. A caller who wants
a knob pinned against the environment sets it in the `Settings` and does not export it.

## Runtime internals: the per-item info span contract

`src/extensions/info_span.hh` implements the per-item convergence-status span; the public
declarations in `blas/extensions.hh` point at it, and the caller-facing convention is in
[the API conventions](api-conventions.md#api-conventions-per-item-info-spans).

- `info` is one `int32` per batch item in the **caller's** memory. It is USM, so a kernel writes it
  in place: there is never a copy back, no tier needs workspace for it, an empty span costs
  nothing, and no `*_buffer_size()` result changes.
- `0` means converged. A value `> 0` is LAPACK-like: the number of off-diagonal elements that
  failed to converge, or `1` where a tier tracks only "did not converge".
- An **empty** span means "not requested": `info_ptr()` yields `nullptr` and every helper is a
  no-op. A too-short span is ignored rather than rejected, because the deducing overloads reject
  short spans up front (`detail::require_info_span` in `blas/options.hh`) and this layer is the
  library's inner-loop spelling, which must not pay a throw path per call.
- **It is an accumulator, not an output register.** Producers only raise a value (`info_report`
  uses `fetch_max`); the span is zeroed exactly once, by the entry point the caller invoked, via
  `info_clear`.

That last rule is the load-bearing one. Nested clears are harmless (`syev -> syev_blocked ->
stedc` clears the same span three times) because no producer runs between them. Two clears with a
producer **between** them is the bug the rule prevents, and it is why `stedc`'s merge kernels
accumulate with `fetch_max`: one `stedc` call runs many merges over the same items, and the answer
wanted is "did any of them fail", not "did the last one". It is also why `stedc` does **not** hand
its `info` to its leaf `steqr` calls: both `stedc` drivers would clear it again underneath
themselves. The recursive driver solves the two halves of the same items with two leaf solves, and
the level-synchronous one solves `leaves * batch` problems in one `steqr` call whose batch axis is
longer than `info`; `info_item()` maps that finer kernel index back to the caller's item, which is
what keeps `info` a plain span of length `batch` rather than a per-level span whose length would
depend on an internal tree shape.

**Single-writer stores.** `info_store` and `info_from_flags` store rather than raise, for a tier
that is the only writer for each item (one CTA kernel per problem; `bdsqr`, `syevx_lobpcg`,
`syevx_filtered`). A store sets a converged item to `0` without a separate clear, which matters
because a clear plus a raising kernel is two submissions, and nothing guarantees the caller's queue
is in order, so the pair can race where a single store cannot. Never use them where several
producers share the span. `info_from_flags`'s `one_means_converged` exists because
`syevx_lobpcg` and `syevx_filtered` write `1` for **converged**, the opposite of LAPACK: copying
either verbatim would report failure on every healthy item and success on every broken one, and a
one-directional test would not catch it. The caller keeps `flags` alive until the kernel has run;
a pool draw outlives the call, a local `UnifiedVector` does not (which is why `syevx_filtered`
waits).

**Vendor status arrays (issue #73).** Every vendor factorisation already allocated a per-item
status array because the vendor call demands one, and before issue #73 it was pool scratch nobody
read; netlib's `potrf`/`getrf`/`getri`/`syev` discarded the LAPACKE return value, so an indefinite
or singular item came back looking exactly like a factorised one. `detail::info_target`
(`src/linalg-impl.hh`) now routes the caller's span in when one is supplied and falls back to the
pool otherwise. Supplying a span only ever **removes** a pool draw, so every `*_vendor_buffer_size`
keeps its unconditional `int` term and stays correct in both modes: a workspace sized without
`info` is never too small for a call made with it. Where the allocation order shifts (a skipped
draw moves later allocations earlier) the sized total becomes an over-estimate, which is the safe
direction. `call_backend_nh_r` exists beside `call_backend_nh` for the same issue: the `_nh` form
discards the return value and is also used for `void` CBLAS calls, and an `auto` deduced from four
branches of which one is `void` is ill-formed once a caller uses the result, so the two stay
apart.

## Runtime internals: matrix layout checks

`src/matrix.cc` validates layout metadata in constructors and conversions. Three of those checks
were added after a silent wrong answer.

**`Matrix` allocating constructor: `ld >= rows`.** `ld` is the distance between successive
columns, so it is at least `rows`, and batch items must be far enough apart to hold a full item.
This used to go unchecked in the allocating constructor (the from-data constructors always checked
it), so an `ld < rows` produced a matrix whose own accessors read into the next column.

**`MatrixView(data, rows, cols, ld, stride, batch_size)`.** This constructor used to perform no
validation (an init list with an empty body). That made an argument-order trap silent: `Matrix`
takes `(rows, cols, batch_size, ld, stride)` but `MatrixView` takes
`(data, rows, cols, ld, stride, batch_size)`, so a caller who learned the order from
`Matrix A(n, n, batch)` writes `MatrixView<float> V(p, n, n, batch)` and gets `ld = batch`,
`stride = batch * n`, `batch_size = 1`: a wrongly strided view with no throw and plausible
numbers. `ld_ < rows` catches exactly that whenever `batch < n`, and the message names the
intended spelling. Deliberately **not** checked:

- a null `data` or `rows == 0`/`cols == 0`: roughly 37 in-repo sites build
  `MatrixView<T, Dense>(nullptr, ...)` as a shape-only stand-in for a workspace query, several with
  the shape `(nullptr, 0, 0, 1, 1, batch)`;
- `stride_ >= ld_ * cols`, which the `Matrix` constructor does check. Two live call sites violate
  it: `src/extensions/ortho.cc`'s transposed CGS view `(A.data_ptr(), i, m, m, A.stride(), batch)`
  (A is `k x m`, `k <= m`, so `A.stride()` is `k*m` against `m*m`; see
  [known defect 1](known-defects.md#defect-1-orthos-transposed-arm-builds-a-view-that-does-not-describe-the-memory))
  and `syevx_lobpcg`'s workspace-sizing dummy `(p, 3*bv, 3*bv, 3*bv, 3*bv*bv, batch)`. A throw here
  would take out `ortho` and `syevx_lobpcg_buffer_size`, and the check adds nothing against the
  trap: for `V(p, n, n, batch)` the resolved stride equals `ld * cols` exactly.

**Row-major pitch is a parameter, not inferred.** `to_column_major` used to infer the pitch
between rows of row-major data from the stored `ld` (`ld > rows ? ld : cols`). That cannot work:
`ld` is the distance between *columns*, so for `rows > cols` a legal pitch `p` in
`cols < p <= rows` is indistinguishable from packed and was silently read at pitch `cols`; and an
inferred pitch `ld > cols` is not even self-consistent, since the allocation is sized `ld * cols`
and the read runs off the end of the buffer. The default (`cols`, packed) is now accepted only
where packed is the **only** layout the matrix can hold: with `ld == rows` and
`stride == rows * cols`, a row-major read at pitch `p` needs `(rows-1)*p + cols <= rows*cols`,
i.e. `p <= cols`, which with `p >= cols` forces `p == cols`. Anywhere else (a padded `ld`, or a
gap between items) a padded buffer fits as well as a packed one, and defaulting is a guess that
reads the wrong elements **in bounds**: for `rows=8, cols=6, ld=8, stride=64` a genuinely padded
(pitch 8) buffer came back with **42 of its 48 elements wrong**. So the default refuses to guess
and names the two spellings that resolve it. The converse makes the rule complete: a padded
row-major layout needs more than `rows * cols` per item, so it cannot be held by a matrix whose
metadata is packed; every padded row-major read either throws or was given its pitch. The helper
also checks that the read stays inside the buffer and does not straddle the next item.
`to_row_major` produces packed data (pitch `cols`, stride `rows * cols`) in a `Matrix` with the
usual column-major metadata (`ld = rows`); the pitch is not recoverable from that metadata, which
is why `to_column_major` takes it as an argument.

**`triangularize`: `uplo` names the triangle to keep.** `Uplo::Upper` zeroes the strict lower
triangle. This used to do the opposite: the decode named `i` the row and `j` the column but
addressed `b*stride + i*ld + j`, and the library is column-major (`data_[b*stride_ + j*ld_ + i]`),
so `i` was really the column and `Upper && i > j` zeroed the strict **upper** triangle. The names
were wrong, not the address arithmetic. The inversion cost PR #66 a day: extracting `R` from a
`geqrf` result silently yielded the Householder reflectors instead, which are easier to
diagonalise, and fabricated an apparent **1.7x** win for QR preconditioning that did not exist.
Two further bugs went with it and were fixed together: the element count came from
`data_.size()` (the span extent, which for a sliced or strided view is not `batch*rows*cols`), and
`n` was taken from `rows_` alone, so a non-square view decoded its own indices wrongly.

**CSR offsets from a dense matrix.** The scan writes offsets `1..rows` of each item and never
element 0, and the CSR allocating constructor reaches its buffer through `UnifiedVector::resize`,
which does not zero. So `row_offsets()[b * (rows + 1)]` was left uninitialised, and anything
reading an item's non-zero count as `offsets[end] - offsets[start]` read garbage (the CSR
`Random` path writes that element explicitly; this path did not). The whole array is now zeroed,
which also makes the value/index padding above each item's own count deterministic: `max_nnz` is
the batch maximum, so on a heterogeneous batch every smaller item has slots the population kernel
never touches.

## Runtime internals: the host BLAS dgemm health check

Some OpenBLAS builds ship a CPU-dispatch kernel that computes `dgemm` wrongly on the machine
auto-detection picks it for. The known case is OpenBLAS 0.3.20's Cooperlake kernel on recent Intel
parts, off by O(1)-O(100) at some sizes while `sgemm` is fine; everything layered on top silently
inherits the garbage (see the Known Pitfalls in `AGENTS.md`).

`cmake/BatchLASBlasHealthCheck.cmake` detects this at configure time and records the
`OPENBLAS_CORETYPE` that repairs it, but a configure-time answer is stale by construction: an
install tree is routinely consumed on another machine. So the value is compiled in as
`BATCHLAS_REQUIRED_OPENBLAS_CORETYPE` and `src/backends/netlib_lapack.cc` re-runs a cheap port of
the probe once, on first double-precision use. The naive reference is evaluated on a strided
sample of `C` (at most 32x32 entries per size), so it costs a few milliseconds: a broken kernel is
wrong by O(1)+ across the result, not in one entry. The sizes are the ones that expose the known
defect (`n = 64` and `n = 256` happen to be correct there, so a single small size proves nothing).
Single precision is skipped: a broken `dgemm` does not make it wrong, and a float-only user should
not be told to change their environment. Paths that call LAPACKE directly rather than through
`submit_host_task` (the netlib `gesvd`) run the guard themselves.

Deliberately **no `setenv()`**: OpenBLAS reads `OPENBLAS_CORETYPE` in its library constructor,
which has run before any BatchLAS code, so setting it would look like it worked and change
nothing; only the environment of the process before it starts can fix it. `OPENBLAS_CORETYPE` is
read straight from the environment for the diagnostic and is not captured into `Settings`, which
owns only this library's knobs. The mode (`BATCHLAS_BLAS_HEALTH`: `off | warn | error`) is a
parsed enum on `settings().unsafe`; `off` suppresses the only detection of a wrong host `dgemm`,
so the unsafe gate refuses it, while `error` is stricter than the default and is let through.

## Runtime internals: build-time structure of the library

`src/CMakeLists.txt` cuts the library into OBJECT libraries, one per area, so that an edit
recompiles one area's objects:

| object library | holds |
| --- | --- |
| `batchlas_core_obj` | `matrix.cc`, `csr_generators.cc`; also the target `generate_export_header()` is keyed on (see @ref design_symbol_visibility) |
| `batchlas_dispatch_obj` | flat kernel selection: `src/select/*.cc`, every `src/ops/<op>/<op>.cc`, and the generated tuned-tables TU (next section) |
| `batchlas_backends_obj`, `batchlas_backends_cuda_obj`, `batchlas_backends_rocm_obj` | the vendor wrapper TUs; the CUDA and ROCm ones exist only when at least one vendor library of that family is enabled |
| `batchlas_extensions_*_obj` (eigen, factorization, symmetric, tridiag, sytrd, latrd, stedc, cta) | the native drivers under `src/extensions/` |
| `batchlas_sycl_obj` | the native SYCL kernels under `src/sycl/` |
| `batchlas_util_obj`, `batchlas_extra_obj` | `src/util/` (queue, settings, SYCL utilities) and `src/extra/` (norms, cond, transpose, random matrices) |

Decisions and traps that go with the cut:

- **Selection is compiled unconditionally.** `batchlas_dispatch_obj` has no vendor condition;
  the vendor calls inside it are `if constexpr`-gated on the `BATCHLAS_HAS_*` macros. That is
  what makes `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` a complete library rather than a link error.
- **A vendor object library with no sources is a hard CMake error,** which is why the CUDA and
  ROCm backend libraries are created only when one of their vendor libraries is on. "There is a
  CUDA device" is not sufficient: with every CUDA math library off the source list is empty, and
  that empty library was the first thing a vendor-free configure used to hit.
- **`batchlas_extensions_cta_obj` is the only object library built without the CPU targets**
  (`NO_CPU_TARGETS`): its kernels are GPU-only (32-lane CTA tiers, the tiny tiers), so a
  native-CPU image of them would be dead weight. Nothing warns if a source moves out of that list;
  it silently acquires a CPU image. `batchlas_sycl_obj` keeps the CPU target on purpose, because
  the native-CPU image is what makes a vendor-free CPU build complete.
- **A source must sit with the sources whose device symbols it calls.** The shared library is the
  device-link unit, so a kernel driver and the TU defining the device functions it shares (for
  example `potrf_blocked.cc` and `potrf_cta.cc`, which share `potrf_cta_body`) must be in the same
  object library. Splitting them links today only because the helpers are inline templates, and
  fails with `ptxas fatal: Unresolved extern function` once one is marked `SYCL_EXTERNAL`. The
  rationale for each placement is in the comments of `src/extensions/CMakeLists.txt`.
- **The tiny-kernel TUs** (`getrf_tiny.cc`, `gesv_tiny.cc`, `posv_tiny.cc`, `trsm_sg_left.cc`) build
  with `-mllvm -pragma-unroll-threshold=262144`; without it LLVM declines `#pragma unroll`
  silently and the register arrays go to the stack.
- **Split by default, monolithic for release.** Each object library becomes its own shared
  library in the default (split) build, and `BATCHLAS_MONOLITHIC_LIBRARY` links them all into one
  `libbatchlas.so`. The build is device-link-bound and the device link is per shared library, so
  the split keeps a one-file edit's relink small; the merged link is a single long
  `sycl-post-link` (measured 575 s for a 273 MiB `.so` on the RTX 4090 box, out of an 11 m 38 s
  `-j16` build). Hidden visibility applies only to the monolithic build, for the reasons in
  [symbol visibility for private headers](#runtime-internals-symbol-visibility-for-private-headers)
  and the vague-linkage note there: in split mode the process-wide inline state would split into
  one copy per `.so`.

## Runtime internals: embedded tuned tables

The tuned selection tables (`tuned/<op>.<dtype>.<device>.txt`, format and provenance in the
`tuned/README.md` and [flat kernel selection](flat-kernel-selection.md)) are compiled into the library, so an installed
BatchLAS needs no data files at run time.

**Generation.** `src/CMakeLists.txt` globs `tuned/*.txt` with `CONFIGURE_DEPENDS`, so adding or
removing a table reconfigures, and runs `cmake/BatchLASEmbedTables.cmake` in script mode as a
custom command whose `DEPENDS` are the tables, so editing one regenerates. The script writes one
TU, `<build>/generated/select/tuned_tables.cc`, that defines
`select::embedded_tables()` (declared in `src/select/select.hh`) over a sorted array of
`{file name, text}` pairs, and it is compiled into `batchlas_dispatch_obj`.

- **Misnamed files are a configure error.** A file not named `<op>.<dtype>.<device>.txt` stops the
  build, because the loader skips any other name and such a table would otherwise silently never
  be used.
- **Raw-string delimiter.** Each table is embedded as `R"batchlas_tbl(...)batchlas_tbl"`; a table
  containing the delimiter is a configure error rather than a broken TU.
- **One `char` array per table, length from `sizeof`.** A `string_view` built from a bare literal
  runs a constexpr `strlen`, and roughly 0.5 MB of tables exceeds clang's constexpr step limit.
  (The design spec's sketch says "a `constexpr std::string_view` per table"; the array is what
  shipped, for this reason.) C++23 `#embed` is not available in DPC++.
- **Zero tables is valid:** the array keeps a sentinel entry and the span is empty.
- **An unchanged table set does not touch the generated file,** so its timestamp holds and the TU
  is not recompiled; regeneration on every configure would cost a device link of
  `batchlas_dispatch_obj`'s library.

**Staleness.** `cmake/BatchLASTunedStaleness.cmake` recomputes, at configure time, the
kernel-source hash that each table's header records (`kernels=<hash>`, over the source list in
`tools/tune/<op>_spec.cc`) and warns when they differ. It is a warning only: a stale table stays
in use, since a slightly stale ranking is still better than none. `.github/ci/check_tuned_tables.py`
and the tuner compute the same hash.

**Run time.** Tables are parsed lazily, per `(op, dtype)`, on the first selection that needs
them, and cached together with the `BATCHLAS_TUNED_DIR` value and a generation counter. A file in
`BATCHLAS_TUNED_DIR` with the same name replaces the embedded one (the trace tag then says
`override`); a non-table file there is skipped with a one-time warning. The cache, its mutex
and the parsed tables are deliberately leaked, like the other process-wide selection state:
`choose()` may run from static destructors, and the coverage writer runs from `atexit`, so
neither may find the tables already destroyed. Tests swap
the embedded set with `select::testing::set_builtin_tables` and restore it with
`use_embedded_tables`, each of which bumps the generation so no cached parse survives.

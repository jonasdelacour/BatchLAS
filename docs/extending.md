# Adding entry points to BatchLAS

> **Covers:** adding a public entry point or a new op: where its declaration and definition go,
> how it joins flat kernel selection (`src/ops/<op>/choice.hh`, `<op>.cc`, tuned tables, a tuner
> spec), and the overload traps every entry point has to avoid.
> **Status:** current (flat kernel selection). The route-era recipe (`RouteTable`, shape builders,
> `src/dispatch/entry_points/`) is history on the
> [vendor-independence page](design/vendor-independence.md#adding-an-op-in-the-route-era).

For work inside the library; the calling conventions are in
[cpp-api.md](cpp-api.md). The selection mechanism this page plugs into is described in
@ref design_flat_selection and, for the code, @ref selection.

An entry point is declared with its backend as an explicit template parameter:

```cpp
template <Backend Back, typename T, ...> Event name(Queue&, ...);
```

There are two halves to an op: the **public declaration** in `include/batchlas/` (the contract; the
sections from [Where the declaration goes](#where-the-declaration-goes) on), and the **definition**
in `src/ops/<op>/`, which validates the arguments, chooses a kernel and launches it
([Where the definition goes](#where-the-definition-goes-srcopsop),
[Adding an op to kernel selection](#adding-an-op-to-kernel-selection)).

## Where the definition goes: `src/ops/<op>/`

- **One directory per op**, `src/ops/<op>/`, holding the vocabulary header `choice.hh` and the op
  file `<op>.cc`. The op file defines the public `<op>` and `<op>_buffer_size` for every backend and
  is listed in `src/CMakeLists.txt` (one `target_sources(batchlas_dispatch_obj PRIVATE
  ops/<op>/<op>.cc)` line). `batchlas_dispatch_obj` is compiled whatever vendor libraries exist, so
  the public symbol exists in every build that has the device.
- **The vendor implementation is a separate symbol**, `backend::<op>_vendor<B, T>` (and
  `backend::<op>_vendor_buffer_size` when it needs scratch), declared in the op's public header and
  defined and instantiated only in the vendor TUs (`src/backends/cublas.cc`, `cusolver.cc`,
  `rocblas.cc`, `netlib_lapack.cc`, …). Never define the public `<op>` in a vendor TU: a build
  without that library would lose the op entirely, not just its vendor path
  ([The entry-point facade](design/vendor-independence.md#the-entry-point-facade)).
- **Instantiate on the device family**, at the end of the op file, with the macros in
  `src/util/template-instantiations.hh` (`BATCHLAS_INSTANTIATE_OP` inside
  `BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS`, as `src/ops/potrf/potrf.cc` does). The vendor arm
  compiles to a throw where the library is absent, so this is correct in every configuration.
- **Kernel bodies stay where they are** (`src/extensions/`, `src/sycl/`, …) and keep their own
  argument checks and launch-parameter functions; the op file only decides which one runs.

`src/ops/potrf/` is the model: read `choice.hh` and `potrf.cc` before writing a new op.

## Adding an op to kernel selection

Every step below is required; each has a check that fails if it is skipped.

1. **The vocabulary, `src/ops/<op>/choice.hh`**, in `namespace batchlas::ops::<op>`:
   - one *family* per kernel driver: `struct Cta : select::NoFields<"cta"> {};` for a driver with no
     knobs, or a struct with integer fields plus static `name` and `fields`, `values()`, `from()` and
     `operator==` (see `Lpanel` in potrf). Anything the driver derives from the shape and device stays
     derived in the driver; only knobs the tuner should choose become fields;
   - `using <Op>Choice = std::variant<...>`, with a `Vendor` family when the op has a vendor library;
   - `candidates<T>()`, every compiled choice once, in tie-break order (simpler first, native before
     `vendor`); `select::all_of<<Op>Choice>()` when no family has fields;
   - `inline constexpr select::OpSpec spec{Op::<op>, select::Lib::<group>}`, with `Lib::none` for an
     op with no vendor family and a third `Rules{last_resort}` argument when the default last-resort
     order (`blocked`, then `vendor`) does not fit;
   - `key_names` (`"uplo:exact"`, `"n:log:3"`, `"batch:log"`, …), exact keys first: the table header
     must declare exactly these;
   - the tuner grid arrays (`grid_n`, `grid_batch`, …).

   A brand-new op also needs its `Op` enumerator and `op_name` case in `include/batchlas/no_route.hh`;
   that name is the table file prefix, the `BATCHLAS_<OP>_ROUTE` pin variable and the coverage `op`
   column.
2. **The op file, `src/ops/<op>/<op>.cc`**, four functions and the public pair:
   - `key_of(...)` returns a `select::Key` with every name in `key_names`;
   - `can_run(choice, device, ...)` is **correctness only** (rule R3): false exactly when that driver
     would throw or answer wrongly, each clause the argument check at the top of the driver; the
     vendor family's is `d.has_vendor`. Never put a speed threshold here;
   - `launch(...)` is one `std::visit(select::overloaded{...})` with one arm per family; the vendor arm
     is `if constexpr (select::has_library<B>(spec.vendor)) return backend::<op>_vendor<B, T>(...);
     else select::no_vendor<B, T>(spec);`;
   - `workspace(...)` is the same visit returning exactly the chosen family's bytes; it must be pure
     (no launch, no read of operand data). A nested op adds its children's public `*_buffer_size`;
   - the public `<op>` calls `<op>_validate_params`, then `select::run<B, T>(spec, q, key, candidates,
     can_run, shape, {}, launch)`; `<op>_buffer_size` calls the same validator, then
     `select::pick<B, T>` with the same key and `can_run`, then `workspace` (rule R5: sizing and
     running make the same choice). Validate only what no family could serve
     ([positional validators](design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve)).
3. **Tables, `tuned/<op>.<dtype>.<device>.txt`**, one per dtype on every device the tree ships tables
   for; `tuned_tables_tests` holds that inventory, parses every table, checks every spelling against
   `candidates<T>()` and the header's keys against `key_names` (register the op in its
   `candidates(op, dtype)` map). Produce them with the tuner, or with `scripts/sweep_to_table.py`
   from a sweep (`--tuner`, or `--transcribe` for an untimed ranking); never by hand
   (@ref tuned_tables_readme). A device without a table borrows another's and warns once.
4. **A tuner spec, `tools/tune/<op>_spec.cc`**, implementing the interface in `tools/tune/spec.hh`
   (problem builder, host verification, grid from `choice.hh`, the kernel-source list between the
   `kernel-sources-begin`/`-end` markers that the staleness hash reads), registered with
   `BATCHLAS_TUNE_REGISTER(<Op>Spec)` and added to `tools/tune/CMakeLists.txt` (@ref tune_tool_readme).
5. **Coverage**: an op with a vendor library gets a row in `append_static_rows` in
   `src/select/coverage.cc` (its vendor predicate and whether a native kernel is linked); the
   `reached` rows come from `select::run` for free.
6. **Tests**: `tests/<op>_candidates_tests.cc` pins every candidate with `select::ScopedPin` on shapes
   that straddle its `can_run` limits in both directions, checks that `can_run` agrees with what the
   driver accepts, runs each candidate with a workspace of exactly `<op>_buffer_size` bytes in a
   poisoned arena, and asserts that a bad pin throws. docs/developer/agent-guide.md §8 applies in full.
7. **Docs**: put `@ingroup selection_ops` on the op's `choice.hh` declarations; the measurements
   behind its families go on `docs/perf/<op>.md` (add the op to `PERF_PAGE` in
   `docs/tools/gen_db_pages.py` if its page has another name). The generated @ref selection_tables
   page picks the op up from its `choice.hh` and tables with no further edit.
8. **Verify the choice, not the timing**: `BATCHLAS_SELECT_TRACE=1` prints each decision and its
   table; when moving an existing op, capture `scripts/route_diff.sh` before and compare after; build
   once with `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` and run the op's tests there.

## Where the declaration goes

- **One header per dense op**, `include/batchlas/blas/functions/<op>.hh`,
  included from the aggregator `include/batchlas/blas/functions.hh` (which
  declares nothing itself). A new op gets its own header and one `#include`
  line there. The extension surface (`steqr`, `stedc`, `syevx`, `lanczos`,
  `ritz_values`, ...) lives in `include/batchlas/blas/extensions.hh` instead.
- **The primary takes views.** Declare it for `MatrixView` / `VectorView` /
  `Span`. Do not hand-write an owning-`Matrix` twin: close the header with
  `BATCHLAS_ACCEPT_OWNING(<op>)` (owning containers accepted wherever the
  primary takes a view) and `BATCHLAS_DISPATCH_ON_QUEUE(<op>)` (backend taken
  from the queue), once per name including the sizing function —
  `include/batchlas/blas/functions/gemv.hh` and `geqrf.hh` show the shape. The
  one hand-written owning twin left, in `ritz_values`, documents in place why the
  generated forwarder cannot take a partially-explicit call.
- **The sizing function is `<op>_buffer_size`** and returns bytes. It must take
  the same arguments as the run path, up to the workspace, and branch
  identically; size it with `BumpAllocator::measuring()` + `required_bytes()`,
  never by re-deriving the sizes by hand (the allocator checks the
  alignment-rounded size but advances by the raw size, so an exact simulation is
  too small). An older `*_workspace` / `*_workspace_size` spelling survives only
  as a `[[deprecated]]` alias (`stedc_workspace_size`, `ritz_values_workspace`).
- **Validation** of shapes and flags belongs in the checked overloads in
  `include/batchlas/blas/options.hh` or a `<op>_validate_params` helper beside
  the declaration, and must be tested through the ordinary spelling (see the
  next section for why).

## Keep the `requires` clause on the queue-dispatch overload

`BATCHLAS_DISPATCH_ON_QUEUE(name)`, in `<batchlas/blas/queue-dispatch.hh>`, adds
the overload that lets callers omit the backend and take it from the queue. It
forwards a variadic pack, so the primary's default arguments still apply, and it
carries a `requires` clause admitting only argument lists the positional entry
point would itself accept.

Keep that clause. An unconstrained pack matches *any* argument list, which
outranks every more specific overload — an option-struct spelling, or one that
relies on a default argument — and then fails to compile inside its own body,
reporting the error at the macro expansion instead of at the call site.

The macro also carries the pointer check (`require_pack_accessible`), whose
diagnostics name argument *positions* rather than parameter names.

## Give the arena overload a different arity from the positional call

Each workspace-taking entry point has two spellings: one ending in a
`Span<std::byte>`, and one that omits it and leases from the queue's arena. The
two must differ in arity, or they are genuinely ambiguous.

An option struct supplies that difference for `potrf`, `getrs` and `syev`:
`potrf(ctx, A, opts)` and `potrf(ctx, A, opts, ws)` are three and four arguments.
An entry point with no options has only the workspace argument to distinguish
them, so its arena spelling drops the workspace and there is no second one to
add — that is the shape of `getrf`, `getri`, `geqrf` and `orgqr`.

## Write two overloads, not one defaulted empty span

Write the two spellings as two overloads. Do not write one function with
`Span<std::byte> ws = {}` and an `if (ws.data() != nullptr)` inside.

A null span means "a zero-length allocation", not "the caller passed nothing".
Code that sub-allocates from a `BumpAllocator` runs its algorithm twice: once in
sizing mode, where every pool allocation hands back an empty span while the
input matrices stay real, and once for real. A null check reads the sizing
pass's empty span as "no workspace given", allocates one, and executes the
algorithm over the caller's live data during the measurement pass.

The argument checker in `queue-dispatch.hh` follows the same rule from the other
side: it skips zero-length spans, which a sizing pass hands out by design and
which carry no pointer worth checking.

## Name the option type when you pass an empty option struct

Inside the library, where calls are written `potrf<B>(...)` with the backend
fixed:

```cpp
potrf<B>(ctx, A, PotrfOptions{}, my_workspace);   // correct
potrf<B>(ctx, A, {}, my_workspace);               // ill-formed: ambiguous
```

`{}` reaches an enum by an exact match and a class type only by a user-defined
conversion, so where an option overload and a positional overload have the same
arity and the positional one's parameter is an enum, a bare `{}` selects the
positional overload and its enum's value-initialised state.

Name the option type at every call site. When you add an entry point whose option
overload and positional overload have the same arity, close the trap the way
`potrf` does — a third overload taking a dedicated *enum* type:

```cpp
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;
```

It sits at the same exact-match rank as the positional overload, so the bare-`{}`
call is ambiguous rather than silently resolved, and the compiler names all three
candidates. Neither `PotrfOptions{}` nor `Uplo::Lower` converts to that type, so
both working spellings still resolve as before.

## Where `T` comes from

The matrix parameters are templates constrained to `Matrix` or `MatrixView`, and
`T` is a *defaulted* template parameter computed from the first of them:

```cpp
template <typename MA, ..., typename T = detail::dense_scalar_t<MA>>
Event gemm(Queue&, const MA& A, ..., const GemmOptions<T>& opts);
```

Keep that shape. Computing `T` from the first matrix parameter fixes it before
the compiler considers the option parameter, which is what makes `{.alpha = 2.0f}`
compile — a braced initialiser needs a concrete type and deduces nothing.

It is also what lets both `Matrix` and `MatrixView` be passed, and mixed. The
positional entry points have a `Matrix` wrapper alongside the `MatrixView`
primary; everything converts to `MatrixView` before the positional call.

## Keep the documentation examples compiling

Every C++ code block in `docs/cpp-api.md` is written out again in
`docs_cpp_api_examples` in `tests/linalg_layer_tests.cc`, which only has to
compile, along with the calls the document names in prose alone —
`trim_workspace`, `WorkspaceLease::release`, `to_row_major`,
`Queue::native_handle`, `is_device_accessible`, the `Vector` factories and the
`linalg` `_into` forms. Change a documented signature and change both together;
add a block to the guard when you add one to the document, and spell the call
there the way the document spells it — template arguments included, so that a
documented spelling which stops compiling fails the build.

The guard is linked into a test binary that is built without regard to which
backends are compiled in, so keep backend-specific code out of it: reach the
backend through `with_backend(ctx, ...)` rather than instantiating, say,
`Backend::CUDA` directly.

# Adding entry points to BatchLAS

> **Status:** current (flat kernel selection). The route-era recipe is history on the
> [vendor-independence page](design/vendor-independence.md#adding-an-op-in-the-route-era).

How to add a public entry point or a new op: where the declaration and definition go, how the op
joins flat kernel selection (@ref design_flat_selection, code: @ref api_selection), and the overload
traps to avoid. Calling conventions are in [cpp-api.md](cpp-api.md).

An entry point takes its backend as an explicit template parameter:

```cpp
template <Backend Back, typename T, ...> Event name(Queue&, ...);
```

An op has two halves: the public declaration in `include/batchlas/` (the contract, see
[Where the declaration goes](#where-the-declaration-goes)) and the definition in `src/ops/<op>/`,
which validates arguments, chooses a kernel and launches it
([Where the definition goes](#where-the-definition-goes-srcopsop)).

## Where the definition goes: `src/ops/<op>/`

- One directory per op holds `choice.hh` (the vocabulary) and `<op>.cc`. The op file defines the
  public `<op>` and `<op>_buffer_size` for every backend and is listed in `src/CMakeLists.txt`
  (`target_sources(batchlas_dispatch_obj PRIVATE ops/<op>/<op>.cc)`). `batchlas_dispatch_obj` is
  built whatever vendor libraries exist, so the public symbol exists in every build with the device.
- The vendor implementation is a separate symbol, `backend::<op>_vendor<B, T>` (plus
  `backend::<op>_vendor_buffer_size` when it needs scratch). It is declared in the op's public
  header and defined only in the vendor TUs (`src/backends/cublas.cc`, `cusolver.cc`, `rocblas.cc`,
  `netlib_lapack.cc`, ...). Never define the public `<op>` in a vendor TU: a build without that
  library would lose the op ([facade](design/vendor-independence.md#the-entry-point-facade)).
- Instantiate on the device family at the end of the op file with the macros in
  `src/util/template-instantiations.hh` (`BATCHLAS_INSTANTIATE_OP` inside
  `BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS`, as `src/ops/potrf/potrf.cc` does). The vendor arm
  compiles to a throw where the library is absent.
- Kernel bodies stay in `src/extensions/`, `src/sycl/`, ... and keep their own argument checks and
  launch-parameter functions; the op file only decides which one runs.

`src/ops/potrf/` is the model: read its `choice.hh` and `potrf.cc` first.

## Adding an op to kernel selection

Every step is required; each has a check that fails if skipped.

1. **Vocabulary, `src/ops/<op>/choice.hh`**, in `namespace batchlas::ops::<op>`:
   - one *family* per kernel driver: `struct Cta : select::NoFields<"cta"> {};` for a driver with no
     knobs, otherwise a struct with integer fields plus static `name` and `fields`, `values()`,
     `from()` and `operator==` (see `Lpanel` in potrf). Anything derivable from shape and device
     stays derived in the driver; only knobs the tuner should choose become fields;
   - `using <Op>Choice = std::variant<...>`, with a `Vendor` family when a vendor library exists;
   - `candidates<T>()`: every compiled choice once, in tie-break order (simpler first, native before
     `vendor`); `select::all_of<<Op>Choice>()` when no family has fields;
   - `inline constexpr select::OpSpec spec{Op::<op>, select::Lib::<group>}`: `Lib::none` for an op
     without a vendor family, and a third `Rules{last_resort}` argument when the default last-resort
     order (`blocked`, then `vendor`) does not fit;
   - `key_names` (`"uplo:exact"`, `"n:log:3"`, `"batch:log"`, ...), exact keys first; the table
     header must declare exactly these;
   - the tuner grid arrays (`grid_n`, `grid_batch`, ...).

   A new op also needs its `Op` enumerator and `op_name` case in `include/batchlas/no_route.hh`. That
   name is the table file prefix, the `BATCHLAS_<OP>_ROUTE` pin variable and the coverage `op` column.
2. **Op file, `src/ops/<op>/<op>.cc`**: four functions plus the public pair.
   - `key_of(...)` returns a `select::Key` with every name in `key_names`.
   - `can_run(choice, device, ...)` is correctness only (rule R3): false exactly when the driver
     would throw or answer wrongly, each clause being the driver's own argument check; the vendor
     family's is `d.has_vendor`. Never put a speed threshold here.
   - `launch(...)` is one `std::visit(select::overloaded{...})`, one arm per family. The vendor arm
     calls `backend::<op>_vendor<B, T>(...)` when `select::has_library<B>(spec.vendor)` holds, and
     `select::no_vendor<B, T>(spec)` otherwise.
   - `%workspace(...)` is the same visit returning exactly the chosen family's bytes. It must be pure
     (no launch, no read of operand data); a nested op adds its children's public `*_buffer_size`.
   - The public `<op>` calls `<op>_validate_params`, then
     `select::run<B, T>(spec, q, key, candidates, can_run, shape, {}, launch)`. `<op>_buffer_size`
     calls the same validator, then `select::pick<B, T>` with the same key and `can_run`, then
     `workspace` (rule R5: sizing and running make the same choice). Validate only what no family
     could serve ([positional validators](design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve)).
3. **Tables, `tuned/<op>.<dtype>.<%device>.txt`**: one per dtype on every device the tree ships
   tables for. `tuned_tables_tests` holds that inventory, parses every table, checks spellings
   against `candidates<T>()` and header keys against `key_names` (register the op in its
   `candidates(op, dtype)` map). Produce tables with the tuner, or `scripts/sweep_to_table.py`
   (`--tuner`, or `--transcribe` for an untimed ranking); never by hand (@ref tuned_tables_readme).
   A device without a table borrows another's and warns once.
4. **Tuner spec, `tools/tune/<op>_spec.cc`**: implements `tools/tune/spec.hh` (problem builder, host
   verification, grid from `choice.hh`, kernel-source list between the `kernel-sources-begin`/`-end`
   markers read by the staleness hash). Register with `BATCHLAS_TUNE_REGISTER(<Op>Spec)` and add it
   to `tools/tune/CMakeLists.txt` (@ref tune_tool_readme).
5. **Coverage**: an op with a vendor library gets a row in `append_static_rows` in
   `src/select/coverage.cc` (its vendor predicate and whether a native kernel is linked). `reached`
   rows come from `select::run` automatically.
6. **Tests**: `tests/<op>_candidates_tests.cc` pins every candidate with `select::ScopedPin` on shapes
   that straddle its `can_run` limits both ways, checks `can_run` against what the driver accepts,
   runs each candidate with a workspace of exactly `<op>_buffer_size` bytes in a poisoned arena, and
   asserts a bad pin throws. docs/developer/agent-guide.md §8 applies in full.
7. **Docs**: `@ingroup %api_selection_ops` on the `choice.hh` declarations; measurements go on
   `docs/perf/<op>.md` (add the op to `PERF_PAGE` in `docs/tools/gen_db_pages.py` if the page is
   named differently). @ref selection_tables picks the op up from `choice.hh` and the tables.
8. **Verify the choice, not the timing**: `BATCHLAS_SELECT_TRACE=1` prints each decision and its
   table. When moving an existing op, capture `scripts/route_diff.sh` before and compare after. Build
   once with `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` and run the op's tests there.

## Where the declaration goes

- One header per dense op, `include/batchlas/blas/functions/<op>.hh`, included from the aggregator
  `include/batchlas/blas/functions.hh` (which declares nothing itself). The extension surface
  (`steqr`, `stedc`, `syevx`, `lanczos`, `ritz_values`, ...) lives in
  `include/batchlas/blas/extensions.hh`.
- The primary takes views (`MatrixView` / `VectorView` / `Span`). Do not hand-write an owning
  `Matrix` twin: close the header with `BATCHLAS_ACCEPT_OWNING(<op>)` and
  `BATCHLAS_DISPATCH_ON_QUEUE(<op>)`, once per name including the sizing function
  (`functions/gemv.hh`, `geqrf.hh` show the shape). The one hand-written owning twin is in
  `ritz_values`, documented in place.
- The sizing function is `<op>_buffer_size` and returns bytes. It takes the same arguments as the run
  path up to the workspace and branches identically. Size it with `BumpAllocator::measuring()` +
  `required_bytes()`, never by re-deriving sizes: the allocator checks the alignment-rounded size
  but advances by the raw size, so an exact simulation is too small. The `*_workspace` /
  `*_workspace_size` spellings survive only as `[[deprecated]]` aliases (`stedc_workspace_size`,
  `ritz_values_workspace`).
- Validation of shapes and flags goes in the checked overloads in `include/batchlas/blas/options.hh`
  or a `<op>_validate_params` helper beside the declaration, and is tested through the ordinary
  spelling.

## Keep the `requires` clause on the queue-dispatch overload

`BATCHLAS_DISPATCH_ON_QUEUE(name)` (`<batchlas/blas/queue-dispatch.hh>`) adds the overload that
takes the backend from the queue. It forwards a variadic pack, so the primary's default arguments
apply, and its `requires` clause admits only argument lists the positional entry point would accept.

An unconstrained pack matches any argument list and outranks every more specific overload (an
option-struct spelling, or one relying on a default argument). It then fails to compile inside its
own body, reporting the error at the macro expansion instead of the call site. The macro also
carries the pointer check (`require_pack_accessible`); its diagnostics name argument positions, not
parameter names.

## Give the arena overload a different arity from the positional call

Each workspace-taking entry point has two spellings: one ending in `Span<std::byte>`, and one that
omits it and leases from the queue's arena. They must differ in arity or they are ambiguous.

- With an option struct (`potrf`, `getrs`, `syev`): `potrf(ctx, A, opts)` vs
  `potrf(ctx, A, opts, ws)`.
- Without options (`getrf`, `getri`, `geqrf`, `orgqr`): only the workspace argument distinguishes
  them, so the arena spelling drops it and there is no second one to add.

## Write two overloads, not one defaulted empty span

Write the two spellings as two overloads. Do not write one function with `Span<std::byte> ws = {}`
and an `if (ws.data() != nullptr)` inside.

A null span means "a zero-length allocation", not "the caller passed nothing". Code that
sub-allocates from a `BumpAllocator` runs its algorithm twice: a sizing pass, where every pool
allocation returns an empty span while the input matrices stay real, then the real pass. A null check
reads the sizing pass's empty span as "no workspace given", allocates one, and executes the
algorithm over the caller's live data during measurement. The argument checker in `queue-dispatch.hh`
skips zero-length spans for the same reason.

## Name the option type when you pass an empty option struct

Inside the library, where the backend is fixed (`potrf<B>(...)`):

```cpp
potrf<B>(ctx, A, PotrfOptions{}, my_workspace);   // correct
potrf<B>(ctx, A, {}, my_workspace);               // ill-formed: ambiguous
```

`{}` reaches an enum by exact match and a class type only by user-defined conversion. Where an
option overload and a positional overload have the same arity and the positional parameter is an
enum, a bare `{}` selects the positional overload with the enum's value-initialised state.

Name the option type at every call site. For a new entry point whose option and positional overloads
have the same arity, close the trap as `potrf` does, with a third overload taking a dedicated enum:

```cpp
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;
```

It has the same exact-match rank as the positional overload, so a bare `{}` is ambiguous and the
compiler names all three candidates. Neither `PotrfOptions{}` nor `Uplo::Lower` converts to it.

## Where `T` comes from

Matrix parameters are templates constrained to `Matrix` or `MatrixView`; `T` is a defaulted
template parameter computed from the first of them:

```cpp
template <typename MA, ..., typename T = detail::dense_scalar_t<MA>>
Event gemm(Queue&, const MA& A, ..., const GemmOptions<T>& opts);
```

Keep that shape. Fixing `T` from the first matrix before the option parameter is considered is what
makes `{.alpha = 2.0f}` compile (a braced initialiser deduces nothing), and what lets `Matrix` and
`MatrixView` be passed and mixed. Positional entry points have a `Matrix` wrapper beside the
`MatrixView` primary; everything converts to `MatrixView` before the positional call.

## Keep the documentation examples compiling

Every C++ block in `docs/cpp-api.md` is repeated in `docs_cpp_api_examples` in
`tests/linalg_layer_tests.cc`, which only has to compile. That test also covers calls named in
prose alone: `trim_workspace`, `WorkspaceLease::release`, `to_row_major`, `Queue::native_handle`,
`is_device_accessible`, the `Vector` factories and the `linalg` `_into` forms.

- Change a documented signature and the guard together; add a guard block for each new document block,
  spelled as the document spells it (template arguments included).
- The guard is built regardless of which backends are compiled in, so keep backend-specific code
  out: reach the backend through `with_backend(ctx, ...)`, never `Backend::CUDA` directly.

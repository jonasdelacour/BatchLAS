# Adding entry points to BatchLAS {#dev_extending}

> **Status:** current · flat kernel selection · checked against `src/` on 2026-10-29

How to add a public entry point or an op to flat kernel selection (@ref design_flat_selection, code:
@ref api_selection), and the overload traps to avoid. Calling conventions: [cpp-api.md](cpp-api.md).
`src/ops/potrf/` is the model: read its `choice.hh` and `potrf.cc` first.

An entry point takes its backend as an explicit template parameter,
`template <Backend Back, typename T, ...> Event name(Queue&, ...)`. Its declaration (the contract)
lives in `include/batchlas/`, its definition (validate, choose a kernel, launch) in `src/ops/<op>/`.

## Where the definition goes: `src/ops/<op>/`

- `src/ops/<op>/<op>.cc` defines the public `<op>` and `<op>_buffer_size` for every backend and is
  added to `batchlas_selection_obj` in `src/CMakeLists.txt`, which is built whatever vendor libraries
  exist. Instantiate at the end with `BATCHLAS_INSTANTIATE_OP` inside
  `BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS` (`src/util/template-instantiations.hh`).
- The vendor call is a separate symbol, `backend::<op>_vendor<B, T>` (plus
  `backend::<op>_vendor_buffer_size`), declared in the op's public header and defined only in the
  vendor TUs (`src/backends/cublas.cc`, `cusolver.cc`, `rocblas.cc`, `netlib_lapack.cc`). Never
  define the public `<op>` in a vendor TU: a build without that library would lose the op.
- Kernel bodies stay in their own TUs with their own argument checks; the op file only decides which
  one runs. A new kernel TU joins one of the OBJECT-library lists in `src/ops/CMakeLists.txt`, chosen
  by whose device symbols it calls, not by folder
  ([build-time structure](design/runtime-internals.md#runtime-internals-build-time-structure)).

| Code | Goes in |
| --- | --- |
| Entry point, `choice.hh`, headers shared by the op's kernels | `src/ops/<op>/` |
| The op's kernel `.cc` files and single-consumer device headers | `src/ops/<op>/kernels/` |
| An internal sub-op of one parent | `src/ops/<parent>/<sub>/`, e.g. `syev/stedc/`, `gesvd/bdsqr/` |
| A public routine without kernel selection | its own `src/ops/<routine>/`, e.g. `ortho/`, `iluk/` |
| Device code used by several ops (`tiny_device.hh`, `sg_partition/`) | `src/kernels/common/` |
| DPC++ vs AdaptiveCpp portability helpers | `src/sycl/` |
| Vendor wrappers and routing-shape helpers | `src/backends/` |
| Queue, settings, matrix utilities | `src/util/` |
| Kernel selection and tuned-table lookup | `src/select/` |
| Verification checks (`batchlas::verify`) | `src/verify/` |

Includes are root-based (`#include <src/ops/potrf/potrf_native.hh>`, `<batchlas/...>`); a quoted or
relative include fails CI.

## Adding an op to kernel selection

Every step is required; each has a check that fails if skipped.

1. **Vocabulary, `src/ops/<op>/choice.hh`** (`namespace batchlas::ops::<op>`):
   - one family per kernel driver: `struct Cta : select::NoFields<"cta"> {};`, or a struct with
     integer fields plus `name`, `fields`, `values()`, `from()`, `operator==` (see potrf's `Lpanel`).
     Only knobs the tuner should choose become fields; the rest is derived in the driver;
   - `using <Op>Choice = std::variant<...>`, with a `Vendor` family when a vendor library exists;
   - `candidates<T>()`: every compiled choice once, in tie-break order (simpler first, native before
     `vendor`), or `select::all_of<<Op>Choice>()`;
   - `inline constexpr select::OpSpec spec{Op::<op>, select::Lib::<group>}` (`Lib::none` without a
     vendor family; a third `{last_resort}` argument when the default `blocked`, then `vendor` does
     not fit);
   - `key_names` (`"uplo:exact"`, `"n:log:3"`, `"batch:log"`), exact keys first, and the tuner grid
     arrays (`grid_n`, `grid_batch`, ...).

   A new op also needs its `Op` enumerator and `op_name` case in `include/batchlas/no_route.hh`; that
   name is the table prefix, the `BATCHLAS_<OP>_ROUTE` pin and the coverage `op` column.
2. **Op file, `src/ops/<op>/<op>.cc`**:
   - `key_of(...)` returns a `select::Key` with every name in `key_names`;
   - `can_run(choice, device, ...)` is correctness only: false exactly when the driver would throw or
     answer wrongly. Never a speed threshold. The vendor family's is `d.has_vendor`;
   - `launch(...)` is one `std::visit(select::overloaded{...})`; the vendor arm calls
     `backend::<op>_vendor<B, T>` when `select::has_library<B>(spec.vendor)`, else
     `select::no_vendor<B, T>(spec)`;
   - `%workspace(...)` is the same visit returning the chosen family's bytes; pure (no launch, no read
     of operand data);
   - the public `<op>` calls `<op>_validate_params`, then `select::run<B, T>(spec, q, key, candidates,
     can_run, shape, {}, launch)`; `<op>_buffer_size` validates, then `select::pick<B, T>` with the
     same key and `can_run`, then `workspace`, so sizing and running choose alike. Validate only what
     no family could serve.
3. **Tables, `tuned/<op>.<device>.txt`**, one per shipped device, dtype the first key. Register the op
   in `candidates(op, dtype)` and `key_spec(op)` in `tests/tuned_tables_tests.cc`. Generate tables with
   `batchlas_tune <op> --tier ...` and `scripts/sweep_to_table.py`, never by hand
   (@ref tuned_tables_readme, @ref design_tuning_data).
4. **Tuner spec, `tools/tune/<op>_spec.cc`** implementing `tools/tune/spec.hh`, with its kernel
   sources between the `kernel-sources-begin`/`-end` markers; `BATCHLAS_TUNE_REGISTER(<Op>Spec)` and
   an entry in `tools/tune/CMakeLists.txt` (@ref tune_tool_readme).
5. **Coverage**: an op with a vendor library gets a row in `append_static_rows`
   (`src/select/coverage.cc`); `reached` rows come from `select::run`.
6. **Tests**, `tests/<op>_candidates_tests.cc`: pin every candidate with `select::ScopedPin` on shapes
   straddling its `can_run` limits both ways, run each with exactly `<op>_buffer_size` bytes of
   poisoned workspace, and assert a bad pin throws. Agent guide §8 applies in full.
7. **Docs**: `@ingroup %api_selection_ops` on the `choice.hh` declarations; measurements on
   `docs/perf/<op>.md` (or map the op in `PERF_PAGE`, `docs/tools/gen_db_pages.py`).
8. **Verify the choice, not the timing**: `BATCHLAS_SELECT_TRACE=1`; `scripts/route_diff.sh` before
   and after when moving an existing op; run the op's tests once in a
   `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` tree.

## Where the declaration goes

- One header per dense op, `include/batchlas/blas/functions/<op>.hh`, included from
  `include/batchlas/blas/functions.hh`; extensions (`steqr`, `stedc`, `syevx`, `lanczos`, ...) live
  in `include/batchlas/blas/extensions.hh`.
- The primary takes views (`MatrixView`, `VectorView`, `Span`); a `Matrix` binds to them, so never
  write an owning twin. Close the header with `BATCHLAS_DISPATCH_ON_QUEUE(<op>)` once per name,
  including `<op>_buffer_size` (see `functions/gemv.hh`).
- `<op>_buffer_size` returns bytes, takes the run path's arguments up to the workspace and branches
  identically. Size with `BumpAllocator::measuring()` + `required_bytes()`, never by re-deriving: the
  allocator checks the aligned size but advances by the raw one.
- Matrix parameters are templates with `T = detail::dense_scalar_t<MA>` taken from the first matrix,
  which is what lets `{.alpha = 2.0f}` compile and `Matrix`/`MatrixView` mix. Positional entry points
  take `MatrixView` only.

## Keep the `requires` clause on the queue-dispatch overload

`BATCHLAS_DISPATCH_ON_QUEUE(name)` (`<batchlas/blas/queue-dispatch.hh>`) forwards a variadic pack to
the backend-templated primary. Its `requires` clause admits only argument lists the primary accepts;
without it the pack outranks every more specific overload and fails inside the macro, not at the
call site.

## Write two overloads, not one defaulted empty span

Each workspace-taking entry point has a spelling ending in `Span<std::byte>` and one that leases from
the queue's arena; they must differ in arity. Write two overloads, never `Span<std::byte> ws = {}`
with a null check: during a `BumpAllocator` sizing pass every allocation is an empty span, so a null
check allocates and runs the algorithm on the caller's live data while measuring.

## Name the option type when you pass an empty option struct

Write `potrf<B>(ctx, A, PotrfOptions{}, ws)`, never `potrf<B>(ctx, A, {}, ws)`: `{}` reaches an enum
by exact match, so it selects the positional overload. A new entry point whose option and positional
overloads share an arity adds potrf's deleted third overload taking
`detail::EmptyBracesAreAmbiguous`, which makes a bare `{}` a compile error.

## Keep the documentation examples compiling

Every C++ block in `docs/cpp-api.md` is repeated in `docs_cpp_api_examples` in
`tests/linalg_layer_tests.cc`, which must compile. Change a documented signature and its guard
together, spelled as the document spells it; reach a backend through `with_backend(ctx, ...)`, never
`Backend::CUDA`.

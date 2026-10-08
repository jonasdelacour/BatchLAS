# API conventions {#design_api_conventions}

> **Status:** current · checked 2026-10-06

The rules behind the shape of the entry points in `include/batchlas/blas/`: the two spellings,
the dispatch and owning-argument overloads, argument checks, workspace and `info` arguments.
The user-facing rules are in [the C++ API guide](../cpp-api.md).

## api conventions: the backend is bound to the Queue by with_backend

`with_backend(ctx, f)` turns the queue's runtime `Backend` into an `std::integral_constant` and
passes it to `f`:

```cpp
with_backend(ctx, [&](auto B) { return gemm<B.value, T>(ctx, ...); });
```

- Only backends compiled into the build have a `case`. Others, and `Backend::MAGMA` and
  `Backend::SYCL`, throw `batchlas::unsupported`.
- Naming a backend never pins a kernel. Flat selection inside the entry point picks the kernel;
  `BATCHLAS_<OP>_ROUTE` pins it. See [flat kernel selection](flat-kernel-selection.md) and
  @ref selection_tables. The level-3 ops hemm, herk and her2k still choose by hand.
- Two spellings: `gemm(ctx, ...)` takes the backend from the queue, `gemm<B>(ctx, ...)` fixes it.
  Library code in `src/extensions/` must use `<B>`, because the runtime spelling reads
  `ctx.backend()` instead of the `B` the algorithm was instantiated for.

## api conventions: the variadic dispatch overload and its requires-clause

`BATCHLAS_DISPATCH_ON_QUEUE(NAME)` defines the backend-deducing overload of
`template <Backend Back, ...> R NAME(Queue&, ...)`, so callers write `NAME(ctx, ...)`.

- It forwards a pack. The signature is not restated, and the primary's default arguments still apply.
- The inner call always passes `Backend` explicitly, so it never re-enters the macro.
- The requires-clause is mandatory. Without it the overload accepts any argument list, beats
  more specific overloads, and fails inside its body. The probe uses `detail::kProbeBackend`.
- Even constrained, the pack binds prvalues and non-const lvalues better than `const MA&`.
  `getrf(ctx, A.view(), pivots)` therefore reaches `getrf<Backend>` past the option overload's
  checks. Repeat checks on both overloads, and test per-call checks through the ordinary spelling.

## api conventions: shape checks live on both option overloads

- Shape preconditions (`detail::require_square`, `require_same_rows`, `require_same_batch`,
  `require_span_at_least`, `require_info_span`; `blas/options.hh`) throw `std::invalid_argument`.
  `std::runtime_error` is reserved for environment and backend failures.
- Checks sit on both the deducing overload and `template <Backend B, ...>`.
- Output spans use `>=`, not `==`. A caller may slice one arena across several calls.
- `geqrf` has no squareness check. Every backend indexes `tau.data() + i * min(m, n)`, so a
  shorter `tau` is read out of bounds.
- `gesvd` checks only `A`. A default-constructed `U` or `Vh` means `SvdVectors::None`.
- Positional primaries in `blas/functions/*.hh` that take an explicit workspace are unchecked.
  `src/extensions/` calls them per iteration.

## api conventions: owning Matrix and Vector arguments

Implicit conversion (`Matrix` to `MatrixView`, `Vector` to `VectorView`) does not help
deduction. With owning matrices, `gemm<Backend::CUDA>(ctx, A, B, C, ...)` deduces nothing and
reports "no matching function".

- `BATCHLAS_ACCEPT_OWNING(NAME)`, placed beside `BATCHLAS_DISPATCH_ON_QUEUE(NAME)`, converts owning
  arguments to views and lets the inner call deduce. Its signature is not copied.
- `BATCHLAS_ACCEPT_OWNING_NB` does the same for entry points not templated on `Backend`
  (`norm`, `transpose`, `francis_sweep`, the `steqr_*_buffer_size` pair).
- `detail::view_of` names the view a parameter becomes. Non-owning arguments pass through.
- It is a last resort. `detail::AnyOwning` keeps it out of all-view calls, and a fixed-parameter
  overload (an option struct, or a fixed arity such as `getrf`'s four-argument form) wins by
  partial ordering.

### api conventions: the owning pack is const Args&, never Args&&

A forwarding-reference pack binds prvalues better than `const MatrixView&`, so `Args&&...` would
beat the checked option overloads. That is how an 8x4 matrix once passed a squareness check.
With `const Args&...` the overloads tie on conversion, and partial ordering selects the checked one.
Nothing is moved through the forwarder, so nothing is lost.

### api conventions: argument lists the owning forwarder cannot reach

- A braced-init-list in any position. The pack deduces nothing, so name the type: `OrthoOptions{}`.
- Partly supplied template arguments. Write `spmm<Back>(ctx, ...)`, not `spmm<Back, T>(...)`.
  `ritz_values` keeps a hand-written twin in `blas/extensions.hh`.

## api conventions: the USM pointer check

A bare pointer from host memory (`std::vector`, `new`, `malloc`) reaches the device as a wild
address. The result is `CUDA_ERROR_ILLEGAL_ADDRESS`, then an abort that no `catch` can stop.
Checked spellings throw `std::invalid_argument` naming the argument: by position from the
dispatch macro, by name from option overloads (`BATCHLAS_CHECK_ARGS`, e.g. `gemm: A`).

- Cost: one USM query per pointer argument, about 70 ns.
- Not checked: an empty `Span`, and a default-constructed view. The latter is the spelling for an
  unused optional matrix, e.g. `syevx(..., JobType::NoEigenVectors, MatrixView<T>(), params)`.
- Off switch: `BATCHLAS_SKIP_POINTER_CHECKS=1`. It is honoured only when `BATCHLAS_ALLOW_UNSAFE_ENV`
  is ON; otherwise the setting is false whatever the environment says.
- Any non-empty value whose first character is not `0` disables the checks, so `=false`, `=off`
  and `=no` all disable them. Do not replace this with `env_truthy`, which would re-enable checks
  for existing users.
- The setting is latched on first use (`detail::pointer_checks_enabled()`). A settings reload does
  not reach it.

## api conventions: two workspace spellings, never a defaulted span

Option overloads come in pairs: one takes `Span<std::byte> ws`, the other leases scratch from the
queue's arena, sized by the matching `*_buffer_size`.

- Do not merge them with `Span<std::byte> ws = {}` and a null check. In a `BumpAllocator` sizing
  pass, pool allocations return empty spans while the inputs stay real. A null check would then
  factorise live data, and the damage shows up later as non-convergence.
- `getrf`, `getri`, `geqrf` and `orgqr` take no options and no workspace parameter. Their only
  option-layer spelling is the arena one.
- Sizing branches exactly as the call does. `ormqr` sizes with the same `block_size_hint`, and
  `gesvd` takes the same Hermitian-or-general branch. Both sizing and running call `select::pick`
  and `select::run`.
- On an out-of-order queue, releasing an arena lease drains the queue. Pass your own span to stay
  asynchronous. See `batchlas/util/workspace.hh`.

## api conventions: T comes from the matrices, never from the option struct

Option structs are templates on `T`. The option overloads deduce `T` from the matrices, through
`T = detail::dense_scalar_t<MA>`. An option struct in a deduced position makes `{.alpha = 2.0f}`
ill-formed.

- `HerkOptions`: `alpha` and `beta` are real. A complex `alpha` would make \f$\alpha A A^H\f$ non-Hermitian.
- `Her2kOptions`: `alpha` is complex and `beta` is real. \f$\alpha A B^H + \bar\alpha B A^H\f$ is Hermitian for any `alpha`.
- `GesvdOptions::hermitian_uplo` is a `std::optional<Uplo>`. An engaged value selects the Hermitian overload.

## api conventions: the bare braces potrf trap

`potrf`'s option and positional overloads both accept `{}`. A bare `{}` matches `Uplo` exactly, so
`potrf(ctx, A, {}, ws)` means `Uplo{}` (Upper), not the default `PotrfOptions`.

A deleted overload taking `detail::EmptyBracesAreAmbiguous` makes the bare `{}` a compile error.
**Write `PotrfOptions{}` or `Uplo::Lower`.** `blas/extensions.hh` has the same guard.

## api conventions: per-item info spans

A trailing `Span<int32_t> info` receives one `int32` per batch item: `0` for success, and for
`potrf` a value `> 0` is the leading minor at which positive definiteness failed. It is the
caller's USM, written in place.

- An empty span means "no status wanted". A non-empty span shorter than the batch is rejected
  (`detail::require_info_span`). Otherwise the backend writes to its own scratch and the caller
  reads stale data, usually zeros, which look like "all factorised".
- `linalg::eigh` and `linalg::svd` always request `info` and return it in the result. At batch
  16384 a single non-converged item is otherwise invisible.

## api conventions: the linalg layer

- Value-returning, backend from the Queue, workspace from the arena: `batchlas::linalg`.
  Out-parameter, caller's workspace: `batchlas::`.
- Qualify every forwarding alias as `\::batchlas::`. An unqualified `inv` inside `linalg::inv`
  recurses forever. Inside `batchlas::linalg`, `detail::` means `linalg::detail`, so write
  `\::batchlas::detail::require_square`.
- Pivots and workspace come from the arena or move into the result. A local `UnifiedVector` is
  freed before the kernels that read it run. `linalg::svd` waits before returning, because
  `~Matrix` frees USM without waiting.
- `MatmulOptions` is not `GemmOptions`. `matmul` allocates `C`, so a caller `beta` would read
  uninitialised memory, and the field is left out.
- `eigh` and `svd` take positional arguments. `SyevOptions` and `GesvdOptions` have no `info`
  field. `eigh` keeps the squareness check that the positional spelling would otherwise lose.

### api conventions: linalg::solve keeps its Transpose branch

`gesv` takes only `NoTrans`. `Trans` and `ConjTrans` use the hand-composed `getrf` + `getrs`.
Do not flatten the branch.

- `gesv` chooses between its fused `tiny` kernel and the `blocked` composition through its own
  table (`src/ops/gesv/gesv.cc`). Do not copy that choice into `batchlas::linalg`.
- `gesv` throws `batchlas::internal_error` on an empty problem (`n`, `nrhs` or `batch` < 1)
  before selection, as `solve_spd` does.

### api conventions: linalg::solve_spd is a separate entry point

`solve_spd` is spelled on `posv`. `solve` (LU with partial pivoting) is backward stable for any
nonsingular `A`. `solve_spd` costs about half the arithmetic but makes the SPD claim the caller's.
A non-positive-definite `A` gives an undefined `X`, as LAPACK `?POSV` does. `solve_spd` passes an
empty `info` span, so the failure is not reported.

## api conventions: cond generators default to CGS2

`random_with_log10_cond_metric` and `random_hermitian_with_log10_cond_metric` default `algo` to
`OrthoAlgorithm::CGS2`. Cholesky-QR squares the condition number, so float items can come out
non-finite. Householder leaves some items singular. Neither reliably hits the requested
\f$\kappa\f$.

## api conventions: generators without a dispatch or owning overload

The `random_*_with_log10_cond_metric` generators in `blas/extra.hh` are not in either macro. Their
`T` is non-deduced, so a macro would create an overload that can never be selected. Call them as
`f<Backend, T>(...)`; see [which spelling each entry point takes](../cpp-api.md#which-spelling-each-entry-point-takes).

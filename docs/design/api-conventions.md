# API conventions {#design_api_conventions}

> **Status:** current · checked 2026-10-06

The rules behind the shape of the entry points in `include/batchlas/blas/`: the two spellings,
the dispatch overload, owning arguments, argument checks, workspace and `info` arguments.
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
  Library code under `src/ops/` must use `<B>`, because the runtime spelling reads
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
  Native drivers under `src/ops/` call them per iteration.

## api conventions: a Matrix is a MatrixView

`Matrix<T, F>` publicly derives from `MatrixView<T, F>` over its own storage, and `Vector<T>` from
`VectorView<T>`. An owning argument binds to a view parameter as its base, and template deduction
accepts it (derived-to-base deduction), so every entry point and every `MatrixView` member takes
it with no forwarder and no `.view()`.

- The owning class holds the `UnifiedVector`s and re-points the view half (`bind()`) after every
  allocation: constructors, copy, move and `Matrix::set_active_dims()`. A copy gets its own
  backend handle; a move leaves the source empty.
- Members the owning class redefines on purpose: `Matrix::fill(T)` waits, `Matrix::astype<U>()`
  keeps the layout (the view's packs it), `Vector::operator[]` is a raw storage index.
- A generic parameter (`auto v`, `const T&`, a CTAD or braced context) takes the `Matrix` itself:
  by value that is a deep copy, and a catch-all `f(const T&)` beats `f(const MatrixView<T>&)`.
  Pass `.view()` there.

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
- Every `linalg` factorisation and solve (`cholesky`, `lu`, `solve`, `solve_spd`, `inv`, `eigh`,
  `svd`) always requests `info` and returns it in the result. At batch 16384 a single failed item
  is otherwise invisible.

## api conventions: the linalg layer

- Value-returning, backend from the Queue, workspace from the arena: `batchlas::linalg`.
  Out-parameter, caller's workspace: `batchlas::`.
- Qualify every forwarding alias as `\::batchlas::`. An unqualified `inv` inside `linalg::inv`
  recurses forever. Inside `batchlas::linalg`, `detail::` means `linalg::detail`, so write
  `\::batchlas::detail::require_square`.
- Pivots and workspace come from the arena or move into the result. `~Matrix` and
  `~UnifiedVector` free USM without waiting, so a wrapper that owns a local copy of `A` (`svd`,
  `solve`, `solve_spd`, `inv`, the full `eigvalsh`) waits before returning.
- `MatmulOptions` is not `GemmOptions`. `matmul` allocates `C`, so a caller `beta` would read
  uninitialised memory, and the field is left out.
- `eigh`, `svd`, `solve` and `solve_spd` call the positional spellings: `SyevOptions` and
  `GesvdOptions` have no `info` field, `gesv` and `posv` have no option overload. `eigh` keeps the
  squareness check that the positional spelling would otherwise lose.
- Option structs are reused, not mirrored: `solve_triangular` takes `TrsmOptions`, `svd` takes
  `GesvdOptions`, the subset `eigh` takes `SyevxParams`.

### api conventions: linalg wrappers order their own launches

A wrapper that launches more than once (copy, then factor) runs its body through
`detail::on_in_order_queue` (`batchlas/util/sycl-device-queue.hh`), the mechanism the multi-launch
drivers use. On an out-of-order `Queue` the factorisation would otherwise read the copy before it is
written. Guard: `LinalgWrappers.OutOfOrder*` in `tests/linalg_layer_tests.cc`.

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
A non-positive-definite item gives an undefined `x`, as LAPACK `?POSV` does, and a non-zero `info`.

## api conventions: cond generators default to CGS2

`random_with_log10_cond_metric` and `random_hermitian_with_log10_cond_metric` default `algo` to
`OrthoAlgorithm::CGS2`. Cholesky-QR squares the condition number, so float items can come out
non-finite. Householder leaves some items singular. Neither reliably hits the requested
\f$\kappa\f$.

## api conventions: generators without a dispatch overload

The `random_*_with_log10_cond_metric` generators in `blas/extra.hh` have no
`BATCHLAS_DISPATCH_ON_QUEUE`. Their `T` is non-deduced, so the macro would create an overload that
can never be selected. Call them as
`f<Backend, T>(...)`; see [which spelling each entry point takes](../cpp-api.md#which-spelling-each-entry-point-takes).

## api conventions: bitwise repeatability under a pinned route

Two calls with identical arguments (the same input bits, `ld`, batch `stride`, batch size, `uplo`
and jobz) return the same bits when they run on the same device, under the same route and from the
same build. The route is the same under a pin (`BATCHLAS_<OP>_ROUTE`, `select::ScopedPin`), or under
Auto with the same tables and settings: a retune, `BATCHLAS_TUNED_DIR` or a different device can
change the route, and a different route gives different rounding. Nothing is promised across
devices, builds, compilers or vendor-library versions.

- **Tested:** `SyevTest.PinnedRouteIsBitwiseRepeatable` (`tests/syev_tests.cc`) pins `cta`
  (n = 17, 32; batch 1024 at n = 32), `blocked` (n = 32, 400) and `two_stage` (n = 40, 400), both
  `uplo`, both jobz, four dtypes, and compares the eigenvalues and the whole matrix buffer with
  `memcmp`. The vendor family is not covered: its repeatability is the vendor library's.
- **Why it holds for syev:** its kernels use no floating-point atomics; the only atomics are
  integer (the `info` `fetch_max` and the grid-barrier word in `latrd_lower_panel.cc`), and every
  reduction has a fixed order for a given work-group size, which the route and device fix.
- **Other ops:** untested. Floating-point atomic sums break the property and exist in the native
  `spmm` scatter (`src/ops/spmm/kernels/spmm_native.cc`), the LOBPCG projections
  (`src/ops/syevx/kernels/syevx_lobpcg.cc`) and the column sums of `norm`
  (`src/util/norm.cc`, work-group local): those results may differ in the last bits between runs.

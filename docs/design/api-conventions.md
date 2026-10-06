# API conventions: spellings, option structs and argument checks {#design_api_conventions}

> **Covers:** why every entry point has a backend-deducing and a compile-time
> spelling, how option structs and positional overloads coexist, the variadic
> `BATCHLAS_DISPATCH_ON_QUEUE` overload and its traps, owning-argument acceptance
> (`BATCHLAS_ACCEPT_OWNING`), the USM pointer check, shape checks, per-item
> `info` spans, the bare-`{}` trap, and the design choices in `batchlas::linalg`.
> **Status:** current. Each section was re-checked against
> `include/batchlas/blas/queue-dispatch.hh`, `include/batchlas/blas/options.hh`,
> `include/batchlas/blas/extra.hh` and `include/batchlas/blas/linalg-ops.hh` on
> 2026-09-30, when the rationale moved here out of those headers' comments.

The user-facing rules (which spelling each entry point takes, the option-struct
fields and defaults, the USM contract as a caller sees it) are in
[the C++ API guide](../cpp-api.md). This page records *why* the surface is shaped
that way, and the traps a future editor of those headers must not re-open. The
API reference for the pieces is in the `options`, `dispatch` and `linalg` groups.

## api conventions: the backend is bound to the Queue by with_backend

Every entry point is templated on `Backend` and explicitly instantiated per
backend. That is right for code generation, but on its own it forces the choice
on the caller at compile time. `with_backend(ctx, f)` turns the queue's runtime
`Backend` back into a compile-time one and hands it to `f` as an
`std::integral_constant`, so the body stays a template:

```cpp
with_backend(ctx, [&](auto B) { return gemm<B.value, T>(ctx, ...); });
```

User code and the convenience layer can then be written once, while the
generated code stays exactly as specialised as before: `with_backend` is a
`switch` over instantiations that already exist, not a virtual call or a
runtime-parameterised kernel. Only backends compiled into the build get a `case`;
the rest reach the `throw batchlas::unsupported`, which is reachable for
`Backend::MAGMA` and `Backend::SYCL` because they are declared in the enum and
have no implementation behind them.

**Two spellings, and which one library code must use.** `gemm(ctx, ...)` takes
the backend from the queue; `gemm<B>(ctx, ...)` fixes it at compile time.
`src/extensions/` is templated on `Backend` and must use the second: the runtime
spelling would silently use `ctx.backend()` instead of the `B` the algorithm was
instantiated for.

## api conventions: the variadic dispatch overload and its requires-clause

`BATCHLAS_DISPATCH_ON_QUEUE(NAME)` defines the backend-deducing overload of an
entry point already declared as
`template <Backend Back, typename T, ...> R NAME(Queue&, ...)`, so that callers
can write `NAME(ctx, ...)`.

**The parameters are forwarded as a pack rather than restated.** The macro then
carries no copy of the signature to drift from the declaration, and because the
inner call names the primary, the primary's *default arguments* still apply to
arguments the caller omitted. Restating the signature would have meant
duplicating every default.

**Overload resolution is unambiguous in both directions.** Called as
`NAME(ctx, args...)`, the Backend-first overloads cannot deduce `Backend` and drop
out, leaving only this one. Called as `NAME<Backend::CUDA>(ctx, args...)`, this
one drops out, because `Backend::CUDA` is a value and `Args` are types. The inner
call always supplies `Backend` explicitly, so it never re-enters the macro's
overload.

**The requires-clause is load-bearing.** Without it the overload accepts *any*
argument list, so it beats a more specific overload (an option-struct spelling,
or one that relies on a default argument) and only then fails, deep inside its
own body, on a call it should never have claimed. Constrained to argument lists
the positional entry point would accept, it drops out of resolution instead: the
difference between "this overload does not apply" and "this overload applies and
is broken". The probe instantiates the call on `detail::kProbeBackend`, any
backend that is compiled in; all backends answer identically because the entry
points are declared once and instantiated per backend.

**Even constrained, it beats option-struct overloads for prvalue and non-const
lvalue arguments.** A forwarding pack `Args&&...` binds a prvalue better than
`const MA&` does, so `getrf(ctx, A.view(), pivots)` resolves to
`getrf<Backend>` *past* every check on the deducing option overload. This is why
the shape checks are repeated on both overloads (next section), and why a test of
a per-call check must go through the ordinary spelling.

## api conventions: shape checks live on both option overloads

The shape preconditions of the LAPACK-style entry points (`detail::require_square`,
`require_same_rows`, `require_same_batch`, `require_span_at_least`,
`require_info_span` in `blas/options.hh`) throw `std::invalid_argument` for a
caller error, never `std::runtime_error`, which is reserved for environment and
backend failures.

They are repeated on **both** the deducing overload and the
`template <Backend B, ...>` overload, deliberately: the variadic dispatch
overload binds a prvalue better than `const MA&`, so a check that lived only on
the deducing overload would be skipped on the ordinary `getrf(ctx, A.view(), pivots)`
call (see the previous section).

The positional primaries in `blas/functions/*.hh` that take an explicit
workspace stay **unchecked**, also deliberately: `src/extensions/` calls them per
iteration and they must not pay a host-side branch.

Two further rules are encoded in the checks:

- **`>=`, not `==`, for output spans.** An oversized pivot or `tau` buffer is
  legitimate: a caller may slice one large arena across several calls.
- **`geqrf` has no squareness check.** Rectangular `A` is the point of `geqrf`.
  What is fixed is the `tau` stride: every backend indexes
  `tau.data() + i * min(m, n)`, and every `ormqr` path reads it at the same stride,
  so a short `tau` is read out of bounds (mirrors `validate_ormqr_dims` in
  `ormqr_blocked.cc`).
- **`gesvd` checks only `A`.** A default-constructed `U` or `Vh` is how the API
  spells `SvdVectors::None`, so any check against `A`'s extents would reject it.

## api conventions: owning Matrix and Vector arguments

`Matrix` converts implicitly to `MatrixView` and `Vector` to `VectorView`
(`blas/matrix.hh`). That suffices wherever the parameter type is already
concrete. It does **not** suffice when the scalar must be deduced from the
argument: template argument deduction does not consider user-defined
conversions, so `gemm<Backend::CUDA>(ctx, A, B, C, ...)` with owning matrices
deduces nothing from `A`, the primary drops out, and the caller gets "no matching
function".

**History.** Every entry point used to carry a hand-written twin whose whole
body was one cast per matrix argument. There was one twin per *overload*, not per
name, so a name with four positional spellings paid for four of them, and each
restated every defaulted argument the primary already had.

**The replacement.** `detail::view_of` names the view a parameter should become;
anything that is not an owning container passes through as itself. That is what
lets the forwarder be variadic and still touch only the arguments that need
converting, and it is also what makes a mixed call such as
`stein(ctx, Vector d, VectorView e, ...)` work, which no hand-written twin
covered. `BATCHLAS_ACCEPT_OWNING(NAME)`, one line beside
`BATCHLAS_DISPATCH_ON_QUEUE(NAME)`, converts the pack argument by argument and
lets the *inner* call do the deducing, so it carries no copy of any signature: a
new overload, a new defaulted argument or a reordered parameter needs no change
in the macro. `BATCHLAS_ACCEPT_OWNING_NB` is the same for entry points that are
not templated on `Backend` (`norm`, `transpose`, `francis_sweep`, the
`steqr_*_buffer_size` pair); those have no dispatch overload either, because there
is no backend to deduce.

**It is a last resort in overload resolution, by construction.**
`detail::AnyOwning` keeps it out of every all-view call (without that gate it
would be an unconstrained variadic that claims every call and then fails in its
own body, the mistake the dispatch macro's requires-clause exists to avoid), and
the requires-clause keeps it out of argument lists the view spelling would not
accept. It therefore applies only where the alternative is a compile error. Where
a more specific overload is also viable (the option-struct spellings in
`blas/options.hh`, or an arity-changing forwarder such as `getrf`'s four-argument
form) partial ordering prefers that one, because a fixed parameter list is more
specialised than a trailing pack.

### api conventions: the owning pack is const Args&, never Args&&

A forwarding-reference pack binds a prvalue **better** than `const MatrixView&`
does, so an `Args&&...` forwarder would beat the checked option overloads in
`blas/options.hh` for a call like `getrf(ctx, A.view(), pivots)`. That is exactly
how an 8x4 matrix once passed through a squareness check. With `const Args&...`
the two rank equally on conversion, and partial ordering picks the more
specialised overload, the one carrying the checks. Nothing is moved through the
forwarder, so nothing is lost by not forwarding: every parameter downstream is a
view, a span, an enum or a small option struct.

### api conventions: argument lists the owning forwarder cannot reach

Two argument lists are deliberately not reached, both because a pack cannot
deduce them; in both cases the alternative is a diagnostic, not a wrong answer.

- **A bare `{}` or other braced-init-list in any position.** A parameter pack
  deduces nothing from one, so the forwarder drops out and the call has to name
  the type it means (`stein_all_counts`, `OrthoOptions{}`). The same property
  keeps `BATCHLAS_DISPATCH_ON_QUEUE` out of `potrf`'s option-struct calls, and it
  is why the deleted bare-`{}` guards in `blas/options.hh` and
  `blas/extensions.hh` still fire.
- **A call that supplies some template arguments and deduces the rest**, e.g.
  `spmm<Back, T>(ctx, Matrix, ...)` with the `MatrixFormat` still deduced: `T`
  lands in the pack and has to match the first argument. Write
  `spmm<Back>(ctx, ...)` and let both deduce. The one in-tree call that needed the
  old spelling keeps a hand-written twin (`ritz_values` in `blas/extensions.hh`).

### api conventions: generators without a dispatch or owning overload

The `random_*_with_log10_cond_metric` generators in `blas/extra.hh` are
deliberately absent from both macros. Their `T` is non-deduced (it appears only
as `float_t<T>`, an alias template, and in the return type), so either macro
would expand to an overload that can never be selected, which reads as if they
were dispatchable when they are not. They keep the explicit `f<Backend, T>(...)`
spelling; see [which spelling each entry point takes](../cpp-api.md#which-spelling-each-entry-point-takes).

## api conventions: the USM pointer check

A `MatrixView` or `Span` takes a bare pointer and cannot check where the memory
came from. Handing ordinary host memory (`std::vector`, `new`, `malloc`) to a GPU
queue used to reach the device as a wild address: `CUDA_ERROR_ILLEGAL_ADDRESS`,
then a `SIGABRT` from inside the runtime during teardown that no `catch` can
stop, while the identical code was correct on the host backend, so a CPU
prototype passed and the GPU run died. The checked spellings turn that into a
thrown `std::invalid_argument` naming the offending argument: by position from
the dispatch macro (it forwards an unnamed pack), by name from the option
overloads (`BATCHLAS_CHECK_ARGS`, "gemm: A").

**Cost.** One USM query per pointer argument, about 70 ns measured, which is
noise against a kernel launch.

**Exemptions.** An argument that addresses no elements is not checked: an empty
`Span` (every allocation of a `BumpAllocator` sizing pass is one) and a
default-constructed view, which is the API's spelling for an optional matrix that
is not in use (`syevx(..., JobType::NoEigenVectors, MatrixView<T>(), params)` is
the documented call, written at about 50 call sites in the repository). Checking
the default view turned every such call into a throw ("the pointer is null"),
which `iluk_tests` caught.

**The off switch.** `BATCHLAS_SKIP_POINTER_CHECKS=1` bypasses the check. It is
one of the knobs the `BATCHLAS_ALLOW_UNSAFE_ENV` build option gates: with that
option OFF the setting is false whatever the environment says, so an embedding
application cannot have its argument validation switched off by ambient process
state.

**The odd acceptance set is deliberate.** It is preserved verbatim in
`settings.cc`: *any* non-empty value whose first character is not `0` skips the
checks, so `=false`, `=off` and `=no` all **disable** them. That is not
`env_truthy` and must not become it: tightening it would silently re-enable
checking for anyone who wrote one of those spellings, a behaviour change at the
site where behaviour changes are most expensive.

**Latched once.** `detail::pointer_checks_enabled()` reads the setting into a
function-local static on first use. It sits on the argument-checking path of
every dispatched call, and the latch keeps it off the per-argument cost. A
settings reload therefore does not reach it.

## api conventions: two workspace spellings, never a defaulted span

The LAPACK-style option overloads come in pairs: one takes `Span<std::byte> ws`,
the other leases scratch from the queue's arena, sized by the matching
`*_buffer_size`. **Do not collapse the pair into one function with
`Span<std::byte> ws = {}` and a null check.** A null span is not a synonym for
"not passed": in a `BumpAllocator` sizing pass every pool allocation hands back
an empty span while the input matrices stay real, so a null check would run the
real factorisation over the caller's live data. Nothing crashes; the corrupted
matrix surfaces later as an algorithm that stops converging.

`getrf`, `getri`, `geqrf` and `orgqr` carry no options, so their only option-layer
spelling is the arena-backed one, and they deliberately take no workspace
parameter: such an overload would be ambiguous with the positional call.

**Sizing must branch exactly as the call does.** `ormqr` sizes with the same
`block_size_hint` the call uses (the hint picks the panel width the workspace is
sized for), and `gesvd` takes the same Hermitian-or-general branch for the query
as for the call, because the two branches pick providers independently and can
need different scratch.

**Releasing an arena lease on an out-of-order queue drains the queue**, so every
arena spelling blocks until the device is idle on such a queue; pass your own
span to keep the call asynchronous. See `batchlas/util/workspace.hh`.

## api conventions: T comes from the matrices, never from the option struct

The option structs are templates on `T` (the dense BLAS ones) and the option
overloads deduce `T` from the matrix arguments, with the struct parameter written
as `const GemmOptions<T>&` where `T` is a *defaulted* template parameter
`T = detail::dense_scalar_t<MA>`. An option struct in a deduced position would
make `{.alpha = 2.0f}` ill-formed, because a braced-init-list deduces nothing.

Two option structs have real-valued scalars on purpose:

- `HerkOptions`: `alpha` and `beta` are real even though the operands are
  complex, because a complex `alpha` would make \f$\alpha A A^H\f$ non-Hermitian.
- `Her2kOptions`: \f$\alpha A B^H + \bar\alpha B A^H\f$ is Hermitian for any
  `alpha`, so `alpha` is complex and only `beta` is real.

`GesvdOptions::hermitian_uplo` is a `std::optional<Uplo>` rather than a `Uplo`
sentinel because an engaged value selects a different overload (the Hermitian
entry point in `blas/functions/gesvd.hh`).

## api conventions: the bare braces potrf trap

`potrf`'s option and positional overloads have the same arity and both accept
`{}`. A bare `{}` matches the `Uplo` enum exactly and the option struct only by a
user-defined conversion, so `potrf(ctx, A, {}, ws)` silently meant
`Uplo{}` == `Upper` and factorised the opposite triangle from the default
`PotrfOptions`.

The fix is a third, deleted overload taking `detail::EmptyBracesAreAmbiguous`, an
empty enum: it is another exact match for `{}`, so the bare-`{}` call becomes
ambiguous (a compile error). `PotrfOptions{}` and `Uplo::Lower` still resolve as
before, and neither converts to the guard type. **Write `PotrfOptions{}`, never a
bare `{}`.** The same guard pattern exists in `blas/extensions.hh`.

## api conventions: per-item info spans

A trailing `Span<int32_t> info` carries the per-item LAPACK status, one `int32`
per batch item (`0` = success; for `potrf`, `> 0` is the leading minor at which
positive-definiteness failed). It is the caller's USM, written in place.

**An empty span spells "no status wanted"**, so only a non-empty one is
measured, and a too-short one is rejected up front
(`detail::require_info_span`). The check exists because a short span otherwise
fails silently: the backend falls back to its own scratch and the caller reads
stale bytes, most often zeros, which read as "all factorised".

The value-returning `linalg::eigh` and `linalg::svd` **always** request `info`
and return it in the result struct. A caller of that layer has no workspace of
their own and no other way to find out, and at batch 16384 a single
non-converged item is otherwise invisible: the values and vectors come back
looking exactly like a converged solve. `info` needs no wait inside `eigh`: it is
written by the kernels, read by the caller after they wait, and moved into the
result.

## api conventions: the linalg layer

**Membership rule.** Value-returning, backend from the Queue, workspace from the
arena: `linalg::`. Out-parameter, workspace yours: `batchlas::`.

**Every forwarding alias is qualified `::batchlas::`.** `batchlas::linalg` is nested
inside `batchlas`, so an unqualified `return inv(ctx, A);` inside `linalg::inv`
is infinite recursion. Likewise `detail::` inside `batchlas::linalg` finds
`batchlas::linalg::detail`, so `eigh` spells `::batchlas::detail::require_square`.

**Scratch lifetime.** Pivots and workspace come from the arena (or are moved
into the result): a local `UnifiedVector` would be freed before the kernels that
read it have run. `linalg::svd` waits before returning because its local copy of
`A` is scratch that `gesvd` overwrites and `~Matrix` frees its USM without
waiting. The `(void)` on a discarded `Event` is deliberate: the queue is
in-order, so the next submission is already ordered after this one.

**`MatmulOptions` is deliberately not `GemmOptions`.** `matmul` allocates `C`, so
a caller-supplied `beta` would read uninitialised memory; omitting the field
makes naming it a compile error.

**`eigh` and `svd` are spelled positionally**, not through `SyevOptions` /
`GesvdOptions`, because those structs carry no `info` field (unlike
`PotrfOptions`). `eigh` keeps the squareness check it had when it went through
the option overload, because the positional spelling has none and a non-square
view would otherwise be factorised as `rows() x rows()`.

### api conventions: linalg::solve keeps its Transpose branch

`gesv` has no `Transpose` parameter (`blas/functions/gesv.hh`), so only `NoTrans`
can reach it; `Trans` and `ConjTrans` keep the hand-composed `getrf` + `getrs`.
**The branch is not stylistic; do not flatten it.** There is also no shape gate
around the `gesv` call: `gesv` itself chooses between its fused kernel and that
same composition (`dispatch/route_gesv.hh`), and a copy of that window in
`batchlas::linalg` would drift from it.

**A deliberate behaviour change.** `route_gesv`'s `supports()` refuses every
route when an extent is degenerate (`n`, `nrhs` or `batch` < 1), and the entry
point then throws (`solve_throw_unroutable`). The earlier hand-composed body
enqueued nothing instead. The change makes `solve` agree with `solve_spd`, which
has thrown on the identical guard in `route_posv.hh` since the P2 work package.

### api conventions: linalg::solve_spd is a separate entry point

`solve_spd` is spelled on `posv`, not folded into `solve`, and the difference is
not stylistic. LU with partial pivoting is backward stable for any nonsingular
`A`, so `solve` is the safe default and stays the one every existing caller
reaches. `solve_spd` trades that generality for roughly half the arithmetic and
makes the SPD claim the caller's; a matrix that is not positive definite comes
back as a non-zero `info` and an undefined `X`, as LAPACK's `?POSV` does.

## api conventions: cond generators default to CGS2

`random_with_log10_cond_metric` and `random_hermitian_with_log10_cond_metric`
default `algo` to `OrthoAlgorithm::CGS2` deliberately. Cholesky-QR squares the
condition number of an uncontrolled random input and returns non-finite items in
float, and Householder leaves some items singular, so both silently fail to honour
the requested \f$\kappa\f$.

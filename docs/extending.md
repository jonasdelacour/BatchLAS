# Adding entry points to BatchLAS

For work inside the library; the calling conventions are in
[cpp-api.md](cpp-api.md).

An entry point is declared with its backend as an explicit template parameter:

```cpp
template <Backend Back, typename T, ...> Event name(Queue&, ...);
```

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

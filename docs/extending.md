# Adding entry points to BatchLAS

This is for people writing new entry points inside the library, not for people
calling it. For the calling conventions, see [cpp-api.md](cpp-api.md).

## Constrain the queue-dispatch overload

An entry point is declared with its backend as an explicit template parameter:

```cpp
template <Backend Back, typename T, ...> Event name(Queue&, ...);
```

`BATCHLAS_DISPATCH_ON_QUEUE(name)`, in `<batchlas/blas/queue-dispatch.hh>`, adds
the overload that lets callers omit the backend and take it from the queue. That
overload forwards a variadic pack rather than restating the signature, so it
carries no copy of the parameter list to drift out of sync, and the primary's
default arguments still apply.

Keep its `requires` clause. The clause admits only argument lists the positional
entry point would itself accept. Without it the pack matches *any* argument list,
so this overload outranks every more specific one — an option-struct spelling, or
one that relies on a default argument — claims the call, and then fails to
compile inside its own body, reporting the error at the macro expansion instead
of at the call site. Constrained, it drops out of overload resolution instead.

The macro also carries the pointer check (`require_pack_accessible`), which is
why it can only name argument *positions* rather than parameter names.

## Never give an arena overload the same arity as the positional call

Each workspace-taking entry point has two spellings: one ending in a
`Span<std::byte>`, and one that omits it and leases from the queue's arena. Give
the arena spelling a different arity, or the two are genuinely ambiguous.

That is why `getrf`, `getri`, `geqrf` and `orgqr` take no workspace parameter at
all in their arena spelling: with no option struct to change the arity, the
workspace argument is the only thing that can distinguish the two. To manage the
workspace yourself for those four, use the positional spelling.

## An empty workspace is a workspace

Write the two spellings as two overloads. Do not write one function with
`Span<std::byte> ws = {}` and an `if (ws.data() != nullptr)` inside.

A null span is not a synonym for "the caller passed nothing". Code that
sub-allocates from a `BumpAllocator` runs its own algorithm once in sizing mode,
where every pool allocation legitimately yields an empty span while the input
matrices stay real. A null check there turns the measurement pass into a real
factorisation over the caller's live data.

The argument checker in `queue-dispatch.hh` follows the same rule from the other
side: it skips zero-length spans, because a sizing pass hands out empty spans by
design and they carry no pointer worth checking.

## Name the option type on an empty option struct

Inside the library, where calls are written `potrf<B>(...)` with the backend
fixed:

```cpp
potrf<B>(ctx, A, PotrfOptions{}, my_workspace);   // correct
potrf<B>(ctx, A, {}, my_workspace);               // ill-formed: ambiguous
```

`potrf`'s two 4-argument overloads take `(ctx, A, const PotrfOptions&,
Span<std::byte>)` and `(ctx, A, Uplo, Span<std::byte>)`. `{}` converts to both,
and not on equal terms: it reaches an enum by an exact match and a class type
only by a user-defined conversion. Left alone, the positional overload wins
silently, `{}` means `Uplo{}` — `Upper`, since `Uplo` is declared
`{Upper, Lower}` — while `PotrfOptions{}.uplo` is `Lower`. The call factorises
the other triangle and returns wrong numbers with no diagnostic.

A third overload closes it: `potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous,
Span<std::byte>) = delete` takes a dedicated *enum* type, so it sits at the same
exact-match rank as the positional one and the bare-`{}` call is ambiguous
rather than silently resolved. The compiler names all three candidates:

```
error: call of overloaded 'potrf(...)' is ambiguous
note: candidate: potrf(Queue&, const MA&, const PotrfOptions&, Span<std::byte>)
note: candidate: potrf(Queue&, const MA&, Uplo, Span<std::byte>)
note: candidate: potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, ...) (deleted)
```

Neither working spelling converts to that type, so both still select the
overloads above.

`potrf` is the only entry point with this collision, and the deleted overload
closes only that one. Name the option type everywhere: an entry point that later
grows a same-arity positional spelling will otherwise reintroduce the trap.

`OptionsApi.NamedEmptyOptionsSelectTheOptionOverload` in
`tests/options_api_tests.cc` pins the two working spellings, and two
`static_assert`s beside it pin the trap type as unreachable from both.

## Where T comes from

The matrix parameters are templates constrained to `Matrix` or `MatrixView`, and
`T` is a *defaulted* template parameter computed from the first of them:

```cpp
template <typename MA, ..., typename T = detail::dense_scalar_t<MA>>
Event gemm(Queue&, const MA& A, ..., const GemmOptions<T>& opts);
```

Keep that shape. It is what makes `{.alpha = 2.0f}` compile: `T` is already fixed
by the time the compiler considers the option parameter, so the braced
initialiser has a concrete type to initialise. Deducing `T` from the option
struct instead would put the option parameter in a deduced context, where a
braced initialiser deduces nothing.

It is also what lets both `Matrix` and `MatrixView` be passed, and mixed. The
positional entry points have a `Matrix` wrapper alongside the `MatrixView`
primary; everything converts to `MatrixView` before the positional call.

## Documentation examples are compile-checked

Most code blocks in `docs/cpp-api.md` have a counterpart in
`docs_cpp_api_examples` in `tests/linalg_layer_tests.cc`. Nothing calls them;
they only have to compile. If you change a signature the document shows, change
them together — and add the block to the guard if it is not there yet.

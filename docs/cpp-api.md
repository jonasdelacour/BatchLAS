# The BatchLAS C++ API

This describes the public C++ calling conventions after the API modernisation.
If you have code written against the older spelling, see
[Migrating from the old API](#migrating-from-the-old-api) at the end — the old
spelling still works, so migration can be incremental.

Every example here has a compile-checked counterpart in
`tests/options_api_tests.cc` and `tests/linalg_layer_tests.cc`.

## The short version

```cpp
#include <blas/linalg.hh>
using namespace batchlas;

Queue ctx(Device::default_device());          // backend resolved from the device
Matrix<float, MatrixFormat::Dense> A(n, n, batch), B(n, n, batch), C(n, n, batch);

gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, A.view(), {.uplo = Uplo::Upper});
ctx.wait();
```

Three things are implicit there, and each is worth knowing about.

## 1. The backend comes from the Queue

A `Queue` carries the backend it dispatches to. You do not pass a `Backend`
template argument to each call.

```cpp
Queue ctx(Device::default_device());                    // AUTO: resolved from the device vendor
Queue host(Device::default_device(), Backend::NETLIB);  // pinned

ctx.set_backend(Backend::CUDA);                         // or change it later
Backend b = ctx.backend();                              // the resolved backend
```

`Backend::AUTO` is resolved once, on first use, from the device vendor: NVIDIA
→ CUDA, AMD → ROCM, Intel → MKL, otherwise NETLIB. If the resolved backend is
not compiled into the build, the call throws `std::runtime_error` naming what
was missing.

To check before committing to a backend:

```cpp
if (Queue::backend_available(Backend::CUDA)) ctx.set_backend(Backend::CUDA);
```

`set_backend` throws rather than silently falling back, because a silent
fallback turns "my CUDA build isn't using CUDA" into a performance mystery
rather than an error.

### Backends are still compile-time internally

Backend selection is a runtime switch over compile-time instantiations, not
virtual dispatch — the vendor call is resolved statically, exactly as before.
The switch happens once per call, in `with_backend` (`blas/queue-dispatch.hh`).

If you need the compile-time backend yourself:

```cpp
with_backend(ctx, [&](auto Back) {
    constexpr Backend B = Back.value;
    gemm<B, float>(ctx, A.view(), B_.view(), C.view(), 1.0f, 0.0f,
                   Transpose::NoTrans, Transpose::NoTrans);
});
```

## 2. Options are structs with defaults

Each entry point takes an option struct, so you write only what differs from
the default. Designated initialisers make the call self-documenting:

```cpp
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f, .transA = Transpose::Trans});
syev(ctx, A.view(), W.to_span(), {.jobz = JobType::NoEigenVectors});
getrs(ctx, LU.view(), X.view(), pivots, {.trans = Transpose::Trans});
```

The available structs are in `include/blas/options.hh`: `GemmOptions<T>`,
`GemvOptions<T>`, `SymmOptions<T>`, `SyrkOptions<T>`, `Syr2kOptions<T>`,
`TrmmOptions<T>`, `TrsmOptions<T>`, `PotrfOptions`, `GetrsOptions`,
`SyevOptions`.

`T` is deduced from the matrix arguments, never from the option struct. That is
what makes `{.alpha = 2.0f}` work: by the time the compiler considers the option
parameter, `T` is already fixed, so the braced initialiser has a concrete type
to initialise. An option struct in a deduced position would not compile — which
is why `alpha` cannot be the thing that determines `T`.

`getrf`, `getri`, `geqrf` and `orgqr` have no options to carry, so they have no
option struct.

## 3. Workspaces come from the queue's arena

The LAPACK-style entry points need scratch space. Leaving the workspace argument
out leases it from a per-`Queue` arena, sized by the matching `*_buffer_size`:

```cpp
potrf(ctx, A.view(), {.uplo = Uplo::Lower});   // workspace leased and returned
```

instead of the older three-step dance:

```cpp
UnifiedVector<std::byte> ws(potrf_buffer_size<Backend::CUDA, float>(ctx, A.view(), Uplo::Lower));
potrf<Backend::CUDA, float>(ctx, A.view(), Uplo::Lower, ws.to_span());
```

The arena grows to the high-water mark of what has been asked of it and then
stops allocating, so a repeated call reuses the same memory rather than
malloc/free-ing device memory each time. `ctx.workspace_capacity()` reports its
current size.

You can also lease explicitly:

```cpp
auto lease = ctx.workspace(n_bytes);
Span<std::byte> bytes = lease.span();
// released when `lease` goes out of scope
```

### When to keep managing the workspace yourself

Passing a span explicitly still works and is still the right thing inside an
algorithm that is already sub-allocating from its own pool:

```cpp
potrf(ctx, A.view(), {.uplo = Uplo::Lower}, my_span);
```

**The one real caveat.** A lease is released when the call returns, and the bytes
go to the next borrower rather than being freed. On an in-order queue — the
default — the next borrower's work is ordered behind this call's, so this is
safe. On an **out-of-order** queue it is not: two overlapping calls could be
handed the same bytes. There, either `wait()` between calls or pass your own
span.

For the same reason, do not build a workspace out of a local `UnifiedVector` and
let it go out of scope before the kernels using it have run — the memory is
freed while the device may still be reading it. Use the arena, whose lifetime is
tied to the queue rather than the enclosing scope.

## 4. The `linalg` convenience layer

`batchlas::linalg` (`include/blas/linalg-ops.hh`) offers value-returning and
elementwise operations. Free functions only; there are no operator overloads.

```cpp
#include <blas/linalg.hh>

auto C = linalg::matmul(ctx, A.view(), B.view());   // allocates and returns C
auto L = linalg::cholesky(ctx, A.view());           // A is not modified
auto X = linalg::solve(ctx, A.view(), B.view());    // A X = B
auto w = linalg::eigvalsh(ctx, A.view());           // eigenvalues only
auto e = linalg::eigh(ctx, A.view());               // e.values, e.vectors
```

and elementwise arithmetic, which the BLAS surface has no place for:

```cpp
auto S = linalg::add(ctx, A.view(), B.view());
auto P = linalg::multiply(ctx, A.view(), B.view());   // Hadamard, NOT matmul
linalg::scale<float>(ctx, A.view(), 2.0f);            // in place
linalg::axpby_into<float>(ctx, 2.0f, A.view(), 3.0f, B.view(), C.view());
```

Every one of these has an `_into` form taking an explicit output, which is what
you want in a loop; the value-returning forms allocate their result.

Use the value-returning forms where clarity matters more than controlling
allocation — setup, tests, exploration. In an inner loop, prefer the
out-parameter forms so the caller owns and reuses the output.

Two behaviours worth knowing:

- `matmul` forces `beta = 0`. The result is freshly allocated, so honouring a
  caller-supplied `beta` would read uninitialised memory.
- `multiply` is elementwise. For square operands both readings are shape-valid,
  so nothing but the values will tell you if you meant `matmul`.

`linalg::qr` does not exist yet: returning `{Q, R}` needs a triangular-extract
kernel, and there is no `triu`/`tril` helper to build it on.

## Migrating from the old API

The old spelling still compiles. Nothing below is urgent.

| Old | New |
| --- | --- |
| `gemm<Backend::CUDA, float>(ctx, A, B, C, alpha, beta, tA, tB, prec)` | `gemm(ctx, A, B, C, {.alpha = alpha, ...})` |
| `potrf<B, T>(ctx, A, uplo, ws)` after sizing `ws` | `potrf(ctx, A, {.uplo = uplo})` |
| `Queue q(dev); f<Backend::CUDA>(q, ...)` | `Queue q(dev, Backend::CUDA); f(q, ...)` |

Mechanically:

1. Move the backend from the call to the `Queue` constructor (or leave it
   `AUTO`).
2. Drop the explicit `<Backend, T>` template arguments. `T` is deduced.
3. Replace the positional trailing arguments with a designated initialiser,
   omitting anything at its default.
4. Delete the `*_buffer_size` call and the `UnifiedVector<std::byte>` unless you
   have a reason to keep owning the workspace.

Step 4 is the one that changes behaviour: workspace memory is now reused across
calls instead of being allocated and freed each time. If you were relying on the
workspace being freshly zeroed, note that neither spelling ever guaranteed that.

### A gotcha if you extend the API

Entry points get their queue-dispatch overload from `BATCHLAS_DISPATCH_ON_QUEUE`
in `blas/queue-dispatch.hh`. That overload is a variadic forwarding template, and
it is **constrained** with a `requires` clause for a reason: an unconstrained
pack matches any argument list, so it would beat every more specific overload
(such as the option-struct ones) and then fail to compile inside its own body,
reporting the error at the macro expansion rather than at your call site. If you
add an entry point, keep the constraint.

For the same reason, never give an arena-backed overload the same arity and
parameter types as the positional call — the two would be genuinely ambiguous.
That is why `getrf`/`getri`/`geqrf`/`orgqr` take no workspace parameter in their
arena spelling.

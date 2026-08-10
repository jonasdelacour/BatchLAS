# The BatchLAS C++ API

This describes the public C++ calling conventions after the API modernisation.
If you have code written against the older spelling, see
[Migrating from the old API](#migrating-from-the-old-api) at the end — the old
spelling still works, so migration can be incremental.

The calling-convention examples here have compile-checked counterparts in
`tests/options_api_tests.cc` and `tests/linalg_layer_tests.cc`.

## The short version

```cpp
#include <batchlas.hh>                            // umbrella header
using namespace batchlas;

Queue ctx(Device::default_device());          // backend resolved from the device
Matrix<float, MatrixFormat::Dense> A(n, n, batch), B(n, n, batch), C(n, n, batch);

gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, A.view(), {.uplo = Uplo::Upper});
ctx.wait();                                   // nothing is readable before this
```

`<batchlas.hh>` is the recommended entry point and is what the rest of this
document uses. It is a one-line wrapper around `<blas/linalg.hh>`, which is the
spelling used inside the repository and in older code; the two are
interchangeable. Neither is collision-proof — the installed headers still spell
their own includes `<blas/...>`, `<util/...>` and `<internal/...>`, so a project
with a header of its own at one of those paths will shadow them.

Before the calling conventions, two things you cannot guess from that snippet:
where the data has to live and how it is laid out, and when results become
readable. Then three things that are implicit in the calls themselves.

## Data layout and memory

### Column-major, always

Matrices are **column-major**, like LAPACK and unlike NumPy. For a dense
`MatrixView V`, element `(i, j)` of batch item `b` lives at

```cpp
V.data_ptr()[b * V.stride() + j * V.ld() + i]
```

`i` is the row, `j` is the column, and all three of `ld`, `stride` and the index
are counted in **elements**, not bytes. Element access computes exactly that
expression: `M(i, j, b)` on an owning `Matrix` (the batch index is not optional
there), and `V.at(i, j, b)` or `V(i, j, b)` on a `MatrixView`, which bounds-checks
and throws `std::out_of_range`.

- **`ld`** — leading dimension, the element distance between column `j` and
  column `j+1`. It defaults to `0`, which means "packed": `ld = rows`. It must be
  at least `rows`; a larger `ld` is how you view a sub-block of a bigger buffer.
- **`stride`** — the element distance between batch item `b` and item `b+1`. It
  defaults to `0`, which means `ld * cols`, i.e. the items are packed back to
  back.

Both defaults are resolved the same way in `Matrix(rows, cols, batch, ld, stride)`
and in `MatrixView(data, rows, cols, ld, stride, batch)`.

If your data really is row-major, `Matrix::to_column_major()` returns a converted
copy (and `to_row_major()` goes back). See the operand-swap recipe below for the
zero-copy alternative on `gemm`.

### Where the memory has to live: the USM contract

**Every pointer you hand to `MatrixView` must be device-accessible for the
backend the `Queue` dispatches to.** The constructor takes a bare `T*` and cannot
check this. Getting it wrong is the nastiest trap in the API:

```cpp
std::vector<float> host(n * n * batch);          // compiles fine
MatrixView<float, MatrixFormat::Dense> A(host.data(), n, n);
gemm(ctx, A, A, A, {});                          // CUDA_ERROR_ILLEGAL_ADDRESS
```

On a CUDA queue that aborts the process. The thrown exception *is* catchable and
`main` can return normally — and then the runtime raises `SIGABRT` during
teardown, which nothing you write can save you from. Worse, the same code is
*correct* on the NETLIB (host) backend, so a CPU prototype passes and the GPU run
dies.

Allocations that work zero-copy, all verified on the CUDA backend:

- `sycl::malloc_device`, `sycl::malloc_shared`, `sycl::malloc_host` — including
  allocations made on your own `sycl::context`, as long as it is the same device;
- `cudaMalloc` and `cudaMallocManaged`.

Allocations that do **not** work on a GPU backend: `malloc`, `new`,
`std::vector`, and anything else backed by ordinary host memory.

### Getting host data in

`Matrix` allocates USM **shared** memory (`sycl::malloc_shared`, on a per-device
context that outlives any individual `Queue`), so the simplest correct recipe is
to let `Matrix` own the memory and fill it from the host:

```cpp
Matrix<float, MatrixFormat::Dense> A(n, n, batch);
for (int b = 0; b < batch; ++b)
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      A(i, j, b) = host_value(i, j, b);          // column-major, host-writable

gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
ctx.wait();
float c00 = C(0, 0, 0);                          // readable after the wait
```

There is also a copying constructor, `Matrix(const T* data, rows, cols, ld,
stride, batch)`, which is the natural "adopt my existing buffer" entry point.
Give it explicit, consistent arguments — for packed host data that means
`ld == rows` and `stride == ld * cols`. Exotic layouts through this constructor
are thinly tested; when in doubt, use the loop above.

`MatrixView` never owns anything. There is no adopting owner: to take ownership
of an existing allocation you copy.

### Row-major data: the operand swap

A column-major view of a row-major `m x k` buffer with row length `k` is exactly
its transpose: `MatrixView<T>(p, /*rows=*/k, /*cols=*/m, /*ld=*/k)` is `Aᵀ`.
Since `Cᵀ = Bᵀ Aᵀ`, feeding `gemm` the transposed views **in the opposite order**
computes the row-major product with no copy and no transpose flags:

```cpp
// Row-major A (m x k), B (k x n), C (m x n), packed, in USM at pa/pb/pc.
MatrixView<float, MatrixFormat::Dense> At(pa, k, m, k);   // = Aᵀ
MatrixView<float, MatrixFormat::Dense> Bt(pb, n, k, n);   // = Bᵀ
MatrixView<float, MatrixFormat::Dense> Ct(pc, n, m, n);   // = Cᵀ
gemm(ctx, Bt, At, Ct, {});                                // C = A B, row-major
```

This does **not** generalise. `potrf`, `getrf` and `syev` have no transpose knob;
a row-major caller of the symmetric routines flips `uplo` instead (a row-major
upper triangle is a column-major lower one). And note that a wrong-major `gemm`
result is not the transpose of the right answer, so "transpose it afterwards"
does not rescue you — you get plausible, wrong numbers.

### `Matrix(n, n, batch)` means something else for CSR

The two owning constructors have the same spelling and different meanings:

```cpp
Matrix<float, MatrixFormat::Dense> D(rows, cols, batch_size, ld, stride);
Matrix<float, MatrixFormat::CSR>   S(rows, cols, nnz, batch_size);
```

The third argument is the **batch size** for dense and the **non-zero count** for
CSR. `Matrix<float, MatrixFormat::CSR> S(n, n, batch)` therefore compiles, means
"one matrix with `batch` non-zeros", and produces no diagnostic.

## Synchronisation and threading

**Every entry point enqueues work and returns immediately**, handing back an
`Event`. The contents of a `Matrix`, `MatrixView` or `UnifiedVector` are not
readable until that work has finished:

```cpp
Event e = gemm(ctx, A.view(), B.view(), C.view(), {});
e.wait();                    // wait on this call, or
ctx.wait();                  // wait on everything enqueued on the queue
ctx.wait_and_throw();        // ... and rethrow asynchronous errors
```

Reading without waiting is the single most common way to get zeros out of a call
that worked. A few paths happen to synchronise internally today — `linalg::eigvalsh`
does, which is why omitting the wait appears to work there — but that is an
implementation detail and not something to rely on.

**A `Queue` is single-threaded.** It owns an unsynchronised workspace arena and a
cached "last event", so handing one `Queue` to a thread pool corrupts both — the
arena double-frees or blows its capacity up by orders of magnitude, and the
cached event aborts inside the SYCL runtime even when you supply your own
workspace. The rule is one `Queue` per thread, and it is enforced rather than
merely documented: the operations that mutate that state compare
`std::this_thread::get_id()` against the thread that constructed the `Queue` and
throw `std::runtime_error` if they differ.

Queues built for the same `Device` share a SYCL context, so per-thread queues
still see each other's USM allocations — what is per-thread is the arena and the
event bookkeeping, not the memory. Moving a `Queue` to another thread and using
it exclusively there is fine; say so with `attach_to_current_thread()`, or the
guard fires on the first call from the new thread.

## Interop with CUDA and with your own SYCL

`Queue::native_handle()` returns the backend-native stream as a `void*` — a
`CUstream` (`cudaStream_t`) on CUDA, a `hipStream_t` on HIP, `nullptr` on every
other backend including CPU. `static_cast` it and use it for `cublasSetStream`,
`cudaMemcpyAsync` or your own kernels. It belongs to the `Queue`: do not destroy
it and do not let it outlive the `Queue`. Work you push onto that stream is
ordered by the stream, so on the default in-order `Queue` it runs after
everything BatchLAS has already submitted. To make BatchLAS wait for *your*
work, call `ctx.create_event_after_external_work()` once you have enqueued it.
No SYCL types are involved, so this needs no extra include.

For SYCL-typed interop, include `<batchlas/sycl_interop.hh>`. It is the one
BatchLAS header that pulls in `<sycl/sycl.hpp>`, and it is deliberately not
reachable from `<batchlas.hh>` or `<blas/linalg.hh>` — that cut is worth about
4 s of compile time per translation unit — so include it only in the TUs that
need it and do not re-export it from your own headers. It provides:

```cpp
batchlas::sycl_queue(const Queue&)   -> sycl::queue&
batchlas::sycl_event(const Event&)   -> sycl::event
batchlas::event_from_sycl(sycl::event) -> Event
```

The last one is what lets a foreign SYCL queue interoperate with no host sync:

```cpp
sycl::event mine = my_queue.submit(/* ... */);
Event e = batchlas::event_from_sycl(mine);
ctx.enqueue(e);                                       // `enqueue` takes an lvalue
batchlas::potrf(ctx, A.view(), {.uplo = Uplo::Lower}); // waits for `mine`

// ... and in the other direction:
my_queue.ext_oneapi_submit_barrier({batchlas::sycl_event(ctx.get_event())});
```

Both queues must live in the same SYCL context. Memory needs none of this:
pointers from `cudaMalloc`, `cudaMallocManaged`, `sycl::malloc_device` and
`sycl::malloc_host` all wrap into `Span`/`MatrixView` zero-copy as long as they
are reachable from that context.

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

This applies to the whole surface, extensions included: `ortho`, `syevx`,
`lanczos`, `steqr`, `stedc`, the `sytrd_*` and `syev_*` family and the rest all
take their backend from the queue. Where an entry point has template parameters
that cannot be deduced from its arguments — `tridiagonal_solver_buffer_size`,
whose arguments are all scalars — it keeps its explicit
`f<Backend, T>(...)` spelling, because there is nothing for the compiler to
work from.

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

Most entry points take an option struct, so you write only what differs from
the default. Designated initialisers make the call self-documenting:

```cpp
UnifiedVector<float>   W(n * batch);        // eigenvalues: the REAL type, even for complex A
UnifiedVector<int64_t> pivots(n * batch);   // getrf/getrs pivots are int64_t

gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f, .transA = Transpose::Trans});
syev(ctx, A.view(), W, {.jobz = JobType::NoEigenVectors});
getrs(ctx, LU.view(), X.view(), pivots, {.trans = Transpose::Trans});
```

`UnifiedVector<T>` converts implicitly to `Span<T>`, which is what those
parameters are, so no `.to_span()` and no `.view()` is needed at those call
sites. `Vector<T>` is a different type — a strided, batched vector — and is not
what these take.

There is one option struct per entry point that has options, and they all live in
`include/blas/options.hh` — read them there rather than trusting a list here.
Today there are thirteen: one per dense BLAS routine (`gemm`, `gemv`, `symm`,
`hemm`, `herk`, `her2k`, `syrk`, `syr2k`, `trmm`, `trsm` — all templated on `T`)
plus `PotrfOptions`, `GetrsOptions` and `SyevOptions`.

Three groups of entry points have **no** option struct:

- `getrf`, `getri`, `geqrf` and `orgqr` have nothing to carry, so their only new
  spelling is the arena-backed one — drop the workspace argument and it is leased
  for you.
- `gesvd`, `ormqr` and `ortho` keep their positional spelling *and* their explicit
  workspace parameter; there is no option struct and no arena overload. You can
  still use the arena for them, by leasing it yourself:

  ```cpp
  auto ws = ctx.workspace(gesvd_buffer_size<Backend::CUDA, float>(
                              ctx, A.view(), S, U.view(), Vh.view(), jobu, jobvh));
  gesvd<Backend::CUDA, float>(ctx, A.view(), S, U.view(), Vh.view(), jobu, jobvh, ws.span());
  ```

- Entry points whose template parameters cannot be deduced from their arguments
  keep the explicit `f<Backend, T>(...)` form; see section 1.

`T` is deduced from the matrix arguments, never from the option struct. That is
what makes `{.alpha = 2.0f}` work: by the time the compiler considers the option
parameter, `T` is already fixed, so the braced initialiser has a concrete type
to initialise. An option struct in a deduced position would not compile — which
is why `alpha` cannot be the thing that determines `T`.

Two consequences worth knowing:

- **You cannot name `T` on an option-struct call.** `syev<B, float>(ctx, ...)`
  works on the positional spelling but not the option one, where the second
  template parameter is the matrix type. Write `syev<B>(ctx, ...)` and let `T`
  deduce, or use the positional spelling.
- **`Matrix` and `MatrixView` are both accepted**, and may be mixed freely.
  Elsewhere in the library, an entry point whose parameter is a `MatrixView<T>`
  cannot deduce `T` from an owning `Matrix<T>` — deduction ignores the implicit
  conversion — so those calls need an explicit `.view()`. `Vector` has a
  `.view()` for the same reason. Note that the `Span`-valued parameters —
  eigenvalues, singular values, `tau`, pivots — are a separate story: pass a
  `UnifiedVector<T>` there, which converts implicitly, not a `Vector<T>`.

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

A repeated call reuses the same memory rather than malloc/free-ing device memory
each time, and the arena never frees on its own — it grows and holds.
`ctx.workspace_capacity()` reports its current size.

"Grows" is worth being precise about, because it is not simply "the high-water
mark". The arena serves from a list of blocks and only ever *appends*: a request
that does not fit in the current block opens a new one (geometrically sized)
rather than replacing it. Repeated same-size calls and a descending sequence of
sizes therefore settle at the peak, as you would expect — but an ascending ramp
(64 → 128 → 256 → 512) retains the **sum** of the blocks it opened, which is
several times the largest single request. Three ways to control it, cheapest
first: pass your own span (capacity stays at 0), destroy the `Queue` (the arena
goes with it), or call `ctx.trim_workspace()`, which frees the blocks and drops
capacity back to nothing. `trim_workspace()` is `[[nodiscard]]`: it returns
`false` and does nothing while any lease is outstanding, and it drains the queue,
so it can throw.

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
default, and what `Queue(device)` gives you — the next borrower's work is ordered
behind this call's, so this is safe and free.

On an **out-of-order** queue nothing orders the two, so the release *drains the
queue* before handing the bytes back. It is still safe; what it costs is
asynchrony. Concretely: on an out-of-order `Queue`, every arena-backed call
blocks until the device is idle before it returns, where the spelling that takes
a caller-supplied span does not. If you chose an out-of-order queue in order to
overlap work, pass your own span; otherwise nothing is wrong, the call is just
synchronous.

Two things the drain does not cover, both worth knowing before relying on it:

- It waits on the queue the lease was taken from, and nothing else. The
  dispatchers that need ordering (`syev`, `gesvd`, `ormqr`, `iluk`) build a
  *derived* in-order queue from your `ctx` and run their kernels there, while the
  lease belongs to `ctx`. What makes that safe is the derived queue being
  destroyed, and so drained, before the dispatcher returns. In general: a lease
  released on queue X does not order against work submitted to a queue derived
  from X.
- Reassigning a live lease (`ws = ctx.workspace(bigger);`) acquires the new loan
  before releasing the old one, so in a loop the arena ratchets instead of
  reusing. Call `ws.release()` first if that matters.

`include/util/workspace.hh` carries the full lifetime rules; this is a summary of
them.

For the same reason, do not build a workspace out of a local `UnifiedVector` and
let it go out of scope before the kernels using it have run — the memory is
freed while the device may still be reading it. Use the arena, whose lifetime is
tied to the queue rather than the enclosing scope.

## 4. The `linalg` convenience layer

`batchlas::linalg` (`include/blas/linalg-ops.hh`) offers value-returning and
elementwise operations. Free functions only; there are no operator overloads.

```cpp
#include <batchlas.hh>

auto C = linalg::matmul(ctx, A.view(), B.view());   // allocates and returns C
auto L = linalg::cholesky(ctx, A.view());           // A is not modified
auto X = linalg::solve(ctx, A.view(), B.view());    // A X = B
auto w = linalg::eigvalsh(ctx, A.view());           // eigenvalues only
auto e = linalg::eigh(ctx, A.view());               // e.values, e.vectors
ctx.wait();                                         // required before reading any of them
```

These allocate and return, but they do not wait: like every other entry point
they enqueue. `C`, `L` and `X` read as zeros until `ctx.wait()`. (`eigvalsh`
happens to synchronise internally today, which makes forgetting the wait look
like it works for that one call. Do not build on that.)

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

### Write `PotrfOptions{}`, never a bare `{}`

Inside the library, where calls are written `potrf<B>(...)` with the backend
fixed, an empty option struct must have its type named:

```cpp
potrf<B>(ctx, A, PotrfOptions{}, my_workspace);   // correct
potrf<B>(ctx, A, {}, my_workspace);               // WRONG: factorises Upper
```

`potrf`'s option overload takes `(ctx, A, const PotrfOptions&, Span<std::byte>)`
and the positional one takes `(ctx, A, Uplo, Span<std::byte>)` — same arity, and
`{}` converts to both. Overload resolution picks the positional one, so `{}`
means `Uplo{}`; since `Uplo` is declared `{Upper, Lower}`, that is **Upper**,
while `PotrfOptions{}.uplo` is **Lower**. The call silently factorises the other
triangle and returns a wrong answer with no diagnostic.

This bit `ortho`'s Cholesky path, where it showed up only as LOBPCG failing to
converge — several layers away from the call. `potrf` is the only entry point
with this collision today; every other option overload differs from its
positional twin in arity or in an argument type. Naming the type costs nothing
and is immune, so do it everywhere.

`OptionsApi.NamedEmptyOptionsSelectTheOptionOverload` in
`tests/options_api_tests.cc` enforces this.

### An empty workspace is a workspace

The workspace-taking and arena-leasing spellings are separate overloads, not one
function with `Span<std::byte> ws = {}` and a null check. If you add an entry
point, keep it that way. A null span is *not* a synonym for "the caller passed
nothing": code that sub-allocates from a `BumpAllocator` runs its algorithm once
in sizing mode, where every pool allocation legitimately yields an empty span
while the input matrices stay real. A null check there turns the measurement
pass into a real factorisation over the caller's live data.

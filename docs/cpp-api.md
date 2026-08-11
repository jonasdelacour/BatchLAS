# The BatchLAS C++ API

This is the reference for the public C++ calling conventions: where your data has
to live, how it is laid out, when results become readable, and how the backend,
options and workspaces are spelled at a call site.

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

Everything installs under `<prefix>/include/batchlas/`, plus the umbrella file
`<prefix>/include/batchlas.hh`. Include `<batchlas.hh>`, or reach in directly:

```cpp
#include <batchlas/blas/linalg.hh>            // what <batchlas.hh> pulls in
#include <batchlas/util/sycl-device-queue.hh>
```

`batchlas` is the only name BatchLAS claims in your include root, and every
public header is spelled `<batchlas/...>`.

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

For row-major source data, use the operand swap below — it costs nothing.
`Matrix::to_column_major()` returns a converted copy and `to_row_major()` goes
back. Both read the source with its own `ld` and `stride` and return a packed
matrix (`ld == rows`, `stride == rows * cols`). Both synchronise before
returning: the no-argument form on a queue it builds itself, the
`to_column_major(ctx)` / `to_row_major(ctx)` form on the queue you pass.

For a *row-major* source there is only one pitch field to carry the row pitch,
and it is `ld`: a padded row-major buffer is spelled `Matrix(data, rows, cols,
ld = row_pitch, ...)`, which the constructor accepts when `row_pitch >= rows`.
When `ld == rows` the buffer can only be packed row-major, and the row pitch is
`cols`.

### Where the memory has to live: the USM contract

**Every pointer you hand to `MatrixView` or `Span` must be device-accessible for
the backend the `Queue` dispatches to.** `MatrixView` takes a bare `T*`, so it
cannot check this at construction; the entry points that take their backend from
the queue check every pointer argument at the call and throw
`std::invalid_argument`. The explicit `f<Backend, T>(...)` spellings do not
check — they go straight to the backend.

```cpp
std::vector<float> host(n * n * batch);          // compiles fine
MatrixView<float, MatrixFormat::Dense> A(host.data(), n, n);
gemm(ctx, A, A, A, GemmOptions<float>{});        // throws std::invalid_argument
```

```
BatchLAS: gemm: argument 1 points to memory that is not reachable from this
Queue's device (NVIDIA GeForce RTX 4090).
It looks like ordinary host memory -- a std::vector, new[] or malloc.
...
Use memory the device can reach:
  - let Matrix<T, MatrixFormat::Dense> own it (it allocates USM shared, ...
```

`Queue::is_device_accessible(ptr)` asks the same question and returns a `bool`
instead of throwing. `BATCHLAS_SKIP_POINTER_CHECKS=1` turns the check off for a
hot loop whose pointers are already validated.

An argument that addresses no elements is exempt, because no kernel can
dereference it. That covers the empty `Span` a sizing pass hands out, and the
default-constructed view that means "this optional matrix is not in use":

```cpp
syevx(ctx, A, W, k, ws, JobType::NoEigenVectors,
      MatrixView<float, MatrixFormat::Dense>(), params);   // fine, not checked
```

Allocations that work zero-copy on a GPU backend:

- `sycl::malloc_device`, `sycl::malloc_shared`, `sycl::malloc_host` — including
  allocations made on your own `sycl::context`, as long as it is the same device;
- `cudaMalloc` and `cudaMallocManaged`.

Allocations that do **not** work on a GPU backend: `malloc`, `new`,
`std::vector`, and anything else backed by ordinary host memory. On a host/CPU
device ordinary host memory *is* what the kernels read, so nothing is rejected
there — which is why a CPU prototype can pass where the GPU run would throw.

### Getting host data in

`Matrix` owns USM **shared** memory (`sycl::malloc_shared`, on a per-device
context that outlives any individual `Queue`), so the host can read and write it
directly. Load it in bulk with the copying constructor:

```cpp
std::vector<float> host(size_t(n) * n * batch);   // column-major, packed
fill_from_wherever(host);

Matrix<float, MatrixFormat::Dense> A(
    Span<const float>(host.data(), host.size()),
    n, n, /*ld=*/n, /*stride=*/0, /*batch_size=*/batch);
```

`(ld, stride)` describe the **source** buffer: element `(i, j, b)` is read from
`data[b * stride + j * ld + i]`, with `ld = 0` meaning `rows` and `stride = 0`
meaning `ld * cols`; `ld` has no default on these constructors and must be
passed. A packed source is copied in a single `std::copy`; a padded `ld` or a
gapped `stride` is copied one column at a time, so neither the padding nor the
gaps are read. The copy keeps your `ld` and packs the batch items back to back.

Prefer the `Span<const T>` overload over the raw-pointer one,
`Matrix(const T* data, rows, cols, ld, stride, batch_size)`. The span knows the
source length, so a shape that would over-read throws `std::invalid_argument`;
the pointer overload cannot check that. Both throw on null data, non-positive
dimensions, `ld < rows`, and a batched `stride` smaller than `ld * cols`.

If the data is generated rather than read in, skip the host entirely. These run
on the device and synchronise before returning:

```cpp
auto R = Matrix<float, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch);
auto I = Matrix<float, MatrixFormat::Dense>::Identity(n, batch);
auto Z = Matrix<float, MatrixFormat::Dense>::Zeros(n, n, batch);
```

`Identity`, `Random`, `RandomTriangular`, `Zeros`, `Ones`, `Diagonal`,
`Triangular` and `TriDiagToeplitz` allocate and fill. To fill a matrix you
already own, the `fill_*` family on `MatrixView` writes in place and returns an
`Event`:

```cpp
UnifiedVector<float> d(n);                        // diagonal values, USM
B.view().fill_diagonal(ctx, d.to_span());         // one kernel, no host loop
B.view().fill_zeros(ctx);
```

`fill`, `fill_zeros`, `fill_ones`, `fill_identity`, `fill_diagonal`,
`fill_triangular`, `fill_tridiag`, `fill_tridiag_toeplitz`, `fill_random` and
`fill_triangular_random` are in `include/batchlas/blas/matrix.hh`, alongside
`fill_random_sparse_hermitian` for CSR. Most have a `(const Queue&, ...)` form
and a form that builds a queue of its own; `fill_identity` and `fill_tridiag`
take a queue.

To refresh a dense matrix from another one, copy view to view. It lowers to
`memcpy` or `ext_oneapi_memcpy2d` where the layouts allow it, falls back to a
3-D kernel where they do not, and is asynchronous:

```cpp
MatrixView<float, MatrixFormat::Dense>::copy(ctx, dst.view(), src.view());
```

Element access — `A(i, j, b)` on an owning `Matrix`, `V.at(i, j, b)` on a view —
works from the host because the memory is shared, but it is one indexed store
into managed memory per element. Use it to set or inspect a handful of entries,
and in tests. Do not use it to load a batch.

`MatrixView` never owns. To own an existing allocation, copy it into a `Matrix`.

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
gemm(ctx, Bt, At, Ct, GemmOptions<float>{});              // C = A B, row-major
```

This does not generalise. `potrf`, `getrf` and `syev` have no transpose knob; a
row-major caller of the symmetric routines flips `uplo` instead, since a
row-major upper triangle is a column-major lower one. A wrong-major result cannot
be repaired by transposing it afterwards.

### The CSR non-zero count has its own type

The two owning constructors line up positionally — shape, then the
format-specific extra, then the batch size — and the CSR non-zero count is
spelled with the `NonZeros` strong typedef:

```cpp
Matrix<float, MatrixFormat::Dense> D(rows, cols, batch_size, ld, stride);
Matrix<float, MatrixFormat::CSR>   S(rows, cols, NonZeros{nnz}, batch_size);
```

`NonZeros` has an `explicit` constructor and no conversion back to `int`, so an
all-`int` CSR call selects a deleted overload whose comment names the fix. The
from-data constructors follow the same order — buffers, shape, `NonZeros{nnz}`,
strides, batch:

```cpp
Matrix<float, MatrixFormat::Dense> D(data, rows, cols, ld, stride, batch_size);
Matrix<float, MatrixFormat::CSR>   S(values, row_offsets, col_indices,
                                     rows, cols, NonZeros{nnz},
                                     matrix_stride, offset_stride, batch_size);
```

`MatrixView` mirrors both.

## Synchronisation and threading

**Every entry point enqueues work and returns immediately**, handing back an
`Event`. The contents of a `Matrix`, `MatrixView` or `UnifiedVector` are not
readable until that work has finished:

```cpp
Event e = gemm(ctx, A.view(), B.view(), C.view(), GemmOptions<float>{});
e.wait();                    // wait on this call, or
ctx.wait();                  // wait on everything enqueued on the queue
ctx.wait_and_throw();        // ... and rethrow asynchronous errors
```

Read a result only after waiting. Without the wait you read the output buffer as
it was before the call — and a fresh `Matrix` is uninitialised, not zeroed; use
`Matrix::Zeros(...)` or `view().fill_zeros(ctx)` if you need a known starting
value.

**A `Queue` is single-threaded.** Use one `Queue` per thread. It owns an
unsynchronised workspace arena and a cached "last event", and the operations that
mutate either — `workspace()`, `trim_workspace()`, submissions, `enqueue()`,
`get_event()`, `create_event_after_external_work()` — compare
`std::this_thread::get_id()` against the thread that constructed the `Queue` and
throw `std::runtime_error` if they differ.

Queues built for the same `Device` share a SYCL context, so per-thread queues
still see each other's USM allocations — what is per-thread is the arena and the
event bookkeeping, not the memory. Moving a `Queue` to another thread and using
it exclusively there is supported: call `attach_to_current_thread()` from the new
owner before its first use, or the guard fires on the first call from the new
thread.

## Interop with CUDA and with your own SYCL

`Queue::native_handle()` returns the backend-native stream as a `void*` — a
`CUstream` (`cudaStream_t`) when the queue's *device* runs on the CUDA SYCL
backend, a `hipStream_t` on HIP, `nullptr` on every other SYCL backend including
CPU. This keys off the device, not off `ctx.backend()` — a queue pinned to
`Backend::NETLIB` on an NVIDIA device still hands you its CUDA stream. Check for
`nullptr`, then `static_cast` it and use
it for `cublasSetStream`, `cudaMemcpyAsync` or your own kernels. It belongs to
the `Queue`: do not destroy it and do not let it outlive the `Queue`. Work you
push onto that stream is ordered by the stream, so on the default in-order
`Queue` it runs after everything BatchLAS has already submitted. To make BatchLAS
wait for *your* work, call `ctx.create_event_after_external_work()` once you have
enqueued it. No SYCL types are involved, so this needs no extra include.

For SYCL-typed interop, include `<batchlas/sycl_interop.hh>`. It is the one
BatchLAS header that pulls in `<sycl/sycl.hpp>`, and it is not reachable from
`<batchlas.hh>`. Include it only in the translation units that move a `Queue` or
an `Event` across the boundary, and do not re-export it from a header of your
own. It provides:

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

## The backend comes from the Queue

A `Queue` carries the backend it dispatches to, and every entry point takes it
from there.

```cpp
Queue ctx(Device::default_device());                    // AUTO: resolved from the device vendor
Queue host(Device::default_device(), Backend::NETLIB);  // pinned

ctx.set_backend(Backend::CUDA);                         // or change it later
Backend b = ctx.backend();                              // the resolved backend
```

`Backend::AUTO` is resolved once, on first use, and cached; `set_backend` resets
the cache. On a **GPU** it takes the vendor's own stack if that backend was
compiled in — NVIDIA → CUDA, AMD → ROCM, Intel → MKL — and otherwise falls back
to NETLIB, as every non-GPU device does. `set_backend` throws
`std::runtime_error` if the *named* backend is not compiled into this build; an
`AUTO` queue throws only when no compiled backend can serve its device at all.
To check first:

```cpp
if (Queue::backend_available(Backend::CUDA)) ctx.set_backend(Backend::CUDA);
```

This applies to the whole surface, extensions included: `ortho`, `syevx`,
`lanczos`, `steqr`, `stedc`, the `sytrd_*` and `syev_*` family and the rest all
take their backend from the queue. An entry point whose template parameters
cannot be deduced from its arguments — `tridiagonal_solver_buffer_size`, whose
arguments are all scalars — keeps the explicit `f<Backend, T>(...)` spelling.

### Getting the compile-time backend

Backend selection is a runtime switch over compile-time instantiations, and it
happens once per call, in `with_backend` (`<batchlas/blas/queue-dispatch.hh>`),
which you can use directly when you need the backend as a constant:

```cpp
with_backend(ctx, [&](auto Back) {
    constexpr Backend Bk = Back.value;
    gemm<Bk>(ctx, A.view(), B.view(), C.view(), 1.0f, 0.0f,
             Transpose::NoTrans, Transpose::NoTrans);
});
```

## Options are structs with defaults

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
parameters are. `Vector<T>` is a different type — a strided, batched vector — and
is not what they take.

There is one option struct per entry point that has options, and they all live in
`include/batchlas/blas/options.hh`: one per dense BLAS routine (`gemm`, `gemv`,
`symm`, `hemm`, `herk`, `her2k`, `syrk`, `syr2k`, `trmm`, `trsm` — all templated
on `T`) plus `PotrfOptions`, `GetrsOptions` and `SyevOptions`.

When you pass an empty option struct *together with* an explicit workspace, name
the type — `potrf(ctx, A.view(), PotrfOptions{}, ws)`. A bare `{}` there is
ambiguous and the compiler says so.

Three groups of entry points have **no** option struct:

- `getrf`, `getri`, `geqrf` and `orgqr` carry nothing, so they have only the
  arena-backed spelling: omit the workspace argument and it is leased for you. To
  manage the workspace yourself, use the positional spelling.
- `gesvd`, `ormqr`, `ortho` and `spmm` take positional arguments and a required
  workspace span. To use the arena for them, lease it yourself:

  ```cpp
  auto ws = ctx.workspace(gesvd_buffer_size<Backend::CUDA, float>(
                              ctx, A.view(), S, U.view(), Vh.view(),
                              SvdVectors::All, SvdVectors::All));
  gesvd<Backend::CUDA, float>(ctx, A.view(), S, U.view(), Vh.view(),
                              SvdVectors::All, SvdVectors::All, ws.span());
  ```

  Wrap that in `with_backend` rather than hardcoding `Backend::CUDA` if the code
  has to run on more than one backend. The explicit `f<Backend, T>(...)` spelling
  does not check its pointer arguments, so a host pointer reaches the vendor call
  instead of throwing.

- Entry points whose template parameters cannot be deduced from their arguments
  keep the explicit `f<Backend, T>(...)` form; see *The backend comes from the
  Queue*.

`T` is deduced from the matrix arguments, never from the option struct. Two
consequences:

- **You cannot name `T` on an option-struct call.** `syev<B, float>(ctx, ...)`
  works on the positional spelling but not the option one, where the second
  template parameter is the matrix type. Write `syev<B>(ctx, ...)` and let `T`
  deduce, or use the positional spelling.
- **`Matrix` and `MatrixView` are both accepted**, and may be mixed freely.
  Elsewhere in the library, an entry point whose parameter is a `MatrixView<T>`
  cannot deduce `T` from an owning `Matrix<T>`, so those calls need an explicit
  `.view()`; `Vector` has a `.view()` for the same reason. The `Span`-valued
  parameters — eigenvalues, singular values, `tau`, pivots — take a
  `UnifiedVector<T>`, which converts implicitly, not a `Vector<T>`.

## Workspaces come from the queue's arena

The LAPACK-style entry points need scratch space. Leaving the workspace argument
out leases it from a per-`Queue` arena, sized by the matching `*_buffer_size`:

```cpp
potrf(ctx, A.view(), {.uplo = Uplo::Lower});   // workspace leased and returned
```

The alternative is to size and own the buffer yourself, and pass it in:

```cpp
UnifiedVector<std::byte> ws(potrf_buffer_size<Backend::CUDA, float>(ctx, A.view(), Uplo::Lower));
potrf<Backend::CUDA, float>(ctx, A.view(), Uplo::Lower, ws.to_span());
ctx.wait();                                    // ws must outlive the kernels
```

This spelling names its backend, so it does not check its pointer arguments
either.

A repeated arena-backed call reuses the same memory rather than malloc/free-ing
device memory each time. The arena never frees on its own: it grows and holds,
and `ctx.workspace_capacity()` reports its current size.

It grows by appending. The arena serves from a list of blocks and never replaces
one: a request that does not fit in the current block opens a new one,
geometrically sized. Repeated same-size calls and a descending sequence of sizes
settle at the peak; an ascending ramp that keeps outgrowing the current block
(128 KB → 256 KB → 512 KB) retains the **sum** of the blocks it opened — always
under twice the largest block, which is itself up to twice the largest single
request. Blocks are never smaller than 64 KB, so a ramp that stays under that
never opens a second one. Three
ways to control it, cheapest first: pass your own span, so capacity stays at 0;
destroy the `Queue`, and the arena goes with it; or call `ctx.trim_workspace()`,
which frees the blocks and drops capacity back to nothing. `trim_workspace()` is
`[[nodiscard]]` — it returns `false` and does nothing while any lease is
outstanding — and it drains the queue, so it can throw.

You can also lease explicitly:

```cpp
auto lease = ctx.workspace(n_bytes);
Span<std::byte> bytes = lease.span();
// released when `lease` goes out of scope
```

### When to keep managing the workspace yourself

Passing a span explicitly is the right thing inside an algorithm that is already
sub-allocating from its own pool:

```cpp
potrf(ctx, A.view(), {.uplo = Uplo::Lower}, my_span);
```

It is also what keeps an arena-backed call asynchronous on an **out-of-order**
queue. A lease is released when the call returns, and the bytes go to the next
borrower rather than being freed. On an in-order queue — the default, and what
`Queue(device)` gives you — the next borrower's work is ordered behind this
call's, so the handover is safe and free. On an out-of-order queue nothing orders
the two, so the release drains the queue before handing the bytes back: every
arena-backed call blocks until the device is idle before it returns. If you chose
an out-of-order queue in order to overlap work, pass your own span.

Two things the drain does not cover:

- It waits on the queue the lease was taken from, and nothing else. A lease
  released on queue X does not order against work submitted to a queue built
  from it with `Queue(X, in_order)`.
- Reassigning a live lease (`ws = ctx.workspace(bigger);`) acquires the new loan
  before releasing the old one, so in a loop the arena ratchets instead of
  reusing. Call `ws.release()` first if that matters.

See `include/batchlas/util/workspace.hh` for the full lifetime rules.

Do not build a workspace out of a local `UnifiedVector` and let it go out of
scope before the kernels using it have run — the memory is freed while the device
may still be reading it. Wait on the queue before it dies, hoist it out of the
call's scope, or use the arena, whose lifetime is tied to the queue.

## The `linalg` convenience layer

`batchlas::linalg` (`include/batchlas/blas/linalg-ops.hh`) offers value-returning
and elementwise operations. Free functions only; there are no operator overloads.
Each takes its backend from the queue and its workspace from the arena.

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
they enqueue.

Elementwise arithmetic:

```cpp
auto S = linalg::add(ctx, A.view(), B.view());
auto P = linalg::multiply(ctx, A.view(), B.view());   // Hadamard, NOT matmul
auto K = linalg::scaled(ctx, A.view(), 2.0f);         // returns a scaled copy
linalg::scale<float>(ctx, A.view(), 2.0f);            // in place
linalg::axpby_into<float>(ctx, 2.0f, A.view(), 3.0f, B.view(), C.view());
```

`add`, `subtract`, `multiply`, `divide` and `scaled` allocate their result.
`add_into`, `subtract_into`, `multiply_into`, `divide_into` and `axpby_into`
write into storage you own, and `scale` works in place. Use the value-returning
forms where clarity matters more than controlling allocation — setup, tests,
exploration. In an inner loop, use the `_into` forms so the caller owns and
reuses the output.

Two behaviours to watch:

- `matmul` ignores `opts.beta` and forces `beta = 0`; the result is freshly
  allocated.
- `multiply` is elementwise (Hadamard). Use `matmul` for the matrix product. For
  square operands both readings are shape-valid.

For QR, use `geqrf` and `orgqr` from the main surface.

---

Adding an entry point to BatchLAS rather than calling one? See
[extending.md](extending.md).

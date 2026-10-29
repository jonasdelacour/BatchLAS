# The BatchLAS C++ API {#guide_cpp_api}

> **Status:** current · checked against `include/batchlas/` on 2026-10-29

Matrices are column-major and batched, every call enqueues work and returns immediately, and the
backend, the workspace and the device all come from the `Queue`. This page teaches that model and
its traps. Signatures, parameters and preconditions are in the API reference (@ref api_reference);
environment variables in @ref design_environment, exceptions in @ref design_error_model, spelling
rules in @ref design_api_conventions.

## The short version

```cpp
#include <batchlas.hh>                        // umbrella header
using namespace batchlas;

int main() {
    const int n = 128, batch = 512;

    Queue ctx(Device::default_device());      // backend resolved from the device
    Matrix<float> A(n, n, batch), B(n, n, batch), C(n, n, batch);

    A.fill_random(ctx, /*hermitian=*/false, /*seed=*/1);
    B.fill_random(ctx, /*hermitian=*/false, /*seed=*/2);

    gemm(ctx, A, B, C, {.alpha = 2.0f});   // C := 2 A B
    ctx.wait();                               // nothing is readable before this

    float c00 = C(0, 0, 0);                   // Matrix owns USM shared memory
}
```

Everything is in namespace `batchlas`. The headers also re-export `Queue`, `Device`, `Event`,
`Span`, `UnifiedVector` and `BumpAllocator` at global scope, a transitional shim that
`BATCHLAS_NO_GLOBAL_NAMES` turns off. `Matrix` and `MatrixView` default to
`<float, MatrixFormat::Dense>`; spell the format only when it is not `Dense`.

### Building and installing BatchLAS

Build the library and every consumer with the same SYCL compiler: DPC++ (developed with intel/llvm
built with `--cuda`) or AdaptiveCpp (`-DBATCHLAS_SYCL_IMPL=ACPP`). The host backend needs LAPACKE
and CBLAS; the CUDA backend the CUDA toolkit. CMake 3.17+ for BatchLAS, 3.21+ for a consumer.

```bash
git clone https://github.com/jonasdelacour/BatchLAS.git && cd BatchLAS
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBATCHLAS_BUILD_TESTS=OFF
cmake --build build -j"$(nproc)"
cmake --install build --prefix "$HOME/inst"   # <prefix> below
```

Every option and preset: @ref guide_build.

### Building against BatchLAS

**Configure the whole consuming project with the compiler BatchLAS was built with.** A mismatch
compiles, then fails at link with undefined references to `batchlas::Matrix<...>` constructors.
The package warns when `CMAKE_CXX_COMPILER` differs from the recorded one;
`-DBATCHLAS_REQUIRE_MATCHING_COMPILER=ON` makes that an error.

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_app CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(BatchLAS CONFIG REQUIRED)        # no COMPONENTS
add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE BatchLAS::batchlas)
```

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ -DCMAKE_PREFIX_PATH=<prefix>
cmake --build build
LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:<prefix>/lib ./build/my_app
```

@ref consumer_example (`examples/consumer/`) is this project, runnable.

- `BatchLAS::batchlas` is the only target. On Linux it propagates `-Wl,--no-as-needed`; keep it if
  you override the link line.
- A translation unit that only calls the documented API needs no SYCL flags. One that includes
  `<sycl/sycl.hpp>` or `<batchlas/sycl_interop.hh>`, or writes kernels, passes `-fsycl
  -fsycl-targets=...` itself, plus `"SHELL:${BatchLAS_SYCL_BACKEND_OPTIONS}"` when the package's
  targets are plain `nvptx64-nvidia-cuda` (otherwise your kernels build for sm_75).

## Devices and queues

A `Queue` names the device, carries the backend, owns a workspace arena and orders the work.
`Device::default_device()` is the first GPU, else the first CPU, else the host device.

```cpp
Queue ctx(Device::get_devices(DeviceType::GPU).at(1));   // the second GPU
Queue cpu("cpu");                                        // "cpu", "gpu", "accelerator"
Queue host(device, Backend::NETLIB);                     // backend pinned
Queue sibling(ctx, /*in_order=*/true);                   // ctx's SYCL context and device
```

- A `Queue` is movable, not copyable, and belongs to one thread
  ([Synchronisation and threading](#synchronisation-and-threading)).
- In-order is the default and is what makes an arena-backed workspace free; on an out-of-order
  queue pass your own workspace spans.

## What the operations compute

Every entry point is batched and applies one operation to every item; **all matrix arguments of
one call must have the same batch size**. What each call computes and writes is in its group:
@ref api_blas2, @ref api_blas3, @ref api_factorizations, @ref api_qr, @ref api_eigen,
@ref api_svd, @ref api_sparse. `pivots` are `int64_t`; `W` and `S` are the real counterpart of `T`.

> **Warning:** `trmm` and `getri` write a second operand (`C`) and leave the input alone, unlike
> reference `?trmm`; `trsm` is in place. `syrk`, `herk`, `syr2k` and `her2k` write only the `uplo`
> triangle of `C`; mirror it with `C.symmetrize(ctx, uplo)` (`hermitize` for `herk`/`her2k`)
> before using it as a full matrix.

### trsm: alpha follows the matrices {#trsm-alpha-moved-next-to-the-matrices}

Positionally, `alpha` is argument 4, as in `trmm`: `trsm<B>(ctx, A, B, alpha, side, uplo, trans,
diag)`. The old alpha-last order is a `= delete` overload, so it fails to compile.

### Which type each parameter takes {#which-type-each-parameter-takes}

- **Matrices** take `Matrix<T>` or `MatrixView<T>`, mixed freely. A `Matrix` is a `MatrixView`;
  pass `.view()` only to a generic (`auto`) parameter, which would otherwise copy.
- **Vectors** (`gemv`'s `x`, `y`; the extension surface's per-item vectors) take `VectorView<T>`;
  `Vector<T>` is one.
- **Flat arrays** (`W`, `S`, `tau`, `pivots`, `info`, workspaces) take `Span<T>`;
  `UnifiedVector<T>` converts implicitly. `VectorView` and `Span` do not interconvert: a strided
  view demoted to a span would read the wrong elements.

> **Warning:** write `Vector<T>(size, batch_size, Stride{s}, Inc{i})`; the bare-int forms are
> deleted, because a swapped `inc`/`stride` pair fits the same buffer and reads wrong elements.

## Options are structs with defaults

```cpp
gemm(ctx, A, B, C, {.alpha = 2.0f, .transA = Transpose::Trans});
syev(ctx, A, W, {.jobz = JobType::NoEigenVectors});
```

The structs are in @ref api_options. Defaults: `alpha = 1`, `beta = 0`, `side = Left`,
`uplo = Lower`, `trans = NoTrans`, `diag = NonUnit`, `jobz = EigenVectors`.

> **Warning:** every `uplo` defaults to `Uplo::Lower`. Fill the upper triangle and call `potrf` or
> `syev` with default options and the lower one is factorised, with no report.

- The transpose field is `transA`/`transB` on `gemm` and `gemv`, `trans` elsewhere.
- Write `PotrfOptions{}`, never a bare `{}`
  ([why](design/api-conventions.md#api-conventions-the-bare-braces-potrf-trap)).
- `*Params` structs (`SyevxParams`, `SteqrParams`, ...) belong to positional entry points with no
  option-struct spelling; read the declaration for their position.

### Which spelling each entry point takes {#which-spelling-each-entry-point-takes}

- **Option struct, backend from the queue:** the dense BLAS calls, and `potrf`, `getrs`, `syev`,
  `ormqr`, `gesvd`, `ortho` with an optional trailing workspace. `getrf`, `getri`, `geqrf`, `orgqr`
  take no options and lease their workspace (`getrf(ctx, A, pivots)`).
- **Positional with a workspace span:** `gesv`, `posv`, `spmm` and the extension surface; lease
  the span yourself ([Workspaces](#workspaces-come-from-the-queues-arena)).
- **Explicit `f<Backend, T>(...)` only:** entry points whose template arguments cannot be deduced,
  such as `tridiagonal_solver_buffer_size` and the `random_*_with_log10_cond_metric` generators.

`T` is deduced from the matrices, never from the option struct.

## Data layout and memory

### Column-major, always {#column-major-always}

Element `(i, j)` of batch item `b` of a dense view `V` lives at
`V.data_ptr()[b * V.stride() + j * V.ld() + i]`, counted in elements. `ld = 0` means `rows` and
`stride = 0` means `ld * cols`; `Matrix(rows, cols, batch)` is packed and unpadded. Access is
`M(i, j, b)`; the batch index is required. Layout rules: @ref design_matrix_model.

> **Warning:** `stride = 0` is the packed default, not cuBLAS's broadcast; BatchLAS has no
> broadcast operand. The batch count is the *sixth* `MatrixView` argument, so
> `MatrixView<float> V(p, n, n, batch)` sets `ld = batch` and throws.

A view can carry a per-item batch mask; see @ref design_masked_batches.

### Row-major source data {#row-major-source-data}

For `gemm`, swap the operands (`Cᵀ = Bᵀ Aᵀ`): no copy.

```cpp
// Row-major A (m x k), B (k x n), C (m x n), packed, in USM at pa/pb/pc.
MatrixView<float> At(pa, k, m, k), Bt(pb, n, k, n), Ct(pc, n, m, n);
gemm(ctx, Bt, At, Ct, GemmOptions<float>{});  // C = A B, row-major
```

Otherwise adopt the buffer packed and convert; `to_row_major()` converts back. Both synchronise.

```cpp
Matrix<float> A(Span<const float>(src, size_t(rows) * cols), rows, cols, /*ld=*/0);
auto col_major = A.to_column_major();     // or to_column_major(row_pitch) for a padded source
```

### Where the memory has to live: the USM contract {#where-the-memory-has-to-live-the-usm-contract}

**Every pointer given to `MatrixView` or `Span` must be device-accessible for the queue's
backend.** Queue-dispatched entry points check each pointer and throw
`batchlas::invalid_argument`; the explicit `f<Backend, T>(...)` spellings do not, and a host
pointer there aborts the process (check with `Queue::is_device_accessible(ptr)`).

Accepted on a GPU: `sycl::malloc_device`/`_shared`/`_host`, `cudaMalloc`, `cudaMallocManaged`.
Not: `malloc`, `new`, `std::vector`. A CPU device accepts host memory, so test on the device you
ship on.

`Matrix` owns USM shared memory, readable from the host. Load host data with the copying
constructor `Matrix<float>(Span<const float>(...), rows, cols, ld, stride, batch)`, whose
`(ld, stride)` describe the source. For device-resident data, wrap `sycl::malloc_device` memory
(from `batchlas::sycl_queue(ctx)`) in a `MatrixView` and pass its `data_ptrs` array (`T**`,
length `batch`): without it, batched vendor calls that need one throw `batchlas::invalid_argument`.

> **Warning:** `Random`, `fill_random` and the other random fills default to seed **42** and are a
> pure function of `(seed, index)`: two default-seeded operands are bit-identical.

### The CSR non-zero count has its own type {#the-csr-non-zero-count-has-its-own-type}

```cpp
Matrix<float, MatrixFormat::CSR> S(rows, cols, NonZeros{nnz}, batch_size);
```

`NonZeros{}` is a per-item **capacity**. `nnz()` is that stride, `nnz(b)` what item `b` stores
(read from its row offsets) and `nnz_capacity()` the slots allocated; on a heterogeneous batch,
loop to `nnz(b)`.

## What gets thrown {#what-gets-thrown}

Every exception BatchLAS raises is a `batchlas::` type from `batchlas/error.hh`, deriving from both
the matching `std::` exception and the tag base `batchlas::exception` (which has no `what()`; use
`e.message()`).

```cpp
try {
    syev(ctx, A, W.to_span());
} catch (const batchlas::workspace_error& e) {   // re-query *_buffer_size or halve the batch
} catch (const batchlas::exception& e) {         // anything BatchLAS diagnosed
} catch (const std::exception& e) {              // plus std::bad_alloc and sycl::exception
}
```

| class | means | retry helps? |
| --- | --- | --- |
| `invalid_argument`, `out_of_range` | The call violates the contract (shape, batch size, short span, non-USM pointer, index). | No. |
| `unsupported` | Nothing in this build on this device serves the request. | With another pin, backend, type or shape. |
| `device_error` | The device or vendor runtime failed. | Sometimes. |
| `workspace_error` | Scratch too small, or the arena ran out. | Yes. |
| `convergence_error` | Iteration or factorisation failed. | Prefer the per-item `info` spans. |
| `internal_error` | BatchLAS is inconsistent. | No; report it. |
| `api_misuse` | Wrong state: a `Queue` used from another thread, `configure()` after a `Queue` exists. | No. |
| `NoRouteError` (`<batchlas/no_route.hh>`) | No kernel in this build serves the call (typically a vendor-free build). | Not in this build. |

`std::bad_alloc` and `sycl::exception` are outside the hierarchy; device-side errors surface at
`ctx.wait_and_throw()`, not at the enqueuing call.

### Per-item status

`potrf`, `getrf`, `getri`, `gesv`, `posv`, `potri`, `sytrf`, `sysv`, `sygvd` and the eigen/SVD
routines take an optional `Span<int32_t> info`, one entry per batch item, LAPACK convention (`0`
good). It is a field on `PotrfOptions` and a trailing parameter elsewhere. Empty (the default)
costs nothing; a non-empty span shorter than the batch throws. `info` flags only an exactly
singular (or not positive-definite) item: a near-singular one reports `0`, so check the residual
when inputs are not known to be well-conditioned.

```cpp
UnifiedVector<int32_t> info(batch);
getrf(ctx, A, pivots.to_span(), info.to_span());
ctx.wait();
if (info[37] != 0) { /* item 37 is singular */ }
```

### Convergence status: `syev`, `syevx`, `gesvd`, `steqr`, `stedc` {#convergence-status-syev-syevx-gesvd-steqr-stedc}

Without `info`, a non-converged item is invisible: the call and `ctx.wait()` return normally. The
span is an **accumulator**: the called entry point zeroes it and nested solves only raise it, so
poison it with `-1` first if the check must be honest.

## Synchronisation and threading

Every entry point enqueues and returns an `Event`. Outputs are not readable until `e.wait()`,
`ctx.wait()` or `ctx.wait_and_throw()` (which also rethrows asynchronous errors). A fresh `Matrix`
is uninitialised, not zeroed.

**A `Queue` is single-threaded.** Its arena and event chain are unsynchronised, and its members
throw `batchlas::api_misuse` from any thread but the one that built it. Use one per thread; to move
one, call `attach_to_current_thread()` from the new owner.

## The backend comes from the Queue

`Backend::Auto` resolves once from the device vendor: `CUDA` on NVIDIA, `ROCM` on AMD, `SYCL`
(native kernels only) on other GPUs, `NETLIB` (host LAPACK) otherwise. `ctx.backend()` returns it;
`Queue(device, Backend::NETLIB)` or `ctx.set_backend(...)` pins it, and an unavailable backend
throws `batchlas::unsupported`. For a compile-time backend use `with_backend`:

```cpp
with_backend(ctx, [&](auto Back) {
    constexpr Backend Bk = Back.value;
    gemm<Bk>(ctx, A, B, C, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
});
```

The backend does not pick the kernel. Each op chooses a native kernel family or `vendor` from a
per-device tuned table, among the families whose correctness predicate admits the call.
`BATCHLAS_<OP>_ROUTE=<spelling>` pins one (a bad pin throws) and `BATCHLAS_SELECT_TRACE=1` prints
each choice. Design: [flat kernel selection](design/flat-kernel-selection.md); families per op:
@ref selection_tables.

## Configuration {#configuration}

Everything BatchLAS reads from the environment is parsed once into `batchlas::settings()`
(`<batchlas/settings.hh>`, @ref api_config); the variables are listed in @ref design_environment.

```cpp
Settings s = batchlas::settings();
s.routing.route("gemm") = EnvValue::of("native");
batchlas::configure(s);                     // before the first Queue
```

- `configure()` after any `Queue` exists throws `batchlas::api_misuse`.
- Code that changes the environment mid-process must call `batchlas::detail::reload_settings()`
  (`ScopedEnvVar` does), and must not do it between a `*_buffer_size()` query and its call.
- Variables that remove a safety check (`BATCHLAS_SKIP_POINTER_CHECKS`,
  `BATCHLAS_LATRD_GRID_FORCE_UNSAFE`, `BATCHLAS_BLAS_HEALTH=off`) are ignored, with a warning,
  unless the build sets `-DBATCHLAS_ALLOW_UNSAFE_ENV=ON` (the `dev*` presets do, `cuda` and
  `release` do not).

## Workspaces come from the queue's arena {#workspaces-come-from-the-queues-arena}

Omit the workspace and the call leases scratch from the queue's arena, sized by its
`*_buffer_size`. To own it, size and pass a span:

```cpp
with_backend(ctx, [&](auto Back) {
    constexpr Backend Bk = Back.value;
    UnifiedVector<std::byte> ws(potrf_buffer_size<Bk, float>(ctx, A, Uplo::Lower));
    potrf<Bk, float>(ctx, A, Uplo::Lower, ws.to_span());
    ctx.wait();                                // ws must outlive the kernels
});
```

- The arena keeps its peak; `ctx.trim_workspace()` frees it when no lease is outstanding.
- Pass your own span on an out-of-order queue or a sibling queue; never one backed by a local
  `UnifiedVector` that dies before the kernels run.

Lifetime rules: @ref design_workspace.

## Interop with CUDA and with your own SYCL

`Queue::native_handle()` returns the native stream (`cudaStream_t` on CUDA, `hipStream_t` on HIP,
else `nullptr`); it belongs to the `Queue`. Work you push to it runs after BatchLAS's; call
`ctx.create_event_after_external_work()` to make BatchLAS wait for yours.

`<batchlas/sycl_interop.hh>`, the one header that includes `<sycl/sycl.hpp>`, provides
`sycl_queue(ctx)`, `sycl_event(e)`, `event_from_sycl(ev)` (pass to `ctx.enqueue`) and
`queue_from_sycl(context, device, in_order)`, which builds a `Queue` in your SYCL context so USM
from that context is used zero-copy. It throws `batchlas::device_error` for a sub-device or a
device outside the context.

## The `linalg` convenience layer

`batchlas::linalg` (@ref api_linalg) returns results by value and leases its workspace:

```cpp
auto C = linalg::matmul(ctx, A, B);   // allocates and returns C
auto s = linalg::solve(ctx, A, B);    // s.x, s.info
auto e = linalg::eigh(ctx, A);        // e.values, e.vectors, e.info
ctx.wait();                           // before reading any of them
```

- Factorisation, solve, eigen and SVD results carry a per-item `info`; check it.
- `multiply` is elementwise; `matmul` is the matrix product, and its options have no `beta`.
- There is no `linalg::qr`; compose `geqrf` and `orgqr`
  ([why](design/known-defects.md#linalgqr-returns-a-wrong-qr-after-an-earlier-call-in-the-process)).

Adding an entry point rather than calling one? See [extending.md](extending.md).

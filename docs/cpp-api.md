# The BatchLAS C++ API

> **Status:** current

Matrices are column-major and batched, every call enqueues work and returns immediately, and the
backend, the workspace and the device all come from the `Queue`. This page teaches that model and
the traps around it. Signatures, option fields and per-op detail are in the generated API reference
(groups in @ref api_reference); environment variables are in @ref design_environment, exceptions in
@ref design_error_model, spelling rules in @ref design_api_conventions.

## The short version

```cpp
#include <batchlas.hh>                        // umbrella header
using namespace batchlas;

int main() {
    const int n = 128, batch = 512;

    Queue ctx(Device::default_device());      // backend resolved from the device
    Matrix<float> A(n, n, batch), B(n, n, batch), C(n, n, batch);

    A.view().fill_random(ctx, /*hermitian=*/false, /*seed=*/1);
    B.view().fill_random(ctx, /*hermitian=*/false, /*seed=*/2);

    gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});   // C := 2 A B
    ctx.wait();                               // nothing is readable before this

    float c00 = C(0, 0, 0);                   // Matrix owns USM shared memory
}
```

Everything is in namespace `batchlas`: `Matrix`, `MatrixView`, `gemm`, `potrf` and the rest of the
numerical surface, plus `Queue`, `Device`, `Event`, `Span`, `UnifiedVector` and `BumpAllocator`.

> **Note:** For one more release, headers re-export `Queue`, `Device`, `Event`, `Span`,
> `UnifiedVector` and `BumpAllocator` at global scope. Define `BATCHLAS_NO_GLOBAL_NAMES` to turn
> that off today.

`Matrix`, `MatrixView` and `VectorView` default to `<float, MatrixFormat::Dense>` (`<float>` for
`VectorView`); `Vector<T>` has no default. Spell the format only when it is not `Dense`:
`Matrix<float, MatrixFormat::CSR> S(n, n, NonZeros{nnz}, batch)`.

### Building and installing BatchLAS

- **A clang-based SYCL compiler, clang 16 or newer.** Developed with intel/llvm DPC++ built with
  `--cuda` at `/opt/dpcpp-cuda`; substitute your prefix. Build the library and every consumer with
  the same compiler, or the link fails with undefined references to the constrained entry points.
- **LAPACKE and CBLAS** for the host backend (`BATCHLAS_ENABLE_NETLIB`, on by default; configure
  warns and builds without it when they are missing).
- **The CUDA toolkit** whenever the CUDA backend is on, which `AUTO` does as soon as the SYCL
  runtime exposes a CUDA device.
- **CMake 3.14+** for BatchLAS, 3.21+ for a consuming project.

```bash
git clone https://github.com/jonasdelacour/BatchLAS.git && cd BatchLAS
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBATCHLAS_BUILD_TESTS=OFF
cmake --build build -j"$(nproc)"
cmake --install build --prefix "$HOME/inst"   # <prefix> in the consumer commands below
```

Options worth knowing: `BATCHLAS_ENABLE_CUDA` (`AUTO` default; `ON` makes configure fail when no
CUDA device is exposed), `BATCHLAS_NVIDIA_ARCH` (detected; pass it only when cross-building),
`BATCHLAS_ENABLE_NETLIB` (`ON`), `BATCHLAS_ENABLE_ROCM` (`OFF`), `BATCHLAS_BUILD_TESTS`,
`BATCHLAS_BUILD_BENCHMARKS`, `BATCHLAS_BUILD_PYTHON`, and `BATCHLAS_ALLOW_UNSAFE_ENV` (`OFF`; see
[Configuration](#configuration)). `CMakePresets.json` has `dev` and `dev-tests` presets.

### Building against BatchLAS

**Configure the whole consuming project with the compiler BatchLAS was built with.** A mismatch
compiles and then fails at link:

```
undefined reference to `batchlas::Matrix<float, (batchlas::MatrixFormat)0>::Matrix<...>(int, int, int, int, int)'
```

The package compares `CMAKE_CXX_COMPILER` to the recorded compiler (in the install's build-tree
`CMakeCache.txt`) by realpath and warns on a difference; `-DBATCHLAS_REQUIRE_MATCHING_COMPILER=ON`
makes it an error.

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_app CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BatchLAS CONFIG REQUIRED)
add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE BatchLAS::batchlas)
```

```bash
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
      -DCMAKE_PREFIX_PATH=<prefix>
cmake --build build
LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:<prefix>/lib ./build/my_app
```

`examples/consumer/` is this project, standalone and runnable (it prints `PASS`).

- `BatchLAS::batchlas` is the only target to link. It propagates `cxx_std_20`, the include root, the
  component libraries and `-Wl,--no-as-needed`; keep that flag if you override the link line.
- Ask for the package whole: no `COMPONENTS`. CUDA and LAPACK link privately, so a CPU-only machine
  can `find_package` a CUDA-enabled install.
- A translation unit that only calls the documented API needs no SYCL flags. One that includes
  `<sycl/sycl.hpp>` or `<batchlas/sycl_interop.hh>`, or writes kernels, passes them itself:

```cmake
target_compile_options(my_app PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
target_link_options(my_app    PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
```

When the building compiler has no `nvidia_gpu_sm_<N>` alias for the GPU (DPC++ stops at sm_90a),
`BatchLAS_SYCL_TARGETS` is `nvptx64-nvidia-cuda` and `BatchLAS_SYCL_BACKEND_OPTIONS` carries the
architecture. Pass it too, or your kernels build for sm_75:

```cmake
target_compile_options(my_app PRIVATE "SHELL:${BatchLAS_SYCL_BACKEND_OPTIONS}")
target_link_options(my_app    PRIVATE "SHELL:${BatchLAS_SYCL_BACKEND_OPTIONS}")
```

At run time the loader must find the DPC++ runtime and its SYCL adapters: export
`LD_LIBRARY_PATH=<dpcpp-prefix>/lib`, or add that directory to `/etc/ld.so.conf.d/` and run
`ldconfig`. Headers install under `<prefix>/include/batchlas/` plus `<prefix>/include/batchlas.hh`.

## Devices and queues

A `Queue` names the device, carries the backend, owns a workspace arena and orders the work.
`Device::default_device()` is the first GPU, else the first CPU, else the first host device.

```cpp
auto gpus = Device::get_devices(DeviceType::GPU);   // CPU, ACCELERATOR, HOST too
Queue ctx(gpus.at(1));                              // the second GPU
Queue cpu("cpu");                                   // "cpu", "gpu", "accelerator"

Queue ooo(device, /*in_order=*/false);              // out-of-order
Queue host(device, Backend::NETLIB);                // backend pinned, in-order
Queue sibling(ctx, /*in_order=*/true);              // ctx's SYCL context and device
```

The string constructor takes the first device of that type; it throws `batchlas::device_error`
when there is none and `batchlas::invalid_argument` for any other string.

- A `Queue` is movable, not copyable, and belongs to one thread (see
  [Synchronisation and threading](#synchronisation-and-threading)).
- `Queue(base, in_order)` shares `base`'s context, so the two see each other's USM allocations; each
  still owns its arena and event chain.
- In-order is the default and is what makes an arena-backed workspace free. On an out-of-order
  queue, pass your own workspace spans.

## What the operations compute

Every entry point is batched and applies the same operation to every item; **all matrix arguments
of one call must have the same batch size**. `α` and `β` are the `alpha` and `beta` option fields,
and `op(A)` is `A`, `Aᵀ` or `Aᴴ` according to `trans`. Shapes and constraints are in the API groups:
@ref api_blas2, @ref api_blas3, @ref api_factorizations, @ref api_qr, @ref api_eigen, @ref api_svd, @ref api_sparse.

| call | computes | written |
| --- | --- | --- |
| `gemm(ctx, A, B, C, opts)` | `C := α·op(A)·op(B) + β·C` | `C` |
| `gemv(ctx, A, x, y, opts)` | `y := α·op(A)·x + β·y` | `y` |
| `symm` / `hemm(ctx, A, B, C, opts)` | `C := α·A·B + β·C` (`Left`) or `α·B·A + β·C` (`Right`), `A` symmetric / Hermitian | `C` |
| `syrk` / `herk(ctx, A, C, opts)` | `C := α·A·Aᵀ + β·C` (`NoTrans`) or `α·Aᵀ·A + β·C`; `herk` uses `ᴴ` | `uplo` triangle of `C` |
| `syr2k` / `her2k(ctx, A, B, C, opts)` | rank-2k update of `C` | `uplo` triangle of `C` |
| `trmm(ctx, A, B, C, opts)` | `C := α·op(A)·B` or `α·B·op(A)` | **`C`**; `B` untouched |
| `trsm(ctx, A, B, opts)` | solves `op(A)·X = α·B` or `X·op(A) = α·B` | **`B`**, overwritten with `X` |
| `potrf(ctx, A, opts)` | Cholesky, `A = L·Lᴴ` or `Uᴴ·U` | `A` in place, `uplo` triangle |
| `getrf(ctx, A, pivots)` | LU with partial pivoting | `A` and `pivots` |
| `getrs(ctx, A, B, pivots, opts)` | solves `op(A)·X = B` from `getrf`'s factors | `B` |
| `getri(ctx, A, C, pivots)` | `C := A⁻¹` from `getrf`'s factors | **`C`**; `A` read-only |
| `geqrf(ctx, A, tau)` / `orgqr(ctx, A, tau)` | QR with Householder reflectors / expand to explicit `Q` | `A`, `tau` |
| `syev(ctx, A, W, opts)` | symmetric/Hermitian eigendecomposition of the `uplo` triangle | `W` ascending; `A` gets eigenvectors when `jobz == JobType::EigenVectors` |

`pivots` is `int64_t` (n·batch), `tau` is `T` (min(m,n)·batch) and `W` is the **real** counterpart
of `T` (n·batch).

> **Warning:** `trmm` and `getri` write a second operand and leave the input alone (reference
> `?trmm` works in place on `B`); `trsm` is in place. Mixing these up gives a wrong answer, not a
> compile error.

> **Warning:** `syrk`, `herk`, `syr2k` and `her2k` write only the `uplo` half of `C`. The other
> half comes back as it went in, including uninitialised memory (a fresh `Matrix` is not zeroed),
> so the result is triangular, not symmetric, and no call reports it.

Mirror the triangle before such use: `symmetrize(ctx, uplo)` copies the named triangle across the
diagonal, `hermitize(ctx, uplo)` its conjugate (the right one for `herk`/`her2k`). Both are one
kernel on a square `MatrixView` and return an `Event`.

```cpp
syrk(ctx, A.view(), C.view(), {.uplo = Uplo::Lower}).wait();
C.view().symmetrize(ctx, Uplo::Lower).wait();   // hermitize() for herk/her2k
```

- `gemm` handles a heterogeneous batch (differing `active_rows`/`active_cols`) on every backend.
- `symm`, `syrk` and `syr2k` are constrained to **real** `T`; `hemm`, `herk` and `her2k` to
  **complex** `T`; `gemm`, `gemv`, `trmm` and `trsm` take both. `herk`'s `α` and `β` and `her2k`'s
  `β` are real.
- Potrf, getrf, getri and the eigen/SVD routines optionally report per-item status
  ([Per-item status](#per-item-status)).

#### trsm: alpha moved next to the matrices

In the positional spelling `alpha` is argument 4, right after the matrices, matching `trmm`:
`trsm<B>(ctx, A, B, alpha, side, uplo, trans, diag)`. Two `= delete` overloads with the old order
(`Side, Uplo, Transpose, Diag, T`) turn a stale call into a "call to deleted function" error. The
option-struct spelling is unaffected; `backend::trsm_vendor` still takes `alpha` last.

### Which type each parameter takes {#which-type-each-parameter-takes}

- **Matrix parameters** take `Matrix<T>` or `MatrixView<T>`, mixed freely, on every spelling.
- **Vector parameters** (`gemv`'s `x`, `y`) take `VectorView<T>`; an owning `Vector<T>` also works.
- **Flat arrays** (`W`, `S`, `tau`, `pivots`, workspaces) take `Span<T>`. `UnifiedVector<T>`
  converts implicitly; `to_span()` is explicit. A `Vector<T>` is not a `Span`.

The extension surface (`steqr`, `stebz`, `stein`, `stedc`, `lanczos`, `ritz_values`) follows the
same rule: `VectorView<T>` for one logical vector per batch item (needs `inc`/`stride`/`batch_size`),
`Span<T>` for a flat array with one entry per item. `stebz` shows both: `d`, `e`, `w` are
`VectorView<T>`, `m` is `Span<int32_t>`. They do not interconvert, because a `VectorView` demoted
to a `Span` would drop the stride and read wrong elements.

| you hold | parameter is `Span<T>` | parameter is `VectorView<T>` |
| --- | --- | --- |
| `UnifiedVector<T>` | pass directly, or `.to_span()` | `VectorView<T>(v, size, batch, Inc{i}, Stride{s})` |
| `Vector<T>` | `.data()`: the whole allocation, so only when `inc == 1` and packed | `.view()` |
| raw pointer | `Span<T>(p, n)` | `VectorView<T>(p, size, batch, Inc{i}, Stride{s})` |

Every entry point accepts owning `Matrix`/`Vector` arguments where a view is expected (a generated
forwarder, `BATCHLAS_ACCEPT_OWNING` in `blas/queue-dispatch.hh`). Two forms fail to compile: a bare
`{}` in any argument position (name the type: `OrthoOptions{}`), and explicit template arguments for
some parameters with deduction for others (`spmm<Backend::CUDA, float>(ctx, A, ...)` with an owning
`A`; write `spmm<Backend::CUDA>(ctx, A, ...)`).

```cpp
UnifiedVector<int64_t> pivots(n * batch);
UnifiedVector<float>   W(n * batch);            // real, even for complex A
Vector<float> x(n, /*batch_size=*/batch), y(m, batch);
gemv(ctx, A.view(), x.view(), y.view(), {.alpha = 1.0f});
```

> **Warning:** `Vector` names `inc` and `stride`; `VectorView` also takes them positionally, in the
> opposite order. Write `Vector<T>(size, batch_size, Stride{s}, Inc{i})`; the bare-int forms are
> deleted. The tags exist because `(inc = n, stride = 1)` and `(inc = 1, stride = n)` fit the same
> buffer, so a swapped pair compiles and reads the wrong elements. `Stride`, `Inc`, `Ld` and
> `BatchSize` live in `batchlas/blas/matrix.hh` beside `NonZeros`; none decays to `int`.

## Options are structs with defaults

Most entry points take an option struct, so you write only what differs:

```cpp
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f, .transA = Transpose::Trans});
syev(ctx, A.view(), W, {.jobz = JobType::NoEigenVectors});
getrs(ctx, LU.view(), X.view(), pivots, {.trans = Transpose::Trans});
```

The structs are in `batchlas/blas/options.hh` (group @ref api_options): `GemmOptions<T>`,
`GemvOptions<T>`, `SymmOptions<T>`, `HemmOptions<T>`, `SyrkOptions<T>`, `HerkOptions<T>`,
`Syr2kOptions<T>`, `Her2kOptions<T>`, `TrmmOptions<T>`, `TrsmOptions<T>`, `PotrfOptions`,
`GetrsOptions`, `SyevOptions`. The BLAS ones are templated on `T`; the three LAPACK ones are not.
Defaults are `alpha = 1`, `beta = 0`, `side = Left`, `uplo = Lower`, `trans = NoTrans`,
`diag = NonUnit`, `jobz = EigenVectors`. The traps:

> **Warning:** Every `uplo` defaults to `Uplo::Lower`. Filling the upper triangle and calling
> `potrf` or `syev` with default options factorises whatever is in the lower one, with no report.
> Fill the lower triangle or pass `{.uplo = Uplo::Upper}`.

- The transpose field is `transA` (and `transB`) in `GemmOptions` and `GemvOptions`, `trans`
  elsewhere. `TrmmOptions` and `TrsmOptions` have no `beta`.
- `HerkOptions` `alpha`/`beta` and `Her2kOptions` `beta` are real (`float_t<T>`), which keeps the
  result Hermitian.
- `ComputePrecision` appears only on `gemm`: `Default` computes in the input type; the others are
  `F32`, `F64`, `F16`, `BF16`, `TF32`, and a backend that cannot serve the one you ask for says so
  at compile time.
- Write `PotrfOptions{}`, never a bare `{}`: see
  [the bare-braces trap](design/api-conventions.md#api-conventions-the-bare-braces-potrf-trap).

### `*Options` and `*Params` are two different things

`*Options` structs belong to the convenience layer: the backend comes from the `Queue`, `T` is
deduced from the matrices, the struct carries every non-matrix argument, and the workspace of
LAPACK-style calls may be omitted.

`*Params` structs (`batchlas/blas/extensions.hh`, `batchlas/blas/functions/iluk.hh`:
`SyevxParams`, `LanczosParams`, `StebzParams`, `SteinParams`, `SteqrParams`, `StedcParams`,
`JacobiParams`, `GesvdjParams`, `SytrdBandReductionParams`, `ILUKParams`) are ordinary arguments to
entry points with no convenience layer: you name the backend (`syevx<Backend::CUDA, float>`) and
pass a workspace. Their position varies, so read the declaration: usually last, but **second to
last** for `SteqrParams` and `StedcParams` (`eigvects` follows), and `SytrdBandReductionParams`
replaces the `int32_t block_size` argument.

### Which spelling each entry point takes {#which-spelling-each-entry-point-takes}

- The dense BLAS calls take an option struct and no workspace.
- `potrf`, `getrs` and `syev` take an option struct plus an optional workspace:
  `potrf(ctx, A, opts)` and `potrf(ctx, A, opts, ws)` both exist.
- `getrf`, `getri`, `geqrf` and `orgqr` carry no options. The arena-backed spelling omits the
  workspace, `getrf(ctx, A, pivots)`; the positional spelling takes it, `getrf<Back, T>(ctx, A, pivots, ws)`.
- `gesvd`, `ormqr`, `ortho` and `spmm` take positional arguments and a workspace span. Lease it from
  the arena yourself:

  ```cpp
  with_backend(ctx, [&](auto Back) {
      constexpr Backend Bk = Back.value;
      auto ws = ctx.workspace(gesvd_buffer_size<Bk, float>(
                                  ctx, A.view(), S, U.view(), Vh.view(),
                                  SvdVectors::All, SvdVectors::All));
      gesvd<Bk, float>(ctx, A.view(), S, U.view(), Vh.view(),
                       SvdVectors::All, SvdVectors::All, ws.span());
  });
  ```

- Entry points whose template parameters cannot be deduced keep the explicit `f<Backend, T>(...)`
  form: `tridiagonal_solver_buffer_size` and the six `random_*_with_log10_cond_metric` generators.
- `stebz_buffer_size`, `stein_buffer_size` and `stedc_buffer_size` take their option struct as a
  required argument: `stebz_buffer_size(ctx, n, batch, StebzParams<float>{})`.

With an empty option struct *and* an explicit workspace, name the type:
`potrf(ctx, A.view(), PotrfOptions{}, ws)`. `T` is deduced from the matrix arguments, never from the
option struct: write `syev<B>(ctx, ...)` or `syev(ctx, ...)` on the option-struct spelling and
`syev<B, float>(ctx, ...)` on the positional one.

## Data layout and memory

### Column-major, always {#column-major-always}

Matrices are **column-major**, like LAPACK and unlike NumPy. For a dense `MatrixView V`, element
`(i, j)` of batch item `b` lives at

```cpp
V.data_ptr()[b * V.stride() + j * V.ld() + i]
```

`ld`, `stride` and the indices count **elements**, not bytes. Access is `M(i, j, b)` on an owning
`Matrix` (the batch index is required) and `V.at(i, j, b)` or `V(i, j, b)` on a `MatrixView`, which
bounds-check and throw `std::out_of_range`.

- **`ld`**: distance between column `j` and `j+1`. `0` means packed (`rows`); otherwise at least
  `rows`. A larger `ld` views a sub-block of a bigger buffer.
- **`stride`**: distance between batch item `b` and `b+1`. `0` means `ld * cols`.

`Matrix(rows, cols, batch)` allocates exactly `rows * cols * batch` elements with `ld() == rows()`
and `stride() == rows() * cols()`; nothing pads. `Matrix(rows, cols, batch, ld, stride)` and
`MatrixView(data, rows, cols, ld, stride, batch)` resolve the defaults identically. Element access
computes `int64_t(b) * stride + j * ld + i`; the within-item offset stays `int` (limit 2³¹
elements, 8 GB of `float`).

`MatrixView` throws `std::invalid_argument` on a negative extent and on a resolved `ld < rows`. The
batch count is the *third* argument on `Matrix` and the *sixth* on `MatrixView`, so
`MatrixView<float> V(p, n, n, batch)` means `ld = batch` and throws. A view accepts a null `data` or
a zero dimension (the shape-only view used for workspace-size queries).

> **Warning:** `stride = 0` is the packed default, not cuBLAS's broadcast. BatchLAS has no
> broadcast operand; unequal batch sizes throw. `MatrixView(dA, n, k, n, /*stride=*/0, batch)` over
> one matrix reads `batch` consecutive items, past the end of the buffer. To multiply one matrix
> against many right-hand sides, fold the batch into columns (packed `B` with `ld == k`,
> `stride == k*n` is one `k × (n·batch)` matrix) or replicate the matrix.

### Row-major source data {#row-major-source-data}

For `gemm`, use the [operand swap](#row-major-data-the-operand-swap) (no copy). Otherwise convert:
`Matrix::to_column_major()` returns a converted packed copy and `to_row_major()` converts back.
Both synchronise before returning; `to_column_major(ctx)` / `to_row_major(ctx)` use your queue.

```cpp
// packed row-major buffer (row pitch cols), adopted with ld = 0
Matrix<float> A(Span<const float>(src, size_t(rows) * cols), rows, cols, /*ld=*/0);
auto col_major = A.to_column_major();          // use only col_major; A is labelled wrongly

// padded row pitch p: a matrix big enough for the padded layout, converted at that pitch
Matrix<float> holder(rows, cols, batch, /*ld=*/rows, /*stride=*/(rows - 1) * p + cols);
auto col_major2 = holder.to_column_major(p);
```

- `to_column_major()` with no pitch requires a packed matrix (`ld() == rows()`,
  `stride() == rows()*cols()`); a padded `ld` or gapped `stride` throws `std::invalid_argument`.
- `to_column_major(row_pitch)` throws for a pitch below `cols`, rows past the allocation, or a
  straddle into the next batch item.
- The copying constructors are column-major: `(ld, stride)` are the source's column pitch and batch
  stride.

### Where the memory has to live: the USM contract {#where-the-memory-has-to-live-the-usm-contract}

**Every pointer given to `MatrixView` or `Span` must be device-accessible for the backend the
`Queue` dispatches to.** `MatrixView` takes a bare `T*` and cannot check at construction; entry
points that take the backend from the queue check every pointer argument at the call and throw
`std::invalid_argument`:

```cpp
std::vector<float> ha(n * n * batch), hb(n * n * batch), hc(n * n * batch);
MatrixView<float> A(ha.data(), n, n, n, n * n, batch);   // B, C likewise
gemm(ctx, A, B, C, GemmOptions<float>{});        // throws std::invalid_argument
```

- The check does **not** run on the `f<Backend, T>(...)` spellings (including `gesvd`, `ormqr`,
  `ortho`, `spmm` and the positional workspace-taking forms): a host pointer reaches the vendor
  call and aborts the process. Validate with `Queue::is_device_accessible(ptr)`.
- `BATCHLAS_SKIP_POINTER_CHECKS=1` turns the check off (an unsafe variable; see
  [Configuration](#configuration)).
- An argument that addresses no elements is exempt: the empty `Span` of a sizing pass and the
  default-constructed `MatrixView<float>()` meaning "optional matrix not in use".

Zero-copy on a GPU backend: `sycl::malloc_device`, `malloc_shared`, `malloc_host` (including on your
own `sycl::context` for the same device), `cudaMalloc`, `cudaMallocManaged`. Not: `malloc`, `new`,
`std::vector`. On a host/CPU device ordinary host memory works and nothing is rejected, so test on
the device you ship on.

### Getting host data in

`Matrix` owns USM **shared** memory, so the host can read and write it directly. Bulk-load with the
copying constructor:

```cpp
Matrix<float> A(Span<const float>(host.data(), host.size()),
                n, n, /*ld=*/n, /*stride=*/0, /*batch_size=*/batch);
```

`(ld, stride)` describe the **source**: element `(i, j, b)` is read from `data[b * stride + j * ld + i]`,
`ld = 0` meaning `rows` and `stride = 0` meaning `ld * cols`. Prefer the `Span<const T>` overload:
the span knows the source length, so an over-reading shape throws `std::invalid_argument`.

Generated data skips the host and synchronises before returning:

```cpp
auto R = Matrix<float>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/7);
auto I = Matrix<float>::Identity(n, batch);
auto Z = Matrix<float>::Zeros(n, n, batch);
```

Other factories: `RandomTriangular`, `Ones`, `Diagonal`, `Triangular`, `TriDiagToeplitz`. To fill a
matrix you own, the `fill_*` family on `MatrixView` writes in place and returns an `Event`
(`fill`, `fill_zeros`, `fill_ones`, `fill_identity`, `fill_diagonal`, `fill_random`, ...; group
@ref api_matrix). Pass your own queue: a default-constructed `Queue` builds a fresh one on
`Device::default_device()`, which targets the wrong device on a multi-GPU box.

> **Warning:** The `seed` of `Random`, `RandomTriangular`, `fill_random` and `fill_triangular_random`
> defaults to **42**, and each element is a pure function of `(seed, index)`. Two default-seeded
> calls return bit-identical matrices; pass distinct seeds for distinct operands. `Random` keys on
> the flat index over the whole allocation (different `ld` or `stride` gives different values at the
> same `(i, j, b)`); `fill_triangular_random` keys on the index within a matrix, so every batch item
> of a `RandomTriangular` result is the same matrix.

View-to-view copy is asynchronous: `MatrixView<float>::copy(ctx, dst.view(), src.view())`. Host
element access is one indexed store into managed memory per element: use it for a handful of entries
and tests, not to load a batch. `MatrixView` never owns.

### Device-resident operands

`Matrix` memory migrates between host and device on demand; a large batch written from the host and
read by many kernels pays for that traffic. For device-resident storage, allocate with
`sycl::malloc_device` and wrap it in a `MatrixView`, which copies nothing:

```cpp
#include <batchlas/sycl_interop.hh>            // this TU needs -fsycl

auto& q = batchlas::sycl_queue(ctx);           // the queue BatchLAS submits on
float*  dA = sycl::malloc_device<float>(size_t(n) * n * batch, q);
float** pA = sycl::malloc_device<float*>(batch, q);   // the batch pointer array
q.memcpy(dA, host.data(), size_t(n) * n * batch * sizeof(float)).wait();

MatrixView<float> A(dA, n, n, /*ld=*/n, /*stride=*/n * n, batch, /*data_ptrs=*/pA);
```

- The allocation must be reachable from the queue's SYCL context.
- **Pass the `data_ptrs` array.** A raw-pointer view without it makes every batched vendor call that
  needs one (`potrf` at batch > 1, `getrf`, `getri`) throw
  `std::runtime_error("data_ptrs target is null")`. It is a `T**` of length `batch_size`, may be
  `malloc_device`, and BatchLAS fills it. Owning `Matrix` objects build it themselves; `gemm` and
  `syev` do not need it.
- Device memory is not host-addressable: read results back with `q.memcpy` and initialise with the
  `fill_*` family. For a device-memory workspace, pass a `Span<std::byte>` over a `malloc_device`
  block to the positional spelling; the arena serves shared memory.

### Row-major data: the operand swap

A column-major view of a row-major `m x k` buffer with row length `k` is its transpose. Since
`Cᵀ = Bᵀ Aᵀ`, passing the transposed views to `gemm` in the opposite order computes the row-major
product with no copy:

```cpp
// Row-major A (m x k), B (k x n), C (m x n), packed, in USM at pa/pb/pc.
MatrixView<float> At(pa, k, m, k), Bt(pb, n, k, n), Ct(pc, n, m, n);
gemm(ctx, Bt, At, Ct, GemmOptions<float>{});  // C = A B, row-major
```

The swap is `gemm`'s. For the symmetric routines, flip `uplo`. For everything else, including
`potrf`, `getrf` and `syev`, convert first ([Row-major source data](#row-major-source-data)).

### The CSR non-zero count has its own type {#the-csr-non-zero-count-has-its-own-type}

The owning constructors line up as shape, format-specific extra, batch size; the CSR non-zero count
is the `NonZeros` strong typedef (a bare `int` does not compile):

```cpp
Matrix<float> D(rows, cols, batch_size, ld, stride);
Matrix<float, MatrixFormat::CSR> S(rows, cols, NonZeros{nnz}, batch_size);
```

`NonZeros{}` is a **capacity**. `convert_to<MatrixFormat::CSR>()` sizes a whole batch by its largest
item, so on a heterogeneous batch `nnz()` over-counts smaller items and a loop to `nnz()` walks past
that item's row range. Three accessors exist on `Matrix`, `MatrixView` and `KernelMatrixView`:

- **`nnz()`**: the per-item stride; what the vendor SpMM descriptors want.
- **`nnz(b)`**: the non-zeros item `b` stores, read from its row offsets (the filling kernel must
  have completed; use the `KernelMatrixView` overload inside a kernel over device memory).
- **`nnz_capacity()`**: the slots allocated per item.

The from-data constructors are in @ref api_matrix and @ref design_matrix_model.

## What gets thrown {#what-gets-thrown}

Every exception BatchLAS raises is a `batchlas::` type from `batchlas/error.hh`. Each derives from
both the `std::` exception that site threw before and an empty tag base, `batchlas::exception`, so
existing `std::` handlers still catch it (and Python raises the same type as before).

```cpp
#include <batchlas/error.hh>

try {
    syev(ctx, A.view(), W.to_span());
} catch (const batchlas::workspace_error& e) {
    // the recoverable one: re-query *_buffer_size, or halve the batch and retry
} catch (const batchlas::exception& e) {
    // anything BatchLAS itself diagnosed. e.message(), not e.what()
} catch (const std::exception& e) {
    // that, plus std::bad_alloc and sycl::exception. e.what()
}
```

| class | derives from | means | retry helps? |
| --- | --- | --- | --- |
| `invalid_argument` | `std::invalid_argument` | The call violates the contract: mismatched shapes or batch sizes, short span, null or non-USM pointer, bad `ld`. | No. |
| `out_of_range` | `std::out_of_range` | Index outside its container (`V.at(i, j, b)`). | No. |
| `error` | `std::runtime_error` | Base of the six below. | n/a |
| `unsupported` | `error` | No kernel or backend in this build on this device serves the request. | Not as asked; try another route pin, backend, type or shape. |
| `device_error` | `error` | The device or vendor runtime failed (cuBLAS/cuSOLVER status, launch failure, null allocation, no device of that type). | Sometimes. |
| `workspace_error` | `error` | Scratch too small, or the arena ran out. | Yes: re-query `*_buffer_size()` or halve the batch. |
| `convergence_error` | `error` | An iterative kernel did not converge or a factorisation broke down (LAPACK `info > 0`). | With different parameters. Prefer the per-item `info` spans below. |
| `internal_error` | `error` | BatchLAS is inconsistent. Caller-reachable today: `gesv` and `posv` (so `linalg::solve`, `linalg::solve_spd`) for an empty problem, `n`, `nrhs` or `batch` below 1, and a heterogeneous batch. | No; report it. |
| `api_misuse` | `error` | Right call, wrong state: a `Queue` used from another thread, `configure()` after a `Queue` exists. | No. |
| `NoRouteError` (`<batchlas/no_route.hh>`) | `std::runtime_error` | Nothing in this build serves the call (typically `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`). `op()`, `backend()`, `scalar()` say which. | Not in this build. |

Properties of `batchlas::exception` that `tests/error_model_tests.cc` asserts, because each fails
silently if broken: it does **not** derive from `std::exception` (a second `std::exception`
subobject would make `catch (const std::exception&)` ambiguous, and an ambiguous catch base is not
diagnosed); it is a **virtual base**; and it has **no `what()`** (use `e.message()` or catch
`std::exception`).

Outside the hierarchy, invisible to `catch (const batchlas::exception&)`: `std::bad_alloc` (where a
`sycl::malloc_*` returned null), `sycl::exception` (including everything reported asynchronously at
`ctx.wait_and_throw()`), and kernel selection: a `BATCHLAS_<OP>_ROUTE` value that does not parse,
names a family not compiled for the scalar type, or cannot run the shape throws
`std::invalid_argument`; a malformed table in `BATCHLAS_TUNED_DIR`, or a call no family can run in a
build that has the op's vendor library, throws `std::runtime_error`. Open item:
[kernel selection throws outside the hierarchy](design/error-model.md#error-model-kernel-selection-throws-outside-the-hierarchy).
A boundary that must let nothing escape needs `catch (const std::exception&)` behind the BatchLAS
one. Device-side errors surface at `ctx.wait_and_throw()`, not at the enqueuing call.

### Argument checks

- BLAS-2/3: the backend validates shapes (`gemm` every batch item; `trsm` also `lda` and `ldb`).
- Queue-dispatching LAPACK-style calls (`potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `syev`
  without an explicit `<Backend>`) check shapes host-side before any device work and throw
  `batchlas::invalid_argument`. `potrf`, `getrf`, `getri` and `syev` require square `A`; `getrs`
  and `getri` also require matching row and batch sizes. `geqrf` and `orgqr` have no squareness check.
- Output spans must be at least `A.rows() * batch_size` (`pivots`, `W`) or
  `min(A.rows(), A.cols()) * batch_size` (`tau`). The test is `>=`, so one arena can be sliced
  across calls.
- The `f<Backend::CUDA>(ctx, ...)` spelling skips these checks by design.

### Per-item status

Eight routines report a per-item status, opt-in, as one `Span<int32_t>` entry per batch item with
LAPACK's convention (`0` good, positive otherwise): `potrf`, `getrf`, `getri` report factorisation
status; `syev`, `syevx`, `gesvd`, `steqr`, `stedc` report convergence status. It is a field on
`PotrfOptions` and a trailing parameter elsewhere:

```cpp
UnifiedVector<int32_t> info(batch);
potrf(ctx, A.view(), {.uplo = Uplo::Lower, .info = info.to_span()});
getrf(ctx, A.view(), pivots.to_span(), info.to_span());
ctx.wait();
if (info[37] != 0) { /* item 37 is rank-deficient; everything downstream is noise */ }
```

An empty span (the default) reports nothing and costs nothing. The span must be device-accessible
USM, is written in place, and changes no workspace size; a non-empty span shorter than `batch_size`
throws `std::invalid_argument`. `geqrf` and `orgqr` have no status: Householder QR has no numerical
failure mode.

### Convergence status: `syev`, `syevx`, `gesvd`, `steqr`, `stedc` {#convergence-status-syev-syevx-gesvd-steqr-stedc}

Each takes a trailing `Span<int32_t> info`, default empty. Without it, a single non-converged item
in a batch of 16384 is invisible: the call and `ctx.wait()` return normally.

```cpp
UnifiedVector<int32_t> info(batch);
syev<Backend::CUDA>(ctx, A.view(), W.to_span(),
                    JobType::EigenVectors, Uplo::Lower, ws.to_span(), info.to_span());
ctx.wait();
for (int b = 0; b < batch; ++b) if (info[b] != 0) { /* item b's eigenvalues are not to be trusted */ }
```

- `0` means converged. A positive value is LAPACK-like: the number of off-diagonal elements that
  failed to converge where the tier counts, `1` where it tracks only failure. Never negative.
- None of the five sizing queries takes `info`.
- The span is an **accumulator**: zeroed once by the entry point you called, and nested solves
  (`syev` → `syev_blocked` → `stedc`, or many `stedc` merges) only raise it. Poison it before the
  call (fill with `-1`) if the check must be honest: a surviving `-1` is a defect.
- `stedc`'s status covers its merges and the leaf `steqr` solves (see
  [stedc: convergence reporting through info](perf/stedc.md#stedc-convergence-reporting-through-info)).
  `stein` runs a fixed iteration count and reports nothing.

The solve-style calls (`getrs`, `linalg::solve`) report nothing. A near-singular item produces
plausible numbers; when inputs are not known to be well-conditioned, check the residual
`‖A·X − B‖` or scan the factor's diagonal for zeros and NaNs per item.

## Synchronisation and threading

Every entry point enqueues work and returns an `Event`. Contents of a `Matrix`, `MatrixView` or
`UnifiedVector` are not readable until the work finishes:

```cpp
Event e = gemm(ctx, A.view(), B.view(), C.view(), GemmOptions<float>{});
e.wait();                    // wait on this call, or
ctx.wait();                  // wait on everything enqueued on the queue
ctx.wait_and_throw();        // ... and rethrow asynchronous errors
```

Without the wait you read the output as it was before the call, and a fresh `Matrix` is
uninitialised, not zeroed (`Matrix::Zeros(...)` or `view().fill_zeros(ctx)` give a known start).

**A `Queue` is single-threaded.** Use one per thread. It owns an unsynchronised arena and a cached
"last event"; `workspace()`, `trim_workspace()`, submissions, `enqueue()`, `get_event()` and
`create_event_after_external_work()` compare `std::this_thread::get_id()` to the constructing thread
and throw `batchlas::api_misuse` on a mismatch. Queues built for the same `Device` share a SYCL
context, so per-thread queues see each other's USM allocations. To move a `Queue` to another thread,
call `attach_to_current_thread()` from the new owner before its first use.

## The backend comes from the Queue

```cpp
Queue ctx(Device::default_device());                    // AUTO: resolved from the device vendor
Queue host(Device::default_device(), Backend::NETLIB);  // pinned
ctx.set_backend(Backend::CUDA);                         // or change it later
Backend b = ctx.backend();                              // the resolved backend
if (Queue::backend_available(Backend::CUDA)) { /* ... */ }
```

`Backend::AUTO` resolves once on first use and is cached; `set_backend` resets the cache. On a GPU
it picks the vendor stack if compiled in (NVIDIA → CUDA, AMD → ROCM, Intel → MKL), else NETLIB, as
every non-GPU device does. `set_backend` throws `batchlas::unsupported` if the named backend is not
compiled in. The nameable backends are `CUDA`, `ROCM`, `MKL`, `NETLIB` and `AUTO`; `Backend::MAGMA`
and `Backend::SYCL` are unavailable on every build.

This covers the whole surface, extensions included. The exceptions are entry points whose template
parameters are not deducible, listed under
[Which spelling each entry point takes](#which-spelling-each-entry-point-takes). When you need the
backend as a compile-time constant, `with_backend` (`<batchlas/blas/queue-dispatch.hh>`) turns the
runtime choice into one:

```cpp
with_backend(ctx, [&](auto Back) {
    constexpr Backend Bk = Back.value;
    gemm<Bk>(ctx, A.view(), B.view(), C.view(), 1.0f, 0.0f,
             Transpose::NoTrans, Transpose::NoTrans);
});
```

### Which kernel runs: flat kernel selection {#which-kernel-runs-flat-kernel-selection}

The backend picks the library build; it does not pick the kernel. After argument validation, each
op chooses one **kernel family** for the call: a native kernel (`tiny`, `cta`, `lpanel:panel=8`,
`blocked`, ...) or `vendor` (the cuBLAS/cuSOLVER/rocBLAS/host LAPACK call). The decision for an op
is `src/ops/<op>/<op>.cc`, with its vocabulary in `src/ops/<op>/choice.hh`.

- **Tables, per device.** `tuned/<op>.<dtype>.<arch>.txt` is embedded at build time. A call finds
  the nearest measured shape and takes the first family in that row whose correctness predicate
  (`can_run`) admits the call; if none does it tries the next table in borrow order, then the op's
  last-resort order. A GPU with no table borrows the nearest and prints one warning per op. The CPU
  never borrows a GPU table. @ref selection_tables lists every op's families; `tuned/README.md` says
  how each table was produced.
- **Correctness is never traded for speed.** `can_run` is false only where the kernel would throw or
  answer wrongly; the table orders admitted families by speed.
- **Sizing agrees with running.** `*_buffer_size` makes the same choice as the call it sizes.
- **Nothing runnable.** Without the op's vendor library the call throws `batchlas::NoRouteError`.
  With it, an exhausted walk throws `std::runtime_error` (`<op>: no runnable kernel on <device>`).

| variable | effect |
| --- | --- |
| `BATCHLAS_<OP>_ROUTE` | Pins the op: `auto`, `native`, `vendor`, or a family spelling from `choice.hh`. A bad spelling throws `std::invalid_argument`; `native` and `vendor` fall back to `auto` with a warning when nothing of that kind can run. |
| `BATCHLAS_SELECT_TRACE=1` | One stderr line per call: op, shape, chosen family, its time and the runner-up's, and which table answered. |
| `BATCHLAS_TUNED_DIR=<dir>` | A same-named table file in `<dir>` replaces the built-in one. |

Nested ops decide for themselves: a blocked factorisation calls the public `gemm` and `trsm`, which
consult their own tables. hemm, herk and her2k are not table-selected. Design:
[flat kernel selection](design/flat-kernel-selection.md).

## Configuration {#configuration}

Everything BatchLAS reads from the process environment lands in one typed struct, parsed once under
`std::call_once` on first call; `settings()` is thread-safe and the only place in the library that
reads the environment. The complete variable list, with types and defaults, is
@ref design_environment.

```cpp
#include <batchlas/settings.hh>
const Settings& s = batchlas::settings();   // parsed from the environment, once
```

`Settings` has five groups: `routing` (the 19 `BATCHLAS_<OP>_ROUTE` pins, via `route(op)`),
`selection` (algorithm choices outside the route vocabulary; three override an explicit API
argument), `geometry` (launch geometry, block widths, iteration counts), `diagnostics` (tracing,
dumping, profiling, opt-in checks; none changes a numeric result) and `unsafe`. Field names follow
variable names (`BATCHLAS_TRSM_OUTER_NB` is `settings().geometry.trsm_outer_nb`). A knob with a
bespoke parser is an `EnvValue` (raw string with `is_set()`, `value()`, `get()`); a default that is
a sentinel (`""`, `0`, `-1`, `nullopt`) means the real default is computed at the call site.
Variables read only by the repository's tests and benchmarks (`BATCHLAS_TEST_BACKEND`,
`BATCHLAS_BENCH_*`, ...) are not in `Settings`.

### Setting it programmatically

```cpp
Settings s = batchlas::settings();          // start from what the environment said
s.routing.route("gemm") = EnvValue::of("native");
s.geometry.trsm_outer_nb = 64;
batchlas::configure(s);                     // before the first Queue
```

- **`configure()` is permitted only until the first `Queue` is constructed**; afterwards it throws
  `batchlas::api_misuse` and changes nothing. A mid-run change would make calls disagree about the
  kernel and under-size a workspace already allocated.
- An explicit `configure()` beats the environment at that moment. A later `reload_settings()` (which
  every `ScopedEnvVar` triggers at both ends of its scope) re-reads the environment over it, so
  call `configure()` once at start-up.
- `kernel_trace_path`, `coverage_out` and `dump_bandr1.dir` are paths the library opens for writing;
  an application inheriting an environment it did not choose should clear them with `configure()`.

### Re-reading the environment, and `ScopedEnvVar`

`batchlas::ScopedEnvVar` (`<batchlas/util/env.hh>`) sets a variable for a scope and restores it on
exit; a null value unsets it for the duration.

```cpp
{
    ScopedEnvVar pin("BATCHLAS_GEMM_ROUTE", "native");
    gemm(ctx, a, b, c, GemmOptions<float>{});   // runs the native kernel
}                                               // ...and back to whatever it was
```

Its constructor and destructor call `batchlas::detail::reload_settings()`. Anything else that writes
the environment mid-process (a raw `setenv`) must call `reload_settings()` too, or the change is
invisible. A reload between a `*_buffer_size()` query and its call desynchronises the workspace from
the block width the call uses: do not let a `ScopedEnvVar` scope straddle a sizing/solve pair.
`env_truthy` accepts exactly `{1, true, TRUE, on, ON}` and `env_falsy` exactly
`{0, false, FALSE, off, OFF}`; an unset variable is neither, which is why some fields are
`std::optional<bool>`.

### `BATCHLAS_ALLOW_UNSAFE_ENV`

Most knobs pick a kernel, geometry or dump path; a wrong one costs a measurement. A few remove a
check, and live in `Settings::unsafe` behind a CMake option (`-DBATCHLAS_ALLOW_UNSAFE_ENV=ON`,
default OFF). With it **OFF** (release and install builds) `settings()` holds the `unsafe` fields at
their safe values whatever the environment says, and prints one warning at first use naming the
variable and the option. It is ON in the `dev`, `dev-tests`, `fast-dev`, `dev-gpu`, `dev-gpu-tests`
and `benchmarks` presets and OFF in `cuda` (the pre-push gate). `tests/settings_tests.cc` asserts
both arms.

| variable | what it disables | how it fails |
| --- | --- | --- |
| `BATCHLAS_SKIP_POINTER_CHECKS` | the one-USM-query-per-argument reachability check (~70 ns) | host memory reaches the device as a wild address: `CUDA_ERROR_ILLEGAL_ADDRESS`, then `SIGABRT` at teardown, which no catch block stops |
| `BATCHLAS_LATRD_GRID_FORCE_UNSAFE` | the co-residency cap on the `latrd` grid path | the grid barrier relies on that cap, so the kernel **hangs** (it looks like slow JIT). Run under `timeout` |
| `BATCHLAS_BLAS_HEALTH=off` | the host-`dgemm` correctness probe | with a known-bad OpenBLAS kernel, every `double` and `complex<double>` host result is silently wrong by O(1) |

The gate refuses only the unsafe direction (`BATCHLAS_BLAS_HEALTH=error` is allowed, `off` is
refused). `configure()` is not gated.

## Workspaces come from the queue's arena {#workspaces-come-from-the-queues-arena}

LAPACK-style entry points need scratch. Leaving the workspace argument out leases it from a
per-`Queue` arena, sized by the matching `*_buffer_size`:

```cpp
potrf(ctx, A.view(), {.uplo = Uplo::Lower});   // workspace leased and returned
```

To size and own the buffer yourself:

```cpp
with_backend(ctx, [&](auto Back) {
    constexpr Backend Bk = Back.value;
    UnifiedVector<std::byte> ws(potrf_buffer_size<Bk, float>(ctx, A.view(), Uplo::Lower));
    potrf<Bk, float>(ctx, A.view(), Uplo::Lower, ws.to_span());
    ctx.wait();                                // ws must outlive the kernels
});
```

- A repeated arena-backed call reuses memory instead of malloc/free-ing device memory.
- The arena grows to the peak it was asked for and holds it; `ctx.workspace_capacity()` reports the
  size. To cap it, pass your own span, destroy the `Queue`, or call `ctx.trim_workspace()`, which
  frees the blocks, drains the queue (so it can throw), and is `[[nodiscard]]`: it returns `false`
  and does nothing while any lease is outstanding.
- Lease explicitly with `auto lease = ctx.workspace(n_bytes); Span<std::byte> bytes = lease.span();`
  (released when `lease` goes out of scope).

When to keep managing the workspace yourself (pass a span explicitly, e.g.
`potrf(ctx, A.view(), {.uplo = Uplo::Lower}, my_span)`):

- **On an out-of-order queue, pass your own span.** A lease's bytes go to the next borrower on
  return; in order that borrower is ordered behind this call, out of order nothing orders them, so
  the release drains the queue and every arena-backed call blocks until the device is idle.
- Call `ws.release()` before reassigning a live lease (`ws = ctx.workspace(...)`); otherwise the
  new loan is taken before the old is returned and the arena ratchets.
- A lease's release orders only against the queue it came from. Pass your own span when the work
  runs on a sibling queue built with `Queue(base, in_order)`.
- Never build a workspace from a local `UnifiedVector` that dies before the kernels ran.

Full lifetime rules: `batchlas/util/workspace.hh`, @ref design_workspace.

## Interop with CUDA and with your own SYCL

`Queue::native_handle()` returns the backend-native stream as `void*`: a `CUstream`
(`cudaStream_t`) when the queue's device runs on the CUDA SYCL backend, a `hipStream_t` on HIP,
`nullptr` otherwise. It keys off the device, not `ctx.backend()`. The stream belongs to the
`Queue`: do not destroy it or outlive the `Queue`. On the default in-order queue, work you push to
it runs after everything BatchLAS has submitted. To make BatchLAS wait for your work, call
`ctx.create_event_after_external_work()` after enqueueing it.

For SYCL-typed interop include `<batchlas/sycl_interop.hh>`, the one BatchLAS header that pulls in
`<sycl/sycl.hpp>` (not reachable from `<batchlas.hh>`; do not re-export it from your headers):

```cpp
batchlas::sycl_queue(const Queue&)   -> sycl::queue&
batchlas::sycl_event(const Event&)   -> sycl::event
batchlas::event_from_sycl(sycl::event) -> Event
```

```cpp
sycl::event mine = my_queue.submit(/* ... */);
Event e = batchlas::event_from_sycl(mine);
ctx.enqueue(e);                                       // `enqueue` takes an lvalue
batchlas::potrf(ctx, A.view(), {.uplo = Uplo::Lower}); // waits for `mine`

// ... and in the other direction:
my_queue.ext_oneapi_submit_barrier({batchlas::sycl_event(ctx.get_event())});
```

Both queues must share a SYCL context. Pointers from `cudaMalloc`, `cudaMallocManaged`,
`sycl::malloc_device` and `sycl::malloc_host` wrap into `Span`/`MatrixView` zero-copy when reachable
from that context.

## The `linalg` convenience layer

`batchlas::linalg` (`batchlas/blas/linalg-ops.hh`, group @ref api_linalg) offers value-returning and
elementwise free functions; there are no operator overloads. Each takes its backend from the queue
and workspace from the arena.

```cpp
auto C = linalg::matmul(ctx, A.view(), B.view());   // allocates and returns C
auto L = linalg::cholesky(ctx, A.view());           // A is not modified
auto X = linalg::solve(ctx, A.view(), B.view());    // A X = B
auto e = linalg::eigh(ctx, A.view());               // e.values, e.vectors
ctx.wait();                                         // required before reading any of them
```

These enqueue and do not wait, except `linalg::norm`, `linalg::cond` and `linalg::svd`.
`add`, `subtract`, `multiply`, `divide` and `scaled` allocate their result; the `_into` forms write
into storage you own (use them in inner loops); `scale` works in place.

- `matmul` takes a `MatmulOptions` with no `beta`: naming it is a compile error.
- `multiply` is elementwise (Hadamard); use `matmul` for the matrix product.
- `eigh` and `svd` return a per-item `info` vector (`0` = converged), always filled; check it.
  `solve`, `solve_spd` and `cholesky` report nothing.
- There is no `linalg::qr`; compose `geqrf` and `orgqr`. The wrapper is withheld because of an
  unexplained wrong-answer defect ([Known defects](design/known-defects.md)).

Design of the layer and option structs: [API conventions](design/api-conventions.md).

---

Adding an entry point rather than calling one? See [extending.md](extending.md).

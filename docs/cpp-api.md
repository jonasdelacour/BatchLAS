# The BatchLAS C++ API

Matrices are column-major and batched, every call enqueues work and returns
immediately, and the backend, the workspace and the device all come from the
`Queue`.

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

`Matrix`, `MatrixView`, `gemm`, `potrf` and the rest of the numerical surface are
in namespace `batchlas`. `Queue`, `Device`, `Event`, `Span` and `UnifiedVector`
are in the **global** namespace: they need no qualification and no
`using namespace batchlas;`.

### The short template spelling

`Matrix`, `MatrixView` and `VectorView` default their template parameters to
`<float, MatrixFormat::Dense>` (`<float>` for `VectorView`), so these pairs name
the same type:

```cpp
Matrix<float>              A(n, n, batch);    // == Matrix<float, MatrixFormat::Dense>
MatrixView<float>          V = A.view();      // == MatrixView<float, MatrixFormat::Dense>
Matrix<std::complex<float>> Z(n, n, batch);   // dense too
Matrix<float, MatrixFormat::CSR> S(n, n, NonZeros{nnz}, batch);
```

Write the format out when it is not `Dense`, and drop it when it is. `Vector<T>`
has no default: spell it `Vector<float>`.

### Building and installing BatchLAS

What the build needs:

- **A clang-based SYCL compiler, clang 16 or newer.** BatchLAS is developed and
  tested with intel/llvm DPC++ built with `--cuda`, installed at
  `/opt/dpcpp-cuda` in the commands below; substitute your own prefix. Build the
  library and everything that consumes it with the same compiler: another one
  links with undefined references to the constrained entry points.
- **oneDPL headers**, which DPC++ does not bundle. Pass
  `-DONEDPL_ROOT=<oneapi-dpl-prefix>`, where
  `<oneapi-dpl-prefix>/include/oneapi/dpl` exists, or set `ONEDPL_ROOT` or
  `DPL_ROOT` in the environment (oneAPI's `setvars.sh` sets `DPL_ROOT`).
  Configure fails without them.
- **LAPACKE and CBLAS** — `liblapacke` plus `libcblas` or `libblas` — for the
  host backend, which `BATCHLAS_ENABLE_NETLIB` turns on by default. Configure
  warns and builds with the host backend off when they are missing.
- **The CUDA toolkit** (cudart, cuBLAS, cuSOLVER, cuSPARSE) whenever the CUDA
  backend is on, which the default `AUTO` does as soon as the SYCL runtime
  exposes a CUDA device.
- **CMake 3.14 or newer** for BatchLAS itself, 3.21 for the consuming project
  below.

```bash
git clone https://github.com/jonasdelacour/BatchLAS.git && cd BatchLAS
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
      -DONEDPL_ROOT=<oneapi-dpl-prefix> \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBATCHLAS_BUILD_TESTS=OFF
cmake --build build -j"$(nproc)"
cmake --install build --prefix "$HOME/inst"   # <prefix> in the consumer commands below
```

Build options:

| option | what it does |
| --- | --- |
| `BATCHLAS_ENABLE_CUDA` | cuBLAS/cuSOLVER backend: `AUTO` (default — on when the SYCL runtime exposes a CUDA device), `ON` (configure fails when it does not), `OFF` |
| `BATCHLAS_NVIDIA_ARCH` | target architecture, e.g. `sm_89`; the build detects the local GPU, so pass this only when cross-building |
| `BATCHLAS_ENABLE_NETLIB` | host BLAS/LAPACK backend, `ON` by default |
| `BATCHLAS_ENABLE_MKL`, `BATCHLAS_ENABLE_ROCM` | the oneMKL and ROCm backends, both `OFF` by default |
| `BATCHLAS_BUILD_TESTS` | on by default for a top-level build; `OFF` when you only want the library |
| `BATCHLAS_BUILD_BENCHMARKS`, `BATCHLAS_BUILD_PYTHON` | off by default |

`CMakePresets.json` carries the development configurations — `cmake --preset dev`
to build the library, `--preset dev-tests` to add the test suite.

### Building against BatchLAS

**Configure the whole consuming project with the same SYCL compiler BatchLAS was
built with.** A mismatch compiles and then fails at link:

```
undefined reference to `batchlas::Matrix<float, (batchlas::MatrixFormat)0>::Matrix<...>(int, int, int, int, int)'
```

The package compares your `CMAKE_CXX_COMPILER` against the recorded one by
realpath and warns when they differ; `-DBATCHLAS_REQUIRE_MATCHING_COMPILER=ON`
makes that a hard error. If you do not know which compiler an install was built
with, look for `CMAKE_CXX_COMPILER` in the `CMakeCache.txt` of its build tree.

The consuming `CMakeLists.txt`:

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

**Start from `examples/consumer/`.** It is that project, standalone and
runnable: `main.cc` is a batched `gemm` on USM-backed `Matrix` operands, checked
against a hand-computed reference and printing `PASS`, and it is the shortest
complete thing to copy. Its `CMakeLists.txt` is the file above plus two options
the example uses for its own testing: `BATCHLAS_CONSUMER_USE_FSYCL` adds
`-fsycl`, which is the recipe a consumer with its own kernels needs, and
`BATCHLAS_CONSUMER_DECOY` puts colliding `blas/`, `util/` and `internal/`
headers on the include path. Build and run it with the two commands above,
pointed at `examples/consumer`.

`BatchLAS::batchlas` is the only target to link. It propagates `cxx_std_20`, the
include root `<prefix>/include`, the component libraries and `-Wl,--no-as-needed`
— keep that flag if you override the link line yourself.

A translation unit that only calls the documented API needs no SYCL flags. One
that includes `<sycl/sycl.hpp>` or `<batchlas/sycl_interop.hh>`, or writes its
own kernels, passes them itself, for the targets the install was built for — the
config file records those in `BatchLAS_SYCL_TARGETS`:

```cmake
target_compile_options(my_app PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
target_link_options(my_app    PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
```

`find_package(BatchLAS)` pulls in `OpenMP` when the install was built with it,
and `MKL` when the MKL backend is on. CUDA and LAPACK are linked privately into
the component libraries, so a CPU-only machine can `find_package` a CUDA-enabled
install. Ask for the package whole — `find_package(BatchLAS CONFIG REQUIRED)`
with no `COMPONENTS`.

At run time the loader must find the DPC++ runtime and the SYCL adapters it
`dlopen`s. Export `LD_LIBRARY_PATH=<dpcpp-prefix>/lib` for interactive use; in a
container, drop a file naming that directory in `/etc/ld.so.conf.d/` and run
`ldconfig`.

### Headers

Everything installs under `<prefix>/include/batchlas/`, plus the umbrella file
`<prefix>/include/batchlas.hh`. Include `<batchlas.hh>`, or reach in directly:

```cpp
#include <batchlas/blas/linalg.hh>            // what <batchlas.hh> pulls in
#include <batchlas/util/sycl-device-queue.hh>
```

Every public header is spelled `<batchlas/...>`; the paths named in this document
— `batchlas/blas/options.hh`, `batchlas/blas/matrix.hh`,
`batchlas/util/workspace.hh` — are the same under the install prefix as in the
source tree.

## Devices and queues

A `Queue` names the device, carries the backend, owns a workspace arena and
orders the work. `Device::default_device()` is the first GPU, else the first
CPU, else the first host device. To pick a particular device, enumerate:

```cpp
auto gpus = Device::get_devices(DeviceType::GPU);   // CPU, ACCELERATOR, HOST too
for (const auto& d : gpus) std::cout << d.get_name() << '\n';

Queue ctx(gpus.at(1));                              // the second GPU
Queue cpu("cpu");                                   // "cpu", "gpu", "accelerator"
```

`get_devices` returns the devices in the runtime's order; the string constructor
takes the first of a type and throws `std::runtime_error` when there is none.
`get_name()`, `get_vendor()` and `get_property(DeviceProperty::GLOBAL_MEM_SIZE)`
describe one.

The `Queue` constructors:

```cpp
Queue ctx;                                       // default device, in-order, backend AUTO
Queue ctx2(device);                              // in-order
Queue ooo(device, /*in_order=*/false);           // out-of-order
Queue host(device, Backend::NETLIB);             // backend pinned, in-order
Queue host2(device, Backend::NETLIB, /*in_order=*/false);
Queue sibling(ctx, /*in_order=*/true);           // ctx's SYCL context and device
```

A `Queue` is movable and not copyable. `Queue(base, in_order)` shares `base`'s
SYCL context and device, so the two see each other's USM allocations — that is
how to get a second queue that can take the same buffers. Each queue still owns
its own arena and event chain, and belongs to one thread (see *Synchronisation
and threading*).

In-order is the default and is what makes an arena-backed workspace free; on an
out-of-order queue, pass your own workspace spans (see *When to keep managing the
workspace yourself*).

## What the operations compute

Every entry point is batched: it applies the same operation to every item of the
batch, and **all matrix arguments to one call must have the same batch size**.
`α` and `β` below are the `alpha` and `beta` fields of the option struct, and
`op(A)` is `A`, `Aᵀ` or `Aᴴ` according to the `trans` field. The "shapes" column
of each table states a requirement: hold to it, and see *What gets thrown* for
which calls check it.

### Dense BLAS

| call | computes | written | shapes, per batch item |
| --- | --- | --- | --- |
| `gemm(ctx, A, B, C, opts)` | `C := α·op(A)·op(B) + β·C` | `C` | `op(A)` m×k, `op(B)` k×n, `C` m×n |
| `gemv(ctx, A, x, y, opts)` | `y := α·op(A)·x + β·y` | `y` | `op(A)` m×n, `x` length n, `y` length m |
| `symm(ctx, A, B, C, opts)` | `C := α·A·B + β·C` (`Side::Left`), `C := α·B·A + β·C` (`Side::Right`); `A` symmetric, only its `uplo` triangle is read | `C` | `B`, `C` m×n; `A` m×m (Left) or n×n (Right) |
| `hemm(ctx, A, B, C, opts)` | as `symm`, with `A` Hermitian: the other triangle is taken as the conjugate transpose and the diagonal's imaginary part as zero, whatever is stored there | `C` | as `symm` |
| `syrk(ctx, A, C, opts)` | `C := α·A·Aᵀ + β·C` (`NoTrans`), `C := α·Aᵀ·A + β·C` (`Trans`) | `C`, `uplo` triangle only | `A` n×k (NoTrans) or k×n (Trans); `C` n×n |
| `herk(ctx, A, C, opts)` | `C := α·A·Aᴴ + β·C` (`NoTrans`), `C := α·Aᴴ·A + β·C` (`ConjTrans`) | `C`, `uplo` triangle only; the diagonal comes out real | as `syrk`. `trans` must be `NoTrans` or `ConjTrans` |
| `syr2k(ctx, A, B, C, opts)` | `C := α·A·Bᵀ + α·B·Aᵀ + β·C` (`NoTrans`), `C := α·Aᵀ·B + α·Bᵀ·A + β·C` (`Trans`) | `C`, `uplo` triangle only | `A`, `B` n×k or k×n; `C` n×n; k > 0 |
| `her2k(ctx, A, B, C, opts)` | `C := α·A·Bᴴ + conj(α)·B·Aᴴ + β·C` (`NoTrans`), `C := α·Aᴴ·B + conj(α)·Bᴴ·A + β·C` (`ConjTrans`) | `C`, `uplo` triangle only; real diagonal | as `syr2k` |
| `trmm(ctx, A, B, C, opts)` | `C := α·op(A)·B` (`Left`), `C := α·B·op(A)` (`Right`); `A` triangular, `uplo` and `diag` describe it | **`C`**; `B` is an input and is not modified | `A` m×m (Left) or n×n (Right); `B`, `C` m×n |
| `trsm(ctx, A, B, opts)` | solves `op(A)·X = α·B` (`Left`) or `X·op(A) = α·B` (`Right`) | **`B`**, overwritten with `X` | `A` m×m (Left) or n×n (Right); `B` m×n |

**`trmm` and `trsm` differ from `?trmm`/`?trsm` and from each other.** `trmm`
takes three matrices and writes the product into `C`, leaving `B` alone — where
the reference BLAS `?trmm` is in place on `B`. `trsm` takes two and is in place:
the solution replaces `B`. Expecting `trmm` to have updated `B`, or expecting
`trsm` to have left it alone, gives a wrong answer, not a compile error.

**"`uplo` triangle only" means the other triangle comes back exactly as it went
in — including uninitialised.** `syrk`, `herk`, `syr2k` and `her2k` write the
named half of `C` and do not touch the other one, and a freshly constructed
`Matrix(rows, cols, batch)` is uninitialised, so the unwritten half holds
whatever was in that memory — not zeros. What comes back is a valid triangular
result and not a symmetric matrix; no call reports this, so `gemm`, `gemv` or
host code that indexes both halves silently produces wrong numbers.

Mirror the triangle before any such use:

```cpp
syrk(ctx, A.view(), C.view(), {.uplo = Uplo::Lower}).wait();
C.view().symmetrize(ctx, Uplo::Lower).wait();   // hermitize() for herk/her2k
```

`symmetrize(ctx, uplo)` copies the named triangle into the other one, and
`hermitize(ctx, uplo)` copies its conjugate transpose (the correct one for
`herk`/`her2k`); both are one kernel on `MatrixView` and return an `Event`.
Zeroing `C` first (`Matrix::Zeros`, `view().fill_zeros(ctx)`) makes the other
half defined, not symmetric — mirror it as well.

**`gemm` handles a heterogeneous batch natively.** When the items of a batch
carry differing `active_rows`/`active_cols`, `gemm` detects that and routes to
the heterogeneous path itself, on every backend — there is no separate entry
point to reach for, and there has never been a `gemm` that could not do this.
(A `gemm_heterogeneous` alias also exists in the C++ headers; it forwards to
`gemm` with an unchanged argument list, so it buys you nothing — call `gemm`.
The Python `gemm_heterogeneous` is a different thing, and does something `gemm`
does not: it accepts a list of differently-shaped arrays.)

`symm`, `syrk` and `syr2k` are constrained to **real** `T` and do not instantiate
for `std::complex`; `hemm`, `herk` and `her2k` are constrained to **complex** `T`
and do not instantiate for real. `gemm`, `gemv`, `trmm` and `trsm` take both.
`herk`'s `α` and `β` are real even though its operands are complex, and `her2k`'s
`β` is real, because that is what keeps the result Hermitian.

### LAPACK-style

These take a workspace. Leave it out and it is leased from the queue's arena; see
*Workspaces come from the queue's arena*.

| call | computes | written | shapes, per batch item |
| --- | --- | --- | --- |
| `potrf(ctx, A, opts)` | Cholesky: `A = L·Lᴴ` (`Uplo::Lower`) or `A = Uᴴ·U` (`Uplo::Upper`) | `A`, in place, `uplo` triangle | `A` n×n |
| `getrf(ctx, A, pivots)` | LU with partial pivoting, `A = P·L·U` | `A` in place, and `pivots` | `A` n×n; `pivots` is a `Span<int64_t>` of n·batch |
| `getrs(ctx, A, B, pivots, opts)` | solves `op(A)·X = B` from `getrf`'s factors and pivots | `B`, overwritten with `X` | `A` n×n already factorised, `B` n×nrhs |
| `getri(ctx, A, C, pivots)` | `C := A⁻¹` from `getrf`'s factors and pivots | **`C`**; `A` is read-only | `A`, `C` both n×n |
| `geqrf(ctx, A, tau)` | QR: `R` in the upper triangle of `A`, the Householder reflectors below it, their scalars in `tau` | `A` in place, and `tau` | `A` m×n; `tau` is a `Span<T>` of min(m,n)·batch |
| `orgqr(ctx, A, tau)` | expands `geqrf`'s reflectors into the explicit `Q` | `A`, overwritten with `Q` | `A` m×n, k = min(m,n) columns of `Q` |
| `syev(ctx, A, W, opts)` | symmetric/Hermitian eigendecomposition of the `uplo` triangle | `W` gets the eigenvalues, ascending; `A` gets the eigenvectors when `jobz == JobType::EigenVectors` | `A` n×n; `W` is a `Span` of n·batch of the **real** type (`float` for `std::complex<float>`) |

`getri`, like `trmm`, writes a second matrix operand and leaves its input alone.

### Which type each parameter takes

- **Matrix parameters** take `Matrix<T>` or `MatrixView<T>`, mixed freely, on
  every spelling. Where an entry point's primary declares a `MatrixView<T>`, an
  owning-`Matrix` overload forwards to it, so `A.view()` is never required.
- **Vector parameters** — `gemv`'s `x` and `y` — take `VectorView<T>`. An owning
  `Vector<T>` converts, and `x.view()` is the spelling that always works.
- **Flat arrays** — eigenvalues `W`, singular values `S`, `tau`, `pivots`, and
  workspaces — take `Span<T>`. `UnifiedVector<T>`, the owning USM array,
  converts implicitly; `to_span()` is explicit. A `Vector<T>` is not a `Span`.

The same rule extends to the extension surface (`steqr`, `stebz`, `stein`,
`stedc`, `lanczos`, `ritz_values`): a parameter takes `VectorView<T>` when it is
one logical vector *per batch item* and so needs `inc`/`stride`/`batch_size`, and
`Span<T>` when it is a flat array with one entry per item or per matrix and no
stride freedom. `stebz` shows both in one call — `d`, `e` and `w` are
`VectorView<T>` (strided, per item), while `m`, the per-item eigenvalue count, is
`Span<int32_t>`. The two are deliberately not interconvertible: a `VectorView`
demoted to a `Span` would drop the stride and read the wrong elements rather than
fail to compile.

What to write at the call site, by owning type:

| you hold | parameter is `Span<T>` | parameter is `VectorView<T>` |
| --- | --- | --- |
| `UnifiedVector<T>` | pass it directly (implicit), or `.to_span()` | `VectorView<T>(v, size, batch, inc, stride)` |
| `Vector<T>` | `.data()` — the whole allocation, so only when `inc == 1` and it is packed | `.view()`, always |
| raw pointer | `Span<T>(p, n)` | `VectorView<T>(p, size, batch, inc, stride)` |

`Vector<T>::view()` is required whenever `T` has to be *deduced* from the
argument, which is every templated entry point: the implicit
`VectorView(const Vector<T>&)` conversion exists but template argument deduction
never considers user-defined conversions. Several entry points also ship an
owning-`Vector` forwarding overload so the bare `Vector` works — `stebz`,
`stein`, `steqr`, `steqr_cta` and `stedc` all do — but `.view()` is the spelling
that works everywhere.

`pivots` is `int64_t`, `tau` is `T`, and `W` is the real counterpart of `T`
(`float` for `std::complex<float>`).

```cpp
UnifiedVector<int64_t> pivots(n * batch);       // owning USM array; also (count, value)
UnifiedVector<float>   tau(std::min(m, n) * batch);
UnifiedVector<float>   W(n * batch);            // real, even for complex A
Span<float>            w = W.to_span();         // or Span<float>(ptr, count); never owns

Vector<float> x(n, /*batch_size=*/batch), y(m, batch);   // owning USM vectors
gemv(ctx, A.view(), x.view(), y.view(), {.alpha = 1.0f});
```

**`Vector` and `VectorView` take `inc` and `stride` in opposite orders.** It is
`Vector<T>(size, batch_size, stride, inc)` and
`VectorView<T>(ptr, size, batch_size, inc, stride)`. Both parameters are `int`
and both default — `inc` to 1, `stride` to `size * inc` — so a pair passed in
the other order compiles and reads the wrong elements. Read the argument order
off the constructor you are calling, every time.

`Vector<T>::zeros(size, batch_size, stride, inc)`, `::ones(...)` and
`::standard_basis(size, index, batch_size, stride)` build one directly.

## Options are structs with defaults

Most entry points take an option struct, so you write only what differs from the
default. Designated initialisers make the call self-documenting:

```cpp
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f, .transA = Transpose::Trans});
syev(ctx, A.view(), W, {.jobz = JobType::NoEigenVectors});
getrs(ctx, LU.view(), X.view(), pivots, {.trans = Transpose::Trans});
```

### The fields and their defaults

The dense BLAS structs are templated on `T`; the three LAPACK ones are not. They
all live in `batchlas/blas/options.hh`.

| struct | fields, with defaults |
| --- | --- |
| `GemmOptions<T>` | `alpha = T(1)`, `beta = T(0)`, `transA = Transpose::NoTrans`, `transB = Transpose::NoTrans`, `precision = ComputePrecision::Default` |
| `GemvOptions<T>` | `alpha = T(1)`, `beta = T(0)`, `transA = Transpose::NoTrans` |
| `SymmOptions<T>` | `alpha = T(1)`, `beta = T(0)`, `side = Side::Left`, `uplo = Uplo::Lower` |
| `HemmOptions<T>` | `alpha = T(1)`, `beta = T(0)`, `side = Side::Left`, `uplo = Uplo::Lower` |
| `SyrkOptions<T>` | `alpha = T(1)`, `beta = T(0)`, `uplo = Uplo::Lower`, `trans = Transpose::NoTrans` |
| `HerkOptions<T>` | `alpha`, `beta` — **real**, `float_t<T>(1)` and `float_t<T>(0)` — `uplo = Uplo::Lower`, `trans = Transpose::NoTrans` |
| `Syr2kOptions<T>` | `alpha = T(1)`, `beta = T(0)`, `uplo = Uplo::Lower`, `trans = Transpose::NoTrans` |
| `Her2kOptions<T>` | `alpha = T(1)` (complex), `beta = float_t<T>(0)` (**real**), `uplo = Uplo::Lower`, `trans = Transpose::NoTrans` |
| `TrmmOptions<T>` | `alpha = T(1)` (**no `beta`**), `side = Side::Left`, `uplo = Uplo::Lower`, `trans = Transpose::NoTrans`, `diag = Diag::NonUnit` |
| `TrsmOptions<T>` | `alpha = T(1)`, `side = Side::Left`, `uplo = Uplo::Lower`, `trans = Transpose::NoTrans`, `diag = Diag::NonUnit` |
| `PotrfOptions` | `uplo = Uplo::Lower` |
| `GetrsOptions` | `trans = Transpose::NoTrans` |
| `SyevOptions` | `jobz = JobType::EigenVectors`, `uplo = Uplo::Lower` |

**Every `uplo` defaults to `Uplo::Lower`.** A default-constructed option struct
therefore reads the *lower* triangle, so fill the lower triangle or say
`{.uplo = Uplo::Upper}`. Populating the upper triangle and calling `potrf` or
`syev` with default options factorises whatever is in the lower one, and reports
nothing.

The field is spelled `transA` (and `transB`) in `GemmOptions` and `GemvOptions`,
and `trans` everywhere else.

`ComputePrecision` appears only on `gemm`. `Default` means "compute in the input
type"; the other values are `F32`, `F64`, `F16`, `BF16` and `TF32`, and a backend
that cannot serve the one you ask for says so at compile time.

### `*Options` and `*Params` are two different things

Two suffixes appear on argument structs and they are not interchangeable.

`*Options` structs live in `batchlas/blas/options.hh` and belong to the
convenience layer: the backend comes from the `Queue`, `T` is deduced from the
matrices, the struct carries every non-matrix argument, and — for the LAPACK
entry points — the workspace may be omitted. `gemm`, `gemv`, `symm`, `hemm`,
`herk`, `her2k`, `syrk`, `syr2k`, `trmm`, `trsm`, `potrf`, `getrs`, `syev`,
`ormqr` and `gesvd` have one, and `ortho`'s lives in `extensions.hh`.

`*Params` structs live in `batchlas/blas/extensions.hh` and
`batchlas/blas/functions/iluk.hh`. They are ordinary arguments to entry points
that have no convenience layer: you still name the backend
(`syevx<Backend::CUDA, float>`) and you still pass a workspace.

The suffix alone does not tell you where the struct sits in the argument list,
so read the declaration. The last two rows are why:

| struct | entry points | where it sits |
| --- | --- | --- |
| `SyevxParams<T>` | `syevx`, `syevx_buffer_size`, `syevx_resolve_range` | last, after `workspace`, `jobz`, `V` |
| `LanczosParams<T>` | `lanczos`, `lanczos_buffer_size` | last |
| `StebzParams<T>` | `stebz`, `stebz_buffer_size` | last, after `ws` |
| `SteinParams<T>` | `stein` | last |
| `SteqrParams<T>` | `steqr`, `steqr_cta` | **second to last** — `eigvects` follows |
| `StedcParams<T>` | `stedc`, `stedc_buffer_size` | **second to last** — `eigvects` follows |
| `JacobiParams<T>` | `syev_jacobi_cta` and its `_buffer_size` | last |
| `GesvdjParams<T>` | `gesvdj_cta` and its `_buffer_size` | last |
| `SytrdBandReductionParams` | `sytrd_band_reduction` | **replaces** the `int32_t block_size` argument |
| `ILUKParams<T>` | `iluk_factorize`, `iluk_buffer_size` | the only non-matrix argument — behaves like an `*Options` struct |

The structs were deliberately not renamed to make this visible in the name. The
rule a rename would have encoded — "`*Options` replaces the positional
arguments, `*Params` is extra tuning appended after the workspace" — is false for
four of the ten above: `ILUKParams` and `SytrdBandReductionParams` play the
`*Options` role exactly, and `SteqrParams` and `StedcParams` are not last.
Renaming on a rule that does not hold would have moved the inaccuracy from the
docs into the type names, where it is harder to correct.

### Which spelling each entry point takes

- The dense BLAS calls — `gemm`, `gemv`, `symm`, `hemm`, `herk`, `her2k`, `syrk`,
  `syr2k`, `trmm`, `trsm` — take an option struct and no workspace.
- `potrf`, `getrs` and `syev` take an option struct, and take the workspace or
  lease it: `potrf(ctx, A, opts)` and `potrf(ctx, A, opts, ws)` both exist.
- `getrf`, `getri`, `geqrf` and `orgqr` carry no options. The arena-backed
  spelling omits the workspace — `getrf(ctx, A, pivots)` — and the positional
  spelling takes it: `getrf<Back, T>(ctx, A, pivots, ws)`.
- `gesvd`, `ormqr`, `ortho` and `spmm` take positional arguments and a workspace
  span. Lease it from the arena yourself:

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

- Entry points whose template parameters cannot be deduced from their arguments
  keep the explicit `f<Backend, T>(...)` form. There are two kinds:
  `tridiagonal_solver_buffer_size`, whose arguments are all scalars; and the six
  `random_*_with_log10_cond_metric` generators, whose scalar type appears only as
  `float_t<T>` — an alias template, so a non-deduced context — and in the return
  type. Spell them `random_with_log10_cond_metric<Backend::CUDA, float>(ctx, …)`.

- The sizing calls whose only `T`-bearing argument is an option struct —
  `stebz_buffer_size`, `stein_buffer_size`, and `stedc_buffer_size` — take that
  struct as a REQUIRED argument, with no default, for exactly this reason: a
  default would make `stebz_buffer_size(ctx, n, batch)` look available while `T`
  had nowhere to come from, and the diagnostic is not a deduction error pointing
  at the declaration but "no matching function" from the queue-deducing wrapper,
  which probes the call in its requires-clause and silently drops itself when the
  substitution fails. Pass `StebzParams<float>{}` for the defaults and the call
  deduces both the backend and `T`:
  `stebz_buffer_size(ctx, n, batch, StebzParams<float>{})`.

When you pass an empty option struct *together with* an explicit workspace, name
the type: `potrf(ctx, A.view(), PotrfOptions{}, ws)`.

`T` is deduced from the matrix arguments, never from the option struct, so on an
option-struct call let it deduce — `syev<B>(ctx, ...)`, or `syev(ctx, ...)` to
take the backend from the queue as well. Name `T` on the positional spelling,
`syev<B, float>(ctx, ...)`, where the second template parameter is the scalar
type.

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
  column `j+1`. Pass `0` for "packed", which resolves to `rows`, or a value of at
  least `rows`; a larger `ld` is how you view a sub-block of a bigger buffer.
- **`stride`** — the element distance between batch item `b` and item `b+1`. It
  defaults to `0`, which means `ld * cols`, i.e. the items are packed back to
  back.

Both defaults are resolved the same way in `Matrix(rows, cols, batch, ld, stride)`
and in `MatrixView(data, rows, cols, ld, stride, batch)`, and the resolution is
deterministic: `Matrix(rows, cols, batch)` allocates exactly `rows * cols * batch`
elements with `ld() == rows()` and `stride() == rows() * cols()`. Nothing pads.

**The shape fields are `int`; the addresses are not.** `rows`, `cols`, `ld`,
`stride` and `batch_size` are `int`, but element access evaluates
`int64_t(b) * stride + j * ld + i`, so the batch term does not overflow. It used
to: a 512×512 `float` matrix has `stride == 262144`, and the old all-`int`
product wrapped at `b == 8192`, a batch size this library is built for. What
remains `int` is `j * ld + i`, the offset *within* one batch item, which can only
overflow on a single matrix above 2³¹ elements (8 GB of `float`) — more than the
library can allocate anyway. `KernelMatrixView::operator()` and `::batch_item`,
`Matrix::operator()`, `MatrixView::at` and `::batch_item`, `Vector::at` and
`VectorView::at` are all widened the same way, and the debug asserts that guard
them now compare `int64_t` against `int64_t` rather than an already-wrapped `int`
against a `size_t`.

`MatrixView`'s constructor validates what it safely can.
`MatrixView(data, rows, cols, ld, stride, batch_size)` throws
`std::invalid_argument` on a negative `rows`, `cols`, `ld`, `stride` or
`batch_size`, and on a resolved `ld` smaller than `rows`. That last check is the
one that matters: the batch count is the *third* argument on `Matrix` and the
*sixth* here, so a caller who learned the order from `Matrix A(n, n, batch)`
writes `MatrixView<float> V(p, n, n, batch)`, which means `ld = batch`,
`stride = batch * n`, `batch_size = 1`. That used to be accepted silently and
produced plausible-looking wrong numbers; it now throws, and the message names
the spelling you meant. Two things it deliberately does *not* reject: a null
`data` pointer or a zero dimension, because a shape-only view is the standard
idiom for a workspace-size query; and `stride < ld * cols`, which the owning
`Matrix` constructor does reject but which some existing views rely on.

**`stride = 0` is the packed default, not cuBLAS's broadcast.** BatchLAS has no
broadcast operand: every operand carries a full batch, and unequal batch sizes
throw (`"GEMM: incompatible matrix dimensions"`). Writing
`MatrixView(dA, n, k, n, /*stride=*/0, batch)` over a buffer that holds one
matrix does not repeat that matrix — it reads `batch` consecutive items, i.e.
past the end of the buffer, and the USM check validates the base pointer only,
not the extent. To multiply one shared matrix against many right-hand sides,
either fold the batch into columns — packed `B` (`ld == k`, `stride == k*n`) is
one `k × (n·batch)` matrix, so the product is a single un-batched `gemm` with no
extra memory — or replicate the shared matrix across the batch.

#### Row-major source data

For `gemm`, use the operand swap below — it copies nothing. Otherwise convert:
`Matrix::to_column_major()` returns a converted copy and `to_row_major()` goes
back, both packed (`ld == rows`, `stride == rows * cols`) and both synchronising
before they return — the no-argument form on a queue they build, the
`to_column_major(ctx)` / `to_row_major(ctx)` form on the queue you pass.

**A packed row-major buffer** (row pitch `cols`) is adopted with `ld = 0` and
converted with the default pitch:

```cpp
// row-major, row pitch cols, no padding
Matrix<float> A(Span<const float>(src, size_t(rows) * cols), rows, cols, /*ld=*/0);
auto col_major = A.to_column_major();          // packed source, default pitch
```

`A` here holds the row-major bytes under a column-major label: `A(i, j, b)`
returns `src[j*rows + i]`, not element `(i, j)`. Use only `col_major` — adopt,
convert, use the result.

**A row-major buffer with a padded row pitch `p`** goes into a matrix you own
that is big enough for the padded layout, and converts at the pitch you state:

```cpp
Matrix<float> holder(rows, cols, batch, /*ld=*/rows, /*stride=*/(rows - 1) * p + cols);
// ... copy the padded rows into holder.view().data_ptr() ...
auto col_major = holder.to_column_major(p);    // or to_column_major(ctx, p)
```

`Matrix(rows, cols, batch, ld, stride)` allocates `stride * batch` elements, so
`stride` is what has to cover the row-major extent `(rows-1)*p + cols`.

The conversion rules:

- `to_column_major()` with no pitch means **packed**, row pitch `cols`. It
  requires a packed matrix — `ld() == rows()` and `stride() == rows()*cols()` —
  which is what `to_row_major()` produces, so the round trip needs no
  bookkeeping. On a padded `ld` or a gapped `stride` it throws
  `std::invalid_argument` naming `rows/cols/ld/stride` and both ways out. A
  one-row matrix reads the same at every pitch and is exempt.
- `to_column_major(row_pitch)` reads at the pitch you state, and throws for a
  pitch below `cols`, one whose rows run past the end of the allocation, or one
  that straddles the next batch item.
- `to_row_major()` reads the source column-major with its own `ld` and `stride`.
- The copying constructors are column-major: `(ld, stride)` are the *source's*
  column pitch and batch stride, and the constructor wants
  `(cols-1)*ld + rows` elements per item.

### Where the memory has to live: the USM contract

**Every pointer you hand to `MatrixView` or `Span` must be device-accessible for
the backend the `Queue` dispatches to.** `MatrixView` takes a bare `T*`, so it
cannot check this at construction; the entry points that take their backend from
the queue check every pointer argument at the call and throw
`std::invalid_argument`.

```cpp
std::vector<float> ha(n * n * batch), hb(n * n * batch), hc(n * n * batch);
MatrixView<float> A(ha.data(), n, n, n, n * n, batch);
MatrixView<float> B(hb.data(), n, n, n, n * n, batch);
MatrixView<float> C(hc.data(), n, n, n, n * n, batch);
gemm(ctx, A, B, C, GemmOptions<float>{});        // throws std::invalid_argument
```

```
BatchLAS: gemm: argument 1 points to memory that is not reachable from this
Queue's device (NVIDIA GeForce RTX 4090).
It looks like ordinary host memory -- a std::vector, new[] or malloc.
...
Use memory the device can reach:
  - let Matrix<T, MatrixFormat::Dense> own it (it allocates USM shared, ...
```

The check runs on the spellings that take the backend from the queue. The
`f<Backend, T>(...)` spellings — including the `gesvd`/`ormqr`/`ortho`/`spmm`
calls and the positional workspace-taking spellings below — go straight to the
backend, where a host pointer reaches the vendor call and aborts the process.
Validate those arguments with `Queue::is_device_accessible(ptr)`, which returns
a `bool` and asks exactly what the checked entry points ask.
`BATCHLAS_SKIP_POINTER_CHECKS=1` turns the check off for a hot loop whose
pointers are already validated.

An argument that addresses no elements is exempt, because no kernel can
dereference it. That covers the empty `Span` a sizing pass hands out, and the
default-constructed view that means "this optional matrix is not in use":

```cpp
SyevxParams<float> params;                     // the last argument of every syevx
syevx(ctx, A, W, k, ws, JobType::NoEigenVectors,
      MatrixView<float>(), params);            // fine, not checked
```

Allocations that work zero-copy on a GPU backend:

- `sycl::malloc_device`, `sycl::malloc_shared`, `sycl::malloc_host` — including
  allocations made on your own `sycl::context`, as long as it is the same device;
- `cudaMalloc` and `cudaMallocManaged`.

Allocations that do **not** work on a GPU backend: `malloc`, `new`,
`std::vector`, and anything else backed by ordinary host memory. On a host/CPU
device that memory *is* what the kernels read and nothing is rejected, so run the
check on the device you will ship on.

### Getting host data in

`Matrix` owns USM **shared** memory (`sycl::malloc_shared`, on a per-device
context that outlives any individual `Queue`), so the host can read and write it
directly. Load it in bulk with the copying constructor:

```cpp
std::vector<float> host(size_t(n) * n * batch);   // column-major, packed
fill_from_wherever(host);

Matrix<float> A(Span<const float>(host.data(), host.size()),
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
dimensions, an `ld` that is neither `0` (packed, meaning `rows`) nor at least
`rows`, and a batched `stride` that is neither `0` (packed, meaning `ld * cols`)
nor at least `ld * cols`.

If the data is generated rather than read in, skip the host entirely. These run
on the device and synchronise before returning:

```cpp
auto R = Matrix<float>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/7);
auto I = Matrix<float>::Identity(n, batch);
auto Z = Matrix<float>::Zeros(n, n, batch);
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
`fill_triangular_random` are in `batchlas/blas/matrix.hh`, alongside
`fill_random_sparse_hermitian` for CSR. Most have a `(const Queue&, ...)` form
and a form that builds a queue of its own; `fill_identity`, `fill_tridiag`,
`fill_zeros` and `fill_ones` take a queue and nothing else. The queue-less
`fill_zeros()` / `fill_ones()` forwarders are gone, along with the `= Queue()`
defaults on `set_access_device`, `set_preferred_location` and `prefetch` on
`Matrix`, `MatrixView`, `Vector` and `VectorView`. A default-constructed `Queue`
is not a handle to a shared queue: it builds a fresh `QueueImpl` on
`Device::default_device()`, so those spellings targeted the wrong device on a
multi-GPU box and carried a throwaway workspace arena that nothing could reuse.
The remaining queue-less `fill_*` forwarders have the same caveat — pass your own
queue.

Two more `MatrixView` helpers work on a matrix that already holds one triangle —
the shape `syrk`, `herk`, `syr2k`, `her2k` and the `uplo`-reading factorisations
leave behind:

```cpp
C.view().symmetrize(ctx, Uplo::Lower);   // lower triangle -> upper: C == Cᵀ
C.view().hermitize(ctx, Uplo::Lower);    // ... conjugated: C == Cᴴ
```

Both take a square matrix, run one kernel over the batch, and return an `Event`.
`symmetrize` copies the named triangle across the diagonal, `hermitize` copies
its conjugate. Use them before handing a one-triangle result to anything that
reads both halves.

#### `Random` is deterministic

The `seed` parameter of `Random`, `RandomTriangular`, `fill_random` and
`fill_triangular_random` defaults to **42**, and each element's value is a pure
function of `(seed, index)` — no entropy, no time, no device state. Two
default-seeded calls return **bit-identical** matrices:

```cpp
auto A = Matrix<float>::Random(n, n, false, batch);   // seed 42
auto B = Matrix<float>::Random(n, n, false, batch);   // seed 42 -> A == B
```

Pass distinct seeds when you want distinct operands:

```cpp
auto A = Matrix<float>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/1);
auto B = Matrix<float>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/2);
```

Two more consequences of keying on the index. `Random` keys on the flat index
over the whole allocation, so the batch items of one matrix differ from each
other, but two matrices of the same logical shape with different `ld` or `stride`
get different values at the same `(i, j, b)`. `fill_triangular_random` keys on
the index *within* a matrix, so every batch item of a `RandomTriangular` result is
the same matrix.

#### Copying between matrices

To refresh a dense matrix from another one, copy view to view. It lowers to
`memcpy` or `ext_oneapi_memcpy2d` where the layouts allow it, falls back to a
3-D kernel where they do not, and is asynchronous:

```cpp
MatrixView<float>::copy(ctx, dst.view(), src.view());
```

Element access — `A(i, j, b)` on an owning `Matrix`, `V.at(i, j, b)` on a view —
works from the host because the memory is shared, but it is one indexed store
into managed memory per element. Use it to set or inspect a handful of entries,
and in tests. Do not use it to load a batch.

`MatrixView` never owns. To own an existing allocation, copy it into a `Matrix`.

### Device-resident operands

`Matrix` owns USM **shared** memory, which is what makes the bulk constructor and
every `at()`-style access work; the runtime migrates its pages between host and
device on demand, and a large batch written from the host and then read by many
kernels pays for that traffic.

For device-resident storage, allocate with `sycl::malloc_device` and wrap it in a
`MatrixView`, which stores the pointer and copies nothing:

```cpp
#include <batchlas/sycl_interop.hh>            // this TU needs -fsycl

auto& q = batchlas::sycl_queue(ctx);           // the queue BatchLAS submits on
const size_t elems = size_t(n) * n * batch;

float* dA = sycl::malloc_device<float>(elems, q);
float** pA = sycl::malloc_device<float*>(batch, q);   // the batch pointer array
q.memcpy(dA, host.data(), elems * sizeof(float)).wait();

MatrixView<float> A(dA, n, n, /*ld=*/n, /*stride=*/n * n, batch, /*data_ptrs=*/pA);
```

Two rules make this work:

- **The allocation must be reachable from the `Queue`'s SYCL context** — which
  `sycl_queue(ctx).get_context()` is. `sycl::malloc_device`, `malloc_host`,
  `cudaMalloc` and `cudaMallocManaged` all qualify. See *Interop with CUDA and
  with your own SYCL*.
- **Pass the `data_ptrs` array.** A `MatrixView` built from a raw pointer without
  it has no pointer array, and every batched vendor call that needs one — `potrf`
  at batch > 1, `getrf`, `getri` — throws
  `std::runtime_error("data_ptrs target is null")`. It is a `T**` of length
  `batch_size`, may itself be `malloc_device`, and BatchLAS fills it for you. An
  owning `Matrix` builds it at construction, so this applies to raw-pointer views
  only — including views over shared USM.

`gemm` and `syev` do not use the pointer array and run on a raw-pointer view
without it.

Device memory is not host-addressable, so read results back with `q.memcpy`, and
initialise the view in place with the `fill_*` family — `fill_random`,
`fill_identity`, `fill_zeros`, `fill_diagonal` and the rest all take a
`MatrixView` — rather than an `at()` loop or a `Matrix` factory, which allocates
its own shared memory.

For a device-memory workspace, pass your own `Span<std::byte>` over a
`malloc_device` block to the positional spelling; the queue's arena serves shared
memory.

### Row-major data: the operand swap

A column-major view of a row-major `m x k` buffer with row length `k` is exactly
its transpose: `MatrixView<T>(p, /*rows=*/k, /*cols=*/m, /*ld=*/k)` is `Aᵀ`.
Since `Cᵀ = Bᵀ Aᵀ`, feeding `gemm` the transposed views **in the opposite order**
computes the row-major product with no copy and no transpose flags:

```cpp
// Row-major A (m x k), B (k x n), C (m x n), packed, in USM at pa/pb/pc.
MatrixView<float> At(pa, k, m, k);            // = Aᵀ
MatrixView<float> Bt(pb, n, k, n);            // = Bᵀ
MatrixView<float> Ct(pc, n, m, n);            // = Cᵀ
gemm(ctx, Bt, At, Ct, GemmOptions<float>{});  // C = A B, row-major
```

The swap is `gemm`'s. For the symmetric routines, flip `uplo` — a row-major upper
triangle is a column-major lower one. For everything else, including `potrf`,
`getrf` and `syev`, convert the data first (see *Row-major source data*).

### The CSR non-zero count has its own type

The two owning constructors line up positionally — shape, then the
format-specific extra, then the batch size — and the CSR non-zero count is
spelled with the `NonZeros` strong typedef:

```cpp
Matrix<float> D(rows, cols, batch_size, ld, stride);
Matrix<float, MatrixFormat::CSR> S(rows, cols, NonZeros{nnz}, batch_size);
```

Spell the count `NonZeros{nnz}`; a bare `int` there does not compile. The
from-data constructors follow the same order — buffers, shape, `NonZeros{nnz}`,
strides, batch:

```cpp
Matrix<float> D(data, rows, cols, ld, stride, batch_size);
Matrix<float, MatrixFormat::CSR> S(values, row_offsets, col_indices,
                                   rows, cols, NonZeros{nnz},
                                   matrix_stride, offset_stride, batch_size);
```

`MatrixView` mirrors both.

`NonZeros{}` is a **capacity**, not a count. `nnz()` returns it, and
`convert_to<MatrixFormat::CSR>()` sizes a whole batch by its *largest* item, so on
a heterogeneous batch — which is the normal result of that conversion — `nnz()`
over-counts every smaller item, and `for (int k = 0; k < S.nnz(); ++k)` walks past
that item's row range into slots the conversion never wrote. Three accessors, on
`Matrix`, `MatrixView` and `KernelMatrixView` alike:

- **`nnz()`** — the per-item stride of the batch. Unchanged, and what the vendor
  SpMM descriptors want: `cusparseCreateCsr` and `rocsparse_create_csr_descr` are
  handed one number for the whole strided batch, and the capacity is the correct
  one there.
- **`nnz(b)`** — the non-zeros batch item `b` actually stores, read from that
  item's row offsets. Requires the kernel that filled the offsets to have
  completed. On a `MatrixView` over `sycl::malloc_device` memory the host cannot
  read the offsets, so use the `KernelMatrixView` overload inside a kernel instead.
- **`nnz_capacity()`** — the slots that were allocated per item. Equal to `nnz()`
  for both allocating constructors, but the from-data constructor lets
  `matrix_stride` exceed the declared count, and it is `matrix_stride` that sizes
  the buffers.

## What gets thrown

The split is by *cause*, not by site: `std::invalid_argument` for anything
determined by the caller's arguments — shapes, `ld`, workspace size, pointer
reachability — and `std::runtime_error` only for environment and backend
failures, such as a backend that is not compiled into this build or a vendor
route with no implementation for the requested arguments. Code that used to
catch `std::runtime_error` on a shape mismatch must now catch
`std::invalid_argument`; through the Python bindings the same paths changed from
`RuntimeError` to `ValueError`.

| exception | thrown for |
| --- | --- |
| `std::invalid_argument` | everything the caller controls: a pointer that is not reachable from the queue's device (from the queue-dispatching entry points); shape and batch-size mismatches from the dense BLAS backends (`"GEMM: incompatible matrix dimensions"`, `"SYMM: batch size mismatch (A=…, B=…, C=…)"`) and `trsm`'s shape, `lda` and `ldb` checks; the LAPACK-style shape preconditions (`"getrf: A must be square, got 100x50"`, `"getrs: A.rows() (8) must equal B.rows() (4)"`, `"geqrf: tau holds 4 elements, needs at least 16"`); a workspace or output span too short for the chosen provider; `Matrix`/`MatrixView` construction and slicing — null data, non-positive dimensions, an `ld` that is neither `0` nor at least `rows`, too short a source span, a `to_column_major` row pitch that does not fit or a defaulted one on a matrix that is not packed; shape errors from `gesvd` and the extension routines |
| `std::runtime_error` | environment and backend failures only: a backend that is not compiled into this build; a route with no implementation for the requested arguments (`"BATCHLAS_TRMM_VARIANT=cublasdx only supports float"`, `gesvd`'s vendor route on thin singular vectors); a `Queue` used from a thread other than its owner; a raw-pointer view with no `data_ptrs` array |
| `std::out_of_range` | `V.at(i, j, b)` / `V(i, j, b)` outside the view |

Catch `std::exception` at the boundary; the message names the routine and the
numbers.

Errors that the device reports asynchronously surface at
`ctx.wait_and_throw()`, not at the call that enqueued the work.

The backend that runs the call validates the BLAS-2 and BLAS-3 shapes: `gemm`
checks every batch item, `symm`, `hemm`, `herk`, `her2k`, `syrk`, `syr2k` and
`trmm` check shapes and batch sizes, and `trsm` checks shapes, `lda` and `ldb`.
The queue-dispatching LAPACK-style calls — `potrf`, `getrf`, `getrs`, `getri`,
`geqrf`, `orgqr`, `syev` written without an explicit `<Backend>` — check their
shapes host-side before any device work, and throw `std::invalid_argument` on a
mismatch. `potrf`, `getrf`, `getri` and `syev` require a square `A`; `getrs`
additionally requires `A.rows() == B.rows()` and a matching batch size, and
`getri` the same of `Ainv`. The output spans must be long enough: `pivots` at
least `A.rows() * batch_size`, `tau` at least
`min(A.rows(), A.cols()) * batch_size`, `W` at least `A.rows() * batch_size`.
Oversized spans are fine — the test is `>=`, so slicing one arena across several
calls still works. `geqrf` and `orgqr` deliberately have no squareness check:
rectangular `A` is the point.

This matters most for `getrf`. A rectangular `A` used to reach the vendor call as
written, and the backends did not agree about what that meant — the netlib path
factorised an `A.rows()` × `A.rows()` block and read and wrote past the end of a
tall `A`'s allocation, cuBLAS's batched `getrf` takes one dimension and is
square-only by construction, and rocSOLVER genuinely handled the rectangular
case. Now all three reject it the same way.

The `f<Backend::CUDA>(ctx, …)` spelling is the library's own inner-loop form and
skips these checks by design, so the cost stays off the hot path.

**Factorisation status is opt-in, and only `potrf`, `getrf` and `getri` have
it.** Each takes an optional `Span<int32_t> info`, one entry per batch item, with
LAPACK's convention: `0` means the item factorised, and a positive value names
the leading minor (`potrf`) or the zero pivot (`getrf`, `getri`) at which it
failed. It is a field on `PotrfOptions` and a trailing parameter on
`getrf`/`getri`:

```cpp
UnifiedVector<int32_t> info(batch);
potrf(ctx, A.view(), {.uplo = Uplo::Lower, .info = info.to_span()});
getrf(ctx, A.view(), pivots.to_span(), info.to_span());
ctx.wait();
if (info[37] != 0) { /* item 37 is rank-deficient; everything downstream is noise */ }
```

Leave `info` out — that is the default — and nothing is reported, exactly as
before. The span has to be device-accessible USM; the vendor writes it in place,
so asking for status costs no extra allocation and does not change the workspace
size. A non-empty span shorter than `batch_size` is rejected with
`std::invalid_argument` rather than partially filled, because the failure would
otherwise be silent: the backend falls back to its own scratch and the caller
reads whatever was already in the buffer, most often zeros — "every item
factorised" — on precisely the batch it was trying to diagnose.

`geqrf` and `orgqr` have no equivalent and will not get one. Householder QR has
no numerical failure mode, so LAPACK's `geqrf` info is only ever `0` or an
illegal-argument code; cuBLAS's `geqrfBatched` takes a single *host* scalar
rather than a per-item device array (it spells the per-item form `devInfoArray`,
as on `gelsBatched`); and rocSOLVER's `geqrf` has no info parameter at all. A
per-item `geqrf` info would be all zeros by construction.

`syev`, `gesvd` and the solve-style calls (`getrs`, `linalg::solve`) still report
nothing. A batch item whose pivot is zero to working precision produces numbers
rather than an exception: `linalg::solve` on a near-singular `A` returns a
plausible-looking result and nothing in the table above fires. Where the inputs
are not known to be well-conditioned, check afterwards — compute the residual
`‖A·X − B‖`, or scan the factor's diagonal for zeros and NaNs — and decide per
batch item.

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

The backends to name are `CUDA`, `ROCM`, `MKL` and `NETLIB`, plus `AUTO`.
`Backend::MAGMA` and `Backend::SYCL` are unavailable on every build:
`Queue::backend_available` reports `false`, and naming either in `set_backend`
or in a `Queue` constructor throws `std::runtime_error`.

To check first:

```cpp
if (Queue::backend_available(Backend::CUDA)) ctx.set_backend(Backend::CUDA);
```

This applies to the whole surface, extensions included: `ortho`, `syevx`,
`lanczos`, `steqr`, `stedc`, the `sytrd_*` and `syev_*` family, `cond` and
`cond_buffer_size` all take their backend from the queue. The exception is the
handful of entry points whose remaining template parameters are not deducible
from their arguments, listed under "Which spelling each entry point takes"
above — they have no queue-deducing overload and must name the backend.

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

Use it rather than hardcoding `Backend::CUDA` in code that has to run on more
than one backend.

## Workspaces come from the queue's arena

The LAPACK-style entry points need scratch space. Leaving the workspace argument
out leases it from a per-`Queue` arena, sized by the matching `*_buffer_size`:

```cpp
potrf(ctx, A.view(), {.uplo = Uplo::Lower});   // workspace leased and returned
```

The alternative is to size and own the buffer yourself, and pass it in:

```cpp
with_backend(ctx, [&](auto Back) {
    constexpr Backend Bk = Back.value;
    UnifiedVector<std::byte> ws(potrf_buffer_size<Bk, float>(ctx, A.view(), Uplo::Lower));
    potrf<Bk, float>(ctx, A.view(), Uplo::Lower, ws.to_span());
    ctx.wait();                                // ws must outlive the kernels
});
```

A repeated arena-backed call reuses the same memory rather than malloc/free-ing
device memory each time. The arena never frees on its own: it grows to the peak
it has been asked for and holds it, and `ctx.workspace_capacity()` reports the
current size. To cap it: pass your own span, so capacity stays at 0; destroy the
`Queue`, and the arena goes with it; or call `ctx.trim_workspace()`, which frees
the blocks and drops capacity back to nothing. `trim_workspace()` is `[[nodiscard]]` — it returns `false` and does
nothing while any lease is outstanding — and it drains the queue, so it can
throw.

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

**On an out-of-order queue, pass your own span.** A lease's bytes go to the next
borrower when the call returns. On an in-order queue that borrower's work is
ordered behind this call's, so the handover is free; on an out-of-order queue
nothing orders the two, so the release drains the queue and every arena-backed
call blocks until the device is idle before it returns.

Two rules for leases you hold yourself:

- Call `ws.release()` before reassigning a live lease (`ws = ctx.workspace(...)`),
  or the new loan is taken before the old one is returned and the arena ratchets
  instead of reusing.
- A lease's release orders only against the queue it was taken from. Pass your
  own span when the work runs on a sibling queue built with `Queue(base, in_order)`.

See `batchlas/util/workspace.hh` for the full lifetime rules.

Do not build a workspace out of a local `UnifiedVector` and let it go out of
scope before the kernels using it have run — the memory is freed while the device
may still be reading it. Wait on the queue before it dies, hoist it out of the
call's scope, or use the arena, whose lifetime is tied to the queue.

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
an `Event` across the boundary or allocate device memory, and do not re-export it
from a header of your own. It provides:

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
are reachable from that context. See *Device-resident operands*.

## The `linalg` convenience layer

`batchlas::linalg` (`batchlas/blas/linalg-ops.hh`) offers value-returning
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
linalg::scale(ctx, A.view(), 2.0f);                   // in place
linalg::axpby_into(ctx, 2.0f, A.view(), 3.0f, B.view(), C.view());
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

---

Adding an entry point to BatchLAS rather than calling one? See
[extending.md](extending.md).

# BatchLAS C++ examples

Self-checking programs for the BatchLAS C++ interface. Each one explains a
slice of the API in comments, then computes something and verifies it against a
plain host reference — so a clean run doubles as a smoke test:

```
[ok  ] batched gemm error: 0.000e+00  (tol 1.000e-12)
```

Lines marked `[FAIL]` mean a check did not hold on your machine, and the program
exits non-zero.

| # | Example | What it covers |
|---|---------|----------------|
| 01 | `01_getting_started.cc` | Backends and devices, `Queue`, `Matrix`/`MatrixView`, column-major storage, the batching convention, `alpha`/`beta`, the workspace contract, scalar types, transposes, asynchrony |

## Building them

### As part of the BatchLAS build

```bash
cmake -B build -S . -DBATCHLAS_BUILD_EXAMPLES=ON
cmake --build build -j"$(nproc)"

./build/examples/cpp/01_getting_started
```

With `BATCHLAS_BUILD_TESTS=ON` as well, the examples are also registered with
CTest as `example_01_getting_started`, so `ctest` runs them alongside the unit
tests.

The example picks a GPU backend when the build and the machine both have one,
and falls back to the host (NETLIB) backend otherwise. Pass `cpu` to force the
host path:

```bash
./build/examples/cpp/01_getting_started cpu
```

### Standalone, against an installed BatchLAS

This is the path to copy when starting your own project — `examples/cpp/CMakeLists.txt`
takes this branch when it is not part of the BatchLAS build:

```bash
cmake --install build --prefix /path/to/install     # once

cmake -S examples/cpp -B build-examples \
      -DCMAKE_PREFIX_PATH=/path/to/install \
      -DCMAKE_CXX_COMPILER=<your SYCL compiler>
cmake --build build-examples -j"$(nproc)"
```

The installed package exports `BatchLAS::batchlas`, which carries the include
directories and the component libraries. You still need the same SYCL compiler
the library was built with — the headers instantiate SYCL kernels in your
translation unit.

## Conventions worth knowing

These are the things the Python facade hides and a C++ caller has to know.

- **The backend is a compile-time template parameter.** Every routine is
  `template <Backend B, typename T>`: `gemm<Backend::CUDA>(...)`,
  `syev<Backend::NETLIB>(...)`. There is no `Backend::AUTO` in the C++ API — the
  library instantiates a fixed set (`NETLIB`, and whichever of `CUDA`/`ROCM`/`MKL`
  the build enabled). Query the build with the macros in
  `<batchlas/backend_config.h>` (`BATCHLAS_HAS_CUDA_BACKEND`,
  `BATCHLAS_HAS_HOST_BACKEND`, …) and pick once at the top of your program.
- **The backend and the device have to agree.** A GPU backend needs a GPU
  `Queue`; `NETLIB` needs the host. `Queue ctx("gpu")` / `Queue ctx("cpu")`, or
  construct from a `Device` out of `Device::get_devices(DeviceType::GPU)`.
- **Storage is column-major.** Element `(i, j)` of batch item `b` lives at
  `data[b*stride + j*ld + i]`. `ld` defaults to `rows` and `stride` to
  `ld * cols`. `Matrix::to_row_major()` / `to_column_major()` convert if you
  have data in the other layout.
- **`Matrix<T>` owns, `MatrixView<T>` does not.** `Matrix` allocates unified
  (USM shared) memory, so the host can read and write elements — `A(i, j, b)` —
  without an explicit transfer. Routines take a `MatrixView`; `Matrix` converts
  implicitly. `A.view()`, `view[i]` (one batch item) and `view(Slice{...},
  Slice{...})` all produce views over the same storage, no copy.
- **Batching is intrinsic.** A `Matrix` carries a `batch_size`; one call handles
  the whole batch. Set per-item dimensions with `set_active_dims` for a
  heterogeneous batch.
- **`alpha`/`beta`.** BLAS-style routines compute
  `C <- alpha * op(A) * op(B) + beta * C`. `C` is an input as well as the
  destination — you always supply it.
- **The workspace contract.** Routines never allocate scratch memory. Ask for
  the size, allocate once, reuse:

  ```cpp
  size_t bytes = syev_buffer_size<B>(ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower);
  UnifiedVector<std::byte> workspace(bytes);
  syev<B>(ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower, workspace.to_span());
  ```

  Pass the same arguments to `*_buffer_size` as to the routine — the size
  depends on them. A size of `0` is legal (some host paths need no scratch).
- **Everything is asynchronous.** Routines enqueue work and return an `Event`.
  Call `event.wait()` or `ctx.wait()` before reading results from the host. On an
  in-order `Queue` (the default) consecutive calls are ordered for you, so one
  wait at the end of a chain is enough.
- **Scalar types.** `float`, `double`, `std::complex<float>`,
  `std::complex<double>`, instantiated for every backend.
- **`uplo`.** Symmetric routines read only the nominated triangle. Passing a
  full symmetric matrix is always safe.

## Relation to the Python examples

`python/examples/` covers the same library from Python in twelve notebooks, in
much more breadth. The API surface is the same; what differs is exactly the list
above. If you want to see what a routine does before writing C++ against it, the
notebooks are the faster read — and `python/examples/README.md` also lists the
currently known library-level defects, which apply equally to the C++ API.

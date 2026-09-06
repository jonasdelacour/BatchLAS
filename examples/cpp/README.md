# BatchLAS C++ examples

Twelve short programs covering the BatchLAS C++ interface, one topic each, in
the same order as the `python/examples/` notebooks. They are meant to be read:
each is a sequence of API calls with the contract explained in comments and the
results printed, so you can see the shapes and conventions a routine expects
without unpicking a test harness.

Run one to watch it work; open it to copy the call you need.

## The examples

| # | Example | What it covers |
|---|---------|----------------|
| 01 | `01_getting_started.cc` | Backends and devices, `Queue`, `Matrix`/`MatrixView`, column-major storage, batching, `alpha`/`beta`, the workspace contract, scalar types, transposes, asynchrony |
| 02 | `02_dense_blas.cc` | `gemm`, `gemv`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`, heterogeneous batches, `ComputePrecision` |
| 03 | `03_linear_solvers.cc` | `potrf`, `getrf`/`getrs`, `getri`, `inv`, solving with a Cholesky factor, complex input |
| 04 | `04_qr_and_orthogonalization.cc` | `geqrf`, `orgqr`, `ormqr`, the `ortho` algorithms, orthogonalising against a basis |
| 05 | `05_svd.cc` | `gesvd`, `gesvd_blocked`, `gesvd_cta`, `gebrd_*`, `bdsqr`, `ormbr` |
| 06 | `06_symmetric_eigensolvers.cc` | The whole `syev` family, the parameter structs, Hermitian input, `uplo` |
| 07 | `07_tridiagonal_reduction.cc` | `sytrd_cta`, `sytrd_blocked`, `sytrd_sy2sb`, `sytrd_sb2st`, `hetrd_hb2st`, `sytrd_band_reduction` |
| 08 | `08_tridiagonal_eigensolvers.cc` | `steqr`, `steqr_cta`, `stedc`, `stedc_flat`, `tridiagonal_solver`, `SteqrParams`/`StedcParams` |
| 09 | `09_sparse_and_iterative.cc` | CSR matrices, `spmm`, `syevx` + instrumentation, `ritz_values`, `lanczos`, ILU(k) |
| 10 | `10_jacobi_relative_accuracy.cc` | Why `syev_jacobi_cta` exists: relative accuracy on graded matrices — and when it does *not* help |
| 11 | `11_generators_and_utilities.cc` | Structured constructors, conditioned random generators, `norm`, `cond`, `transpose`, `lascl`, `astype`, `convert_to` |
| 12 | `12_choosing_a_variant.cc` | Batching speed-up, throughput scaling, picking a `syev` variant, workspace reuse |

`example_utils.hh` holds the only shared code: section headings and the
backend/device selection every example needs. It is not part of the library API.
Example 01 spells the backend selection out inline instead of using it, because
choosing a backend is one of the things you have to understand to use the C++
interface.

Setup uses the library's own generators — `Matrix::Random`, `Identity`,
`TriDiagToeplitz`, `RandomTriangular` and friends — so the examples stay short
and exercise more of the API. Where a result is worth checking, they check it
with BatchLAS itself: example 07 confirms a tridiagonal reduction preserved the
spectrum by feeding it to `stedc`, and example 08 prints the closed-form
eigenvalues of a Toeplitz matrix next to what the solvers found.

## Building them

### As part of the BatchLAS build

```bash
cmake -B build -S . -DBATCHLAS_BUILD_EXAMPLES=ON
cmake --build build -j"$(nproc)"

./build/examples/cpp/01_getting_started
```

With `BATCHLAS_BUILD_TESTS=ON` as well, each example is registered with CTest as
`example_<name>`, so a broken build or a crashing routine shows up in `ctest`.

Every example picks a GPU backend when the build and the machine both have one,
and falls back to the host (NETLIB) backend otherwise. Pass `cpu` to force the
host path:

```bash
./build/examples/cpp/06_symmetric_eigensolvers cpu
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
  `BATCHLAS_HAS_HOST_BACKEND`, …) and pick once at the top of your program. The
  `*_cta` routines are instantiated for GPU backends only, so calling one with
  `Backend::NETLIB` is a *link* error — guard with `if constexpr`, not a runtime
  check.
- **The backend and the device have to agree.** A GPU backend needs a GPU
  `Queue`; `NETLIB` needs the host. `Queue ctx("gpu")` / `Queue ctx("cpu")`, or
  construct from a `Device` out of `Device::get_devices(DeviceType::GPU)`.
- **Storage is column-major.** Element `(i, j)` of batch item `b` lives at
  `data[b*stride + j*ld + i]`. `ld` defaults to `rows` and `stride` to
  `ld * cols`. `Matrix::to_row_major()` / `to_column_major()` convert.
- **`Matrix<T>` owns, `MatrixView<T>` does not.** `Matrix` allocates unified
  (USM shared) memory, so the host can read and write elements — `A(i, j, b)` —
  without an explicit transfer. Routines take a `MatrixView`; `Matrix` converts
  implicitly. `A.view()`, `view[i]` (one batch item) and `view(Slice{...},
  Slice{...})` all produce views over the same storage, no copy.
- **Batching is intrinsic.** A `Matrix` carries a `batch_size`; one call handles
  the whole batch. For a *heterogeneous* batch, allocate at the maximum shape
  and declare per-item dimensions with `set_active_dims`; `A.rows()` is then the
  capacity and `A.rows(b)` the active count. See example 02.
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

  Pass the same arguments — including any parameter struct — to `*_buffer_size`
  as to the routine; the size depends on them. A size of `0` is legal.
- **Everything is asynchronous.** Routines enqueue work and return an `Event`.
  Call `event.wait()` or `ctx.wait()` before reading results from the host. On an
  in-order `Queue` (the default) consecutive calls are ordered for you.
- **Solvers overwrite their input.** `syev` and friends replace `A` with the
  eigenvectors; `getrf` replaces it with the factors; `trsm` overwrites its
  right-hand side. Clone first if you need the original — and note that a
  benchmark loop that calls a solver repeatedly on one buffer is re-solving its
  own output after the first iteration.
- **Eigenvalues and singular values are real.** For complex `T` the value span is
  `Span<float_t<T>>`, not `Span<T>`.
- **Scalar types.** `float`, `double`, `std::complex<float>`,
  `std::complex<double>`, instantiated for every backend.
- **Parameter structs.** Tuning knobs are plain structs with defaulted members —
  `SteqrParams`, `StedcParams`, `JacobiParams`, `SyevxParams`, `LanczosParams`,
  `ILUKParams`, `SytrdBandReductionParams` — passed by value.
- **Tridiagonal input.** `(d, e)` with `d` of length `n` and `e` of length
  `n - 1`. Several routines validate this and throw on a mismatch.
- **Band storage.** `(kd + 1) x n`, lower LAPACK convention: `AB(i - j, j)`
  holds `A(i, j)`. `examples::band_to_dense` expands it.

## Known issues visible in these examples

These are library-level problems, not mistakes in the examples. Each was found
while writing the example named, which works around it and says so in a comment
at the call site.

- **`syev_two_stage` with `JobType::NoEigenVectors`** returns wrong eigenvalues
  for `n >= 32`, silently. `n = 16` is fine, and `JobType::EigenVectors` is
  correct at every size tested. Ask for vectors and discard them. (06, 12)
- **`stedc` / `stedc_flat` with `JobType::NoEigenVectors`** throws
  `Invalid slice dimensions` — it slices an eigenvector output it was told not
  to produce. Same workaround. (08)
- **`stedc_flat` eigenvectors on CUDA.** Eigenvalues are correct and the columns
  are orthonormal, but they do not satisfy `T V = V diag(w)` — the residual is
  order 1. An orthogonality check passes, so only a residual check exposes it.
  The host backend is unaffected. (08)
- **`tridiagonal_solver` accuracy.** Its QR iteration does not converge
  reliably; the error on a matrix with an exactly known spectrum is around
  `1e-2`. Prefer `steqr` or `stedc`. (08)
- **`uplo = Uplo::Upper` with a half-filled matrix on CUDA.** `syev` is correct
  for a full symmetric matrix and for `Uplo::Lower` with only the lower triangle
  filled, but not for `Uplo::Upper` with only the upper triangle filled. (06)
- **`gesvd`'s `Uplo` overload on NETLIB** ignores the triangle hint and reads
  the whole matrix, so a half-filled symmetric input gives wrong singular
  values. Correct on CUDA. Passing the full symmetric matrix is safe
  everywhere. (05)
- **`ortho(Transpose::Trans)` on NETLIB** is wrong for wide input (`k < m`):
  LAPACK reports an illegal `DORGQR` argument and the result is not orthonormal.
  Square input is fine. The host backend also appears to route every
  `OrthoAlgorithm` through the same QR-based path, so the algorithm choice has
  no effect there. (04)
- **`ortho(algorithm = Householder)` on CUDA** returns a non-orthonormal result
  if an earlier call in the same process consumed a `geqrf` workspace;
  `ortho_buffer_size` sizes its sub-workspaces from a placeholder view rather
  than the real `A` (`src/extensions/ortho.cc`). The examples use the other
  algorithms. (04)
- **`Matrix<T, CSR>::RandomSparseHermitian`** (and
  `csr_generators::random_sparse_hermitian_csr`) is not symmetric at moderate
  sizes — `|A - A^T|` is order 1 at `n = 64, density = 0.1` — and can emit
  duplicate column indices within a row. Example 09 builds its CSR matrices by
  hand instead. (09)
- **`getrf`/`getri` on a single matrix.** `Matrix` allocates its array of
  per-batch-item base pointers only when `batch_size > 1`, so routines whose
  vendor path takes a pointer array throw `data_ptrs target is null` for
  `batch_size == 1`. Build the `MatrixView` yourself with an explicit
  `UnifiedVector<T*>`. Note also that `Matrix::data_ptrs(Queue&)` cannot fix
  this: it delegates to a temporary view and throws the same way. (03)
- **`getrf` pivot values are not portable.** The pivots land in a
  `Span<int64_t>`, but the netlib path widens LAPACK's `int32` to `int64` while
  cuBLAS writes `int32` straight into the buffer. Passing the span back to
  `getrs` is always fine; *reading* the values needs to know the backend. (03)
- **`lanczos` with `JobType::NoEigenVectors`** and a default-constructed `V`
  throws `sort: invalid batch layout for eigenvalues span`, because the internal
  sort reads `V.batch_size()`. Pass a real `V`, or set
  `LanczosParams::sort_enabled = false`. (09)
- **`steqr_buffer_size` under-reports on NETLIB** for the larger blocked
  settings (`block_size = 32` with `block_rotations = true`): the routine then
  throws `Attempted to allocate ... from a BumpAllocator with only ... bytes
  remaining`. The buffer-size query and the routine disagree, which is the one
  thing the workspace contract has to get right. (08)
- **Values-only modes on the SYCL native-CPU device.** `steqr` with
  `JobType::NoEigenVectors` fails with `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`,
  and `syevx` with `JobType::NoEigenVectors` faults with
  `UR_RESULT_ERROR_INVALID_NULL_POINTER`. Both work on CUDA. Ask for vectors
  and ignore them when targeting the host. (08, 09)
- **`lanczos` on NETLIB** does not converge: on a matrix whose spectrum is
  known, the extreme Ritz values come back several units away. Correct on
  CUDA. (09)
- **`cond` cannot be used with an explicit workspace.** `blas/extra.hh` declares
  a non-template `cond_buffer_size` taking a `MatrixView<float>` that has no
  definition (a link error), while the real one is a template the header never
  declares. Use the allocating overload. (11)

## Device requirements

The `*_cta` routines map one work-group onto one matrix and need a sub-group
width of 32, so they are GPU-only and limited to `n <= 32`. `syev_blocked`,
`syev_two_stage` and the `sytrd_*` reduction stages are also GPU-only, and
accept `Uplo::Lower` only. The examples check the device and print a
`(skipped)` line rather than failing.

Timings in example 12 come from a run on an RTX 4090 and will differ on your
hardware.

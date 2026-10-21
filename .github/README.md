<div align="center">
  <img src="https://raw.githubusercontent.com/jonasdelacour/BatchLAS/main/BatchLAS_logo_transparent.png" alt="BatchLAS Logo" width="200">
</div>

# BatchLAS

Batched dense and sparse linear algebra in SYCL. Every operation takes a batch of matrices and
picks, per shape, between BatchLAS's own kernels and the vendor library (cuBLAS, cuSOLVER,
cuSPARSE, host LAPACK), using per-GPU tuned tables.

**Documentation: <https://jonasdelacour.github.io/BatchLAS/>** (C++ API, guides, design notes and
every measurement behind a routing decision).

## Status

- **Research code, no release tag.** Pin to a commit. `main` carries known test failures, tracked
  in `tests/known-failures.txt` ([CI](../docs/ci.md)).
- **Scalar types**: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
- **SYCL implementations**: DPC++ (`intel/llvm` built with CUDA) and AdaptiveCpp 25.10
  (`-DBATCHLAS_SYCL_IMPL=ACPP`) from one tree
  ([two SYCL implementations](../docs/design/sycl-implementations.md)).

| Backend | Runs on | Status |
| --- | --- | --- |
| `CUDA` | NVIDIA GPUs: native SYCL kernels plus cuBLAS/cuSOLVER/cuSPARSE | RTX 4090 (`sm_89`) in daily CI; RTX PRO 6000 (`sm_120`) built and tested |
| `SYCL` | a GPU without a vendor library: native kernels only | Apple M5 Max through AdaptiveCpp Metal; `float` and `complex<float>` only; not in CI |
| `NETLIB` | host LAPACKE/CBLAS plus native kernels on a CPU device | built and tested beside the GPU backends |
| `ROCM` | AMD GPUs: rocBLAS/rocSOLVER/rocSPARSE | code paths exist; syntax-checked only, not built or run |

A vendor-free build (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) runs native kernels only; its gaps are in
[vendor-free status](../docs/design/vendor-free-status.md).

## Supported operations

| Area | Operations |
| --- | --- |
| BLAS | `gemm`, `gemv`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`, `hemm`, `herk`, `her2k` |
| Factorizations and solvers | `potrf`, `posv`, `getrf`, `getrs`, `getri`, `gesv`, `inv`, `geqrf`, `orgqr`, `ormqr` |
| Eigenvalues | `syev`, `syevx` (selected range: direct, filtered, LOBPCG), `lanczos`, `ritz_values` |
| SVD | `gesvd` (one-sided Jacobi and blocked), `gebrd`, `bdsqr`, `bdsdc`, `ormbr` |
| Tridiagonal and band reduction | `sytrd`, `steqr`, `stedc`, `stebz`, `stein`, `tridiagonal_solver` |
| Sparse | `spmm` (CSR), `iluk` |
| Utilities | `ortho`, `norm`, `cond`, `transpose`, `lascl`, matrix generators; the `linalg::` layer (`matmul`, `solve`, `eigh`, `svd`, elementwise ops) |
| Python | optional pybind11 package with NumPy/SciPy interop (`-DBATCHLAS_BUILD_PYTHON=ON`) |

Which kernel runs for each op, and where the vendor still wins, is on the
[performance pages](../docs/perf/README.md).

## Install

Requirements: CMake 3.17+, a SYCL 2020 compiler **with a backend for your GPU** (for NVIDIA, a
DPC++ built with `--cuda` or oneAPI plus the Codeplay NVIDIA plugin; stock `icpx` silently builds
CPU-only), and optionally LAPACKE/CBLAS and the CUDA toolkit.

```bash
git clone https://github.com/jonasdelacour/BatchLAS.git && cd BatchLAS
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++   # your SYCL compiler
cmake --build build -j8                                                 # heavy TUs: keep -j low
cmake --install build --prefix "$HOME/inst"
```

Configure output must name your GPU (`Using SYCL targets: nvidia_gpu_sm_89`); `spir64_x86_64`
alone means a CPU-only build. Consume it from CMake **with the same compiler**:

```cmake
find_package(BatchLAS CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE BatchLAS::batchlas)
```

```cpp
#include <batchlas.hh>
using namespace batchlas;

Queue ctx(Device::default_device());                 // backend comes from the device
Matrix<float> A(n, n, batch), B(n, n, batch), C(n, n, batch), S(n, n, batch);
A.fill_random(ctx, /*hermitian=*/false, /*seed=*/1);
B.fill_random(ctx, /*hermitian=*/false, /*seed=*/2);
S.fill_random_spd(ctx);

gemm(ctx, A, B, C, {.alpha = 2.0f});                 // C := 2 A B
potrf(ctx, S, PotrfOptions{.uplo = Uplo::Lower});    // S := its Cholesky factor
ctx.wait();                                          // nothing is readable before this
```

Matrices are column-major in device-accessible (USM) memory.

- Toolchain, presets, CMake options, tests, benchmarks, Python, macOS:
  [building guide](../docs/guide/build.md).
- The API contract and the consumer recipe: [C++ API](../docs/cpp-api.md); a runnable outside
  project: [examples/consumer/](../examples/consumer/README.md).

## Tune

Each op chooses its kernel from `tuned/<op>.<device>.txt`. A GPU without tables borrows
the nearest device's and warns once, so retune on new hardware:

```bash
cmake --preset benchmarks && cmake --build --preset benchmarks --target batchlas_tune
B=build/presets/benchmarks/tools/tune/batchlas_tune
$B --list                                                   # ops, candidates, grids
$B all --tier preview --dtype float,double --devices 0 --plan   # cells and time estimate, no GPU
$B all --tier coarse  --dtype float,double --devices 0 --out tuned
```

- `BATCHLAS_TUNED_DIR=<dir>` loads tables from another directory without a rebuild;
  `BATCHLAS_SELECT_TRACE=1` prints every choice and its runner-up.
- `BATCHLAS_<OP>_ROUTE=<choice>` pins one op's kernel (e.g. `BATCHLAS_GEMM_ROUTE=vendor`).
- Tiers, ledger and flags: [batchlas_tune](../tools/tune/README.md); table format:
  [tuned/](../tuned/README.md); the n-dependent constants in `tuning_params.hh`:
  [tuning constants](../docs/perf/tuning.md).

## License

MIT. See `LICENSE`.

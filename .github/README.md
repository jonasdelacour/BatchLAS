<div align="center">
  <img src="https://raw.githubusercontent.com/jonasdelacour/BatchLAS/main/BatchLAS_logo_transparent.png" alt="BatchLAS Logo" width="200">
</div>

# BatchLAS

Batched dense and sparse linear algebra in SYCL. Every operation takes a batch of matrices and
picks, per shape, between BatchLAS's own kernels and the vendor library (cuBLAS, cuSOLVER,
cuSPARSE), using per-GPU tuned tables.

**Documentation: <https://jonasdelacour.github.io/BatchLAS/>** (C++ API, guides, design notes and
every measurement behind a routing decision).

## Status

- **Research code, no release tag.** Pin to a commit. `main` carries known test failures, tracked
  in `tests/known-failures.txt` ([CI](../docs/ci.md)).
- **Tested on NVIDIA only**: RTX 4090 (`sm_89`) daily in CI, RTX PRO 6000 (`sm_120`), with a
  CUDA-enabled DPC++ and CUDA 13.2. ROCm code paths exist but are not built or run; CPU-only builds
  configure and build but are not exercised.
- **Scalar types**: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
- **Backends**: CUDA GPUs (native SYCL kernels plus cuBLAS/cuSOLVER/cuSPARSE) and a host netlib
  BLAS/LAPACK backend. A vendor-free build (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) runs native kernels
  only; its gaps are listed in [vendor-free status](../docs/design/vendor-free-status.md).

## Supported operations

| Area | Operations |
| --- | --- |
| BLAS | `gemm`, `gemv`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`, `hemm`, `herk`, `her2k` |
| Factorizations and solvers | `potrf`, `posv`, `getrf`, `getrs`, `getri`, `gesv`, `geqrf`, `orgqr`, `ormqr` |
| Eigenvalues and SVD | `syev`, `syevx` (partial; direct, filtered and LOBPCG), `gesvd`, `lanczos`, `ritz_values` |
| Tridiagonal and reductions | `sytrd`, `steqr`, `stedc`, `stebz`, `stein`, `tridiagonal_solver`, `gebrd` |
| Sparse | `spmm` (CSR), `iluk` |
| Utilities | `ortho`, `norm`, `cond`, `transpose`, matrix generators; the `linalg::` layer (`solve`, `eigh`, elementwise ops) |
| Python | optional pybind11 package with NumPy/SciPy interop (`-DBATCHLAS_BUILD_PYTHON=ON`) |

Which kernel runs for each op, and where the vendor still wins, is on the
[performance pages](../docs/perf/README.md).

## Install

Requirements: CMake 3.14+, a SYCL 2020 compiler **with a backend for your GPU** (for NVIDIA, a
DPC++ built with `--cuda` or oneAPI plus the Codeplay NVIDIA plugin; stock `icpx` silently builds
CPU-only), oneDPL headers, and optionally LAPACKE/CBLAS and the CUDA toolkit.

```bash
git clone https://github.com/jonasdelacour/BatchLAS.git && cd BatchLAS
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++   # your SYCL compiler
cmake --build build -j"$(nproc)"
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

Queue ctx(Device::default_device());               // backend comes from the device
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, A.view(), {.uplo = Uplo::Lower});
ctx.wait();                                        // nothing is readable before this
```

Matrices are column-major in device-accessible (USM) memory.

- Toolchain, presets, CMake options, tests, benchmarks, Python: [building guide](../docs/guide/build.md).
- The API contract and the consumer recipe: [C++ API](../docs/cpp-api.md); a runnable outside
  project: [examples/consumer/](../examples/consumer/README.md).

## Tune

Each op chooses its kernel from `tuned/<op>.<dtype>.<device>.txt`. A GPU without tables borrows
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

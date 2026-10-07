# BatchLAS documentation {#mainpage}

BatchLAS is a SYCL-first library of **batched** dense and sparse linear algebra:
thousands of small-to-medium problems solved in one call, with native SYCL
kernels competing against vendor libraries (cuBLAS/cuSOLVER/cuSPARSE, rocBLAS,
netlib) through measured routing windows.

This site is both the reference manual and the project's **record of evidence
and decisions**. Code comments carry invariants and a pointer
`evidence: docs/<page>.md#<section>`; the measurement, the rejected
alternatives and the reasoning live on the page it points to.

| If you want to… | Go to |
| --- | --- |
| call BatchLAS from C++ or Python | @ref guide_index |
| look up a function, type or option | @ref api "API reference" (grouped by operation family) |
| know why the code is shaped the way it is | @ref architecture_index |
| read the mathematics behind an algorithm | @ref algorithms_index |
| know which kernel runs for a shape, and the numbers behind that choice | @ref perf_evidence |
| find a raw measurement grid, or what depends on a piece of evidence | @ref results_index |
| build, test, benchmark, tune or document | @ref developer_index |

Source, issues and releases: [github.com/jonasdelacour/BatchLAS](https://github.com/jonasdelacour/BatchLAS).

## What it covers

- **Batched BLAS:** gemm, gemv, symm, syrk, syr2k, trmm, trsm, and Hermitian hemm, herk, her2k.
- **Factorizations and solves:** Cholesky (potrf, posv), LU (getrf, getrs, getri, gesv), QR (geqrf, orgqr, ormqr).
- **Eigenvalues and SVD:** symmetric syev, selected-range syevx, gesvd (one-sided Jacobi and blocked).
- **Sparse and spectral:** spmm, lanczos, tridiagonal steqr and stedc, ritz_values, ILU(k).
- **Utilities:** ortho with several algorithms, matrix generators, norms, condition numbers.
- **Python:** a pybind11 package with NumPy and SciPy interop (@ref python_examples).

## Performance against cuSOLVER

Each call is routed per op, size and batch between BatchLAS's own kernels and
the vendor library (@ref selection_tables), so where cuSOLVER is faster for a
shape, that is what runs. Two committed single-machine measurements (float,
RTX 4090, CUDA backend, large batch):

- **syev with eigenvectors:** 67.8 µs vs cuSOLVER's 203.0 µs per matrix at
  n = 320, batch 819 (3.0×); 2.1× at n = 448, batch 585. Conditions:
  [the headline measurement](../perf/syev.md#syev-the-blocked-over-cusolver-headline-measurement).
- **gesvd vs `cusolverDnXgesvdjBatched`:** 0.0064 vs 0.339 µs per matrix at
  n = 8, batch 16384; 0.468 vs 0.768 µs at n = 32. Conditions:
  [the headline comparison](../perf/gesvd.md#gesvd-readme-headline-jacobi-vs-gesvdjbatched-per-matrix).

## A minimal call

```cpp
#include <batchlas.hh>
using namespace batchlas;

Queue ctx(Device::default_device());          // the backend comes from the device
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, S.view(), PotrfOptions{.uplo = Uplo::Upper});
ctx.wait();                                   // nothing is readable before this
```

Every entry point enqueues and returns an Event; matrices are column-major and
must live in device-accessible (USM) memory; one Queue per thread. The full
contract is in @ref md_docs_2cpp-api.

@subpage guide_index
@subpage architecture_index
@subpage algorithms_index
@subpage perf_evidence
@subpage results_index
@subpage developer_index

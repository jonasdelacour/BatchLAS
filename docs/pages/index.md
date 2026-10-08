# BatchLAS documentation {#mainpage}

\htmlonly
<div class="bl-hero"><p>Batched dense and sparse linear algebra for SYCL: thousands of small-to-medium problems in one call. Each call is routed, per op, size and batch, between native SYCL kernels and the vendor library (cuBLAS, cuSOLVER, cuSPARSE, rocBLAS, netlib), whichever measured faster.</p></div>
<div class="bl-cards">
<a class="bl-card" href="guide_index.html"><p class="bl-card-title">User guide</p><p>Build, install, and call BatchLAS from C++ or Python.</p><span class="bl-card-link">Get started &rarr;</span></a>
<a class="bl-card" href="group__api__reference.html"><p class="bl-card-title">API reference</p><p>Every function, type and option, grouped by operation family.</p><span class="bl-card-link">Browse the API &rarr;</span></a>
<a class="bl-card" href="architecture_index.html"><p class="bl-card-title">Architecture</p><p>Matrix model, kernel selection, workspaces, errors and the environment.</p><span class="bl-card-link">Read the design &rarr;</span></a>
<a class="bl-card" href="algorithms_index.html"><p class="bl-card-title">Algorithms</p><p>The mathematics behind the solvers.</p><span class="bl-card-link">See the algorithms &rarr;</span></a>
<a class="bl-card" href="perf_evidence.html"><p class="bl-card-title">Performance</p><p>Which kernel runs for a shape, and the measurements behind that choice.</p><span class="bl-card-link">See the evidence &rarr;</span></a>
<a class="bl-card" href="developer_index.html"><p class="bl-card-title">Developer guide</p><p>Build, test, benchmark, tune and document the library.</p><span class="bl-card-link">Contribute &rarr;</span></a>
</div>
\endhtmlonly

## A minimal call

```cpp
#include <batchlas.hh>
using namespace batchlas;

Queue ctx(Device::default_device());          // the backend comes from the device
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, S.view(), PotrfOptions{.uplo = Uplo::Upper});
ctx.wait();                                   // nothing is readable before this
```

Every entry point enqueues and returns an Event. Matrices are column-major in device-accessible
(USM) memory. Use one Queue per thread. The full contract is in @ref md_docs_2cpp-api.

## Operations

| Family | Operations |
| --- | --- |
| Batched BLAS | gemm, gemv, symm, syrk, syr2k, trmm, trsm, hemm, herk, her2k |
| Factorizations and solves | potrf, posv, getrf, getrs, getri, gesv, geqrf, orgqr, ormqr |
| Eigenvalues and SVD | syev, syevx (selected range), gesvd (one-sided Jacobi and blocked) |
| Sparse and spectral | spmm, lanczos, steqr, stedc, ritz_values, ILU(k) |
| Utilities | ortho, matrix generators, norms, condition numbers |
| Python | pybind11 package with NumPy and SciPy interop (@ref python_examples) |

## Performance against cuSOLVER

Float, RTX 4090, CUDA backend, large batch.

| Op | BatchLAS | cuSOLVER | Speed-up | Conditions |
| --- | --- | --- | --- | --- |
| syev with eigenvectors, n = 320, batch 819 | 67.8 µs/matrix | 203.0 µs/matrix | 3.0× | [measurement](../perf/syev.md#syev-the-blocked-over-cusolver-headline-measurement) |
| syev with eigenvectors, n = 448, batch 585 | (ratio only) | (ratio only) | 2.1× | same |
| gesvd vs `cusolverDnXgesvdjBatched`, n = 8, batch 16384 | 0.0064 µs/matrix | 0.339 µs/matrix | 53× | [comparison](../perf/gesvd.md#gesvd-readme-headline-jacobi-vs-gesvdjbatched-per-matrix) |
| gesvd vs `cusolverDnXgesvdjBatched`, n = 32 | 0.468 µs/matrix | 0.768 µs/matrix | 1.6× | same |

Where the vendor library is faster for a shape, it runs instead (@ref selection_tables).

Source, issues and releases: [github.com/jonasdelacour/BatchLAS](https://github.com/jonasdelacour/BatchLAS).

@subpage guide_index

@subpage architecture_index

@subpage algorithms_index

@subpage perf_evidence

@subpage results_index

@subpage developer_index

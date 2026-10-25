# Ritz values (Rayleigh quotients) {#guide_ritz_values}

> **Status:** current · signatures checked against `include/batchlas/blas/extensions.hh` and
> `src/ops/ritz_values/ritz_values.cc` on 2026-10-21.

`ritz_values` returns the Rayleigh quotient of every trial-vector column, for dense or CSR
operators and batched inputs.

## What it computes

For square \f$A\f$ (dense or CSR) and a dense block \f$V = [v_1, \dots, v_k]\f$, per column and
batch item:

\f[
  \theta_j \;=\; \operatorname{Re}\,\frac{v_j^H A\, v_j}{v_j^H v_j}, \qquad j = 1, \dots, k .
\f]

- \f$v_j^H = v_j^T\f$ for real types. The quotient is formed in the full scalar type and only the
  real part is stored, so the output is `float` for `std::complex<float>` and `double` for
  `std::complex<double>`.
- For Hermitian \f$A\f$ and \f$v_j\f$ near an eigenvector, \f$\theta_j\f$ is second-order accurate.
  Python exposes it as `batchlas.ritz_values(a, vectors)`.
- Columns are independent: the result is \f$\operatorname{diag}(V^H A V)\f$ divided by
  \f$\operatorname{diag}(V^H V)\f$, not the eigenvalues of \f$V^H A V\f$. Columns need not be
  normalised or orthogonal.

> **Note:** For a Rayleigh-Ritz projection, form \f$V^H A V\f$ with `gemm` and call `syev` on it.

## How it is computed

1. \f$W = A V\f$ into workspace: `gemm` for dense \f$A\f$, `spmm` for CSR (its workspace is carved
   from the same buffer).
2. One kernel, one work-group per (batch item, column) pair, grid-strided. Each work-group reduces
   \f$v_j^H w_j\f$ and \f$v_j^H v_j\f$ through local memory; work-item 0 writes
   \f$\operatorname{Re}(\text{num}/\text{den})\f$.

## Calling it

### Allocating overload

Allocates the output and workspace itself and blocks (`ctx.wait()` before and after). Keep it out
of loops.

```cpp
#include <batchlas.hh>
using namespace batchlas;

Queue ctx(Device::default_device());
Matrix<float> A = /* n x n, batch items */;
Matrix<float> V = /* n x k trial vectors, same batch */;

// Explicit backend and scalar type; MFormat is deduced from A.
Vector<float> theta = ritz_values<Backend::CUDA, float>(ctx, A, V);
float theta_j_b = theta(j, b);   // column j of batch item b
```

It is declared for `MatrixView`; an owning `Matrix` binds to it as its base, so
`ritz_values<B, T>(ctx, A, V)` takes either.

### Workspace form

```cpp
Vector<float> theta(k, batch);   // one value per column per batch item

size_t bytes = ritz_values_buffer_size(ctx, A, V, theta);
UnifiedVector<std::byte> ws(bytes);
ritz_values(ctx, A, V, theta, ws);   // backend taken from the queue
ctx.wait();                          // theta is not readable before this
```

```cpp
template <Backend B, typename T, MatrixFormat MFormat>
Event ritz_values(Queue& ctx,
                  const MatrixView<T, MFormat>& A,
                  const MatrixView<T, MatrixFormat::Dense>& V,
                  const VectorView<typename base_type<T>::type>& ritz_vals,
                  Span<std::byte> workspace);

template <Backend B, typename T, MatrixFormat MFormat>
size_t ritz_values_buffer_size(Queue& ctx,
                               const MatrixView<T, MFormat>& A,
                               const MatrixView<T, MatrixFormat::Dense>& V,
                               const VectorView<typename base_type<T>::type>& ritz_vals);
```

Both are also reachable without the backend (`BATCHLAS_DISPATCH_ON_QUEUE`) and take owning
`Matrix` arguments directly. For sparse \f$A\f$ pass a CSR matrix, e.g.
`auto As = A.convert_to<MatrixFormat::CSR>();`.

## Parameters and layout

| Parameter | Meaning |
| --- | --- |
| `A` | \f$n \times n\f$ per batch item, `Dense` or `CSR`; must be square |
| `V` | dense \f$n \times k\f$ per batch item, same batch count as `A`; columns are the trial vectors |
| `ritz_vals` | real `VectorView`, size \f$\ge k\f$, same batch count; entry \f$j\f$ of item \f$b\f$ is `ritz_vals(j, b)` |
| `%workspace` | at least `ritz_values_buffer_size(...)` bytes: \f$A V\f$ (\f$n k \cdot \text{batch}\f$ scalars), a batch pointer array, and the `spmm` workspace for CSR |

Shape mismatches fail an `assert`, so a release build does not diagnose them.

## Supported configurations

- Scalar types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
- Formats of `A`: `Dense`, `CSR`.
- Backends: `NETLIB`, `CUDA`, `ROCM` and `SYCL`, each only when compiled in; `SYCL` instantiates
  `float` and `complex<float>` only, unless the build has device fp64. There is no `Backend::MKL`
  instantiation.

## Tests

`tests/ritz_values_tests.cc`, over every compiled backend and scalar type:

| Case | Operator | Checks |
| --- | --- | --- |
| `DiagonalMatrix` | dense \f$5 \times 5\f$ diagonal, identity trial vectors, batch 1 | the diagonal |
| `TridiagonalMatrix` | dense \f$10 \times 10\f$ tridiagonal Toeplitz (\f$a = 2, b = c = -1\f$), batch 2 | \f$\lambda_j = a - 2\sqrt{bc}\cos\!\big(j\pi/(n+1)\big)\f$ |
| `SparseExactWorkspaceInPoisonedArena` | CSR \f$37 \times 37\f$ Toeplitz, \f$k = 5\f$ eigenvectors, batch 3 | the analytic values, in a workspace of exactly `ritz_values_buffer_size` bytes filled with `0x7f` |

`lanczos` and `syevx_lobpcg.cc` compute their Ritz values internally; the in-tree callers of
`ritz_values` are its tests and the Python binding. Not implemented: an optional residual norm
\f$\lVert A v - \theta v \rVert\f$, skipping the denominator for orthonormal \f$V\f$, and blocked
evaluation for very large \f$k\f$.

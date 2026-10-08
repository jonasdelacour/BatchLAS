# Ritz values (Rayleigh quotients) {#guide_ritz_values}

> **Status:** current · signatures verified against
> `include/batchlas/blas/extensions.hh` (lines 1621-1672) and `src/extensions/ritz_values.cc`
> on 2026-09-30.

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
  `stedc_tests` uses it to recompute eigenvalues from returned eigenvectors; Python exposes it as
  `batchlas.ritz_values(a, vectors)`.
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

It is declared for `MatrixView`, plus one owning-`Matrix` twin so that the partially-explicit
`ritz_values<B, T>(ctx, A, V)` compiles (`extensions.hh:1646-1664`).

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

Both are also reachable without the backend (`BATCHLAS_DISPATCH_ON_QUEUE`) and with owning
arguments (`BATCHLAS_ACCEPT_OWNING`). `ritz_values_workspace` is a deprecated alias of
`ritz_values_buffer_size`. For sparse \f$A\f$ pass a CSR matrix, e.g.
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
- Backends: `NETLIB`, `CUDA`, `ROCM`, `MKL`, each only when compiled in
  (`src/extensions/ritz_values.cc:202-216`). There is no `Backend::SYCL` instantiation.

## Tests

`tests/ritz_values_tests.cc` checks a diagonal matrix with identity trial vectors and a
\f$10 \times 10\f$ tridiagonal Toeplitz matrix (\f$a = 2, b = c = -1\f$) at batch 2 against
\f$\lambda_j = a - 2\sqrt{bc}\cos\!\big(j\pi/(n+1)\big)\f$, over every compiled backend and scalar
type. Both are dense: the CSR path has no test (open debt; `spmm_tests` covers `spmm` only).

`lanczos` and `syevx_lobpcg.cc` compute their Ritz values internally; the in-tree callers of
`ritz_values` are the tests and the Python binding. Not implemented: an optional residual norm
\f$\lVert A v - \theta v \rVert\f$, skipping the denominator for orthonormal \f$V\f$, and blocked
evaluation for very large \f$k\f$.

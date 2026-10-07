# Ritz values (Rayleigh quotients) {#guide_ritz_values}

> **Covers:** the `ritz_values` extension: what it computes, its two C++
> spellings, workspace sizing, and the traps. **Status:** current; signatures
> verified against `include/batchlas/blas/extensions.hh:1621-1672` and the
> implementation in `src/extensions/ritz_values.cc` on 2026-09-30.
> Migrated from the root `RITZ_VALUES.md` and `IMPLEMENTATION_SUMMARY.md`.

## What `ritz_values` computes

For a square matrix \f$A\f$ (dense or CSR) and a dense block of trial vectors
\f$V = [v_1, \dots, v_k]\f$, `ritz_values` returns, for every column and every
batch item, the Rayleigh quotient

\f[
  \theta_j \;=\; \operatorname{Re}\,\frac{v_j^H A\, v_j}{v_j^H v_j}, \qquad j = 1, \dots, k .
\f]

For real types \f$v_j^H = v_j^T\f$. The quotient is computed in the full scalar
type and only its real part is stored, so the output is always real
(`float` for `std::complex<float>`, `double` for `std::complex<double>`).

When \f$v_j\f$ is close to an eigenvector of a Hermitian \f$A\f$, \f$\theta_j\f$
is a second-order accurate approximation of its eigenvalue, which is why the
routine is used to check eigenvectors: `stedc_tests` recomputes eigenvalues from
the returned eigenvectors this way, and the Python facade exposes the same
operation as `batchlas.ritz_values(a, vectors)`.

**Each column is independent.** This is the diagonal of
\f$V^H A V\f$ divided by the diagonal of \f$V^H V\f$, not the eigenvalues of the
projected matrix \f$V^H A V\f$ (a Rayleigh–Ritz step). The two agree only when
the columns of \f$V\f$ are eigenvectors of the projection, for example exact
eigenvectors of \f$A\f$. The columns need not be normalised (the denominator
takes care of that) and need not be orthogonal.

> **Corrected claim.** `RITZ_VALUES.md` listed "compute eigenvalues of the
> projected matrix \f$V^T A V\f$ in subspace methods" as a use case and advised
> orthogonal trial vectors "for best results". Neither holds for a per-column
> quotient; for a Rayleigh–Ritz projection, form \f$V^H A V\f$ with `gemm` and
> call `syev` on it.

## How it is computed

Two steps (`src/extensions/ritz_values.cc`):

1. \f$W = A V\f$ into workspace, with `gemm` for a dense \f$A\f$ and `spmm` for
   CSR; `spmm` gets its own workspace carved from the same buffer.
2. One kernel with one work-group per (batch item, column) pair, grid-strided:
   each work-group reduces \f$v_j^H w_j\f$ and \f$v_j^H v_j\f$ through local
   memory and work-item 0 writes \f$\operatorname{Re}(\text{num}/\text{den})\f$.

## Calling it

### The allocating convenience overload

The shortest spelling allocates the output vector and the workspace itself and
**blocks** (`ctx.wait()` before and after), so the returned values are readable
immediately:

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

It is declared for `MatrixView` arguments, plus one owning-`Matrix` twin that
exists only so that the partially-explicit spelling `ritz_values<B, T>(ctx, A, V)`
compiles (`extensions.hh:1646-1664` explains why the generic owning forwarder
cannot take it). Because it waits on the queue and allocates on every call, keep
it out of loops; use the workspace form there.

### The workspace form

```cpp
Vector<float> theta(k, batch);   // one value per column per batch item

size_t bytes = ritz_values_buffer_size(ctx, A, V, theta);
UnifiedVector<std::byte> ws(bytes);
ritz_values(ctx, A, V, theta, ws);   // backend taken from the queue
ctx.wait();                          // theta is not readable before this
```

The primary declarations are

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

and, like every entry point, both are also reachable without the backend
(`BATCHLAS_DISPATCH_ON_QUEUE`) and with owning `Matrix` / `Vector` arguments
(`BATCHLAS_ACCEPT_OWNING`). The spelling `ritz_values_workspace` is a deprecated
alias of `ritz_values_buffer_size`, kept so an out-of-tree caller gets a warning
instead of a link error.

For a sparse \f$A\f$ the call is the same with a CSR matrix, e.g.
`auto As = A.convert_to<MatrixFormat::CSR>();`; `MFormat` follows the argument.

## Parameters and layout

| parameter | meaning |
| --- | --- |
| `A` | \f$n \times n\f$ per batch item, `Dense` or `CSR`; must be square |
| `V` | dense \f$n \times k\f$ per batch item, same batch count as `A`; columns are the trial vectors |
| `ritz_vals` | `VectorView` of the real type, size \f$\ge k\f$, same batch count; entry \f$j\f$ of item \f$b\f$ is `ritz_vals(j, b)` |
| `workspace` | at least `ritz_values_buffer_size(...)` bytes: \f$A V\f$ (\f$n k \cdot \text{batch}\f$ scalars), a batch pointer array, and `spmm`'s workspace for CSR |

Shape mismatches are `assert`s in the implementation, so a release build does
not diagnose them.

## Supported configurations

- **Scalar types:** `float`, `double`, `std::complex<float>`, `std::complex<double>`.
- **Formats of `A`:** `Dense`, `CSR`.
- **Backends:** `NETLIB`, `CUDA`, `ROCM`, `MKL`, each only when compiled in
  (`src/extensions/ritz_values.cc:202-216`).

> **Corrected claim.** The earlier documents listed `SYCL` among the backends
> and wrote every example as `ritz_values<Backend::SYCL, ...>`. There is no
> `Backend::SYCL` instantiation, so those examples would not link.

## Tests

`tests/ritz_values_tests.cc` checks a diagonal matrix with identity trial
vectors, a \f$10 \times 10\f$ tridiagonal Toeplitz matrix
(\f$a = 2, b = c = -1\f$) against its analytic eigenpairs
\f$\lambda_j = a - 2\sqrt{bc}\cos\!\big(j\pi/(n+1)\big)\f$ at batch 2, over
every compiled backend and scalar type. Both are dense: the `SparseCSRMatrix`
case that `IMPLEMENTATION_SUMMARY.md` described is not in the file today, so the
CSR path of `ritz_values` itself has no test. (`spmm_tests` has an nrhs ladder
that names `ritz_values` as a consumer, `tests/spmm_tests.cc:813`, but it tests
`spmm`, not the Rayleigh-quotient kernel.) Adding a CSR case is an open debt.

Ideas recorded when the routine was added and never implemented: an optional
residual norm \f$\lVert A v - \theta v \rVert\f$, skipping the denominator for
orthonormal \f$V\f$, kernel work for very large \f$k\f$, and blocked evaluation
for memory-constrained cases.

`IMPLEMENTATION_SUMMARY.md` also named Lanczos and LOBPCG (`syevx`) as
integration points. Neither calls `ritz_values` today: `lanczos` and
`syevx_lobpcg.cc` compute their Ritz values internally (the LOBPCG
`store_ritz_values` instrumentation flag records its own history), so the
routine's in-tree callers are the tests and the Python binding.

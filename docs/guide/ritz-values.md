# Ritz values (Rayleigh quotients) {#guide_ritz_values}

> **Status:** current

`ritz_values` returns the Rayleigh quotient of every trial-vector column, for dense or CSR
operators and batched inputs. Signatures: the @ref api_eigen group.

## What it computes

For square \f$A\f$ (dense or CSR) and a dense block \f$V = [v_1, \dots, v_k]\f$, per column and
batch item:

\f[
  \theta_j \;=\; \operatorname{Re}\,\frac{v_j^H A\, v_j}{v_j^H v_j}, \qquad j = 1, \dots, k .
\f]

- The quotient is formed in the full scalar type and only the real part is stored, so the output
  is `float` for `std::complex<float>` and `double` for `std::complex<double>`.
- Columns are independent: the result is \f$\operatorname{diag}(V^H A V)\f$ divided by
  \f$\operatorname{diag}(V^H V)\f$, not the eigenvalues of \f$V^H A V\f$. Columns need not be
  normalised or orthogonal. For a Rayleigh-Ritz projection, form \f$V^H A V\f$ with `gemm` and
  call `syev` on it.
- \f$W = A V\f$ goes through `gemm` (dense) or `spmm` (CSR), then one work-group per (item,
  column) reduces the two dot products.

## Calling it

The allocating overload allocates output and workspace and blocks; keep it out of loops.

```cpp
Vector<float> theta = ritz_values<Backend::CUDA, float>(ctx, A, V);
float theta_j_b = theta(j, b);   // column j of batch item b
```

The workspace form is asynchronous:

```cpp
Vector<float> theta(k, batch);
size_t bytes = ritz_values_buffer_size(ctx, A, V, theta);
UnifiedVector<std::byte> ws(bytes);
ritz_values(ctx, A, V, theta, ws);   // backend taken from the queue
ctx.wait();                          // theta is not readable before this
```

For sparse \f$A\f$ pass a CSR matrix (`A.convert_to<MatrixFormat::CSR>()`). Python:
`batchlas.ritz_values(a, vectors)`.

- Shape mismatches fail an `assert`, so a release build does not diagnose them.
- `Backend::SYCL` instantiates `float` and `complex<float>` only unless the build has device fp64.
